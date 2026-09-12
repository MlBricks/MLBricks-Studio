from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import json
import ast
import math
import copy
import random
import re
import time
import inspect
import itertools
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .import_pool import IMPORT_POOL
from .api_graph_runtime import API_COMPONENTS
from .security import safe_torch_load
from .version import __version__


# Tokenizers are immutable during Studio inference. Reusing the already parsed
# tokenizer avoids repeated Hugging Face startup work on every cold generation.
_TOKENIZER_CACHE: dict[str, Any] = {}


class ModelCompileError(RuntimeError):
    pass


class TrainingStopped(RuntimeError):
    pass


class ExistingModelArtifactError(RuntimeError):
    """Existing model weights could not be loaded safely for retraining."""

    def __init__(self, path, message):
        self.path = str(path)
        super().__init__(message)


def _bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _none(v: Any):
    if v is None: return None
    if isinstance(v, str) and v.strip().lower() in {"", "none", "null"}: return None
    return v


def _literal_or_text(value: Any):
    """Parse JSON/Python literals from Builder text fields without eval()."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(text)
        except Exception:
            pass
    return text


def _scalar_or_sequence(value: Any, *, cast: Callable[[Any], Any], label: str):
    parsed = _literal_or_text(value)
    if isinstance(parsed, str) and "," in parsed:
        parsed = [part.strip() for part in parsed.split(",") if part.strip()]
    if isinstance(parsed, (list, tuple)):
        try:
            return [cast(item) for item in parsed]
        except Exception as exc:
            raise ValueError(f"{label} contains an invalid value: {parsed!r}") from exc
    try:
        return cast(parsed)
    except Exception as exc:
        raise ValueError(f"{label} is invalid: {value!r}") from exc


def _config_value(value: Any, *, label: str):
    parsed = _literal_or_text(value)
    if parsed is None:
        return None
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, (list, tuple)) and all(isinstance(item, dict) for item in parsed):
        return list(parsed)
    raise ValueError(f"{label} must be a JSON object or a list of JSON objects.")


def _missing_number(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def runtime_int(
    value: Any,
    default: int | None,
    label: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Convert runtime/UI values without leaking int(None) to the user."""
    if _missing_number(value):
        if default is None:
            raise ValueError(f"{label} is required.")
        value = default
    try:
        number = int(float(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a number; received {value!r}.") from exc
    if minimum is not None and number < minimum:
        raise ValueError(f"{label} must be at least {minimum}; received {number}.")
    if maximum is not None and number > maximum:
        raise ValueError(f"{label} must be at most {maximum}; received {number}.")
    return number


def runtime_float(
    value: Any,
    default: float | None,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Float counterpart to runtime_int with field-specific errors."""
    if _missing_number(value):
        if default is None:
            raise ValueError(f"{label} is required.")
        value = default
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a number; received {value!r}.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite; received {value!r}.")
    if minimum is not None and number < minimum:
        raise ValueError(f"{label} must be at least {minimum}; received {number}.")
    if maximum is not None and number > maximum:
        raise ValueError(f"{label} must be at most {maximum}; received {number}.")
    return number


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "model")).strip("-.")
    return value or "model"


def resolve_device(requested: str | None) -> torch.device:
    value = str(requested or "auto").strip().lower()
    if value == "auto":
        if torch.cuda.is_available(): return torch.device("cuda:0")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available(): return torch.device("mps")
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available(): return torch.device("xpu:0")
        return torch.device("cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"{value} was selected but CUDA is unavailable.")
    return device


def resolve_precision(name: str | None, device: torch.device) -> tuple[str, torch.dtype | None]:
    value = str(name or "auto").strip().lower()
    if value == "auto":
        value = "fp16" if device.type == "cuda" else "fp32"
    # Float16 CPU kernels are frequently unsupported or dramatically slower
    # than float32. Saved GPU-oriented projects may still request fp16 after
    # Auto falls back to CPU, so make that fallback safe and usable.
    if device.type == "cpu" and value == "fp16":
        value = "fp32"
    mapping = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if value not in mapping:
        raise ValueError(f"Unsupported precision: {name!r}")
    return value, mapping[value]


def _topological(nodes: list[dict], edges: list[dict]) -> list[dict]:
    by_id={n["id"]:n for n in nodes}
    incoming={n["id"]:0 for n in nodes}
    outgoing={n["id"]:[] for n in nodes}
    for e in edges:
        a,b=e.get("source"),e.get("target")
        if a in by_id and b in by_id:
            outgoing[a].append(b); incoming[b]+=1
    q=[n["id"] for n in nodes if incoming[n["id"]]==0]
    order=[]
    while q:
        nid=q.pop(0); order.append(by_id[nid])
        for nxt in outgoing[nid]:
            incoming[nxt]-=1
            if incoming[nxt]==0:q.append(nxt)
    if len(order)!=len(nodes): raise ModelCompileError("Graph contains a cycle.")
    return order


class _Identity(nn.Module):
    def forward(self,x): return x


def _shape_tuple(value: Any, *, label: str, allow_zero: bool = False) -> tuple[int, ...]:
    """Parse Builder shape fields such as ``"4, 8"`` without eval()."""
    parsed = _literal_or_text(value)
    if isinstance(parsed, str):
        cleaned = parsed.lower().replace("x", ",").replace(" ", "")
        parsed = [part for part in cleaned.split(",") if part != ""]
    if isinstance(parsed, (int, float)):
        parsed = [parsed]
    if not isinstance(parsed, (list, tuple)) or not parsed:
        raise ValueError(f"{label} must contain one or more dimensions, for example 4,1.")
    dims = []
    for raw in parsed:
        try:
            dim = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} contains an invalid dimension {raw!r}.") from exc
        if allow_zero:
            if dim < -1:
                raise ValueError(f"{label} dimensions must be -1, 0, or positive integers.")
        elif dim < 1:
            raise ValueError(f"{label} dimensions must be positive integers.")
        dims.append(dim)
    if allow_zero and dims.count(-1) > 1:
        raise ValueError(f"{label} may contain at most one inferred (-1) dimension.")
    return tuple(dims)


class _LearnableTensor(nn.Module):
    """Source node that exposes a trainable tensor to the visual graph."""
    def __init__(self, shape, *, init="normal", scale=0.02):
        super().__init__()
        shape = tuple(int(v) for v in shape)
        init = str(init or "normal").strip().lower()
        scale = float(scale)
        if init == "zeros":
            value = torch.zeros(shape)
        elif init == "ones":
            value = torch.ones(shape)
        elif init == "uniform":
            value = torch.empty(shape).uniform_(-abs(scale), abs(scale))
        elif init == "normal":
            value = torch.empty(shape).normal_(mean=0.0, std=abs(scale))
        else:
            raise ValueError("Learnable Parameter initialization must be zeros, ones, normal, or uniform.")
        self.value = nn.Parameter(value)

    def forward(self, _x):
        return self.value


class _ConstantTensor(nn.Module):
    """Source node that exposes a fixed tensor constant."""
    def __init__(self, shape, *, value=0.0):
        super().__init__()
        self.register_buffer("value", torch.full(tuple(int(v) for v in shape), float(value)))

    def forward(self, _x):
        return self.value


class _LogisticRegression(nn.Module):
    def __init__(self, in_features: int, out_features: int = 1, *, bias=True, output="probability"):
        super().__init__()
        self.linear = nn.Linear(int(in_features), int(out_features), bias=bool(bias))
        output = str(output or "probability").strip().lower()
        if output not in {"probability", "logits"}:
            raise ValueError("Logistic Regression output must be probability or logits.")
        self.output = output

    def forward(self, x):
        logits = self.linear(x)
        return torch.sigmoid(logits) if self.output == "probability" else logits


class _PolynomialFeatures(nn.Module):
    """Small educational polynomial expansion for [B,D] or [...,D] tensors."""
    def __init__(self, degree=2, *, include_bias=False, interaction_only=False):
        super().__init__()
        self.degree = int(degree)
        self.include_bias = bool(include_bias)
        self.interaction_only = bool(interaction_only)
        if self.degree < 1 or self.degree > 4:
            raise ValueError("Polynomial Features degree must be between 1 and 4 for Studio graphs.")

    def forward(self, x):
        if x.ndim < 1:
            raise ValueError("Polynomial Features expects a tensor with a final feature dimension.")
        width = int(x.shape[-1])
        terms = []
        if self.include_bias:
            terms.append(torch.ones_like(x[..., :1]))
        terms.extend([x[..., i:i+1] for i in range(width)])
        for degree in range(2, self.degree + 1):
            iterator = (
                itertools.combinations(range(width), degree)
                if self.interaction_only
                else itertools.combinations_with_replacement(range(width), degree)
            )
            for combo in iterator:
                term = x[..., combo[0]:combo[0]+1]
                for idx in combo[1:]:
                    term = term * x[..., idx:idx+1]
                terms.append(term)
        return torch.cat(terms, dim=-1)




class _FittedBufferModule(nn.Module):
    """Resize fitted buffers before strict state-dict restore."""
    fitted_buffer_names = ()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for name in self.fitted_buffer_names:
            key = prefix + name
            if key in state_dict:
                current = getattr(self, name, None)
                loaded = state_dict[key]
                if isinstance(current, torch.Tensor) and tuple(current.shape) != tuple(loaded.shape):
                    setattr(self, name, torch.empty_like(loaded))
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class _KNNClassifier(_FittedBufferModule):
    """Small fit-based KNN classifier used by the educational Studio runtime.

    Fitted samples are buffers so ``state_dict``/MLBricks lifecycle persistence
    retains the complete classical model without a scikit-learn dependency.
    """
    fitted_buffer_names = ("train_x", "train_y", "classes_")
    def __init__(self, neighbors=5, *, weights="uniform", p=2.0):
        super().__init__()
        self.neighbors = int(neighbors)
        self.weights = str(weights or "uniform").strip().lower()
        self.p = float(p)
        if self.neighbors < 1:
            raise ValueError("KNN neighbors must be >= 1.")
        if self.weights not in {"uniform", "distance"}:
            raise ValueError("KNN voting must be uniform or distance.")
        if self.p <= 0:
            raise ValueError("KNN distance P must be > 0.")
        self.register_buffer("train_x", torch.empty(0, 0))
        self.register_buffer("train_y", torch.empty(0, dtype=torch.long))
        self.register_buffer("classes_", torch.empty(0, dtype=torch.long))

    @property
    def is_fitted(self):
        return self.train_x.numel() > 0 and self.train_y.numel() > 0

    @torch.no_grad()
    def fit(self, x, y):
        x = torch.as_tensor(x, dtype=torch.float32).detach().cpu()
        y = torch.as_tensor(y, dtype=torch.long).reshape(-1).detach().cpu()
        if x.ndim != 2:
            raise ValueError("KNN fit expects a 2D feature matrix [samples, features].")
        if x.size(0) != y.numel():
            raise ValueError("KNN feature and label sample counts do not match.")
        if x.size(0) < 1:
            raise ValueError("KNN needs at least one training sample.")
        self.train_x = x.contiguous()
        self.train_y = y.contiguous()
        self.classes_ = torch.unique(y, sorted=True)
        return self

    def forward(self, x):
        if not self.is_fitted:
            raise RuntimeError("KNN Classifier has not been fitted. Use Start Training/Fit first.")
        original_device = x.device if isinstance(x, torch.Tensor) else self.train_x.device
        x = torch.as_tensor(x, dtype=self.train_x.dtype, device=self.train_x.device)
        squeeze = x.ndim == 1
        if squeeze:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.size(-1) != self.train_x.size(-1):
            raise ValueError(
                f"KNN expected [B,{self.train_x.size(-1)}] features, received {tuple(x.shape)}."
            )
        distances = torch.cdist(x, self.train_x, p=self.p)
        k = min(self.neighbors, int(self.train_x.size(0)))
        nearest_dist, nearest_idx = torch.topk(distances, k=k, dim=1, largest=False)
        nearest_y = self.train_y[nearest_idx]
        classes = self.classes_
        votes = nearest_y.unsqueeze(-1).eq(classes.view(1, 1, -1))
        if self.weights == "distance":
            weights = 1.0 / nearest_dist.clamp_min(1e-12)
            score = (votes.to(weights.dtype) * weights.unsqueeze(-1)).sum(dim=1)
        else:
            score = votes.sum(dim=1)
        result = classes[score.argmax(dim=-1)].to(original_device)
        return result[0] if squeeze else result


class _DecisionTreeClassifier(_FittedBufferModule):
    """Compact CART-style classifier with tensor-buffer persistence."""
    fitted_buffer_names = ("feature_", "threshold_", "left_", "right_", "value_", "is_leaf_", "classes_", "feature_count_")
    def __init__(self, max_depth=5, *, min_samples_split=2, min_samples_leaf=1, criterion="gini"):
        super().__init__()
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.criterion = str(criterion or "gini").strip().lower()
        if self.max_depth < 1:
            raise ValueError("Decision Tree max depth must be >= 1.")
        if self.min_samples_split < 2:
            raise ValueError("Decision Tree min samples split must be >= 2.")
        if self.min_samples_leaf < 1:
            raise ValueError("Decision Tree min samples leaf must be >= 1.")
        if self.criterion not in {"gini", "entropy"}:
            raise ValueError("Decision Tree criterion must be gini or entropy.")
        self.register_buffer("feature_", torch.empty(0, dtype=torch.long))
        self.register_buffer("threshold_", torch.empty(0, dtype=torch.float32))
        self.register_buffer("left_", torch.empty(0, dtype=torch.long))
        self.register_buffer("right_", torch.empty(0, dtype=torch.long))
        self.register_buffer("value_", torch.empty(0, dtype=torch.long))
        self.register_buffer("is_leaf_", torch.empty(0, dtype=torch.bool))
        self.register_buffer("classes_", torch.empty(0, dtype=torch.long))
        self.register_buffer("feature_count_", torch.tensor(0, dtype=torch.long))

    @property
    def is_fitted(self):
        return self.feature_.numel() > 0

    def _impurity(self, labels):
        if labels.numel() == 0:
            return 0.0
        _, counts = torch.unique(labels, return_counts=True)
        probs = counts.float() / float(labels.numel())
        if self.criterion == "entropy":
            return float((-(probs * probs.clamp_min(1e-12).log2())).sum())
        return float(1.0 - (probs * probs).sum())

    @torch.no_grad()
    def fit(self, x, y):
        x = torch.as_tensor(x, dtype=torch.float32).detach().cpu()
        y = torch.as_tensor(y, dtype=torch.long).reshape(-1).detach().cpu()
        if x.ndim != 2:
            raise ValueError("Decision Tree fit expects a 2D feature matrix [samples, features].")
        if x.size(0) != y.numel():
            raise ValueError("Decision Tree feature and label sample counts do not match.")
        if x.size(0) < 1:
            raise ValueError("Decision Tree needs at least one training sample.")
        self.classes_ = torch.unique(y, sorted=True)
        features=[]; thresholds=[]; lefts=[]; rights=[]; values=[]; leaves=[]

        def majority(labels):
            classes, counts = torch.unique(labels, return_counts=True)
            return int(classes[counts.argmax()].item())

        def build(indices, depth):
            node_index=len(features)
            labels=y[indices]
            features.append(-1); thresholds.append(0.0); lefts.append(-1); rights.append(-1)
            values.append(majority(labels)); leaves.append(True)
            if depth >= self.max_depth or indices.numel() < self.min_samples_split or torch.unique(labels).numel() <= 1:
                return node_index
            parent_impurity=self._impurity(labels)
            best=None
            for feature in range(x.size(1)):
                vals=x[indices,feature]
                unique=torch.unique(vals,sorted=True)
                if unique.numel() <= 1:
                    continue
                thresholds_here=(unique[:-1]+unique[1:])*0.5
                if thresholds_here.numel()>64:
                    pick=torch.linspace(0,thresholds_here.numel()-1,steps=64).round().long().unique()
                    thresholds_here=thresholds_here[pick]
                for threshold in thresholds_here:
                    mask=vals <= threshold
                    n_left=int(mask.sum())
                    n_right=int(mask.numel()-n_left)
                    if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                        continue
                    left_idx=indices[mask]; right_idx=indices[~mask]
                    weighted=(n_left*self._impurity(y[left_idx])+n_right*self._impurity(y[right_idx]))/float(mask.numel())
                    gain=parent_impurity-weighted
                    if best is None or gain>best[0]+1e-12:
                        best=(gain,feature,float(threshold),left_idx,right_idx)
            if best is None or best[0] <= 1e-12:
                return node_index
            _,feature,threshold,left_idx,right_idx=best
            leaves[node_index]=False;features[node_index]=feature;thresholds[node_index]=threshold
            lefts[node_index]=build(left_idx,depth+1)
            rights[node_index]=build(right_idx,depth+1)
            return node_index

        build(torch.arange(x.size(0),dtype=torch.long),0)
        self.feature_=torch.tensor(features,dtype=torch.long)
        self.threshold_=torch.tensor(thresholds,dtype=torch.float32)
        self.left_=torch.tensor(lefts,dtype=torch.long)
        self.right_=torch.tensor(rights,dtype=torch.long)
        self.value_=torch.tensor(values,dtype=torch.long)
        self.is_leaf_=torch.tensor(leaves,dtype=torch.bool)
        self.feature_count_=torch.tensor(int(x.size(1)),dtype=torch.long)
        return self

    def forward(self, x):
        if not self.is_fitted:
            raise RuntimeError("Decision Tree has not been fitted. Use Start Training/Fit first.")
        original_device=x.device if isinstance(x,torch.Tensor) else self.feature_.device
        x=torch.as_tensor(x,dtype=torch.float32,device=self.feature_.device)
        squeeze=x.ndim==1
        if squeeze:x=x.unsqueeze(0)
        expected=int(self.feature_count_.item())
        if x.ndim!=2 or x.size(-1)!=expected:
            raise ValueError(f"Decision Tree expected [B,{expected}] features, received {tuple(x.shape)}.")
        outputs=[]
        # Traversal is intentionally explicit for Learn Mode readability.
        for row in x:
            node=0
            while not bool(self.is_leaf_[node].item()):
                feature=int(self.feature_[node].item())
                node=int((self.left_[node] if row[feature] <= self.threshold_[node] else self.right_[node]).item())
            outputs.append(self.value_[node])
        result=torch.stack(outputs).to(original_device)
        return result[0] if squeeze else result


class _KMeans(_FittedBufferModule):
    """Dependency-free K-Means with persisted centroids."""
    fitted_buffer_names = ("centroids_", "inertia_", "n_iter_")
    def __init__(self, clusters=3, *, max_iter=100, tolerance=1e-4, seed=42):
        super().__init__()
        self.clusters=int(clusters);self.max_iter=int(max_iter);self.tolerance=float(tolerance);self.seed=int(seed)
        if self.clusters < 1: raise ValueError("K-Means clusters must be >= 1.")
        if self.max_iter < 1: raise ValueError("K-Means max iterations must be >= 1.")
        if self.tolerance < 0: raise ValueError("K-Means tolerance must be >= 0.")
        self.register_buffer("centroids_",torch.empty(0,0))
        self.register_buffer("inertia_",torch.tensor(float("nan")))
        self.register_buffer("n_iter_",torch.tensor(0,dtype=torch.long))

    @property
    def is_fitted(self): return self.centroids_.numel()>0

    @torch.no_grad()
    def fit(self,x,y=None):
        x=torch.as_tensor(x,dtype=torch.float32).detach().cpu()
        if x.ndim!=2: raise ValueError("K-Means fit expects a 2D feature matrix [samples, features].")
        if x.size(0)<self.clusters: raise ValueError("K-Means needs at least as many samples as clusters.")
        gen=torch.Generator(device="cpu");gen.manual_seed(self.seed)
        first=int(torch.randint(0,x.size(0),(1,),generator=gen).item())
        chosen=[first]
        while len(chosen)<self.clusters:
            current=x[torch.tensor(chosen,dtype=torch.long)]
            min_sq=torch.cdist(x,current,p=2).pow(2).min(dim=1).values
            total=float(min_sq.sum())
            if total<=1e-12:
                remaining=[i for i in range(x.size(0)) if i not in chosen]
                chosen.append(remaining[0]);continue
            idx=int(torch.multinomial(min_sq/min_sq.sum(),1,generator=gen).item())
            if idx in chosen:
                remaining=[i for i in range(x.size(0)) if i not in chosen]
                idx=remaining[0]
            chosen.append(idx)
        centroids=x[torch.tensor(chosen,dtype=torch.long)].clone()
        iterations=0
        for iteration in range(1,self.max_iter+1):
            distances=torch.cdist(x,centroids,p=2)
            labels=distances.argmin(dim=1)
            updated=[]
            for cluster in range(self.clusters):
                members=x[labels==cluster]
                updated.append(members.mean(dim=0) if members.numel() else centroids[cluster])
            updated=torch.stack(updated)
            shift=float(torch.norm(updated-centroids,dim=1).max())
            centroids=updated;iterations=iteration
            if shift<=self.tolerance:break
        final_dist=torch.cdist(x,centroids,p=2)
        final_labels=final_dist.argmin(dim=1)
        inertia=((x-centroids[final_labels])**2).sum()
        self.centroids_=centroids.contiguous()
        self.inertia_=inertia.float()
        self.n_iter_=torch.tensor(iterations,dtype=torch.long)
        return self

    def forward(self,x):
        if not self.is_fitted: raise RuntimeError("K-Means has not been fitted. Use Start Training/Fit first.")
        original_device=x.device if isinstance(x,torch.Tensor) else self.centroids_.device
        x=torch.as_tensor(x,dtype=self.centroids_.dtype,device=self.centroids_.device)
        squeeze=x.ndim==1
        if squeeze:x=x.unsqueeze(0)
        if x.ndim!=2 or x.size(-1)!=self.centroids_.size(-1):
            raise ValueError(f"K-Means expected [B,{self.centroids_.size(-1)}] features, received {tuple(x.shape)}.")
        result=torch.cdist(x,self.centroids_,p=2).argmin(dim=1).to(original_device)
        return result[0] if squeeze else result


class _PCA(_FittedBufferModule):
    """Principal component analysis fitted with torch.linalg.svd."""
    fitted_buffer_names = ("mean_", "components_", "explained_variance_", "explained_variance_ratio_")
    def __init__(self, components=2, *, center=True, whiten=False):
        super().__init__()
        self.components=int(components);self.center=bool(center);self.whiten=bool(whiten)
        if self.components < 1: raise ValueError("PCA components must be >= 1.")
        self.register_buffer("mean_",torch.empty(0))
        self.register_buffer("components_",torch.empty(0,0))
        self.register_buffer("explained_variance_",torch.empty(0))
        self.register_buffer("explained_variance_ratio_",torch.empty(0))

    @property
    def is_fitted(self): return self.components_.numel()>0

    @torch.no_grad()
    def fit(self,x,y=None):
        x=torch.as_tensor(x,dtype=torch.float32).detach().cpu()
        if x.ndim!=2: raise ValueError("PCA fit expects a 2D feature matrix [samples, features].")
        if x.size(0)<2: raise ValueError("PCA needs at least two samples.")
        k=min(self.components,int(x.size(0)),int(x.size(1)))
        if k<1: raise ValueError("PCA could not resolve a valid component count.")
        mean=x.mean(dim=0) if self.center else torch.zeros(x.size(1),dtype=x.dtype)
        centered=x-mean
        _,singular,vh=torch.linalg.svd(centered,full_matrices=False)
        variance=(singular**2)/max(int(x.size(0))-1,1)
        total=variance.sum().clamp_min(1e-12)
        self.mean_=mean.contiguous()
        self.components_=vh[:k].contiguous()
        self.explained_variance_=variance[:k].contiguous()
        self.explained_variance_ratio_=(variance[:k]/total).contiguous()
        return self

    def forward(self,x):
        if not self.is_fitted: raise RuntimeError("PCA has not been fitted. Use Start Training/Fit first.")
        original_device=x.device if isinstance(x,torch.Tensor) else self.components_.device
        x=torch.as_tensor(x,dtype=self.components_.dtype,device=self.components_.device)
        squeeze=x.ndim==1
        if squeeze:x=x.unsqueeze(0)
        if x.ndim!=2 or x.size(-1)!=self.components_.size(-1):
            raise ValueError(f"PCA expected [B,{self.components_.size(-1)}] features, received {tuple(x.shape)}.")
        transformed=(x-self.mean_)@self.components_.transpose(0,1)
        if self.whiten:
            transformed=transformed/self.explained_variance_.sqrt().clamp_min(1e-12)
        transformed=transformed.to(original_device)
        return transformed[0] if squeeze else transformed


class _FlexibleBatchNorm1d(nn.Module):
    """BatchNorm1d that can treat the final dimension as the feature axis."""
    def __init__(self, num_features, *, eps=1e-5, momentum=0.1, layout="features_last"):
        super().__init__()
        self.num_features = int(num_features)
        self.layout = str(layout or "features_last").strip().lower()
        if self.layout not in {"features_last", "channels_first"}:
            raise ValueError("BatchNorm 1D layout must be features_last or channels_first.")
        self.norm = nn.BatchNorm1d(self.num_features, eps=float(eps), momentum=float(momentum))

    def forward(self, x):
        if self.layout == "channels_first":
            return self.norm(x)
        if x.shape[-1] != self.num_features:
            raise ValueError(
                f"BatchNorm 1D expected final feature width {self.num_features}, received {x.shape[-1]}."
            )
        original = x.shape
        y = self.norm(x.reshape(-1, self.num_features))
        return y.reshape(original)


class _SequenceRecurrent(nn.Module):
    """Unified educational RNN/LSTM/GRU wrapper with batch-first input."""
    def __init__(self, kind, input_size, hidden_size, *, num_layers=1, dropout=0.0,
                 bidirectional=False, output="sequence", nonlinearity="tanh"):
        super().__init__()
        kind = str(kind).lower()
        layers = int(num_layers)
        hidden = int(hidden_size)
        input_size = int(input_size)
        if layers < 1 or hidden < 1 or input_size < 1:
            raise ValueError("RNN/LSTM/GRU sizes and layer count must be positive.")
        effective_dropout = float(dropout) if layers > 1 else 0.0
        common = dict(
            input_size=input_size, hidden_size=hidden, num_layers=layers,
            batch_first=True, dropout=effective_dropout,
            bidirectional=bool(bidirectional),
        )
        if kind == "rnn":
            self.cell = nn.RNN(nonlinearity=str(nonlinearity or "tanh"), **common)
        elif kind == "lstm":
            self.cell = nn.LSTM(**common)
        elif kind == "gru":
            self.cell = nn.GRU(**common)
        else:
            raise ValueError(f"Unsupported recurrent type: {kind}")
        self.output = str(output or "sequence").strip().lower()
        if self.output not in {"sequence", "last"}:
            raise ValueError("Recurrent Output must be sequence or last.")

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("RNN/LSTM/GRU expects [batch, sequence, features] input.")
        sequence, _state = self.cell(x)
        return sequence if self.output == "sequence" else sequence[:, -1, :]


class _SelfAttention(nn.Module):
    def __init__(self, dim, heads, *, dropout=0.0, causal=False, bias=True):
        super().__init__()
        self.dim = int(dim)
        self.heads = int(heads)
        if self.dim < 1 or self.heads < 1 or self.dim % self.heads:
            raise ValueError("Self Attention embedding dim must be positive and divisible by Heads.")
        self.causal = bool(causal)
        self.attn = nn.MultiheadAttention(
            self.dim, self.heads, dropout=float(dropout), bias=bool(bias), batch_first=True
        )

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError(f"Self Attention expects [B,T,{self.dim}] input.")
        mask = None
        if self.causal:
            length = int(x.shape[1])
            mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1)
        y, _weights = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        return y


class _TensorBinaryOp(nn.Module):
    def __init__(self, op, *, dim=-1, epsilon=0.0):
        super().__init__()
        self.op = str(op)
        self.dim = int(dim)
        self.epsilon = float(epsilon)

    def forward(self, a, b):
        if self.op == "matmul": return torch.matmul(a, b)
        if self.op == "add": return a + b
        if self.op == "subtract": return a - b
        if self.op == "multiply": return a * b
        if self.op == "divide": return a / (b + self.epsilon)
        if self.op == "concat": return torch.cat((a, b), dim=self.dim)
        raise RuntimeError(f"Unknown tensor binary operation: {self.op}")


class _TensorReduce(nn.Module):
    def __init__(self, op, *, dim=-1, keepdim=False):
        super().__init__()
        self.op = str(op)
        self.dim = int(dim)
        self.keepdim = bool(keepdim)

    def forward(self, x):
        if self.op == "mean": return torch.mean(x, dim=self.dim, keepdim=self.keepdim)
        if self.op == "sum": return torch.sum(x, dim=self.dim, keepdim=self.keepdim)
        if self.op == "max": return torch.amax(x, dim=self.dim, keepdim=self.keepdim)
        if self.op == "min": return torch.amin(x, dim=self.dim, keepdim=self.keepdim)
        raise RuntimeError(f"Unknown tensor reduction: {self.op}")


class _SafeUnary(nn.Module):
    def __init__(self, op, *, epsilon=0.0):
        super().__init__()
        self.op = str(op)
        self.epsilon = float(epsilon)

    def forward(self, x):
        if self.op == "exp": return torch.exp(x)
        if self.op == "log": return torch.log(x.clamp_min(self.epsilon)) if self.epsilon > 0 else torch.log(x)
        if self.op == "sqrt": return torch.sqrt(x.clamp_min(self.epsilon)) if self.epsilon > 0 else torch.sqrt(x)
        raise RuntimeError(f"Unknown tensor unary operation: {self.op}")


class _Transpose(nn.Module):
    def __init__(self, dim0=-2, dim1=-1):
        super().__init__(); self.dim0=int(dim0); self.dim1=int(dim1)
    def forward(self, x): return x.transpose(self.dim0, self.dim1)


class _Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__(); self.shape=tuple(int(v) for v in shape)
    def forward(self, x):
        resolved=[]
        for index, dim in enumerate(self.shape):
            if dim == 0:
                if index >= x.ndim:
                    raise ValueError(f"Reshape dimension {index} uses 0 but input has only {x.ndim} dimensions.")
                resolved.append(int(x.shape[index]))
            else:
                resolved.append(dim)
        return x.reshape(tuple(resolved))


class _Unsqueeze(nn.Module):
    def __init__(self, dim=1): super().__init__(); self.dim=int(dim)
    def forward(self, x): return x.unsqueeze(self.dim)


class _Squeeze(nn.Module):
    def __init__(self, dim=None): super().__init__(); self.dim=dim
    def forward(self, x): return x.squeeze() if self.dim is None else x.squeeze(int(self.dim))


class _ClassifierHead(nn.Module):
    """Trainable Studio classification head.

    Two-dimensional input ``[B,D]`` is projected directly. Sequence input
    ``[B,T,D]`` is mean-pooled across T before projection so the node produces
    one class-score vector per sample.
    """

    def __init__(self, dim: int, classes: int):
        super().__init__()
        self.dim = int(dim)
        self.classes = int(classes)
        if self.dim < 1:
            raise ValueError("Classifier hidden dim must be >= 1.")
        if self.classes < 1:
            raise ValueError("Classifier classes must be >= 1.")
        self.proj = nn.Linear(self.dim, self.classes)

    def forward(self, x):
        if x.ndim == 3:
            x = x.mean(dim=-2)
        elif x.ndim != 2:
            raise ValueError("Classifier Head expects [B,D] or [B,T,D] input.")

        if x.size(-1) != self.dim:
            raise ValueError(
                f"Classifier Head expected feature width {self.dim}, "
                f"received {x.size(-1)}."
            )
        return self.proj(x)


class _FPNFusion(nn.Module):
    """Small reusable top-down feature-pyramid fusion block."""
    def __init__(self, high_channels=64, lateral_channels=32, out_channels=32, fusion="add"):
        super().__init__()
        self.fusion = str(fusion or "add").lower()
        if self.fusion not in {"add", "concat"}:
            raise ValueError("FPN fusion must be 'add' or 'concat'.")
        self.high_proj = nn.Conv2d(int(high_channels), int(out_channels), kernel_size=1)
        self.lateral_proj = nn.Conv2d(int(lateral_channels), int(out_channels), kernel_size=1)
        merged = int(out_channels) if self.fusion == "add" else int(out_channels) * 2
        self.smooth = nn.Conv2d(merged, int(out_channels), kernel_size=3, padding=1)

    def forward(self, high, lateral):
        if high.ndim != 4 or lateral.ndim != 4:
            raise ValueError("FPN Fusion expects [B,C,H,W] tensors for High and Lateral inputs.")
        high = self.high_proj(high)
        high = F.interpolate(high, size=lateral.shape[-2:], mode="nearest")
        lateral = self.lateral_proj(lateral)
        fused = high + lateral if self.fusion == "add" else torch.cat([high, lateral], dim=1)
        return self.smooth(fused)


class _PANFusion(nn.Module):
    """Small reusable bottom-up path-aggregation fusion block."""
    def __init__(self, fine_channels=32, coarse_channels=64, out_channels=64, fusion="concat"):
        super().__init__()
        self.fusion = str(fusion or "concat").lower()
        if self.fusion not in {"add", "concat"}:
            raise ValueError("PAN fusion must be 'add' or 'concat'.")
        self.fine_proj = nn.Conv2d(int(fine_channels), int(out_channels), kernel_size=1)
        self.coarse_proj = nn.Conv2d(int(coarse_channels), int(out_channels), kernel_size=1)
        merged = int(out_channels) if self.fusion == "add" else int(out_channels) * 2
        self.smooth = nn.Conv2d(merged, int(out_channels), kernel_size=3, padding=1)

    def forward(self, fine, coarse):
        if fine.ndim != 4 or coarse.ndim != 4:
            raise ValueError("PAN Fusion expects [B,C,H,W] tensors for Fine and Coarse inputs.")
        fine = F.adaptive_avg_pool2d(fine, output_size=coarse.shape[-2:])
        fine = self.fine_proj(fine)
        coarse = self.coarse_proj(coarse)
        fused = fine + coarse if self.fusion == "add" else torch.cat([fine, coarse], dim=1)
        return self.smooth(fused)


class _DetectionHead(nn.Module):
    """Anchor-free grid detection head with multiple prediction slots per cell.

    One slot keeps the historical ``[B, 5 + classes, H, W]`` layout for
    compatibility. Multiple slots return ``[B, slots, 5 + classes, H, W]`` so
    several objects may be assigned to the same spatial cell without requiring
    fixed anchor box shapes.
    """
    def __init__(self, in_channels=64, classes=3, anchors=1):
        super().__init__()
        self.in_channels = int(in_channels)
        self.classes = int(classes)
        self.anchors = max(1, int(anchors))
        if self.classes < 1:
            raise ValueError("Detection Head classes must be >= 1.")
        self.pred = nn.Conv2d(self.in_channels, self.anchors * (5 + self.classes), kernel_size=1)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Detection Head expects [B,C,H,W] feature maps.")
        if x.size(1) != self.in_channels:
            raise ValueError(f"Detection Head expected {self.in_channels} channels, received {x.size(1)}.")
        raw = self.pred(x)
        if self.anchors == 1:
            return raw
        b, _, h, w = raw.shape
        return raw.view(b, self.anchors, 5 + self.classes, h, w)


class _DetectionPyramidHead(nn.Module):
    """Three-scale anchor-free detector head for P3/P4/P5 feature maps."""
    def __init__(self, p3_channels=32, p4_channels=64, p5_channels=96, classes=3, slots=3):
        super().__init__()
        self.channels = (int(p3_channels), int(p4_channels), int(p5_channels))
        self.classes = int(classes)
        self.slots = max(1, int(slots))
        if self.classes < 1:
            raise ValueError("Detection Pyramid Head classes must be >= 1.")
        width = self.slots * (5 + self.classes)
        self.p3 = nn.Conv2d(self.channels[0], width, kernel_size=1)
        self.p4 = nn.Conv2d(self.channels[1], width, kernel_size=1)
        self.p5 = nn.Conv2d(self.channels[2], width, kernel_size=1)

    def _head(self, layer, x, expected, label):
        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            raise ValueError(f"Detection Pyramid Head {label} expects [B,C,H,W].")
        if int(x.size(1)) != int(expected):
            raise ValueError(f"Detection Pyramid Head {label} expected {expected} channels, received {x.size(1)}.")
        raw = layer(x)
        b, _, h, w = raw.shape
        return raw.view(b, self.slots, 5 + self.classes, h, w)

    def forward(self, p3, p4, p5):
        return (
            self._head(self.p3, p3, self.channels[0], "P3"),
            self._head(self.p4, p4, self.channels[1], "P4"),
            self._head(self.p5, p5, self.channels[2], "P5"),
        )


class _DetectionNMS(nn.Module):
    """Inference-only class-aware NMS block for raw detector predictions."""
    def __init__(self, score_threshold=0.25, iou_threshold=0.5, max_detections=100):
        super().__init__()
        self.score_threshold = float(score_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_detections = max(1, int(max_detections))

    def forward(self, prediction):
        return decode_detection_predictions(
            prediction,
            score_threshold=self.score_threshold,
            iou_threshold=self.iou_threshold,
            max_detections=self.max_detections,
        )


class _SignalFFT(nn.Module):
    """Differentiable real-FFT magnitude feature block for signal graphs."""
    def __init__(self, *, bins=16, log_scale=False, remove_dc=False, flatten_channels=True):
        super().__init__()
        self.bins = max(0, int(bins or 0))
        self.log_scale = bool(log_scale)
        self.remove_dc = bool(remove_dc)
        self.flatten_channels = bool(flatten_channels)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise ValueError("FFT Magnitude expects a tensor input.")
        if x.ndim not in {2, 3}:
            raise ValueError(f"FFT Magnitude expects [B,T] or [B,C,T], received {tuple(x.shape)}.")
        mag = torch.fft.rfft(x.float(), dim=-1).abs()
        if self.remove_dc and mag.size(-1) > 1:
            mag = mag[..., 1:]
        if self.bins:
            mag = mag[..., :min(self.bins, int(mag.size(-1)))]
        if self.log_scale:
            mag = torch.log1p(mag)
        if x.ndim == 3 and self.flatten_channels:
            mag = mag.flatten(1)
        return mag


class _JEPAMask(nn.Module):
    """Visible context/target masking primitive for educational JEPA graphs."""
    def __init__(self, *, mask_ratio=0.35, mask_value=0.0, mode="auto"):
        super().__init__()
        self.mask_ratio = float(mask_ratio)
        self.mask_value = float(mask_value)
        self.mode = str(mode or "auto").strip().lower()
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError("JEPA Mask ratio must be between 0 and 1.")
        if self.mode not in {"auto", "random", "contiguous", "spatiotemporal"}:
            raise ValueError("JEPA Mask mode must be auto, random, contiguous, or spatiotemporal.")

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise ValueError("JEPA Mask expects a tensor input.")
        target = x.clone()
        context = x.clone()
        ratio = self.mask_ratio

        if x.ndim == 2:  # text ids, audio windows, signal windows: [B,T]
            length = int(x.size(1))
            width = max(1, min(length, int(round(length * ratio))))
            for b in range(int(x.size(0))):
                start = int(torch.randint(0, max(1, length - width + 1), (1,), device=x.device).item())
                context[b, start:start + width] = self.mask_value
        elif x.ndim == 4:  # image [B,C,H,W]
            h, w = int(x.size(-2)), int(x.size(-1))
            side_ratio = math.sqrt(ratio)
            mh = max(1, min(h, int(round(h * side_ratio))))
            mw = max(1, min(w, int(round(w * side_ratio))))
            for b in range(int(x.size(0))):
                top = int(torch.randint(0, max(1, h - mh + 1), (1,), device=x.device).item())
                left = int(torch.randint(0, max(1, w - mw + 1), (1,), device=x.device).item())
                context[b, :, top:top + mh, left:left + mw] = self.mask_value
        elif x.ndim == 5:  # video [B,T,C,H,W]
            frames, h, w = int(x.size(1)), int(x.size(-2)), int(x.size(-1))
            mt = max(1, min(frames, int(round(frames * max(0.25, ratio)))))
            spatial_ratio = min(0.75, math.sqrt(ratio))
            mh = max(1, min(h, int(round(h * spatial_ratio))))
            mw = max(1, min(w, int(round(w * spatial_ratio))))
            for b in range(int(x.size(0))):
                ts = int(torch.randint(0, max(1, frames - mt + 1), (1,), device=x.device).item())
                top = int(torch.randint(0, max(1, h - mh + 1), (1,), device=x.device).item())
                left = int(torch.randint(0, max(1, w - mw + 1), (1,), device=x.device).item())
                context[b, ts:ts + mt, :, top:top + mh, left:left + mw] = self.mask_value
        else:
            mask = torch.rand_like(context.float()) < ratio
            context = context.masked_fill(mask, self.mask_value)
        return {"main": context, "context": context, "target": target}


class _JEPAEncoder(nn.Module):
    """Small modality-aware encoder shared by context and EMA target branches."""
    def __init__(self, *, modality="image", role="context", latent_dim=64, hidden_dim=64, vocab_size=257, in_channels=1):
        super().__init__()
        self.modality = str(modality or "image").strip().lower()
        self.role = str(role or "context").strip().lower()
        self.latent_dim = int(latent_dim)
        hidden = int(hidden_dim)
        if self.modality not in {"image", "video", "text", "audio", "signal"}:
            raise ValueError("JEPA Encoder modality must be image, video, text, audio, or signal.")
        if self.role not in {"context", "target"}:
            raise ValueError("JEPA Encoder role must be context or target.")
        if self.latent_dim < 1 or hidden < 1:
            raise ValueError("JEPA Encoder dimensions must be positive.")

        if self.modality in {"image", "video"}:
            channels = int(in_channels)
            self.visual = nn.Sequential(
                nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden, hidden, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.project = nn.Linear(hidden, self.latent_dim)
        elif self.modality == "text":
            self.embedding = nn.Embedding(int(vocab_size), hidden, padding_idx=0)
            self.norm = nn.LayerNorm(hidden)
            self.project = nn.Linear(hidden, self.latent_dim)
        else:
            self.temporal = nn.Sequential(
                nn.Conv1d(1, hidden, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(hidden, hidden, kernel_size=5, stride=2, padding=2),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.project = nn.Linear(hidden, self.latent_dim)

        if self.role == "target":
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    def forward(self, x):
        if self.modality == "image":
            if x.ndim == 3:
                x = x.unsqueeze(1)
            if x.ndim != 4:
                raise ValueError(f"Image JEPA Encoder expects [B,C,H,W], received {tuple(x.shape)}.")
            h = self.visual(x.float()).flatten(1)
            return self.project(h)
        if self.modality == "video":
            if x.ndim == 4:  # [B,T,H,W]
                x = x.unsqueeze(2)
            if x.ndim != 5:
                raise ValueError(f"Video JEPA Encoder expects [B,T,C,H,W], received {tuple(x.shape)}.")
            b, tt, c, h, w = x.shape
            features = self.visual(x.float().reshape(b * tt, c, h, w)).flatten(1)
            features = features.reshape(b, tt, -1).mean(dim=1)
            return self.project(features)
        if self.modality == "text":
            if x.ndim != 2:
                raise ValueError(f"Text JEPA Encoder expects token ids [B,T], received {tuple(x.shape)}.")
            ids = x.long().clamp(min=0, max=self.embedding.num_embeddings - 1)
            emb = self.embedding(ids)
            keep = (ids != 0).to(emb.dtype).unsqueeze(-1)
            denom = keep.sum(dim=1).clamp_min(1.0)
            pooled = (emb * keep).sum(dim=1) / denom
            return self.project(self.norm(pooled))
        if x.ndim == 3 and x.size(1) == 1:
            signal = x.float()
        elif x.ndim == 2:
            signal = x.float().unsqueeze(1)
        else:
            raise ValueError(f"{self.modality.title()} JEPA Encoder expects [B,T] or [B,1,T], received {tuple(x.shape)}.")
        return self.project(self.temporal(signal).squeeze(-1))


class _JEPAPredictor(nn.Module):
    def __init__(self, *, latent_dim=64, hidden_dim=128, dropout=0.0):
        super().__init__()
        latent_dim, hidden_dim = int(latent_dim), int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError(f"JEPA Predictor expects [B,D] latent vectors, received {tuple(x.shape)}.")
        return self.net(x)


class _JEPALatentLoss(nn.Module):
    def __init__(self, *, loss="mse", normalize=True):
        super().__init__()
        self.loss = str(loss or "mse").strip().lower()
        self.normalize = bool(normalize)
        if self.loss not in {"mse", "smooth_l1", "cosine"}:
            raise ValueError("JEPA Latent Loss must be mse, smooth_l1, or cosine.")

    def forward(self, prediction, target):
        target = target.detach()
        if tuple(prediction.shape) != tuple(target.shape):
            raise ValueError(f"JEPA latent shapes must match, got {tuple(prediction.shape)} vs {tuple(target.shape)}.")
        if self.normalize:
            prediction = F.normalize(prediction, dim=-1)
            target = F.normalize(target, dim=-1)
        if self.loss == "smooth_l1":
            return F.smooth_l1_loss(prediction, target)
        if self.loss == "cosine":
            return (1.0 - F.cosine_similarity(prediction, target, dim=-1)).mean()
        return F.mse_loss(prediction, target)


class _PreviousValueBuffer(nn.Module):
    """Builder utility for previous physical-depth tensors.

    ``hold`` keeps the connected tensor available as the previous-depth value
    for a later node. ``zero_init`` creates the correctly shaped zero vector
    used by the first recurrent physical depth. No trainable parameters or
    hidden mutable Python state are introduced.
    """
    def __init__(self, *, mode: str = "hold", width: int = 0):
        super().__init__()
        mode = str(mode or "hold").strip().lower()
        if mode not in {"hold", "zero_init"}:
            raise ValueError("Previous Value Buffer mode must be 'hold' or 'zero_init'.")
        self.mode = mode
        self.width = int(width or 0)
        if self.width < 0:
            raise ValueError("Previous Value Buffer width must be 0 (Auto) or a positive integer.")

    def forward(self, x):
        if self.mode == "hold":
            return x
        width = self.width or int(x.shape[-1])
        return x.new_zeros(*x.shape[:-1], width)


class _StateAwareESAStack(nn.Module):
    """Exact StateAware ESA depth stack used by the supplied 200M notebook."""
    def __init__(self, *, dim, state_dim, layers, heads, block, batch, depth_dim,
                 compass, backend, precision, update_ratio_start=0.20,
                 update_ratio_end=0.14, stream_ratio=1.08):
        super().__init__()
        # Resolve only the APIs this compound component needs. The shared
        # import pool prefers canonical submodules and caches them for reuse.
        ESA = IMPORT_POOL.resolve_component("esa")
        RMSNorm = IMPORT_POOL.resolve_component("rmsnorm")
        StateAwareFFN = IMPORT_POOL.resolve_component("saffn")
        ResController = IMPORT_POOL.resolve_component("rescontroller")
        self.dim=int(dim); self.state_dim=int(state_dim); self.layer_count=int(layers)
        if self.layer_count < 1: raise ValueError("StateAware ESA layers must be >= 1.")
        self.layers=nn.ModuleList(); self.write_gates=nn.ParameterList()
        for index in range(self.layer_count):
            depth=index/max(self.layer_count-1,1)
            ratio=float(update_ratio_start)+depth*(float(update_ratio_end)-float(update_ratio_start))
            self.layers.append(nn.ModuleDict({
                "norm": RMSNorm(self.dim),
                "esa": ESA(embd=self.dim, head=int(heads), batch=int(batch), block=int(block),
                           backend=backend, precision=precision, compass=int(compass), auto_compile=False),
                "ffn": StateAwareFFN(d_model=self.dim, state_dim=self.state_dim,
                                     depth_embedding_dim=int(depth_dim), layer_index=index,
                                     total_layers=self.layer_count, backend="pytorch"),
                "residual": ResController(update_ratio=ratio, stream_ratio=float(stream_ratio),
                                          update_softness=8.0, stream_softness=8.0, backend="pytorch"),
            }))
            self.write_gates.append(nn.Parameter(torch.tensor(-1.0)))
    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def forward(self,x):
        state=x.new_zeros(*x.shape[:-1],self.state_dim); previous_esa=torch.zeros_like(x)
        for layer,write_gate in zip(self.layers,self.write_gates):
            z=layer["norm"](x); esa_update=layer["esa"](z)
            ffn_update,state=layer["ffn"](z,esa_update,previous_esa,state)
            update=torch.sigmoid(write_gate)*(esa_update.float()+ffn_update.float())
            x=layer["residual"](x,update); previous_esa=esa_update
        return x


def _custom_coerce(value: Any, type_name: str, *, label: str) -> Any:
    kind = str(type_name or "str").lower()
    if kind in {"int", "integer", "number-int"}:
        try:
            return int(value)
        except Exception as exc:
            raise ModelCompileError(f"{label} must be an integer, got {value!r}.") from exc
    if kind in {"float", "number", "number-float"}:
        try:
            return float(value)
        except Exception as exc:
            raise ModelCompileError(f"{label} must be a number, got {value!r}.") from exc
    if kind in {"bool", "boolean"}:
        return _bool(value)
    if kind in {"json", "dict", "list", "tuple"}:
        parsed = _literal_or_text(value)
        if kind == "dict" and not isinstance(parsed, dict):
            raise ModelCompileError(f"{label} must be a JSON object.")
        if kind in {"list", "tuple"} and not isinstance(parsed, (list, tuple)):
            raise ModelCompileError(f"{label} must be a JSON list.")
        return tuple(parsed) if kind == "tuple" else parsed
    if kind in {"none", "null"}:
        return None
    return str(value) if value is not None else ""


def _bound_parameter_value(spec: dict[str, Any], params: dict[str, Any], runtime: dict[str, Any], x=None, skip=None, extra=None):
    name = str(spec.get("name") or spec.get("key") or "argument")
    source = str(spec.get("source") or "user").lower()
    if source in {"input", "main"}:
        return x
    if source == "skip":
        return skip
    if source == "extra":
        return extra
    if source == "device":
        return str(runtime.get("device") or "auto")
    if source in {"dtype", "precision"}:
        precision = str(runtime.get("precision") or "fp16")
        return {"fp16": torch.float16, "float16": torch.float16, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp32": torch.float32, "float32": torch.float32}.get(precision.lower(), precision)
    if source == "model_dim":
        raw = runtime.get("model_dim", params.get(name, spec.get("default")))
    elif source == "heads":
        raw = runtime.get("heads", params.get(name, spec.get("default")))
    elif source == "context":
        raw = runtime.get("context_length", params.get(name, spec.get("default")))
    elif source == "batch":
        raw = runtime.get("batch_size", params.get(name, spec.get("default")))
    else:
        raw = params.get(name, spec.get("default"))
    if (raw is None or (isinstance(raw, str) and raw == "")) and not spec.get("required"):
        return None
    return _custom_coerce(raw, str(spec.get("type") or "str"), label=name)


def _api_binding_import_path(binding: dict[str, Any]) -> str:
    path = str(binding.get("import_path") or "").strip()
    if path:
        return path
    module = str(binding.get("module_path") or "").strip().strip(".")
    symbol = str(binding.get("symbol") or "").strip().strip(".")
    return ".".join(part for part in (module, symbol) if part)


class _LayerBlock(nn.Module):
    """Explicit ESA layer with separate Signal and Residual lanes.

    Studio historically flattened nested layers into one main tensor lane.  This
    block keeps the residual stream explicit at the graph boundary so layer-to-
    layer wiring cannot lose or reinterpret the skip path.  Signal Out and
    Residual Out intentionally carry the same post-residual tensor for the next
    canonical Pre-LN block; they remain separate graph ports for architecture
    visibility and future state-aware variants.
    """

    def __init__(
        self, *, dim: int, heads: int, ffn_dim: int, batch: int, block: int,
        backend: str, precision: str, compass: int = 16, activation: str = "gelu",
        dropout: float = 0.0, norm_eps: float = 1e-5, device: str = "auto",
    ):
        super().__init__()
        LayerNorm = IMPORT_POOL.resolve_component("layernorm")
        ESA = IMPORT_POOL.resolve_component("esa")
        FFN = IMPORT_POOL.resolve_component("ffn")

        self.dim = int(dim)
        self.norm1 = LayerNorm(self.dim, eps=float(norm_eps), elementwise_affine=True, bias=True)
        self.esa = ESA(
            embd=self.dim, head=int(heads), batch=int(batch), block=int(block),
            backend=backend, precision=precision, compass=int(compass),
            dropout=float(dropout), gate_min=0.8, gate_max=0.995, eps=1e-5,
            device=device, auto_compile=False, auto_move_input=True, strict_checks=False,
        )
        self.norm2 = LayerNorm(self.dim, eps=float(norm_eps), elementwise_affine=True, bias=True)
        self.ffn = FFN(
            self.dim, int(ffn_dim), activation=activation, dropout=float(dropout),
            bias=True, gated=False,
        )
        self.update_dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

    @staticmethod
    def _result(value):
        # Named ports plus universal-lane aliases keep both explicit wiring and
        # older Main/Skip graph consumers compatible.
        return {
            "main": value,
            "skip": value,
            "signal": value,
            "residual": value,
        }

    def _finish(self, signal, residual, esa_update):
        base = signal if residual is None else residual
        residual_mid = base + self.update_dropout(esa_update)
        ffn_update = self.ffn(self.norm2(residual_mid))
        residual_out = residual_mid + self.update_dropout(ffn_update)
        return self._result(residual_out)

    def forward(self, signal, residual=None):
        if signal is None:
            raise ModelCompileError("Layer Block requires Signal In.")
        esa_update = self.esa(self.norm1(signal))
        return self._finish(signal, residual, esa_update)

    @torch.no_grad()
    def prefill(self, signal, residual=None):
        prefill = getattr(self.esa, "prefill", None)
        if not callable(prefill):
            raise ModelCompileError("Layer Block ESA does not expose prefill().")
        esa_update, state = prefill(self.norm1(signal))
        return self._finish(signal, residual, esa_update), state

    @torch.no_grad()
    def decode_step(self, signal, state, residual=None):
        decode = getattr(self.esa, "decode_step", None)
        if not callable(decode):
            raise ModelCompileError("Layer Block ESA does not expose decode_step().")
        esa_update, state = decode(self.norm1(signal), state)
        return self._finish(signal, residual, esa_update), state


class _APILaneOutputs:
    """Internal three-lane result produced by an API/User Function node."""
    __slots__ = ("main", "skip", "extra")

    def __init__(self, main=None, skip=None, extra=None):
        self.main = main
        self.skip = skip
        self.extra = extra


class _APINamedOutputs:
    """Arbitrary named outputs produced by a User Defined Function node."""
    __slots__ = ("values", "main")

    def __init__(self, values):
        self.values = dict(values or {})
        self.main = next(iter(self.values.values()), None)


class _APIMixedOutputs:
    """Standard Main/Skip/Extra outputs plus user-created named terminals.

    Custom API terminals are additive: the six universal Studio sockets remain
    available while extra named outputs can be wired independently.
    """
    __slots__ = ("standard", "values")

    def __init__(self, standard, values):
        self.standard = standard
        self.values = dict(values or {})


def _lane_output(value, lane):
    if isinstance(value, dict):
        if lane in value:
            return value[lane]
        if lane == "main" and "main" in value:
            return value["main"]
        raise ModelCompileError(f"API output lane {lane!r} is connected but has no mapped return value.")
    if isinstance(value, _APIMixedOutputs):
        return _lane_output(value.standard, lane)
    if isinstance(value, _APILaneOutputs):
        selected = getattr(value, lane, None)
        if selected is None:
            raise ModelCompileError(f"API output lane {lane!r} is connected but has no mapped return value.")
        return selected
    if isinstance(value, _APINamedOutputs):
        if lane in value.values:
            return value.values[lane]
        if lane == "main" and value.main is not None:
            return value.main
        raise ModelCompileError(f"Named User Function output has no {lane!r} value.")
    return value


def _named_output(value, key):
    key = str(key or "").strip()
    if isinstance(value, dict):
        lookup = "main" if key in {"", "output"} else key
        if lookup in value:
            return value[lookup]
        raise ModelCompileError(f"API contract output has no named output port {key!r}.")
    if isinstance(value, _APIMixedOutputs):
        if key in value.values:
            return value.values[key]
        if key in {"", "main", "skip", "extra", "output"}:
            return _lane_output(value.standard, "main" if key in {"", "output"} else key)
        raise ModelCompileError(f"Custom API has no named output terminal {key!r}.")
    if isinstance(value, _APINamedOutputs):
        if key in value.values:
            return value.values[key]
        raise ModelCompileError(f"User Function has no named output port {key!r}.")
    if isinstance(value, _APILaneOutputs) and key in {"main", "skip", "extra"}:
        return _lane_output(value, key)
    if key in {"", "main", "skip", "extra", "output"}:
        return _lane_output(value, "main" if key in {"", "output"} else key)
    raise ModelCompileError(f"Output {key!r} is not available from the connected source node.")


class _APIOperation(nn.Module):
    """One Python/PyTorch operation inside a user-authored API Component graph.

    V1.0 object-aware API nodes distinguish imports from object instances.  A
    node can create/register an object once, call normal/static/class methods,
    or reuse an object created by another node.  The registry is scoped to one
    TensorGraph instance so state is preserved across later API nodes without
    re-constructing the object.
    """

    _CALL_TYPES = {"function", "user_function", "user_class", "static_method", "class_method", "instance_method", "constructor"}

    def __init__(self, *, binding, params, runtime, label, object_registry=None):
        super().__init__()
        self.binding = deepcopy(binding or {})
        self.params = deepcopy(params or {})
        self.runtime = deepcopy(runtime or {})
        self.label = str(label or "API Function")
        self.object_registry = object_registry if object_registry is not None else {}
        self.parameters = list(self.binding.get("parameters") or [])

        legacy_kind = str(self.binding.get("target_kind") or "module").lower()
        call_type = str(self.binding.get("call_type") or "").strip().lower()
        if not call_type:
            call_type = "function" if legacy_kind == "function" else "instance_method"
        if call_type not in self._CALL_TYPES:
            raise ModelCompileError(f"API function {self.label!r} has unsupported call type {call_type!r}.")
        self.call_type = call_type
        self.object_mode = "existing" if str(self.binding.get("object_mode") or "new").lower() == "existing" else "new"
        self.object_id = str(self.binding.get("object_id") or f"object::{self.label}").strip()
        self.object_name = str(self.binding.get("object_name") or self.label).strip()
        self.object_ref = str(self.binding.get("object_ref") or "").strip()
        self.result_object_id = str(self.binding.get("result_object_id") or f"result::{self.label}").strip()
        self.result_object_name = str(self.binding.get("result_object_name") or f"{self.label} result").strip()
        self.auto_main_input = bool(self.binding.get("auto_main_input", True))
        self.register_result_object = bool(self.binding.get("register_result_object"))
        self.result_output_mode = "passthrough" if str(self.binding.get("result_output_mode") or "result").lower() == "passthrough" else "result"
        self.multi_output = bool(self.binding.get("multi_output"))
        self.output_map = deepcopy(self.binding.get("output_map") or {})
        _port_mode = str(self.binding.get("port_mode") or "standard").lower()
        self.port_mode = _port_mode if _port_mode in {"standard", "extended", "named"} else "standard"
        self.input_ports = deepcopy(self.binding.get("input_ports") or [])
        self.output_ports = deepcopy(self.binding.get("output_ports") or [])

        self.api_target = None
        self.created_object = None

        if not _bool(self.runtime.get("allow_user_code", True)):
            raise ModelCompileError(
                f"Executable API component {self.label!r} is blocked because this project is untrusted. "
                "Review the project's Python source/import bindings, then call builder.trust_project() "
                "in the current session before training, generation, or serving."
            )

        # User-authored source is embedded in the component/project cache.  It is
        # compiled in the user's active Python environment.  Third-party imports
        # are deliberately not installed by Studio; missing libraries produce an
        # explicit install-the-dependency error instead.
        if self.call_type in {"user_function", "user_class"}:
            is_class = self.call_type == "user_class"
            source_key = "user_class_code" if is_class else "user_code"
            name_key = "user_class_name" if is_class else "user_function_name"
            kind_label = "User Class" if is_class else "User Function"
            source = str(self.binding.get(source_key) or "").strip()
            entry_name = str(self.binding.get(name_key) or "").strip()
            if not source:
                raise ModelCompileError(f"{kind_label} {self.label!r} has no Python source code.")
            if not entry_name:
                raise ModelCompileError(f"{kind_label} {self.label!r} has no entry name configured.")
            namespace = {"torch": torch, "nn": nn}
            try:
                exec(compile(source, f"<MLB Studio:{self.label}>", "exec"), namespace, namespace)
            except ModuleNotFoundError as exc:
                missing = getattr(exc, "name", None) or str(exc)
                raise ModelCompileError(
                    f"{kind_label} {self.label!r} needs dependency {missing!r}. "
                    f"Install it explicitly in the active Python environment before building this component."
                ) from exc
            except Exception as exc:
                raise ModelCompileError(f"Could not compile {kind_label} {self.label!r}: {exc}") from exc
            target = namespace.get(entry_name)
            if not callable(target):
                raise ModelCompileError(f"{kind_label} {self.label!r} did not define callable {entry_name!r}.")
            self.api_target = target
            if is_class:
                init_specs = [spec for spec in self.parameters if str(spec.get("stage") or "init").lower() == "init"]
                init_args, init_kwargs = self._build_arguments(init_specs, x=None)
                try:
                    instance = target(*init_args, **init_kwargs)
                except Exception as exc:
                    raise ModelCompileError(
                        f"Could not construct user object {self.object_name!r} from class {entry_name!r}: {exc}"
                    ) from exc
                self.created_object = instance
                self.object_registry[self.object_id] = instance
            return

        # Reusing an existing object does not need another import or constructor.
        if self.call_type == "instance_method" and self.object_mode == "existing":
            if not self.object_ref:
                raise ModelCompileError(f"API instance method {self.label!r} has no existing object selected.")
            return

        import_path = _api_binding_import_path(self.binding)
        if not import_path:
            raise ModelCompileError(f"API function {self.label!r} has no import/module + function/class configured.")
        target = IMPORT_POOL.resolve_external(import_path)
        self.api_target = target

        if self.call_type in {"constructor", "instance_method"} and self.object_mode == "new":
            init_specs = [spec for spec in self.parameters if str(spec.get("stage") or "init").lower() == "init"]
            init_args, init_kwargs = self._build_arguments(init_specs, x=None)
            if not callable(target):
                raise ModelCompileError(f"API target {import_path} is not constructible/callable.")
            try:
                instance = target(*init_args, **init_kwargs)
            except Exception as exc:
                raise ModelCompileError(
                    f"Could not construct API object {self.object_name!r} for {self.label!r} from {import_path}: {exc}"
                ) from exc
            # If this is an nn.Module, assigning it here registers its parameters
            # exactly once with PyTorch.  Reuser nodes keep only the plain shared
            # registry reference and therefore do not duplicate module ownership.
            self.created_object = instance
            self.object_registry[self.object_id] = instance
        elif self.call_type == "function":
            if not callable(target):
                raise ModelCompileError(f"API target {import_path} is not callable.")
        elif self.call_type in {"static_method", "class_method"}:
            method = str(self.binding.get("call_method") or "").strip()
            if not method:
                raise ModelCompileError(f"API {self.call_type.replace('_', ' ')} {self.label!r} needs a method name.")
            try:
                candidate = getattr(target, method)
            except Exception as exc:
                raise ModelCompileError(f"API target {import_path} has no method {method!r}.") from exc
            if not callable(candidate):
                raise ModelCompileError(f"API target {import_path}.{method} is not callable.")

    def _build_arguments(self, specs, x, skip=None, extra=None):
        positional = []
        keywords = {}
        for spec in specs:
            name = str(spec.get("name") or spec.get("key") or "").strip()
            if not name and not spec.get("positional"):
                continue
            value = _bound_parameter_value(spec, self.params, self.runtime, x=x, skip=skip, extra=extra)
            if value is None and not spec.get("required"):
                continue
            if bool(spec.get("positional")):
                positional.append(value)
            else:
                keywords[name] = value
        return positional, keywords

    @staticmethod
    def _select_output(value, selector):
        text = str(selector or "auto").strip()
        if text in {"", "auto"}:
            if torch.is_tensor(value):
                return value
            if isinstance(value, (list, tuple)) and value and torch.is_tensor(value[0]):
                return value[0]
            return value
        if isinstance(value, (list, tuple)):
            try:
                return value[int(text)]
            except Exception:
                pass
        if isinstance(value, dict) and text in value:
            return value[text]
        if hasattr(value, text):
            return getattr(value, text)
        raise ModelCompileError(f"API output selector {text!r} could not be applied for {self.label!r}.")

    def _resolve_call_target(self):
        method = str(self.binding.get("call_method") or "").strip()
        if self.call_type in {"function", "user_function"}:
            return self.api_target
        if self.call_type in {"static_method", "class_method"}:
            return getattr(self.api_target, method)
        if self.call_type == "instance_method":
            if self.object_mode == "existing":
                if self.object_ref not in self.object_registry:
                    raise ModelCompileError(
                        f"API node {self.label!r} references object {self.object_ref!r}, but it is not available yet. "
                        "Create/register that object in an upstream API node first."
                    )
                instance = self.object_registry[self.object_ref]
            else:
                instance = self.created_object
            if instance is None:
                raise ModelCompileError(f"API node {self.label!r} has no object instance available.")
            if method and method not in {"__call__", "forward"}:
                try:
                    return getattr(instance, method)
                except Exception as exc:
                    raise ModelCompileError(
                        f"Object {self.object_name!r} used by {self.label!r} has no method {method!r}."
                    ) from exc
            return instance
        raise ModelCompileError(f"API node {self.label!r} is a constructor and has no call target.")

    def forward(self, x, skip=None, extra=None, named_inputs=None):
        # A constructor is an object-lifecycle node, not a tensor transform.  It
        # creates/registers the object once in __init__ and transparently passes
        # the Main lane through at execution time.
        if self.call_type in {"constructor", "user_class"}:
            return x

        call_specs = [spec for spec in self.parameters if str(spec.get("stage") or "call").lower() == "call"]
        if call_specs:
            args, kwargs = self._build_arguments(call_specs, x=x, skip=skip, extra=extra)
            tensor_bound = any(
                str(spec.get("source") or "user").lower() in {"input", "main", "skip", "extra"}
                for spec in call_specs
            )
            if not tensor_bound and self.auto_main_input:
                args = [x, *args]
        else:
            args, kwargs = ([x], {}) if self.auto_main_input else ([], {})

        # User-created terminals are additive in ``extended`` mode: Main/Skip/
        # Extra keep their universal behavior and these values are appended to
        # the call using the configured function-parameter mapping.  ``named``
        # remains supported for legacy designs whose custom ports replaced the
        # universal layout.
        if self.input_ports and self.port_mode in {"extended", "named"}:
            named_inputs = dict(named_inputs or {})
            for port in self.input_ports:
                port_id = str(port.get("id") or "").strip()
                port_name = str(port.get("name") or port_id).strip()
                parameter = str(port.get("parameter") or port_name).strip()
                if port_id not in named_inputs:
                    if port.get("required", True):
                        raise ModelCompileError(
                            f"API function {self.label!r} input terminal {port_name!r} is not connected."
                        )
                    continue
                value = named_inputs[port_id]
                if bool(port.get("positional")):
                    args.append(value)
                else:
                    kwargs[parameter] = value

        target = self._resolve_call_target()
        try:
            out = target(*args, **kwargs)
        except Exception as exc:
            raise RuntimeError(f"API function {self.label!r} failed: {exc}") from exc

        custom_outputs = {}
        if self.output_ports and self.port_mode in {"extended", "named"}:
            for port in self.output_ports:
                port_id = str(port.get("id") or port.get("name") or "output").strip()
                selector = port.get("selector", "auto")
                custom_outputs[port_id] = self._select_output(out, selector)

        if self.register_result_object:
            self.object_registry[self.result_object_id] = out
            if self.result_output_mode == "passthrough":
                standard_result = x
                return _APIMixedOutputs(standard_result, custom_outputs) if custom_outputs else standard_result

        # Legacy named-only layouts keep their historical behavior.
        if self.port_mode == "named":
            if not custom_outputs:
                custom_outputs = {"output": self._select_output(out, "auto")}
            return _APINamedOutputs(custom_outputs)

        if self.multi_output:
            mapping = self.output_map or {}
            def selected(lane, default=""):
                selector = str(mapping.get(lane, default) or "").strip()
                if not selector:
                    return None
                return self._select_output(out, selector)
            main_value = selected("main", "0")
            if main_value is None:
                raise ModelCompileError(f"User/API function {self.label!r} needs a Main output mapping.")
            standard_result = _APILaneOutputs(
                main=main_value,
                skip=selected("skip"),
                extra=selected("extra"),
            )
        else:
            standard_result = self._select_output(out, self.binding.get("output_selector"))

        if custom_outputs:
            return _APIMixedOutputs(standard_result, custom_outputs)
        return standard_result


class _APIBoundComponent(nn.Module):
    """Reusable API Component represented as an explicit mixed execution DAG.

    API Components may contain ``api_step`` nodes, supported built-in Builder
    components, and reusable graph Modules. Nested API Components are kept out
    of the authoring UI, while the shared custom-definition stack still guards
    against circular Module dependencies. Legacy single-binding API Components
    remain supported.
    """

    def __init__(self, *, definition, params, runtime, custom_components=None, _custom_stack=()):
        super().__init__()
        self.definition = deepcopy(definition)
        self.params = deepcopy(params or {})
        self.runtime = deepcopy(runtime or {})
        self.custom_components = custom_components or {}
        self._custom_stack = tuple(_custom_stack or ())
        nodes = [deepcopy(n) for n in (self.definition.get("nodes") or [])]
        self.legacy = None
        self.graph = None
        if not nodes:
            binding = self.definition.get("api_binding") or {}
            self.legacy = _APIOperation(
                binding=binding,
                params=self.params,
                runtime=self.runtime,
                label=self.definition.get("name") or "API Component",
                object_registry={},
            )
            return

        # Apply exposed API parameter values to their owning api_step before the
        # generic TensorGraph builds modules. Built-in component parameters are
        # already serialized directly on their nodes.
        for node in nodes:
            if str(node.get("type") or "") != "api_step":
                continue
            binding = deepcopy(node.get("api_binding") or {})
            node_params = node.setdefault("params", {})
            for spec in binding.get("parameters") or []:
                name = str(spec.get("name") or spec.get("key") or "").strip()
                if not name:
                    continue
                expose_key = str(spec.get("expose_key") or f"{node.get('id')}::{name}")
                if expose_key in self.params:
                    node_params[name] = self.params[expose_key]
                elif name not in node_params and spec.get("default") is not None:
                    node_params[name] = spec.get("default")

        # TensorGraph is defined below this class and is available by the time a
        # compiled model instantiates an API component.
        self.graph = TensorGraph(
            nodes=nodes,
            edges=deepcopy(self.definition.get("edges") or []),
            custom_components=self.custom_components,
            runtime=self.runtime,
            _custom_stack=self._custom_stack,
        )

    def forward(self, x, skip=None, extra=None):
        if self.legacy is not None:
            return self.legacy(x, skip=skip, extra=extra)
        return self.graph(x, graph_skip=skip, graph_extra=extra)


class _AbstractLayerComponent(nn.Module):
    """Editable layer shell backed by an internal TensorGraph.

    The outer component always keeps Studio's three fixed lanes (Main / Skip /
    Extra). Up to five additive custom named inputs and outputs are described by
    ``definition["interface"]`` and are carried through the two boundary nodes
    inside the graph.
    """

    def __init__(self, *, definition, params, runtime, custom_components=None, _custom_stack=()):
        super().__init__()
        self.definition = deepcopy(definition or {})
        self.params = deepcopy(params or {})
        self.runtime = deepcopy(runtime or {})
        self.custom_components = custom_components or {}
        self._custom_stack = tuple(_custom_stack or ())
        nodes = [deepcopy(n) for n in (self.definition.get("nodes") or [])]
        if not any(str(n.get("type") or "") == "abstract_input" for n in nodes):
            raise ModelCompileError(f"Abstract Layer {self.definition.get('name')!r} has no Layer Inputs boundary.")
        if not any(str(n.get("type") or "") == "abstract_output" for n in nodes):
            raise ModelCompileError(f"Abstract Layer {self.definition.get('name')!r} has no Layer Outputs boundary.")
        self.graph = TensorGraph(
            nodes=nodes,
            edges=deepcopy(self.definition.get("edges") or []),
            custom_components=self.custom_components,
            runtime=self.runtime,
            _custom_stack=self._custom_stack,
        )

    def forward(self, x, *, skip=None, extra=None, named_inputs=None):
        return self.graph(
            x, graph_skip=skip, graph_extra=extra, graph_named=named_inputs or {}
        )

    @torch.no_grad()
    def prefill(self, x, *, skip=None, extra=None, named_inputs=None, capacity=None):
        return self.graph.prefill(
            x, capacity=capacity, graph_skip=skip, graph_extra=extra,
            graph_named=named_inputs or {},
        )

    @torch.no_grad()
    def decode_step(self, x, cache, *, skip=None, extra=None, named_inputs=None, position=None):
        return self.graph.decode_step(
            x, cache, position=position, graph_skip=skip, graph_extra=extra,
            graph_named=named_inputs or {},
        )

    def recurrent_generation_support(self):
        return self.graph.recurrent_generation_support()

    def recurrent_generation_algorithms(self):
        return self.graph.recurrent_generation_algorithms()


class _SpeakerEmbedding(nn.Module):
    def __init__(self, num_speakers=4, embedding_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(max(1, int(num_speakers)), max(1, int(embedding_dim)))
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if x.ndim > 1:
            x = x.reshape(x.shape[0], -1)[:, 0]
        return self.embedding(x.long().clamp_min(0).clamp_max(self.embedding.num_embeddings - 1))


class _AudioCodecEncoder(nn.Module):
    """Small differentiable waveform encoder for Studio's educational audio graphs."""
    def __init__(self, latent_dim=32, hidden_channels=16):
        super().__init__()
        h=max(4,int(hidden_channels)); d=max(1,int(latent_dim))
        self.net=nn.Sequential(
            nn.Conv1d(1,h,7,stride=2,padding=3), nn.SiLU(),
            nn.Conv1d(h,h*2,5,stride=2,padding=2), nn.SiLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(1), nn.Linear(h*2,d), nn.Tanh(),
        )
    def forward(self, x):
        if x.ndim==2: x=x.unsqueeze(1)
        if x.ndim!=3: raise ValueError(f"Audio Codec Encoder expects [B,S] or [B,1,S], received {tuple(x.shape)}.")
        if x.shape[1]!=1:
            x=x.mean(dim=1,keepdim=True)
        return self.net(x.float())


class _AudioTokenPredictor(nn.Module):
    def __init__(self, in_features=32, latent_dim=64, hidden_dim=64):
        super().__init__()
        i=max(1,int(in_features)); h=max(1,int(hidden_dim)); d=max(1,int(latent_dim))
        self.net=nn.Sequential(nn.Linear(i,h),nn.SiLU(),nn.Linear(h,d),nn.Tanh())
    def forward(self,x):
        return self.net(x.float())


class _AudioCodecDecoder(nn.Module):
    """Latent-to-waveform educational decoder used by TTS/music/sound demos."""
    def __init__(self, latent_dim=64, output_samples=256, hidden_dim=128):
        super().__init__()
        d=max(1,int(latent_dim)); h=max(4,int(hidden_dim)); s=max(16,int(output_samples))
        self.output_samples=s
        self.net=nn.Sequential(nn.Linear(d,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,s),nn.Tanh())
    def forward(self,x):
        if x.ndim>2: x=x.reshape(x.shape[0],-1)
        return self.net(x.float())


class TensorGraph(nn.Module):
    """Small tensor DAG compiler for the model components Builder can execute today."""
    def __init__(self, *, nodes, edges, custom_components, runtime, vocab_override=None, _custom_stack=()):
        super().__init__()
        self.nodes=deepcopy(nodes)
        self.edges=deepcopy(edges)
        self.custom_components=custom_components
        self.runtime=runtime
        self.vocab_override=vocab_override
        self._custom_stack=tuple(_custom_stack or ())
        self.order=_topological(self.nodes,self.edges)
        self.by_id={n["id"]:n for n in self.nodes}
        # Shared only by API operation nodes in this graph instance.  Directly
        # created objects are registered during module construction; results can
        # be registered during forward for later upstream-connected API nodes.
        self.api_object_registry={}
        self.in_main={n["id"]:[] for n in self.nodes}
        self.in_skip={n["id"]:[] for n in self.nodes}
        self.in_extra={n["id"]:[] for n in self.nodes}
        self.in_main_edges={n["id"]:[] for n in self.nodes}
        self.in_skip_edges={n["id"]:[] for n in self.nodes}
        self.in_extra_edges={n["id"]:[] for n in self.nodes}
        self.in_named={n["id"]:[] for n in self.nodes}
        self.outgoing={n["id"]:[] for n in self.nodes}
        for e in self.edges:
            a,b=e.get("source"),e.get("target")
            if a not in self.by_id or b not in self.by_id: continue
            kind=str(e.get("kind") or "main").lower()
            target_port=str(e.get("target_port") or "").strip().lower()

            # Destination routing is determined by the *target* port, not by
            # whether the source happens to be a named API output. A stateful
            # component may expose ``named_out:main`` and feed that tensor into
            # an ordinary downstream ``main_in`` (for example SAFFN -> ESA).
            if target_port.startswith("named_in:"):
                self.in_named[b].append(deepcopy(e))
            elif target_port in {"main", "main_in"}:
                self.in_main[b].append(a);self.in_main_edges[b].append(deepcopy(e))
            elif target_port in {"skip", "skip_in"}:
                self.in_skip[b].append(a);self.in_skip_edges[b].append(deepcopy(e))
            elif target_port in {"extra", "extra_in"}:
                self.in_extra[b].append(a);self.in_extra_edges[b].append(deepcopy(e))
            elif kind == "named":
                # Backward compatibility for older serialized named-input
                # edges that omitted an explicit named_in:* target port.
                self.in_named[b].append(deepcopy(e))
            elif kind in {"residual","skip"}:
                self.in_skip[b].append(a);self.in_skip_edges[b].append(deepcopy(e))
            elif kind in {"main",""}:
                self.in_main[b].append(a);self.in_main_edges[b].append(deepcopy(e))
            elif kind in {"aux","extra"}:
                self.in_extra[b].append(a);self.in_extra_edges[b].append(deepcopy(e))
            else:
                self.in_main[b].append(a);self.in_main_edges[b].append(deepcopy(e))
            self.outgoing[a].append(b)
        self.mods=nn.ModuleDict()
        for node in self.nodes:
            mod=self._module_for(node)
            if mod is not None:self.mods[node["id"]]=mod
        self._apply_weight_tying()

    def _module_for(self,node):
        # MLBricks symbols are imported lazily per component. Adding/using one
        # component must not depend on unrelated top-level exports being present.
        t=node.get("type"); p=deepcopy(node.get("params") or {})
        device=str(self.runtime.get("device") or "auto")
        backend=str(self.runtime.get("backend") or "pytorch")
        precision=str(self.runtime.get("precision") or "fp16")
        if precision=="auto": precision="fp16" if resolve_device(device).type=="cuda" else "fp32"
        contract = API_COMPONENTS.get(t)
        if contract is not None:
            try:
                return contract.instantiate(node, {**self.runtime, "device": device, "backend": backend, "precision": precision})
            except (TypeError, ValueError) as exc:
                raise ModelCompileError(f"Could not construct {node.get('name') or t} from its MLBricks API contract: {exc}") from exc
        if t=="api_step":
            binding=deepcopy(node.get("api_binding") or {})
            step_params=deepcopy(node.get("params") or {})
            return _APIOperation(
                binding=binding,
                params=step_params,
                runtime=self.runtime,
                label=node.get("name") or "API Function",
                object_registry=self.api_object_registry,
            )
        if t in {"text_input","image_input","audio_input","video_input","signal_input","feature_input","stream_input","text_output","audio_output","tensor_output","logits_output","abstract_input","abstract_output"}: return _Identity()
        if t=="classifier":
            default_dim = runtime_int(
                self.runtime.get("model_dim"),
                384,
                f"{node.get('name','Classifier Head')} runtime model dim",
                minimum=1,
            )
            return _ClassifierHead(
                dim=runtime_int(
                    p.get("dim") or p.get("hidden_size"),
                    default_dim,
                    f"{node.get('name','Classifier Head')} hidden dim",
                    minimum=1,
                ),
                classes=runtime_int(
                    p.get("classes"),
                    10,
                    f"{node.get('name','Classifier Head')} classes",
                    minimum=1,
                ),
            )
        if t=="signal_fft":
            return _SignalFFT(
                bins=runtime_int(p.get("bins"),16,f"{node.get('name','FFT Magnitude')} bins",minimum=0),
                log_scale=_bool(p.get("log_scale",False)),
                remove_dc=_bool(p.get("remove_dc",False)),
                flatten_channels=_bool(p.get("flatten_channels",True)),
            )
        if t=="jepa_mask":
            return _JEPAMask(
                mask_ratio=runtime_float(p.get("mask_ratio"),0.35,f"{node.get('name','JEPA Mask')} mask ratio",minimum=0.000001,maximum=0.999999),
                mask_value=runtime_float(p.get("mask_value"),0.0,f"{node.get('name','JEPA Mask')} mask value"),
                mode=str(p.get("mode") or "auto"),
            )
        if t=="jepa_encoder":
            return _JEPAEncoder(
                modality=str(p.get("modality") or "image"), role=str(p.get("role") or "context"),
                latent_dim=runtime_int(p.get("latent_dim"),64,f"{node.get('name','JEPA Encoder')} latent dim",minimum=1),
                hidden_dim=runtime_int(p.get("hidden_dim"),64,f"{node.get('name','JEPA Encoder')} hidden dim",minimum=1),
                vocab_size=runtime_int(p.get("vocab_size"),257,f"{node.get('name','JEPA Encoder')} vocab size",minimum=2),
                in_channels=runtime_int(p.get("in_channels"),1,f"{node.get('name','JEPA Encoder')} input channels",minimum=1),
            )
        if t=="jepa_predictor":
            return _JEPAPredictor(
                latent_dim=runtime_int(p.get("latent_dim"),64,f"{node.get('name','JEPA Predictor')} latent dim",minimum=1),
                hidden_dim=runtime_int(p.get("hidden_dim"),128,f"{node.get('name','JEPA Predictor')} hidden dim",minimum=1),
                dropout=runtime_float(p.get("dropout"),0.0,f"{node.get('name','JEPA Predictor')} dropout",minimum=0.0,maximum=1.0),
            )
        if t=="jepa_latent_loss":
            return _JEPALatentLoss(loss=str(p.get("loss") or "mse"), normalize=_bool(p.get("normalize",True)))
        if t=="fpn_fusion":
            return _FPNFusion(
                high_channels=runtime_int(p.get("high_channels"),64,f"{node.get('name','FPN Fusion')} high channels",minimum=1),
                lateral_channels=runtime_int(p.get("lateral_channels"),32,f"{node.get('name','FPN Fusion')} lateral channels",minimum=1),
                out_channels=runtime_int(p.get("out_channels"),32,f"{node.get('name','FPN Fusion')} output channels",minimum=1),
                fusion=str(p.get("fusion") or "add"),
            )
        if t=="pan_fusion":
            return _PANFusion(
                fine_channels=runtime_int(p.get("fine_channels"),32,f"{node.get('name','PAN Fusion')} fine channels",minimum=1),
                coarse_channels=runtime_int(p.get("coarse_channels"),64,f"{node.get('name','PAN Fusion')} coarse channels",minimum=1),
                out_channels=runtime_int(p.get("out_channels"),64,f"{node.get('name','PAN Fusion')} output channels",minimum=1),
                fusion=str(p.get("fusion") or "concat"),
            )
        if t=="detection_head":
            return _DetectionHead(
                in_channels=runtime_int(p.get("in_channels"),64,f"{node.get('name','Detection Head')} input channels",minimum=1),
                classes=runtime_int(p.get("classes"),3,f"{node.get('name','Detection Head')} classes",minimum=1),
                anchors=runtime_int(p.get("anchors"),1,f"{node.get('name','Detection Head')} prediction slots",minimum=1),
            )
        if t=="detection_pyramid_head":
            return _DetectionPyramidHead(
                p3_channels=runtime_int(p.get("p3_channels"),32,f"{node.get('name','Detection Pyramid Head')} P3 channels",minimum=1),
                p4_channels=runtime_int(p.get("p4_channels"),64,f"{node.get('name','Detection Pyramid Head')} P4 channels",minimum=1),
                p5_channels=runtime_int(p.get("p5_channels"),96,f"{node.get('name','Detection Pyramid Head')} P5 channels",minimum=1),
                classes=runtime_int(p.get("classes"),3,f"{node.get('name','Detection Pyramid Head')} classes",minimum=1),
                slots=runtime_int(p.get("slots"),3,f"{node.get('name','Detection Pyramid Head')} slots",minimum=1),
            )
        if t=="detection_nms":
            return _DetectionNMS(
                score_threshold=runtime_float(p.get("score_threshold"),0.25,f"{node.get('name','Detection NMS')} score threshold",minimum=0.0,maximum=1.0),
                iou_threshold=runtime_float(p.get("iou_threshold"),0.5,f"{node.get('name','Detection NMS')} IoU threshold",minimum=0.0,maximum=1.0),
                max_detections=runtime_int(p.get("max_detections"),100,f"{node.get('name','Detection NMS')} max detections",minimum=1),
            )
        if t=="value_buffer":
            return _PreviousValueBuffer(
                mode=str(p.get("mode") or "hold"),
                width=runtime_int(p.get("width"),0,f"{node.get('name','Previous Value Buffer')} output width",minimum=0),
            )

        # Builder-native educational ML / DL / tensor components. Keep these
        # independent of the external MLBricks import registry so a student can
        # assemble first-principles graphs with stock PyTorch operations.
        if t=="linear_regression":
            return nn.Linear(
                runtime_int(p.get("in_features"),4,f"{node.get('name','Linear Regression')} input features",minimum=1),
                runtime_int(p.get("out_features"),1,f"{node.get('name','Linear Regression')} outputs",minimum=1),
                bias=_bool(p.get("bias",True)),
            )
        if t=="logistic_regression":
            return _LogisticRegression(
                runtime_int(p.get("in_features"),4,f"{node.get('name','Logistic Regression')} input features",minimum=1),
                runtime_int(p.get("out_features"),1,f"{node.get('name','Logistic Regression')} outputs",minimum=1),
                bias=_bool(p.get("bias",True)), output=str(p.get("output") or "probability"),
            )
        if t=="polynomial_features":
            return _PolynomialFeatures(
                degree=runtime_int(p.get("degree"),2,f"{node.get('name','Polynomial Features')} degree",minimum=1,maximum=4),
                include_bias=_bool(p.get("include_bias",False)), interaction_only=_bool(p.get("interaction_only",False)),
            )
        if t=="knn_classifier":
            return _KNNClassifier(
                neighbors=runtime_int(p.get("neighbors"),5,f"{node.get('name','KNN Classifier')} neighbors",minimum=1),
                weights=str(p.get("weights") or "uniform"),
                p=runtime_float(p.get("p"),2.0,f"{node.get('name','KNN Classifier')} distance P",minimum=0.000001),
            )
        if t=="decision_tree_classifier":
            return _DecisionTreeClassifier(
                max_depth=runtime_int(p.get("max_depth"),5,f"{node.get('name','Decision Tree')} max depth",minimum=1),
                min_samples_split=runtime_int(p.get("min_samples_split"),2,f"{node.get('name','Decision Tree')} min samples split",minimum=2),
                min_samples_leaf=runtime_int(p.get("min_samples_leaf"),1,f"{node.get('name','Decision Tree')} min samples leaf",minimum=1),
                criterion=str(p.get("criterion") or "gini"),
            )
        if t=="kmeans":
            return _KMeans(
                clusters=runtime_int(p.get("clusters"),3,f"{node.get('name','K-Means')} clusters",minimum=1),
                max_iter=runtime_int(p.get("max_iter"),100,f"{node.get('name','K-Means')} max iterations",minimum=1),
                tolerance=runtime_float(p.get("tolerance"),1e-4,f"{node.get('name','K-Means')} tolerance",minimum=0.0),
                seed=runtime_int(p.get("seed"),42,f"{node.get('name','K-Means')} seed"),
            )
        if t=="pca":
            return _PCA(
                components=runtime_int(p.get("components"),2,f"{node.get('name','PCA')} components",minimum=1),
                center=_bool(p.get("center",True)),
                whiten=_bool(p.get("whiten",False)),
            )
        if t=="relu": return nn.ReLU()
        if t=="leaky_relu":
            return nn.LeakyReLU(runtime_float(p.get("negative_slope"),0.01,f"{node.get('name','Leaky ReLU')} negative slope",minimum=0.0))
        if t=="gelu": return nn.GELU(approximate=str(p.get("approximate") or "none"))
        if t=="silu": return nn.SiLU()
        if t=="sigmoid": return nn.Sigmoid()
        if t=="tanh": return nn.Tanh()
        if t=="softmax": return nn.Softmax(dim=runtime_int(p.get("dim"),-1,f"{node.get('name','Softmax')} dimension"))
        if t=="batchnorm1d":
            return _FlexibleBatchNorm1d(
                runtime_int(p.get("num_features"),128,f"{node.get('name','BatchNorm 1D')} features",minimum=1),
                eps=runtime_float(p.get("eps"),1e-5,f"{node.get('name','BatchNorm 1D')} epsilon",minimum=0.0),
                momentum=runtime_float(p.get("momentum"),0.1,f"{node.get('name','BatchNorm 1D')} momentum",minimum=0.0),
                layout=str(p.get("layout") or "features_last"),
            )
        if t=="batchnorm2d":
            return nn.BatchNorm2d(
                runtime_int(p.get("num_features"),32,f"{node.get('name','BatchNorm 2D')} channels",minimum=1),
                eps=runtime_float(p.get("eps"),1e-5,f"{node.get('name','BatchNorm 2D')} epsilon",minimum=0.0),
                momentum=runtime_float(p.get("momentum"),0.1,f"{node.get('name','BatchNorm 2D')} momentum",minimum=0.0),
            )
        if t in {"conv1d","conv2d","conv3d"}:
            cls={"conv1d":nn.Conv1d,"conv2d":nn.Conv2d,"conv3d":nn.Conv3d}[t]
            kwargs=dict(
                in_channels=runtime_int(p.get("in_channels"),1 if t=="conv1d" else 3,f"{node.get('name',t)} input channels",minimum=1),
                out_channels=runtime_int(p.get("out_channels"),16 if t!="conv2d" else 32,f"{node.get('name',t)} output channels",minimum=1),
                kernel_size=runtime_int(p.get("kernel_size"),3,f"{node.get('name',t)} kernel size",minimum=1),
                stride=runtime_int(p.get("stride"),1,f"{node.get('name',t)} stride",minimum=1),
                padding=runtime_int(p.get("padding"),1,f"{node.get('name',t)} padding",minimum=0),
                bias=_bool(p.get("bias",True)),
            )
            if t in {"conv1d","conv2d"}:
                kwargs["dilation"]=runtime_int(p.get("dilation"),1,f"{node.get('name',t)} dilation",minimum=1)
                kwargs["groups"]=runtime_int(p.get("groups"),1,f"{node.get('name',t)} groups",minimum=1)
            return cls(**kwargs)
        if t in {"maxpool1d","maxpool2d","avgpool1d","avgpool2d"}:
            cls={"maxpool1d":nn.MaxPool1d,"maxpool2d":nn.MaxPool2d,"avgpool1d":nn.AvgPool1d,"avgpool2d":nn.AvgPool2d}[t]
            kernel=runtime_int(p.get("kernel_size"),2,f"{node.get('name',t)} kernel size",minimum=1)
            stride_raw=runtime_int(p.get("stride"),0,f"{node.get('name',t)} stride",minimum=0)
            return cls(kernel_size=kernel,stride=(stride_raw or None),padding=runtime_int(p.get("padding"),0,f"{node.get('name',t)} padding",minimum=0))
        if t=="adaptive_avgpool1d":
            return nn.AdaptiveAvgPool1d(runtime_int(p.get("output_size"),1,f"{node.get('name','Adaptive AvgPool 1D')} output size",minimum=1))
        if t=="adaptive_avgpool2d":
            out=runtime_int(p.get("output_size"),1,f"{node.get('name','Adaptive AvgPool 2D')} output size",minimum=1)
            return nn.AdaptiveAvgPool2d((out,out))
        if t in {"rnn","lstm","gru"}:
            return _SequenceRecurrent(
                t,
                runtime_int(p.get("input_size"),128,f"{node.get('name',t.upper())} input size",minimum=1),
                runtime_int(p.get("hidden_size"),128,f"{node.get('name',t.upper())} hidden size",minimum=1),
                num_layers=runtime_int(p.get("num_layers"),1,f"{node.get('name',t.upper())} layers",minimum=1),
                dropout=runtime_float(p.get("dropout"),0.0,f"{node.get('name',t.upper())} dropout",minimum=0.0,maximum=1.0),
                bidirectional=_bool(p.get("bidirectional",False)), output=str(p.get("output") or "sequence"),
                nonlinearity=str(p.get("nonlinearity") or "tanh"),
            )
        if t=="self_attention":
            return _SelfAttention(
                runtime_int(p.get("dim"),128,f"{node.get('name','Self Attention')} embedding dim",minimum=1),
                runtime_int(p.get("heads"),4,f"{node.get('name','Self Attention')} heads",minimum=1),
                dropout=runtime_float(p.get("dropout"),0.0,f"{node.get('name','Self Attention')} dropout",minimum=0.0,maximum=1.0),
                causal=_bool(p.get("causal",False)), bias=_bool(p.get("bias",True)),
            )
        if t=="learnable_parameter":
            return _LearnableTensor(
                _shape_tuple(p.get("shape") or "4,1",label=f"{node.get('name','Learnable Parameter')} shape"),
                init=str(p.get("init") or "normal"),
                scale=runtime_float(p.get("scale"),0.02,f"{node.get('name','Learnable Parameter')} init scale",minimum=0.0),
            )
        if t=="constant":
            return _ConstantTensor(
                _shape_tuple(p.get("shape") or "1",label=f"{node.get('name','Constant')} shape"),
                value=runtime_float(p.get("value"),0.0,f"{node.get('name','Constant')} value"),
            )
        if t in {"matmul","tensor_add","tensor_subtract","tensor_multiply","tensor_divide","concat"}:
            op={"matmul":"matmul","tensor_add":"add","tensor_subtract":"subtract","tensor_multiply":"multiply","tensor_divide":"divide","concat":"concat"}[t]
            return _TensorBinaryOp(
                op, dim=runtime_int(p.get("dim"),-1,f"{node.get('name',t)} dimension"),
                epsilon=runtime_float(p.get("epsilon"),0.0,f"{node.get('name',t)} epsilon",minimum=0.0),
            )
        if t in {"reduce_mean","reduce_sum","reduce_max","reduce_min"}:
            op={"reduce_mean":"mean","reduce_sum":"sum","reduce_max":"max","reduce_min":"min"}[t]
            return _TensorReduce(op,dim=runtime_int(p.get("dim"),-1,f"{node.get('name',t)} dimension"),keepdim=_bool(p.get("keepdim",False)))
        if t in {"tensor_exp","tensor_log","tensor_sqrt"}:
            op={"tensor_exp":"exp","tensor_log":"log","tensor_sqrt":"sqrt"}[t]
            return _SafeUnary(op,epsilon=runtime_float(p.get("epsilon"),0.0,f"{node.get('name',t)} epsilon",minimum=0.0))
        if t=="transpose":
            return _Transpose(runtime_int(p.get("dim0"),-2,f"{node.get('name','Transpose')} dimension A"),runtime_int(p.get("dim1"),-1,f"{node.get('name','Transpose')} dimension B"))
        if t=="reshape":
            return _Reshape(_shape_tuple(p.get("shape") or "0,-1",label=f"{node.get('name','Reshape')} shape",allow_zero=True))
        if t=="flatten":
            return nn.Flatten(
                start_dim=runtime_int(p.get("start_dim"),1,f"{node.get('name','Flatten')} start dimension"),
                end_dim=runtime_int(p.get("end_dim"),-1,f"{node.get('name','Flatten')} end dimension"),
            )
        if t=="unsqueeze": return _Unsqueeze(runtime_int(p.get("dim"),1,f"{node.get('name','Unsqueeze')} dimension"))
        if t=="squeeze":
            raw_dim=p.get("dim")
            dim=None if _missing_number(raw_dim) else runtime_int(raw_dim,None,f"{node.get('name','Squeeze')} dimension")
            return _Squeeze(dim)
        if t=="speaker_embedding":
            return _SpeakerEmbedding(
                runtime_int(p.get("num_speakers"),4,f"{node.get('name','Speaker Embedding')} speaker count",minimum=1),
                runtime_int(p.get("embedding_dim"),16,f"{node.get('name','Speaker Embedding')} dimension",minimum=1),
            )
        if t=="audio_codec_encoder":
            return _AudioCodecEncoder(
                latent_dim=runtime_int(p.get("latent_dim"),32,f"{node.get('name','Audio Codec Encoder')} latent dim",minimum=1),
                hidden_channels=runtime_int(p.get("hidden_channels"),16,f"{node.get('name','Audio Codec Encoder')} hidden channels",minimum=1),
            )
        if t=="audio_token_predictor":
            return _AudioTokenPredictor(
                in_features=runtime_int(p.get("in_features"),32,f"{node.get('name','Audio Token Predictor')} input dim",minimum=1),
                latent_dim=runtime_int(p.get("latent_dim"),64,f"{node.get('name','Audio Token Predictor')} latent dim",minimum=1),
                hidden_dim=runtime_int(p.get("hidden_dim"),64,f"{node.get('name','Audio Token Predictor')} hidden dim",minimum=1),
            )
        if t=="audio_codec_decoder":
            return _AudioCodecDecoder(
                latent_dim=runtime_int(p.get("latent_dim"),64,f"{node.get('name','Audio Codec Decoder')} latent dim",minimum=1),
                output_samples=runtime_int(p.get("output_samples"),256,f"{node.get('name','Audio Codec Decoder')} output samples",minimum=16),
                hidden_dim=runtime_int(p.get("hidden_dim"),128,f"{node.get('name','Audio Codec Decoder')} hidden dim",minimum=1),
            )
        if t=="embedding":
            vocab=int(self.vocab_override or p.get("vocab_size") or p.get("num_embeddings") or 32000)
            dim=int(p.get("embedding_dim") or p.get("hidden_size") or 384)
            try:
                Embedding = IMPORT_POOL.resolve_component("embedding")
                return Embedding(vocab,dim)
            except Exception:
                # Keep educational/custom graphs usable without the optional
                # mlbricks-kit package while preferring its public component
                # whenever it is installed.
                return nn.Embedding(vocab,dim)
        if t=="lm_head":
            LMHead = IMPORT_POOL.resolve_component("lm_head")
            vocab=int(self.vocab_override or p.get("vocab_size") or 32000)
            hidden=int(p.get("hidden_size") or p.get("dim") or 384)
            # ``tie_to`` is a module reference in the MLBricks Python API. Builder
            # resolves that reference after the graph modules have been created.
            return LMHead(hidden,vocab,bias=_bool(p.get("bias",False)))
        if t=="learned_position":
            LearnedPosition = IMPORT_POOL.resolve_component("learned_position")
            return LearnedPosition(
                runtime_int(p.get("dim"),384,f"{node.get('name','Learned Position')} dim",minimum=1),
                runtime_int(p.get("max_seq_len"),65536,f"{node.get('name','Learned Position')} max sequence length",minimum=1),
            )
        if t=="sinusoidal_position":
            SinusoidalPosition = IMPORT_POOL.resolve_component("sinusoidal_position")
            return SinusoidalPosition(
                runtime_int(p.get("dim"),384,f"{node.get('name','Sinusoidal Position')} dim",minimum=1),
                runtime_int(p.get("max_seq_len"),65536,f"{node.get('name','Sinusoidal Position')} max sequence length",minimum=1),
                base=runtime_float(p.get("base"),10000.0,f"{node.get('name','Sinusoidal Position')} base",minimum=1e-9),
            )
        if t=="esa":
            ESA = IMPORT_POOL.resolve_component("esa")
            return ESA(
                embd=runtime_int(p.get("embd"),384,f"{node.get('name','ESA')} embedding size",minimum=1),
                head=runtime_int(p.get("head"),4,f"{node.get('name','ESA')} head count",minimum=1),
                batch=_none(p.get("batch")), block=_none(p.get("block")),
                backend=backend, precision=precision, compass=p.get("compass","auto"),
                dropout=runtime_float(p.get("dropout"),0.0,f"{node.get('name','ESA')} dropout",minimum=0.0),
                gate_min=runtime_float(p.get("gate_min"),0.8,f"{node.get('name','ESA')} gate min"),
                gate_max=runtime_float(p.get("gate_max"),0.995,f"{node.get('name','ESA')} gate max"),
                eps=runtime_float(p.get("eps"),1e-5,f"{node.get('name','ESA')} epsilon",minimum=0.0),
                device=device, auto_compile=False, auto_move_input=True,
                strict_checks=_bool(p.get("strict_checks",False)),
            )
        if t=="stateaware_esa_stack":
            return _StateAwareESAStack(
                dim=runtime_int(p.get("dim"),384,f"{node.get('name','StateAware ESA')} model dim",minimum=1),
                state_dim=runtime_int(p.get("state_dim"),2749,f"{node.get('name','StateAware ESA')} state dim",minimum=1),
                layers=runtime_int(p.get("layers"),8,f"{node.get('name','StateAware ESA')} layer count",minimum=1),
                heads=runtime_int(p.get("heads"),6,f"{node.get('name','StateAware ESA')} head count",minimum=1),
                block=runtime_int(p.get("block"),256,f"{node.get('name','StateAware ESA')} block size",minimum=1),
                batch=runtime_int(p.get("batch"),16,f"{node.get('name','StateAware ESA')} batch",minimum=1),
                depth_dim=runtime_int(p.get("depth_dim"),64,f"{node.get('name','StateAware ESA')} depth embedding dim",minimum=1),
                compass=runtime_int(p.get("compass"),16,f"{node.get('name','StateAware ESA')} compass",minimum=1),
                backend=backend, precision=precision,
                update_ratio_start=runtime_float(p.get("update_ratio_start"),0.20,f"{node.get('name','StateAware ESA')} update ratio start",minimum=1e-9),
                update_ratio_end=runtime_float(p.get("update_ratio_end"),0.14,f"{node.get('name','StateAware ESA')} update ratio end",minimum=1e-9),
                stream_ratio=runtime_float(p.get("stream_ratio"),1.08,f"{node.get('name','StateAware ESA')} stream ratio",minimum=1e-9),
            )
        if t=="soup":
            SOUP = IMPORT_POOL.resolve_component("soup")
            depth = runtime_int(p.get("depth"), 2, f"{node.get('name','SOUP')} depth", minimum=1)
            width = _scalar_or_sequence(
                p.get("width", 1116), cast=int, label=f"{node.get('name','SOUP')} state width"
            )
            mixer = _scalar_or_sequence(
                p.get("mixer", "esa"), cast=lambda v: str(v).strip().lower(),
                label=f"{node.get('name','SOUP')} mixer",
            )
            ffn = _scalar_or_sequence(
                p.get("ffn", "saffn"), cast=lambda v: str(v).strip().lower(),
                label=f"{node.get('name','SOUP')} FFN",
            )
            return SOUP(
                dim=runtime_int(p.get("dim"), 512, f"{node.get('name','SOUP')} model dim", minimum=1),
                width=width,
                depth=depth,
                mixer=mixer,
                ffn=ffn,
                mixer_config=_config_value(p.get("mixer_config"), label=f"{node.get('name','SOUP')} mixer config"),
                ffn_config=_config_value(p.get("ffn_config"), label=f"{node.get('name','SOUP')} FFN config"),
                backend=backend,
                precision=precision,
                memory_dim=runtime_int(p.get("memory_dim"), 128, f"{node.get('name','SOUP')} memory dim", minimum=1),
                fusion_hidden=runtime_int(p.get("fusion_hidden"), 768, f"{node.get('name','SOUP')} fusion hidden", minimum=1),
            )
        if t=="rmsnorm":
            RMSNorm = IMPORT_POOL.resolve_component("rmsnorm")
            shape=p.get("normalized_shape",p.get("hidden_size",384))
            shape=runtime_int(shape,384,f"{node.get('name','RMSNorm')} normalized shape",minimum=1)
            return RMSNorm(shape,eps=runtime_float(p.get("eps"),1e-6,f"{node.get('name','RMSNorm')} epsilon",minimum=0.0),elementwise_affine=_bool(p.get("elementwise_affine",True)))
        if t=="layernorm":
            LayerNorm = IMPORT_POOL.resolve_component("layernorm")
            shape=p.get("normalized_shape",p.get("hidden_size",p.get("dim",384)))
            shape=runtime_int(shape,384,f"{node.get('name','LayerNorm')} normalized shape",minimum=1)
            return LayerNorm(shape,eps=runtime_float(p.get("eps"),1e-5,f"{node.get('name','LayerNorm')} epsilon",minimum=0.0),elementwise_affine=_bool(p.get("elementwise_affine",True)),bias=_bool(p.get("bias",True)))
        if t=="linear":
            # Deep Learning Core must remain usable even when the optional
            # mlbricks-kit package is unavailable (for example in a clean
            # educational/offline Studio environment). Prefer the public
            # MLBricks Linear implementation when installed, otherwise fall
            # back to the equivalent PyTorch primitive.
            Linear = IMPORT_POOL.resolve_component("linear", required=False) or nn.Linear
            return Linear(
                runtime_int(p.get("in_features"),384,f"{node.get('name','Linear')} input features",minimum=1),
                runtime_int(p.get("out_features"),384,f"{node.get('name','Linear')} output features",minimum=1),
                bias=_bool(p.get("bias",True)),
            )
        if t=="ffn":
            FFN = IMPORT_POOL.resolve_component("ffn")
            return FFN(
                runtime_int(p.get("hidden_size"),384,f"{node.get('name','FFN')} hidden size",minimum=1),
                runtime_int(
                    p.get("intermediate_size"),
                    4*runtime_int(p.get("hidden_size"),384,f"{node.get('name','FFN')} hidden size",minimum=1),
                    f"{node.get('name','FFN')} intermediate size",minimum=1,
                ),
                activation=p.get("activation") or "gelu",
                dropout=runtime_float(p.get("dropout"),0.0,f"{node.get('name','FFN')} dropout",minimum=0.0),
                bias=_bool(p.get("bias",True)), gated=_bool(p.get("gated",False)),
            )
        if t=="residual":
            Residual = IMPORT_POOL.resolve_component("residual")
            return Residual(dropout=runtime_float(p.get("dropout"),0.0,f"{node.get('name','Residual')} dropout",minimum=0.0))
        if t=="dropout":
            probability=p.get("p") if p.get("p") is not None else p.get("dropout")
            return nn.Dropout(runtime_float(probability,0.1,f"{node.get('name','Dropout')} probability",minimum=0.0,maximum=1.0))
        if t=="layer_block":
            dim=runtime_int(p.get("dim") or p.get("hidden_size"),384,f"{node.get('name','Layer Block')} hidden dim",minimum=1)
            return _LayerBlock(
                dim=dim,
                heads=runtime_int(p.get("heads") or p.get("head"),4,f"{node.get('name','Layer Block')} ESA heads",minimum=1),
                ffn_dim=runtime_int(p.get("ffn_dim") or p.get("intermediate_size"),4*dim,f"{node.get('name','Layer Block')} FFN hidden dim",minimum=1),
                batch=runtime_int(p.get("batch"),16,f"{node.get('name','Layer Block')} batch",minimum=1),
                block=runtime_int(p.get("block"),512,f"{node.get('name','Layer Block')} block size",minimum=1),
                backend=backend, precision=precision,
                compass=runtime_int(p.get("compass"),16,f"{node.get('name','Layer Block')} compass",minimum=1),
                activation=str(p.get("activation") or "gelu"),
                dropout=runtime_float(p.get("dropout"),0.0,f"{node.get('name','Layer Block')} dropout",minimum=0.0,maximum=1.0),
                norm_eps=runtime_float(p.get("norm_eps"),1e-5,f"{node.get('name','Layer Block')} norm epsilon",minimum=0.0),
                device=device,
            )
        if t=="custom":
            did=node.get("definition_id"); definition=deepcopy(self.custom_components.get(did) or {})
            if not definition: raise ModelCompileError(f"Custom component definition not found for {node.get('name')}.")
            if did in self._custom_stack:
                chain=" -> ".join([*self._custom_stack,did])
                raise ModelCompileError(f"Circular custom component dependency detected: {chain}")
            implementation = str(definition.get("implementation") or "graph")
            if implementation == "api":
                return _APIBoundComponent(
                    definition=definition, params=p, runtime=self.runtime,
                    custom_components=self.custom_components,
                    _custom_stack=(*self._custom_stack,did),
                )
            if implementation == "abstract_layer":
                return _AbstractLayerComponent(
                    definition=definition, params=p, runtime=self.runtime,
                    custom_components=self.custom_components,
                    _custom_stack=(*self._custom_stack,did),
                )
            by_id={n["id"]:n for n in definition.get("nodes") or []}
            for exposed in definition.get("exposed_api") or []:
                key=exposed.get("key"); sid=exposed.get("source_node")
                if key in p and sid in by_id: by_id[sid].setdefault("params",{})[key]=p[key]
            return TensorGraph(
                nodes=list(by_id.values()),edges=definition.get("edges") or [],custom_components=self.custom_components,
                runtime=self.runtime,vocab_override=self.vocab_override,_custom_stack=(*self._custom_stack,did),
            )
        # Not silently faking execution for unsupported advanced blocks.
        raise ModelCompileError(
            f"Training compiler does not yet support component {node.get('name')!r} ({t}). "
            "Supported today: declarative MLBricks API components (including BOLT and ResController), API Function, Text/Image/Audio Input/Output, "
            "Embedding, Learned/Sinusoidal Position, ESA, StateAware ESA Stack, SOUP, RMSNorm, LayerNorm, Linear, FFN, Residual, "
            "Dropout, Previous Value Buffer, ML/DL foundation blocks, tensor/math operations, LM Head, nested custom components, and API-bound custom components. "
            "ElasticBit is a post-training/inference runtime component, not a differentiable training layer."
        )

    def _nearest_upstream_embedding(self, node_id, expected_shape):
        def upstream_ids(nid):
            ids=[]
            ids.extend(self.in_main.get(nid,[]))
            ids.extend(self.in_skip.get(nid,[]))
            ids.extend(self.in_extra.get(nid,[]))
            ids.extend(
                edge.get("source")
                for edge in self.in_named.get(nid,[])
                if edge.get("source") in self.by_id
            )
            # Preserve graph order while avoiding duplicate traversal caused by
            # one source feeding multiple named API arguments.
            return list(dict.fromkeys(ids))

        queue=upstream_ids(node_id)
        seen=set()
        while queue:
            current=queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            node=self.by_id.get(current) or {}
            if node.get("type")=="embedding" and current in self.mods:
                weight=getattr(self.mods[current],"weight",None)
                if weight is not None and tuple(weight.shape)==tuple(expected_shape):
                    return self.mods[current]
            queue.extend(upstream_ids(current))
        return None

    def _apply_weight_tying(self):
        """Resolve Builder LM-head tying to the nearest compatible embedding.

        MLBricks exposes ``LMHead(..., tie_to=<Embedding>)``. A visual graph
        cannot serialize a live module reference, so Builder stores the
        equivalent ``tie_embeddings`` flag and resolves the module here.
        """
        for node in self.nodes:
            if node.get("type")!="lm_head" or node.get("id") not in self.mods:
                continue
            p=node.get("params") or {}
            explicit=p.get("tie_embeddings")
            if explicit is None:
                tie_to=p.get("tie_to")
                enabled=tie_to not in {None,"","none","None","null"}
            else:
                enabled=_bool(explicit)
            if not enabled:
                continue
            head=self.mods[node["id"]]
            expected=(getattr(head,"vocab_size",head.out_features),getattr(head,"hidden_size",head.in_features))
            embedding=self._nearest_upstream_embedding(node["id"],expected)
            if embedding is None:
                raise ModelCompileError(
                    f"{node.get('name','LM Head')} has weight tying enabled but no compatible upstream Embedding "
                    f"with weight shape {expected} was found."
                )
            head.tie_weights(embedding)

    def forward(self, graph_input, graph_skip=None, graph_extra=None, graph_named=None):
        values={}
        def edge_value(edge, lane):
            source_id=edge.get("source")
            source_port=str(edge.get("source_port") or "")
            if source_port.startswith("named_out:"):
                return _named_output(values[source_id], source_port.replace("named_out:","",1))
            return _lane_output(values[source_id], lane)
        for node in self.order:
            nid=node["id"]; t=node.get("type"); mod=self.mods[nid]
            main_sources=self.in_main[nid]; skip_sources=self.in_skip[nid]; extra_sources=self.in_extra[nid]; named_edges=self.in_named[nid]
            named_inputs={}
            for named_edge in named_edges:
                source_id=named_edge.get("source")
                source_port=str(named_edge.get("source_port") or "main_out")
                if source_port.startswith("named_out:"):
                    source_key=source_port.replace("named_out:","",1)
                elif "skip" in source_port:
                    source_key="skip"
                elif "extra" in source_port:
                    source_key="extra"
                else:
                    source_key="main"
                target_key=str(named_edge.get("target_port") or "").replace("named_in:","",1)
                if source_id not in values:
                    raise ModelCompileError(f"Named input for {node.get('name')} is not available from upstream node {source_id!r}.")
                named_inputs[target_key]=_named_output(values[source_id],source_key)
            if main_sources:
                if len(main_sources)!=1: raise ModelCompileError(f"{node.get('name')} has {len(main_sources)} Main inputs; merge execution is not implemented.")
                x=edge_value(self.in_main_edges[nid][0], "main")
            else:
                input_key=str((node.get("params") or {}).get("input_key") or "").strip()
                if input_key and graph_named is not None and input_key in graph_named:
                    x=graph_named[input_key]
                else:
                    x=graph_input
            stored_value = None
            contract = API_COMPONENTS.get(t)
            if t == "abstract_input":
                y = _APIMixedOutputs(
                    _APILaneOutputs(main=graph_input, skip=graph_skip, extra=graph_extra),
                    dict(graph_named or {}),
                )
                stored_value = y
            elif t == "abstract_output":
                if not main_sources:
                    raise ModelCompileError("Layer Outputs requires a Main input from the internal layer graph.")
                if len(skip_sources) > 1 or len(extra_sources) > 1:
                    raise ModelCompileError("Layer Outputs accepts at most one Skip and one Extra input.")
                skip_value = edge_value(self.in_skip_edges[nid][0], "skip") if skip_sources else None
                extra_value = edge_value(self.in_extra_edges[nid][0], "extra") if extra_sources else None
                y = _APIMixedOutputs(
                    _APILaneOutputs(main=x, skip=skip_value, extra=extra_value),
                    named_inputs,
                )
                stored_value = y
            elif contract is not None:
                declared=set(contract.input_ports)
                contract_inputs={}

                if "main" in declared:
                    contract_inputs["main"] = x
                elif main_sources:
                    raise ModelCompileError(
                        f"{node.get('name')} received a Main input, but its MLBricks API contract does not declare a Main port."
                    )

                if "skip" in declared:
                    if len(skip_sources)>1:
                        raise ModelCompileError(f"{node.get('name')} has {len(skip_sources)} Skip inputs; exactly one is allowed.")
                    skip_value=edge_value(self.in_skip_edges[nid][0], "skip") if skip_sources else graph_skip
                    if skip_value is None:
                        api_arg=contract.input_ports.get("skip", "skip")
                        raise ModelCompileError(
                            f"{node.get('name')} requires the Skip input for MLBricks argument {api_arg!r}."
                        )
                    contract_inputs["skip"] = skip_value
                elif skip_sources:
                    raise ModelCompileError(
                        f"{node.get('name')} received a Skip input, but its MLBricks API contract does not declare a Skip port."
                    )

                if "extra" in declared:
                    if len(extra_sources)>1:
                        raise ModelCompileError(f"{node.get('name')} has {len(extra_sources)} Extra inputs; exactly one is allowed.")
                    extra_value=edge_value(self.in_extra_edges[nid][0], "extra") if extra_sources else graph_extra
                    if extra_value is None:
                        api_arg=contract.input_ports.get("extra", "extra")
                        raise ModelCompileError(
                            f"{node.get('name')} requires the Extra input for MLBricks argument {api_arg!r}."
                        )
                    contract_inputs["extra"] = extra_value
                elif extra_sources:
                    raise ModelCompileError(
                        f"{node.get('name')} received an Extra input, but its MLBricks API contract does not declare an Extra port."
                    )

                undeclared_named=sorted(set(named_inputs)-declared)
                if undeclared_named:
                    raise ModelCompileError(
                        f"{node.get('name')} received undeclared named input port(s): {', '.join(undeclared_named)}."
                    )
                named_declared = declared - {"main", "skip", "extra"}
                # Resolve every explicitly connected named input first so a
                # declarative initializer can reference another logical input
                # regardless of Python set iteration order.
                for port in named_declared:
                    if port in named_inputs:
                        contract_inputs[port]=named_inputs[port]

                for port in named_declared:
                    if port in contract_inputs:
                        continue
                    default_value, initialized = contract.initialize_input(
                        port, contract_inputs, node, mod
                    )
                    if initialized:
                        contract_inputs[port]=default_value
                        continue

                    api_arg=contract.input_ports.get(port, port)
                    raise ModelCompileError(
                        f"{node.get('name')} requires named input {port!r} for MLBricks argument {api_arg!r}."
                    )

                result = contract.execute(mod, contract_inputs)
                y = result.get("main")
                repeat=max(1,int(node.get("repeat") or 1))
                for _ in range(1,repeat):
                    if "main" not in contract_inputs or "main" not in result:
                        raise ModelCompileError(
                            f"{node.get('name')} cannot Repeat because its MLBricks API contract has no Main input/output feedback path."
                        )
                    contract_inputs["main"] = y
                    result = contract.execute(mod, contract_inputs)
                    y = result.get("main")
                stored_value = result if len(result) > 1 or set(result) != {"main"} else y
            elif t=="layer_block":
                signal_value=named_inputs.get("signal", x)
                residual_value=named_inputs.get("residual")
                if residual_value is None:
                    if len(skip_sources)>1:
                        raise ModelCompileError(f"Layer Block {node.get('name')} has multiple Residual inputs.")
                    residual_value=edge_value(self.in_skip_edges[nid][0], "skip") if skip_sources else signal_value
                y=mod(signal_value,residual_value)
                repeat=max(1,int(node.get("repeat") or 1))
                for _ in range(1,repeat):
                    feedback=_named_output(y,"signal")
                    residual_feedback=_named_output(y,"residual")
                    y=mod(feedback,residual_feedback)
                stored_value=y
            elif t=="fpn_fusion":
                high=named_inputs.get("high")
                lateral=named_inputs.get("lateral")
                if high is None or lateral is None:
                    missing=[]
                    if high is None: missing.append("High-Level Feature")
                    if lateral is None: missing.append("Lateral Feature")
                    raise ModelCompileError(f"{node.get('name')} requires connected input(s): {', '.join(missing)}.")
                if main_sources or skip_sources or extra_sources:
                    raise ModelCompileError(f"{node.get('name')} uses named High/Lateral terminals; disconnect legacy Main/Skip/Extra inputs.")
                y=mod(high,lateral)
            elif t=="pan_fusion":
                fine=named_inputs.get("fine")
                coarse=named_inputs.get("coarse")
                if fine is None or coarse is None:
                    missing=[]
                    if fine is None: missing.append("Fine Feature")
                    if coarse is None: missing.append("Coarse Feature")
                    raise ModelCompileError(f"{node.get('name')} requires connected input(s): {', '.join(missing)}.")
                if main_sources or skip_sources or extra_sources:
                    raise ModelCompileError(f"{node.get('name')} uses named Fine/Coarse terminals; disconnect legacy Main/Skip/Extra inputs.")
                y=mod(fine,coarse)
            elif t=="detection_pyramid_head":
                p3=named_inputs.get("p3")
                p4=named_inputs.get("p4")
                p5=named_inputs.get("p5")
                if p3 is None or p4 is None or p5 is None:
                    missing=[]
                    if p3 is None: missing.append("P3")
                    if p4 is None: missing.append("P4")
                    if p5 is None: missing.append("P5")
                    raise ModelCompileError(f"{node.get('name')} requires connected input(s): {', '.join(missing)}.")
                if main_sources or skip_sources or extra_sources:
                    raise ModelCompileError(f"{node.get('name')} uses named P3/P4/P5 terminals; disconnect legacy Main/Skip/Extra inputs.")
                y=mod(p3,p4,p5)
            elif t=="jepa_latent_loss":
                prediction=named_inputs.get("prediction")
                target=named_inputs.get("target")
                if prediction is None or target is None:
                    missing=[]
                    if prediction is None: missing.append("Prediction")
                    if target is None: missing.append("Target")
                    raise ModelCompileError(f"{node.get('name')} requires connected input(s): {', '.join(missing)}.")
                if main_sources or skip_sources or extra_sources:
                    raise ModelCompileError(f"{node.get('name')} uses named Prediction/Target terminals; disconnect legacy Main/Skip/Extra inputs.")
                y=mod(prediction,target)
            elif t in {"matmul","tensor_add","tensor_subtract","tensor_multiply","tensor_divide","concat"}:
                # These first-principles math blocks intentionally expose two
                # logical named inputs so both operands are explicit on the
                # canvas instead of hiding one operand in a special residual lane.
                a=named_inputs.get("a")
                b=named_inputs.get("b")
                if a is None or b is None:
                    missing=[]
                    if a is None: missing.append("A")
                    if b is None: missing.append("B")
                    raise ModelCompileError(f"{node.get('name')} requires connected operand(s): {', '.join(missing)}.")
                if main_sources or skip_sources or extra_sources:
                    raise ModelCompileError(f"{node.get('name')} uses its named A/B operand terminals; disconnect legacy Main/Skip/Extra inputs.")
                y=mod(a,b)
            elif t=="residual":
                if len(skip_sources)!=1: raise ModelCompileError(f"Residual {node.get('name')} needs exactly one Skip input.")
                if extra_sources: raise ModelCompileError(f"Residual {node.get('name')} does not accept an Extra input.")
                y=mod(edge_value(self.in_skip_edges[nid][0], "skip"),x)
            elif t=="api_step":
                if len(skip_sources)>1 or len(extra_sources)>1:
                    raise ModelCompileError(f"API function {node.get('name')} accepts at most one Skip and one Extra tensor lane.")
                skip_value=edge_value(self.in_skip_edges[nid][0], "skip") if skip_sources else graph_skip
                extra_value=edge_value(self.in_extra_edges[nid][0], "extra") if extra_sources else graph_extra
                y=mod(x,skip=skip_value,extra=extra_value,named_inputs=named_inputs)
                repeat=max(1,int(node.get("repeat") or 1))
                if repeat>1 and getattr(mod,"port_mode","standard")=="named":
                    raise ModelCompileError(f"Named-port User Function {node.get('name')} cannot use Repeat > 1; connect another function node instead.")
                for _ in range(1,repeat):
                    repeat_input=_lane_output(y,"main")
                    y=mod(repeat_input,skip=skip_value,extra=extra_value,named_inputs=named_inputs)
            elif t=="custom" and str((self.custom_components.get(node.get("definition_id")) or {}).get("implementation") or "graph") == "abstract_layer":
                if len(skip_sources)>1 or len(extra_sources)>1:
                    raise ModelCompileError(f"Abstract Layer {node.get('name')} accepts at most one Skip and one Extra tensor lane.")
                skip_value=edge_value(self.in_skip_edges[nid][0], "skip") if skip_sources else graph_skip
                extra_value=edge_value(self.in_extra_edges[nid][0], "extra") if extra_sources else graph_extra
                y=mod(x,skip=skip_value,extra=extra_value,named_inputs=named_inputs)
                repeat=max(1,int(node.get("repeat") or 1))
                for _ in range(1,repeat):
                    y=mod(
                        _lane_output(y,"main"),
                        skip=(getattr(getattr(y,"standard",None),"skip",None) if isinstance(y,_APIMixedOutputs) else None),
                        extra=(getattr(getattr(y,"standard",None),"extra",None) if isinstance(y,_APIMixedOutputs) else None),
                        named_inputs=(dict(y.values) if isinstance(y,_APIMixedOutputs) else {}),
                    )
                stored_value=y
            elif t=="custom" and str((self.custom_components.get(node.get("definition_id")) or {}).get("implementation") or "graph") == "api":
                if len(skip_sources)>1 or len(extra_sources)>1:
                    raise ModelCompileError(f"API component {node.get('name')} accepts at most one Skip and one Extra tensor lane.")
                skip_value=edge_value(self.in_skip_edges[nid][0], "skip") if skip_sources else None
                extra_value=edge_value(self.in_extra_edges[nid][0], "extra") if extra_sources else None
                y=mod(x,skip=skip_value,extra=extra_value)
                repeat=max(1,int(node.get("repeat") or 1))
                for _ in range(1,repeat): y=mod(y,skip=skip_value,extra=extra_value)
            else:
                if skip_sources: raise ModelCompileError(f"{node.get('name')} has a Skip input but this component does not consume Skip tensors.")
                if extra_sources: raise ModelCompileError(f"{node.get('name')} has an Extra input but this component does not consume Extra tensors.")
                y=mod(x)
                repeat=max(1,int(node.get("repeat") or 1))
                for _ in range(1,repeat): y=mod(y)
            values[nid]=stored_value if stored_value is not None else y
        sinks=[n for n in self.order if not self.outgoing[n["id"]]]
        if not sinks: raise ModelCompileError("Graph has no output node.")
        if len(sinks)>1: raise ModelCompileError("Training compiler currently requires one tensor output.")
        sink=sinks[0]
        value=values[sink["id"]]
        if str(sink.get("type") or "") == "abstract_output":
            return value
        return _lane_output(value, "main")

    def recurrent_generation_support(self):
        """Return whether this visual graph has an exact token-step execution path.

        Generation is deliberately capability based.  Sequence mixers must expose
        an explicit prefill/decode contract; pointwise layers are safe to run on
        one token; arbitrary API/user functions are not guessed to be causal.
        """
        pointwise = {
            "text_input", "image_input", "audio_input", "video_input", "signal_input", "stream_input", "text_output", "logits_output",
            "embedding", "lm_head", "learned_position", "sinusoidal_position",
            "rmsnorm", "layernorm", "linear", "ffn", "residual", "dropout",
            "value_buffer", "abstract_input", "abstract_output",
        }
        for node in self.order:
            nid=node["id"]
            component_type=str(node.get("type") or "")
            module=self.mods[nid]
            implementation = str((self.custom_components.get(node.get("definition_id")) or {}).get("implementation") or "graph") if component_type == "custom" else ""
            if self.in_named.get(nid) and component_type != "layer_block" and not (component_type == "custom" and implementation == "abstract_layer") and component_type != "abstract_output":
                return False, f"{node.get('name', component_type)} uses named state ports"
            if component_type == "esa":
                if not callable(getattr(module,"prefill",None)) or not callable(getattr(module,"decode_step",None)):
                    return False, f"{node.get('name','ESA')} does not expose prefill/decode_step"
            elif component_type == "bolt":
                required=("prefill_with_cache","project_decode_state","decode_append_projected")
                if not all(callable(getattr(module,name,None)) for name in required):
                    return False, f"{node.get('name','BOLT')} does not expose fixed-cache decoding"
            elif component_type == "soup":
                if not callable(getattr(module,"prefill",None)) or not callable(getattr(module,"decode_step",None)):
                    return False, f"{node.get('name','SOUP')} does not expose recurrent generation"
            elif component_type == "layer_block":
                if not callable(getattr(module,"prefill",None)) or not callable(getattr(module,"decode_step",None)):
                    return False, f"{node.get('name','Layer Block')} does not expose recurrent ESA generation"
            elif component_type == "custom" and isinstance(module,_AbstractLayerComponent):
                supported,reason=module.recurrent_generation_support()
                if not supported:
                    return False, f"{node.get('name','Abstract Layer')}: {reason}"
            elif component_type == "custom" and isinstance(module,TensorGraph):
                if self.in_skip.get(nid) or self.in_extra.get(nid):
                    return False, f"{node.get('name','Module')} uses external Skip/Extra generation inputs"
                supported,reason=module.recurrent_generation_support()
                if not supported:
                    return False, f"{node.get('name','Module')}: {reason}"
            elif component_type in pointwise:
                pass
            elif component_type == "rescontroller":
                # ResController is a pointwise residual/update merge.  BOLT is
                # the only other declarative API component with a recurrent path.
                pass
            else:
                return False, f"{node.get('name', component_type)} has no verified token-step contract"
        return True, None

    def recurrent_generation_algorithms(self):
        algorithms=[]
        for node in self.order:
            component_type=str(node.get("type") or "")
            module=self.mods[node["id"]]
            if component_type == "esa": algorithms.append("ESA Thunder prefill + Lightning decode")
            elif component_type == "bolt": algorithms.append("BOLT fixed C/rho cache decode")
            elif component_type == "soup": algorithms.append("SOUP recurrent generation")
            elif component_type == "layer_block": algorithms.append("ESA Layer Block recurrent generation")
            elif component_type == "learned_position": algorithms.append("Cyclic learned-position continuation")
            elif component_type == "custom" and isinstance(module,_AbstractLayerComponent):
                algorithms.extend(module.recurrent_generation_algorithms())
            elif component_type == "custom" and isinstance(module,TensorGraph):
                algorithms.extend(module.recurrent_generation_algorithms())
        return list(dict.fromkeys(algorithms))

    def _terminal_generation_head(self,nid):
        """True when every consumer after an LM head is only an output identity."""
        pending=list(self.outgoing.get(nid) or [])
        seen=set()
        while pending:
            current=pending.pop()
            if current in seen: continue
            seen.add(current)
            node=self.by_id[current]
            if node.get("type") not in {"text_output","logits_output"}: return False
            pending.extend(self.outgoing.get(current) or [])
        return True

    @staticmethod
    def _bolt_prefill(module,x,capacity,start_pos=0):
        y,(prefix_c,prefix_rho)=module.prefill_with_cache(x,start_pos=int(start_pos))
        prefix=int(prefix_c.size(2))
        capacity=max(prefix,int(capacity))
        c=prefix_c.new_empty(prefix_c.size(0),prefix_c.size(1),capacity,prefix_c.size(3))
        rho=prefix_rho.new_empty(prefix_rho.size(0),prefix_rho.size(1),capacity)
        c[:,:,:prefix,:].copy_(prefix_c)
        rho[:,:,:prefix].copy_(prefix_rho)
        return y,{"c":c,"rho":rho,"length":prefix}

    @staticmethod
    def _bolt_decode(module,x,state,position):
        q,c_now,rho_now=module.project_decode_state(x,start_pos=int(position))
        y=module.decode_append_projected(
            q,c_now,rho_now,state["c"],state["rho"],position=int(position)
        )[:,None,:]
        state["length"]=max(int(state.get("length",0)),int(position)+1)
        return y,state

    def _generation_transform(self,node,module,x,run):
        phase=run["phase"]
        nid=node["id"]
        component_type=str(node.get("type") or "")
        repeat=max(1,int(node.get("repeat") or 1))
        states=[] if phase=="prefill" else list(run["states"].get(nid) or [])
        output=x
        next_states=[]
        for index in range(repeat):
            state=states[index] if index<len(states) else None
            if component_type == "esa":
                if phase=="prefill": output,state=module.prefill(output)
                else: output,state=module.decode_step(output,state)
                next_states.append(state)
            elif component_type == "bolt":
                if phase=="prefill":
                    output,state=self._bolt_prefill(module,output,run["capacity"],run["position"])
                else:
                    output,state=self._bolt_decode(module,output,state,run["position"])
                next_states.append(state)
            elif component_type == "soup":
                if phase=="prefill":
                    prepare=getattr(module,"prepare_generation",None)
                    if callable(prepare): prepare(fast=True)
                    output,state=module.prefill(output)
                else: output,state=module.decode_step(output,state)
                next_states.append(state)
            elif component_type == "custom" and isinstance(module,TensorGraph):
                if phase=="prefill":
                    output,state=module.prefill(output,capacity=run["capacity"])
                else: output,state=module.decode_step(output,state,position=run["position"])
                next_states.append(state)
            elif component_type in {"learned_position","sinusoidal_position"}:
                start_pos=int(run["position"])
                if component_type=="learned_position":
                    # Recurrent state mixers can decode beyond the training
                    # context without replaying it. Absolute learned-position
                    # tables are finite, so cycle their index rather than doing
                    # a full 512-token rebuild for every overflow token.
                    params=node.get("params") or {}
                    max_positions=int(
                        params.get("max_seq_len") or params.get("max_length")
                        or params.get("num_embeddings") or 0
                    )
                    if max_positions>0: start_pos%=max_positions
                output=module(output,start_pos=start_pos)
            elif component_type == "lm_head" and self._terminal_generation_head(nid):
                # The full-sequence hidden states are useful to earlier layers,
                # but generation only needs vocabulary scores for the last one.
                output=module(output[:,-1:,:])
            else:
                output=module(output)
        if next_states: run["next_states"][nid]=next_states
        return output

    def _generation_execute(self,graph_input,run,graph_skip=None,graph_extra=None,graph_named=None):
        values={}
        def edge_value(edge,lane):
            source_id=edge.get("source")
            source_port=str(edge.get("source_port") or "")
            if source_port.startswith("named_out:"):
                return _named_output(values[source_id], source_port.replace("named_out:","",1))
            return _lane_output(values[source_id],lane)
        for node in self.order:
            nid=node["id"]; component_type=node.get("type"); module=self.mods[nid]
            main_sources=self.in_main[nid]; skip_sources=self.in_skip[nid]; extra_sources=self.in_extra[nid]
            named_inputs={}
            for named_edge in self.in_named[nid]:
                source_id=named_edge.get("source")
                source_port=str(named_edge.get("source_port") or "main_out")
                if source_port.startswith("named_out:"):
                    source_key=source_port.replace("named_out:","",1)
                elif "skip" in source_port:
                    source_key="skip"
                elif "extra" in source_port:
                    source_key="extra"
                else:
                    source_key="main"
                target_key=str(named_edge.get("target_port") or "").replace("named_in:","",1)
                named_inputs[target_key]=_named_output(values[source_id],source_key)
            if main_sources:
                if len(main_sources)!=1: raise ModelCompileError(f"{node.get('name')} has {len(main_sources)} Main inputs; merge execution is not implemented.")
                x=edge_value(self.in_main_edges[nid][0],"main")
            else: x=graph_input

            contract=API_COMPONENTS.get(component_type)
            if component_type=="abstract_input":
                y=_APIMixedOutputs(
                    _APILaneOutputs(main=graph_input,skip=graph_skip,extra=graph_extra),
                    dict(graph_named or {}),
                )
            elif component_type=="abstract_output":
                if not main_sources:
                    raise ModelCompileError("Layer Outputs requires a Main input from the internal layer graph.")
                if len(skip_sources)>1 or len(extra_sources)>1:
                    raise ModelCompileError("Layer Outputs accepts at most one Skip and one Extra input.")
                skip_value=edge_value(self.in_skip_edges[nid][0],"skip") if skip_sources else None
                extra_value=edge_value(self.in_extra_edges[nid][0],"extra") if extra_sources else None
                y=_APIMixedOutputs(
                    _APILaneOutputs(main=x,skip=skip_value,extra=extra_value),
                    named_inputs,
                )
            elif component_type=="bolt":
                if skip_sources or extra_sources: raise ModelCompileError("BOLT recurrent generation accepts only its Main input.")
                y=self._generation_transform(node,module,x,run)
            elif component_type=="layer_block":
                signal_value=named_inputs.get("signal",x)
                residual_value=named_inputs.get("residual")
                if residual_value is None:
                    residual_value=edge_value(self.in_skip_edges[nid][0],"skip") if skip_sources else signal_value
                states=[] if run["phase"]=="prefill" else list(run["states"].get(nid) or [])
                state=states[0] if states else None
                if run["phase"]=="prefill":
                    y,state=module.prefill(signal_value,residual_value)
                else:
                    y,state=module.decode_step(signal_value,state,residual_value)
                run["next_states"][nid]=[state]
            elif contract is not None:
                declared=set(contract.input_ports); inputs={}
                if "main" in declared: inputs["main"]=x
                if "skip" in declared:
                    if len(skip_sources)>1: raise ModelCompileError(f"{node.get('name')} has multiple Skip inputs.")
                    value=edge_value(self.in_skip_edges[nid][0],"skip") if skip_sources else graph_skip
                    if value is None: raise ModelCompileError(f"{node.get('name')} requires its Skip input.")
                    inputs["skip"]=value
                if "extra" in declared:
                    if len(extra_sources)>1: raise ModelCompileError(f"{node.get('name')} has multiple Extra inputs.")
                    value=edge_value(self.in_extra_edges[nid][0],"extra") if extra_sources else graph_extra
                    if value is None: raise ModelCompileError(f"{node.get('name')} requires its Extra input.")
                    inputs["extra"]=value
                result=contract.execute(module,inputs)
                y=result.get("main")
                for _ in range(1,max(1,int(node.get("repeat") or 1))):
                    inputs["main"]=y; result=contract.execute(module,inputs); y=result.get("main")
            elif component_type=="residual":
                if len(skip_sources)!=1: raise ModelCompileError(f"Residual {node.get('name')} needs exactly one Skip input.")
                y=module(edge_value(self.in_skip_edges[nid][0],"skip"),x)
            elif component_type=="custom" and isinstance(module,_AbstractLayerComponent):
                if len(skip_sources)>1 or len(extra_sources)>1:
                    raise ModelCompileError(f"Abstract Layer {node.get('name')} accepts at most one Skip and one Extra generation input.")
                skip_value=edge_value(self.in_skip_edges[nid][0],"skip") if skip_sources else graph_skip
                extra_value=edge_value(self.in_extra_edges[nid][0],"extra") if extra_sources else graph_extra
                states=[] if run["phase"]=="prefill" else list(run["states"].get(nid) or [])
                state=states[0] if states else None
                if run["phase"]=="prefill":
                    y,state=module.prefill(
                        x,skip=skip_value,extra=extra_value,named_inputs=named_inputs,capacity=run["capacity"]
                    )
                else:
                    y,state=module.decode_step(
                        x,state,skip=skip_value,extra=extra_value,named_inputs=named_inputs,position=run["position"]
                    )
                run["next_states"][nid]=[state]
            elif component_type=="custom" and isinstance(module,TensorGraph):
                y=self._generation_transform(node,module,x,run)
            else:
                if skip_sources or extra_sources:
                    raise ModelCompileError(f"{node.get('name')} has unsupported auxiliary inputs during recurrent generation.")
                y=self._generation_transform(node,module,x,run)
            values[nid]=y
        sinks=[n for n in self.order if not self.outgoing[n["id"]]]
        if len(sinks)!=1: raise ModelCompileError("Recurrent generation requires exactly one tensor output.")
        sink=sinks[0]
        value=values[sink["id"]]
        if str(sink.get("type") or "") == "abstract_output":
            return value
        return _lane_output(value,"main")

    @torch.no_grad()
    def prefill(self,graph_input,*,capacity=None,graph_skip=None,graph_extra=None,graph_named=None):
        supported,reason=self.recurrent_generation_support()
        if not supported: raise ModelCompileError(reason or "Graph has no recurrent generation path.")
        if graph_input.ndim<2 or graph_input.size(1)<1: raise ValueError("prefill expects a non-empty [B,T,...] input")
        run={"phase":"prefill","states":{},"next_states":{},"position":0,
             "capacity":max(int(capacity or graph_input.size(1)),int(graph_input.size(1)))}
        output=self._generation_execute(graph_input,run,graph_skip,graph_extra,graph_named)
        cache={"states":run["next_states"],"position":int(graph_input.size(1)),"capacity":run["capacity"]}
        return output,cache

    @torch.no_grad()
    def decode_step(self,graph_input,cache,*,position=None,graph_skip=None,graph_extra=None,graph_named=None):
        if graph_input.ndim<2 or graph_input.size(1)!=1: raise ValueError("decode_step expects one token per batch")
        position=int(cache.get("position",0) if position is None else position)
        if position>=int(cache.get("capacity",position+1)): raise ValueError("generation cache capacity exceeded")
        run={"phase":"decode","states":cache.get("states") or {},"next_states":{},
             "position":position,"capacity":int(cache.get("capacity",position+1))}
        output=self._generation_execute(graph_input,run,graph_skip,graph_extra,graph_named)
        cache={"states":run["next_states"],"position":position+1,"capacity":run["capacity"]}
        return output,cache



@dataclass
class CompiledModel:
    # ``model`` is the inference/evaluation model. ``training_model`` is a
    # loss-wrapped whole-model graph used only for training when requested.
    model: nn.Module
    raw_model: nn.Module
    training_model: nn.Module | None
    device: torch.device
    precision: str
    vocab_size: int
    parameter_count: int
    compile_used: bool
    compile_error: str | None


class _CausalLMTrainingGraph(nn.Module):
    """Whole language-model training graph including cross entropy.

    Keeping the loss inside this wrapper lets ``torch.compile`` capture the
    model + LM head + loss as one graph, matching the benchmark notebook rather
    than compiling only the logits-producing portion.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_ids)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
        )


def _root_model(state):
    ws=(state.get("workspaces") or {}).get("model") or {}
    root_id=ws.get("root_component_id") or state.get("root_component_id")
    comp=(state.get("components") or {}).get(root_id)
    if not comp: raise ModelCompileError("Model Builder graph was not found.")
    return comp


def _graph_vocab(model_graph):
    sizes=[]
    for n in model_graph.get("nodes") or []:
        p=n.get("params") or {}
        if n.get("type") in {"embedding","lm_head"} and p.get("vocab_size"):
            sizes.append(int(p["vocab_size"]))
    return max(sizes) if sizes else 0


def _tokenizer_for(meta, *, local_only_first=True, tokenizer_path=None):
    tok_cfg=((meta or {}).get("pipeline") or {}).get("tokenizer") or {}
    name=tok_cfg.get("tokenizer_name") or "gpt2"
    preferred=None
    if tokenizer_path:
        try:
            candidate=Path(str(tokenizer_path)).expanduser()
            if candidate.exists(): preferred=str(candidate.resolve())
        except Exception:
            preferred=None
    cache_key=preferred or str(name)
    cached=_TOKENIZER_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Training/generation needs transformers. Install transformers in the notebook.") from exc
    errors=[]
    tok=None
    if preferred:
        try:
            tok=AutoTokenizer.from_pretrained(preferred,local_files_only=True)
        except Exception as exc:
            errors.append(exc)
    if tok is None and local_only_first:
        try: tok=AutoTokenizer.from_pretrained(name,local_files_only=True)
        except Exception as exc: errors.append(exc); tok=None
    if tok is None:
        try: tok=AutoTokenizer.from_pretrained(name)
        except Exception as exc:
            detail=str(errors[-1]) if errors else ""
            raise RuntimeError(f"Tokenizer {name!r} is unavailable. {detail} {exc}") from exc
    if tok.pad_token_id is None:
        if tok.eos_token_id is not None: tok.pad_token=tok.eos_token
        elif tok.unk_token_id is not None: tok.pad_token=tok.unk_token
        else: tok.add_special_tokens({"pad_token":"<|pad|>"})
    _TOKENIZER_CACHE[cache_key]=tok
    # Also cache by canonical tokenizer name when a saved tokenizer directory was
    # used. A later runtime with the same tokenizer can reuse it immediately.
    _TOKENIZER_CACHE.setdefault(str(name),tok)
    return tok


def _model_requires_tokenizer(model_entry):
    requirements = dict((model_entry or {}).get("requirements") or {})
    training_mode = str(requirements.get("training_mode") or "").strip().lower()
    training_task = str(requirements.get("training_task") or "").strip().lower()
    # Text JEPA uses the visible JEPA Preparation byte-token pipeline and does
    # not require a Hugging Face tokenizer or a causal-LM loss wrapper.
    if training_mode in {"jepa", "audio_generation", "multimodal"} or training_task in {"jepa", "multimodal_jepa", "sensor_vision_fusion", "tts", "voice_tts", "voice_clone", "music_generation", "sound_generation"}:
        return False
    modality = str(requirements.get("modality") or "text").strip().lower()
    # Older saved Studio builds did not record modality. Preserve their language
    # model behavior by treating unknown/blank as text.
    return modality in {"", "unknown", "text"}


def compile_builder_model(state, model_entry, dataset_meta, runtime, *, progress=None, for_training=False):
    """Build a Builder model against the current public MLBricks runtime.

    Training compilation wraps the *entire* causal-LM forward including loss
    and uses one ``torch.compile`` call with ``fullgraph=True`` and
    ``dynamic=False`` so the requested benchmark flags reach Dynamo unchanged. Inference/generation compilation remains shape-friendly
    and compiles only the model forward.
    """
    graph=copy.deepcopy((model_entry or {}).get("architecture") or _root_model(state))
    custom_components=copy.deepcopy(state.get("custom_components") or {})
    custom_components.update(
        copy.deepcopy((model_entry or {}).get("custom_components_snapshot") or {})
    )
    device=resolve_device(runtime.get("device","auto"))
    precision,dtype=resolve_precision(runtime.get("precision","fp16"),device)
    tokenizer=None
    # Only text/language graphs require a tokenizer. Generic supervised image,
    # signal and tabular training must compile without trying to construct a
    # Hugging Face tokenizer. Older text models remain text by default.
    requires_tokenizer=_model_requires_tokenizer(model_entry)
    if requires_tokenizer:
        if progress:
            progress({
                "status":"running","runtime_kind":"train" if for_training else "generate",
                "phase":"tokenizer","overall":0,
                "message":f"Loading tokenizer for {device}…",
            })
        tokenizer=_tokenizer_for(dataset_meta, tokenizer_path=(model_entry or {}).get("tokenizer_path"))
    graph_vocab=_graph_vocab(graph)
    tokenizer_vocab=len(tokenizer) if tokenizer is not None else graph_vocab
    effective_vocab=max(graph_vocab,tokenizer_vocab)
    if progress:
        msg=f"Building model on {device}"
        if tokenizer is not None and effective_vocab!=graph_vocab: msg+=f" · vocab {graph_vocab:,} → {effective_vocab:,} to match tokenizer"
        progress({"status":"running","runtime_kind":"train" if for_training else "generate","phase":"compile","overall":1,"message":msg})

    # Preflight only the MLBricks APIs actually referenced by this graph.
    # This produces a single clear import error before model construction and
    # warms the same cache later used by each component constructor.
    import_checks = IMPORT_POOL.ensure_graph(
        graph.get("nodes") or [], custom_components
    )
    # Some educational components deliberately have a stock-PyTorch fallback.
    # Their optional MLBricks import must not make the whole graph fail during
    # preflight when the fallback is exactly what _module_for() will use.
    native_fallback_types = {"linear", "embedding"}
    import_failures = [
        status for status in import_checks.values()
        if not status.get("ok") and str(status.get("component_type") or "") not in native_fallback_types
    ]
    if import_failures:
        details = "; ".join(
            f"{item.get('component_type')}: {item.get('error')}"
            for item in import_failures
        )
        raise RuntimeError(f"MLBricks component import preflight failed: {details}")

    try:
        model_settings = copy.deepcopy((model_entry or {}).get("model_settings") or {})
        runtime_context = {
            **runtime,
            "device": str(device),
            "precision": precision,
            "model_dim": model_settings.get("embedding_size") or model_settings.get("model_dim"),
            "heads": model_settings.get("heads"),
            "context_length": (model_entry or {}).get("context_length") or runtime.get("context_length"),
            "batch_size": runtime.get("batch_size") or (model_entry or {}).get("batch_size"),
        }
        raw=TensorGraph(
            nodes=graph.get("nodes") or [],
            edges=graph.get("edges") or [],
            custom_components=custom_components,
            runtime=runtime_context,
            vocab_override=effective_vocab,
        )
    except RuntimeError as exc:
        if str(runtime.get("backend") or "pytorch").lower()=="native" and "native extension is unavailable" in str(exc).lower():
            raise RuntimeError(
                "Backend 'native' is selected, but the optional MLBricks native extension is unavailable. "
                "Open Training Setup and set Backend to 'pytorch', or install/build the native MLBricks extension."
            ) from exc
        raise
    # AMP training needs FP32 master parameters; autocast supplies reduced-
    # precision activations and GradScaler safely scales their gradients. Moving
    # trainable parameters themselves to FP16 makes GradScaler fail during
    # unscale_. Inference has no optimizer/scaler, so it can use the requested
    # reduced parameter dtype directly.
    parameter_dtype = (
        torch.float32
        if for_training and precision in {"fp16", "bf16"}
        else dtype
    )
    raw.to(device=device,dtype=parameter_dtype)
    params=sum(p.numel() for p in raw.parameters())
    inference_model=raw
    is_text_training=bool(for_training and requires_tokenizer)
    training_model=_CausalLMTrainingGraph(raw) if is_text_training else None
    compile_used=False
    compile_error=None

    if str(runtime.get("execution_mode","eager"))=="compiled":
        if not hasattr(torch,"compile"):
            raise RuntimeError("Compiled execution was selected, but torch.compile is unavailable in this PyTorch build.")
        mode=str(runtime.get("compile_mode") or "default")
        # TensorGraph and the training wrapper are ordinary nn.Modules. Use one
        # explicit torch.compile wrapper here so the requested mode/fullgraph/
        # dynamic flags are guaranteed to reach Dynamo exactly as configured.
        # (MLBricks' package-level compile API remains useful for models with
        # their own compile hook; Builder's visual TensorGraph has no such hook.)
        # No eager fallback: a compile failure is surfaced to the user.
        if is_text_training:
            training_model=torch.compile(
                training_model, mode=mode, dynamic=False, fullgraph=True
            )
        else:
            # Generic supervised training computes the selected loss in the
            # Studio trainer, so compile the visual model forward itself. The
            # compiled module shares the same parameters as ``raw`` and the
            # optimizer continues to update the canonical graph weights.
            inference_model=torch.compile(
                raw, mode=mode, dynamic=False if for_training else None, fullgraph=False
            )
        compile_used=True

    return CompiledModel(
        inference_model,raw,training_model,device,precision,effective_vocab,
        params,compile_used,compile_error,
    ),tokenizer


def _universal_tensor_for_runtime(value, *, kind, device, precision):
    """Move a universal numeric/media sample onto the model runtime device."""
    if not isinstance(value, torch.Tensor):
        return value
    tensor=value
    if kind in {"image", "audio", "video", "signal"} and not tensor.is_floating_point():
        tensor=tensor.float()
    if tensor.is_floating_point():
        dtype={"fp16":torch.float16,"bf16":torch.bfloat16,"fp32":torch.float32}.get(str(precision),torch.float32)
        # CPU FP16 kernels are frequently incomplete; keep CPU media/signal
        # inference in FP32 unless BF16 was explicitly selected.
        if device.type=="cpu" and dtype==torch.float16:
            dtype=torch.float32
        tensor=tensor.to(device=device,dtype=dtype)
    else:
        tensor=tensor.to(device=device)
    return tensor


def _tensor_image_data_uri(value):
    """Best-effort PNG preview for image-like model output tensors."""
    import base64
    import io
    import numpy as np
    from PIL import Image

    tensor=value.detach().float().cpu()
    if tensor.ndim==4 and tensor.shape[0]==1:
        tensor=tensor[0]
    if tensor.ndim==3 and tensor.shape[0] in {1,3,4}:
        tensor=tensor.permute(1,2,0)
    if tensor.ndim==2:
        tensor=tensor.unsqueeze(-1)
    if tensor.ndim!=3 or tensor.shape[-1] not in {1,3,4}:
        return None
    arr=tensor.numpy()
    finite=np.isfinite(arr)
    if not finite.any():
        return None
    arr=np.nan_to_num(arr,nan=0.0,posinf=1.0,neginf=0.0)
    lo=float(arr.min()); hi=float(arr.max())
    if lo<0.0 or hi>1.0:
        span=hi-lo
        arr=(arr-lo)/(span if span>1e-12 else 1.0)
    arr=(arr.clip(0.0,1.0)*255.0).astype("uint8")
    if arr.shape[-1]==1:
        arr=arr[...,0]
    image=Image.fromarray(arr)
    buff=io.BytesIO(); image.save(buff,format="PNG")
    payload=base64.b64encode(buff.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}", {"width":image.width,"height":image.height,"format":"png"}


def _tensor_audio_data_uri(value, *, sample_rate=16000):
    """Encode a small waveform tensor as PCM16 WAV for Studio playback."""
    import base64, io, wave
    tensor=value.detach().float().cpu()
    if tensor.ndim>1:
        tensor=tensor.reshape(-1)
    if tensor.numel()<1:
        return None
    tensor=torch.nan_to_num(tensor,nan=0.0,posinf=1.0,neginf=-1.0).clamp(-1,1)
    pcm=(tensor*32767.0).round().to(torch.int16).numpy().tobytes()
    buff=io.BytesIO()
    with wave.open(buff,"wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(int(sample_rate)); wav.writeframes(pcm)
    payload=base64.b64encode(buff.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{payload}", {"sample_rate":int(sample_rate),"samples":int(tensor.numel()),"format":"wav"}


def _encode_audio_condition_text(value, *, length=64):
    if isinstance(value, torch.Tensor):
        return value.long()
    text=str(value or "")
    raw=list(text.encode("utf-8",errors="replace"))[:max(1,int(length))]
    ids=[b+1 for b in raw]
    ids += [0]*(max(1,int(length))-len(ids))
    return torch.tensor([ids],dtype=torch.long)


def universal_output_envelope(value, *, output_type="unknown", input_kind="unknown", task="run"):
    """Serialize arbitrary model results into the Studio universal output contract."""
    output_type=str(output_type or "unknown").lower()
    input_kind=str(input_kind or "unknown").lower()
    task=str(task or "run").lower()
    # Common image-generation pipelines return PIL images directly.
    try:
        from PIL import Image
        if isinstance(value, Image.Image):
            import base64
            import io
            buff=io.BytesIO(); value.save(buff,format="PNG")
            data_uri="data:image/png;base64,"+base64.b64encode(buff.getvalue()).decode("ascii")
            return {"kind":"image","mime":"image/png","data":data_uri,"metadata":{"width":value.width,"height":value.height,"format":"png"}}
    except Exception:
        pass
    try:
        import numpy as np
        if isinstance(value,np.ndarray):
            value=torch.from_numpy(value)
    except Exception:
        pass
    if isinstance(value, torch.Tensor):
        tensor=value.detach().cpu()
        shape=list(tensor.shape)
        meta={"shape":shape,"dtype":str(tensor.dtype).replace("torch.","")}
        image_hint=("image" in output_type or task in {"edit","generate_image","image_edit"})
        if image_hint:
            preview=_tensor_image_data_uri(tensor)
            if preview is not None:
                data_uri,image_meta=preview
                meta.update(image_meta)
                return {"kind":"image","mime":"image/png","data":data_uri,"metadata":meta}
        if output_type=="audio_output" or task in {"tts","voice_tts","voice_clone","music_generation","sound_generation","generate_audio"}:
            preview=_tensor_audio_data_uri(tensor,sample_rate=16000)
            if preview is not None:
                data_uri,audio_meta=preview; meta.update(audio_meta)
                return {"kind":"audio","mime":"audio/wav","data":data_uri,"metadata":meta}
        if input_kind in {"signal","audio"} and tensor.numel()<=65536:
            flat=tensor.reshape(-1).float().tolist()
            meta["samples"]=len(flat)
            return {"kind":"signal","mime":"application/x-mlbricks-signal","data":flat,"metadata":meta}
        flat=tensor.reshape(-1)
        limit=4096
        preview_values=flat[:limit].tolist()
        if flat.numel()>limit:
            meta["truncated"]=True
            meta["total_values"]=int(flat.numel())
        kind="classification" if output_type in {"classifier","classification","logits_output"} else "tensor"
        return {"kind":kind,"mime":"application/x-mlbricks-tensor","data":preview_values,"metadata":meta}
    if isinstance(value, dict):
        return {"kind":"json","mime":"application/json","data":value,"metadata":{}}
    if isinstance(value, (list,tuple)):
        try:
            data=[item.detach().cpu().tolist() if isinstance(item,torch.Tensor) else item for item in value]
        except Exception:
            data=[str(item) for item in value]
        return {"kind":"json","mime":"application/json","data":data,"metadata":{}}
    if isinstance(value, (bytes,bytearray)):
        import base64
        encoded=base64.b64encode(bytes(value)).decode("ascii")
        return {"kind":"file","mime":"application/octet-stream","data":f"data:application/octet-stream;base64,{encoded}","metadata":{"bytes":len(value)}}
    if isinstance(value,str):
        return {"kind":"text","mime":"text/plain","data":value,"metadata":{"characters":len(value)}}
    if value is None:
        return {"kind":"json","mime":"application/json","data":None,"metadata":{}}
    return {"kind":"json","mime":"application/json","data":str(value),"metadata":{"python_type":type(value).__name__}}


def _call_universal_adapter(fn, sample, *, prompt, task, metadata):
    """Call a custom model adapter with only the context it declares."""
    kwargs={"prompt":prompt,"task":task,"metadata":metadata or {}}
    try:
        signature=inspect.signature(fn)
        accepts_kwargs=any(p.kind==inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        if not accepts_kwargs:
            kwargs={key:value for key,value in kwargs.items() if key in signature.parameters}
    except (TypeError,ValueError):
        kwargs={}
    return fn(sample,**kwargs)


def run_universal_inference(compiled, value, *, input_kind, output_type="unknown", task="run", prompt="", metadata=None):
    """Run one universal input through a compiled/runtime model.

    Custom/pretrained modules may expose ``process_input`` or ``predict`` and
    receive prompt/task/metadata when their signatures accept those fields.
    Visual TensorGraph models fall back to ordinary tensor ``forward``.
    """
    multimodal_named={}
    if str(input_kind or "").lower()=="multimodal" and isinstance(value,dict):
        if "image" not in value:
            raise ValueError("Multimodal runtime input requires an 'image' field for the current Step 8 reference models.")
        image_value=value.get("image")
        if not isinstance(image_value,torch.Tensor): image_value=torch.as_tensor(image_value,dtype=torch.float32)
        sample=_universal_tensor_for_runtime(image_value,kind="image",device=compiled.device,precision=compiled.precision)
        if isinstance(sample,torch.Tensor) and sample.ndim==2: sample=sample.unsqueeze(0).unsqueeze(0)
        elif isinstance(sample,torch.Tensor) and sample.ndim==3: sample=sample.unsqueeze(0)
        if value.get("text") is not None:
            multimodal_named["text"]=_encode_audio_condition_text(value.get("text"),length=int((metadata or {}).get("text_length",32))).to(compiled.device)
        if value.get("sensor") is not None:
            sensor_value=value.get("sensor")
            if not isinstance(sensor_value,torch.Tensor): sensor_value=torch.as_tensor(sensor_value,dtype=torch.float32)
            sensor=_universal_tensor_for_runtime(sensor_value,kind="signal",device=compiled.device,precision=compiled.precision)
            if isinstance(sensor,torch.Tensor) and sensor.ndim==1: sensor=sensor.unsqueeze(0)
            multimodal_named["sensor"]=sensor
    elif str(output_type or "").lower()=="audio_output" and str(input_kind or "").lower()=="text":
        sample=_encode_audio_condition_text(value,length=int((metadata or {}).get("text_length",64))).to(compiled.device)
    else:
        sample=_universal_tensor_for_runtime(
            value,kind=input_kind,device=compiled.device,precision=compiled.precision
        )
    raw=compiled.raw_model
    with torch.inference_mode():
        process=getattr(raw,"process_input",None)
        predict=getattr(raw,"predict",None)
        if callable(process):
            result=_call_universal_adapter(process,sample,prompt=prompt,task=task,metadata=metadata)
        elif callable(predict):
            result=_call_universal_adapter(predict,sample,prompt=prompt,task=task,metadata=metadata)
        else:
            if not isinstance(sample,torch.Tensor):
                raise RuntimeError(
                    f"{input_kind.title()} input produced {type(sample).__name__}; this model needs a custom process_input()/predict() adapter or a tensor-compatible input component."
                )
            named=dict(multimodal_named)
            if str(output_type or "").lower()=="audio_output":
                meta=metadata or {}
                named["speaker_id"]=torch.tensor([int(meta.get("speaker_id",0))],dtype=torch.long,device=compiled.device)
                if meta.get("reference_audio") is not None:
                    named["reference_audio"]=_universal_tensor_for_runtime(torch.as_tensor(meta.get("reference_audio"),dtype=torch.float32),kind="audio",device=compiled.device,precision=compiled.precision).reshape(1,-1)
            with _autocast_context(compiled.device,compiled.precision):
                result=compiled.model(sample,graph_named=named) if named else compiled.model(sample)
    if str(output_type or "").lower() in {"detection_head","detection_pyramid_head"} or "detection" in str(task or "").lower():
        detections=decode_detection_predictions(
            result,
            score_threshold=float((metadata or {}).get("score_threshold",0.25)),
            iou_threshold=float((metadata or {}).get("nms_iou_threshold",0.5)),
            max_detections=int((metadata or {}).get("max_detections",100)),
        )
        payload=[[
            {"box_xyxy":[float(v) for v in row[:4].detach().cpu().tolist()],"score":float(row[4].item()),"class_id":int(row[5].item())}
            for row in det
        ] for det in detections]
        # Object detection is a visual result.  Return the processed input image
        # alongside normalized boxes so Studio can render the familiar annotated
        # image instead of exposing raw JSON as the primary output.
        detection_meta={
            "coordinate_space":"normalized_xyxy",
            "detections":sum(len(batch) for batch in payload),
            "batches":len(payload),
        }
        # Prefer the pre-resize source preview supplied by Universal Input.
        # Detection models may operate on tiny educational tensors (for example
        # 16x16), which made the visual result appear as a few pixels.  Boxes are
        # normalized, so they can be overlaid safely on the higher-resolution
        # source preview.
        display_image=(metadata or {}).get("display_image")
        if isinstance(display_image,str) and display_image.startswith("data:image/"):
            detection_meta["input_image"]=display_image
            detection_meta.update({
                "image_width":int((metadata or {}).get("display_width") or (metadata or {}).get("source_width") or 0) or None,
                "image_height":int((metadata or {}).get("display_height") or (metadata or {}).get("source_height") or 0) or None,
                "image_format":str((metadata or {}).get("display_format") or "jpeg"),
                "preview_source":"pre_resize",
            })
        elif isinstance(sample,torch.Tensor):
            preview=_tensor_image_data_uri(sample)
            if preview is not None:
                data_uri,image_meta=preview
                detection_meta["input_image"]=data_uri
                detection_meta.update({
                    "image_width":image_meta.get("width"),
                    "image_height":image_meta.get("height"),
                    "image_format":image_meta.get("format","png"),
                    "preview_source":"model_tensor",
                })
        class_names=(metadata or {}).get("class_names")
        if isinstance(class_names,(list,tuple)):
            detection_meta["class_names"]=[str(v) for v in class_names]
        return {
            "kind":"detection",
            "mime":"application/x-mlbricks-detection",
            "data":payload,
            "metadata":detection_meta,
        }
    # Classification is a semantic result, not a raw tensor dump. Convert
    # logits into probabilities/predicted class and, for image models, carry
    # a preview of the exact processed model input so Studio can present the
    # prediction visually.
    normalized_output_type=str(output_type or "").lower()
    if normalized_output_type in {"classifier","classification","logits_output"} and str(input_kind or "").lower() not in {"signal","audio"}:
        if isinstance(result, torch.Tensor):
            logits=result.detach().float().cpu()
            if logits.ndim==0:
                logits=logits.reshape(1,1)
            elif logits.ndim==1:
                logits=logits.unsqueeze(0)
            elif logits.ndim>2:
                logits=logits.reshape(logits.shape[0],-1)
            first=logits[0]
            if first.numel()==1:
                positive=float(torch.sigmoid(first[0]).item())
                probabilities=[1.0-positive,positive]
                raw_logits=[float(first[0].item())]
            else:
                probabilities=[float(v) for v in torch.softmax(first,dim=-1).tolist()]
                raw_logits=[float(v) for v in first.tolist()]
            predicted_class=int(max(range(len(probabilities)),key=lambda idx: probabilities[idx])) if probabilities else 0
            confidence=float(probabilities[predicted_class]) if probabilities else 0.0
            class_meta={
                "predicted_class":predicted_class,
                "confidence":confidence,
                "probabilities":probabilities,
                "logits":raw_logits,
            }
            class_names=(metadata or {}).get("class_names")
            if isinstance(class_names,(list,tuple)):
                class_meta["class_names"]=[str(v) for v in class_names]
            output_meta={
                "shape":list(result.shape),
                "dtype":str(result.dtype).replace("torch.",""),
                "classes":len(probabilities),
            }
            if str(input_kind or "").lower()=="image":
                display_image=(metadata or {}).get("display_image")
                if isinstance(display_image,str) and display_image.startswith("data:image/"):
                    output_meta["input_image"]=display_image
                    output_meta.update({
                        "image_width":int((metadata or {}).get("display_width") or (metadata or {}).get("source_width") or 0) or None,
                        "image_height":int((metadata or {}).get("display_height") or (metadata or {}).get("source_height") or 0) or None,
                        "image_format":str((metadata or {}).get("display_format") or "jpeg"),
                        "preview_source":"pre_resize",
                    })
                elif isinstance(sample,torch.Tensor):
                    preview=_tensor_image_data_uri(sample)
                    if preview is not None:
                        data_uri,image_meta=preview
                        output_meta["input_image"]=data_uri
                        output_meta.update({
                            "image_width":image_meta.get("width"),
                            "image_height":image_meta.get("height"),
                            "image_format":image_meta.get("format","png"),
                            "preview_source":"model_tensor",
                        })
            if "class_names" in class_meta:
                output_meta["class_names"]=class_meta["class_names"]
            return {
                "kind":"classification",
                "mime":"application/x-mlbricks-classification",
                "data":class_meta,
                "metadata":output_meta,
            }

    # Image reconstruction models (for example the Studio Autoencoder) return
    # an image-shaped tensor through a generic Tensor Output node.  Treat an
    # output that exactly matches the image input shape as a semantic
    # reconstruction result instead of dumping tens of thousands of scalar
    # values into the runtime panel.
    if (
        str(input_kind or "").lower()=="image"
        and normalized_output_type in {"tensor_output","image_output","reconstruction"}
        and isinstance(result,torch.Tensor)
        and isinstance(sample,torch.Tensor)
    ):
        reconstructed=result.detach().float().cpu()
        model_input=sample.detach().float().cpu()
        image_like=False
        if reconstructed.ndim==4 and reconstructed.shape[0]==1 and reconstructed.shape[1] in {1,3,4}:
            image_like=True
        elif reconstructed.ndim==3 and reconstructed.shape[0] in {1,3,4}:
            image_like=True
        same_shape=tuple(reconstructed.shape)==tuple(model_input.shape)
        if image_like and same_shape:
            output_preview=_tensor_image_data_uri(reconstructed)
            input_preview=_tensor_image_data_uri(model_input)
            if output_preview is not None:
                output_uri,output_image_meta=output_preview
                input_uri=None
                input_image_meta={}
                if input_preview is not None:
                    input_uri,input_image_meta=input_preview
                delta=torch.nan_to_num(reconstructed-model_input,nan=0.0,posinf=0.0,neginf=0.0)
                mse=float(delta.pow(2).mean().item())
                mae=float(delta.abs().mean().item())
                psnr=None
                if mse>0.0:
                    psnr=float(10.0*math.log10(1.0/max(mse,1e-12)))
                reconstruction_meta={
                    "shape":list(reconstructed.shape),
                    "dtype":str(result.dtype).replace("torch.",""),
                    "input_image":input_uri,
                    "output_width":output_image_meta.get("width"),
                    "output_height":output_image_meta.get("height"),
                    "input_width":input_image_meta.get("width"),
                    "input_height":input_image_meta.get("height"),
                    "mse":mse,
                    "mae":mae,
                    "psnr_db":psnr,
                }
                source_image=(metadata or {}).get("display_image")
                if isinstance(source_image,str) and source_image.startswith("data:image/"):
                    reconstruction_meta["source_image"]=source_image
                return {
                    "kind":"reconstruction",
                    "mime":"application/x-mlbricks-reconstruction",
                    "data":{
                        "reconstructed_image":output_uri,
                        "mse":mse,
                        "mae":mae,
                        "psnr_db":psnr,
                    },
                    "metadata":reconstruction_meta,
                }

    return universal_output_envelope(
        result,output_type=output_type,input_kind=input_kind,task=task
    )


class _PackedLMBatcher:
    """Produce exact ``[batch, context]`` causal-LM batches without padding.

    Tokenized examples are concatenated into a stream with an EOS separator and
    sliced into ``context+1`` blocks. Eager and compiled execution therefore see
    identical fixed shapes and every reported token corresponds to real compute.
    """
    def __init__(self, dataset, *, context, separator_id, rng):
        if len(dataset)<=0:
            raise RuntimeError("Selected split has no rows.")
        self.dataset=dataset
        self.context=int(context)
        self.separator_id=int(separator_id)
        self.rng=rng
        self.buffer=[]
        self.offset=0

    def _append_row(self):
        attempts=0
        while attempts<100:
            attempts+=1
            row=self.dataset[self.rng.randrange(len(self.dataset))]
            ids=row.get("input_ids") if isinstance(row,dict) else None
            if ids is None:
                raise RuntimeError("Prepared data has no input_ids. Add Tokenize Text to the Data Processing pipeline.")
            ids=[int(v) for v in list(ids)]
            # Prepared tokenizer max length is not the model training context.
            # If the dataset was padded during tokenization, remove padding via
            # attention_mask before appending the row to the repackable stream.
            mask=row.get("attention_mask") if isinstance(row,dict) else None
            if mask is not None:
                mask=list(mask)
                if len(mask)==len(ids):
                    ids=[token_id for token_id,keep in zip(ids,mask) if int(keep)!=0]
            if not ids:
                continue
            self.buffer.extend(ids)
            if ids[-1]!=self.separator_id:
                self.buffer.append(self.separator_id)
            return
        raise RuntimeError("Could not read a non-empty tokenized row from the selected split.")

    def batch(self,batch_size,device):
        batch_size=int(batch_size)
        block=self.context+1
        needed=batch_size*block
        while len(self.buffer)-self.offset<needed:
            self._append_row()
        flat=self.buffer[self.offset:self.offset+needed]
        self.offset+=needed
        if self.offset>262144:
            self.buffer=self.buffer[self.offset:]
            self.offset=0
        packed=torch.tensor(flat,dtype=torch.long).view(batch_size,block)
        x=packed[:,:-1].contiguous()
        y=packed[:,1:].contiguous()
        if device.type=="cuda":
            try:
                x=x.pin_memory(); y=y.pin_memory()
            except RuntimeError:
                pass
        return x.to(device,non_blocking=True),y.to(device,non_blocking=True),batch_size*self.context


def _sample_batch(dataset,batch_size,context,pad_id,device,rng,*,fixed_length=True):
    # Backward-compatible helper used by callers outside the training loop.
    # New LM training always uses packed fixed-shape data for both eager and
    # compiled modes; ``pad_id`` is used as the stream separator when EOS is not
    # otherwise available.
    del fixed_length
    return _PackedLMBatcher(
        dataset,context=context,separator_id=pad_id,rng=rng
    ).batch(batch_size,device)


def _autocast_context(device,precision):
    if device.type=="cuda" and precision in {"fp16","bf16"}:
        return torch.autocast(device_type="cuda",dtype=torch.float16 if precision=="fp16" else torch.bfloat16)
    if device.type=="cpu" and precision=="bf16": return torch.autocast(device_type="cpu",dtype=torch.bfloat16)
    from contextlib import nullcontext
    return nullcontext()


def _perplexity(loss_value):
    if loss_value is None:
        return None
    try:
        value=float(loss_value)
    except (TypeError,ValueError,OverflowError):
        return None
    if not math.isfinite(value):
        return None
    # exp(20) is already ~4.85e8; cap only to keep telemetry finite.
    return math.exp(min(value,20.0))


def _sync_device(device):
    if device.type=="cuda":
        try: torch.cuda.synchronize(device)
        except Exception: pass


def _memory_snapshot(device):
    empty={
        "memory_allocated_gb":None,"memory_reserved_gb":None,
        "memory_peak_gb":None,"memory_total_gb":None,
    }
    if device.type!="cuda":
        return empty
    try:
        scale=float(1024**3)
        props=torch.cuda.get_device_properties(device)
        return {
            "memory_allocated_gb":torch.cuda.memory_allocated(device)/scale,
            "memory_reserved_gb":torch.cuda.memory_reserved(device)/scale,
            "memory_peak_gb":torch.cuda.max_memory_allocated(device)/scale,
            "memory_total_gb":float(props.total_memory)/scale,
        }
    except Exception:
        return empty


def _optimizer(model,config):
    name=str(config.get("optimizer") or "adamw").lower()
    lr=runtime_float(config.get("learning_rate"),5e-4,"Learning Rate",minimum=0.0)
    wd=runtime_float(config.get("weight_decay"),0.1,"Weight Decay",minimum=0.0)
    beta1=runtime_float(config.get("beta1"),0.9,"Adam Beta 1",minimum=0.0,maximum=1.0)
    beta2=runtime_float(config.get("beta2"),0.95,"Adam Beta 2",minimum=0.0,maximum=1.0)
    if name in {"adamw","adam"}:
        try:
            cls = IMPORT_POOL.resolve_api("optim.AdamW" if name == "adamw" else "optim.Adam")
        except ImportError:
            # Keep Studio graphs trainable in standalone/dev environments while
            # preferring the MLBricks optimizer whenever mlbricks-kit is present.
            cls = torch.optim.AdamW if name == "adamw" else torch.optim.Adam
        return cls(model.parameters(),lr=lr,betas=(beta1,beta2),weight_decay=wd)
    if name=="sgd":
        return torch.optim.SGD(model.parameters(),lr=lr,weight_decay=wd,momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {name}")


def _evaluate(
    loss_model, raw_model, batcher, *, steps, batch_size, device, precision,
    progress_callback=None, stop_event=None,
):
    """Evaluate a bounded number of validation batches with live progress.

    Validation used to be a silent block between the final training step and the
    terminal event. On short smoke/retrain runs that made Studio sit at 99% while
    20 validation batches (and then sample generation) were still executing. The
    work was finite, but the UI looked hung. Keep evaluation bounded and surface
    each completed batch without loading any additional dataset state.
    """
    if batcher is None:
        return None
    total=max(1,int(steps))
    raw_model.eval();loss_model.eval();losses=[]
    try:
        with torch.no_grad():
            for index in range(total):
                if stop_event is not None and stop_event.is_set():
                    raise TrainingStopped("Training stopped.")
                x,y,_=batcher.batch(batch_size,device)
                _sync_device(device)
                with _autocast_context(device,precision):
                    loss=loss_model(x,y)
                losses.append(float(loss.detach().float().cpu()))
                if progress_callback is not None:
                    try:
                        progress_callback(index+1,total,sum(losses)/len(losses))
                    except TrainingStopped:
                        raise
                    except Exception:
                        # Telemetry must never make validation fail.
                        pass
    finally:
        raw_model.train();loss_model.train()
    return sum(losses)/len(losses)


def _sample_next(logits,temperature,top_k,top_p,generator=None):
    temperature=max(runtime_float(temperature,0.8,"Temperature",minimum=1e-5),1e-5)
    logits=logits/temperature
    top_k=runtime_int(top_k,50,"Top K",minimum=0)
    candidate_ids=None
    if 0<top_k<logits.size(-1):
        # Restrict first, then run nucleus sampling inside this small candidate
        # set.  The old path masked to top-k but still sorted/scattered the whole
        # vocabulary every token.
        logits,candidate_ids=torch.topk(logits,top_k,dim=-1,sorted=True)
    top_p=runtime_float(top_p,0.95,"Top P",minimum=0.0,maximum=1.0)
    if 0<top_p<1:
        if candidate_ids is None:
            logits,sorted_ids=torch.sort(logits,descending=True)
            candidate_ids=sorted_ids
        probs=torch.softmax(logits,dim=-1)
        cum=torch.cumsum(probs,dim=-1)
        mask=cum>float(top_p)
        mask[...,1:]=mask[...,:-1].clone();mask[...,0]=False
        logits=logits.masked_fill(mask,float('-inf'))
    selected=torch.multinomial(torch.softmax(logits,dim=-1),1,generator=generator)
    return selected if candidate_ids is None else candidate_ids.gather(-1,selected)


def generate_text(model,tokenizer,prompt,*,max_new_tokens,context,device,precision,temperature=.8,top_k=50,top_p=.95,seed=42,progress=None,stop_event=None,stream_every_token=False):
    ids=tokenizer.encode(str(prompt),add_special_tokens=True)
    if not ids: ids=[tokenizer.eos_token_id or tokenizer.pad_token_id or 0]
    generated=list(ids)
    was_training=bool(model.training)
    model.eval()
    try:
        generator_device=device if device.type in {"cpu","cuda"} else torch.device("cpu")
        seed=runtime_int(seed,42,"Seed")
        max_new_tokens=runtime_int(max_new_tokens,128,"New Token Count",minimum=1)
        context=runtime_int(context,512,"Model Context",minimum=2)
        gen=torch.Generator(device=generator_device);gen.manual_seed(seed)

        # torch.compile wrappers expose the original TensorGraph through
        # ``_orig_mod``.  Recurrent methods live on that graph, while ordinary
        # unsupported models keep using the possibly-compiled full forward.
        recurrent_model=model
        while isinstance(getattr(recurrent_model,"_orig_mod",None),nn.Module):
            recurrent_model=recurrent_model._orig_mod
        support=getattr(recurrent_model,"recurrent_generation_support",None)
        supported=False; fallback_reason=None; algorithms=[]
        if callable(support):
            supported,fallback_reason=support()
            if supported:
                report=getattr(recurrent_model,"recurrent_generation_algorithms",None)
                algorithms=report() if callable(report) else []
        mode="recurrent-cache" if supported else "full-context"
        mode_label=("cached prefill/decode" if supported else "full-context compatibility")
        started=time.perf_counter()
        last_stream_at=0.0
        # Notebook widget transports cannot consume hundreds of growing JSON
        # payloads per second. Coalesce only the UI transport; token generation
        # itself remains unthrottled and the browser still updates at 12.5 FPS.
        stream_interval_seconds=0.08
        active_ids=list(ids[-context:])

        with torch.inference_mode(),_autocast_context(device,precision):
            if supported:
                x=torch.tensor([active_ids],dtype=torch.long,device=device)
                recurrent_capacity=max(context,len(active_ids)+max_new_tokens)
                logits,cache=recurrent_model.prefill(x,capacity=recurrent_capacity)
            else:
                cache=None
                x=torch.tensor([active_ids],dtype=torch.long,device=device)
                logits=model(x)
            if isinstance(logits,(tuple,list)): logits=logits[0]

            if progress:
                progress({
                    "status":"running","runtime_kind":"generate","phase":"prefill","overall":0,
                    "generated_tokens":0,"prefill_tokens":len(active_ids),"generation_mode":mode,
                    "generation_algorithms":algorithms,"fallback_reason":fallback_reason,
                    "message":f"Prompt ready · {mode_label}",
                    "generated_text":tokenizer.decode(generated,skip_special_tokens=True),
                    "generated_output":{"kind":"text","mime":"text/plain","data":tokenizer.decode(generated,skip_special_tokens=True)},
                    "generated_output_kind":"text","generated_output_mime":"text/plain",
                    "generated_output_meta":{"tokens": 0},
                })

            for i in range(max_new_tokens):
                if stop_event is not None and stop_event.is_set(): raise TrainingStopped("Generation stopped.")
                next_token=_sample_next(logits[:,-1,:].float(),temperature,top_k,top_p,generator=gen)
                next_id=int(next_token[0,0].item())
                generated.append(next_id)
                active_ids.append(next_id)
                now=time.perf_counter()
                elapsed=max(now-started,1e-9)
                terminal=(
                    (tokenizer.eos_token_id is not None and next_id==tokenizer.eos_token_id)
                    or i+1==max_new_tokens
                )
                if progress and not terminal and (stream_every_token or i==0 or now-last_stream_at>=stream_interval_seconds):
                    last_stream_at=now
                    progress({
                        "status":"running","runtime_kind":"generate","phase":"generate",
                        "overall":round((i+1)/max_new_tokens*100),"generated_tokens":i+1,
                        "tokens_per_sec":float(i+1)/elapsed,"generation_mode":mode,
                        "generation_algorithms":algorithms,"fallback_reason":fallback_reason,
                        "message":f"Generated {i+1}/{max_new_tokens} tokens · {mode_label}",
                        "generated_text":tokenizer.decode(generated,skip_special_tokens=True),
                        "generated_output":{"kind":"text","mime":"text/plain","data":tokenizer.decode(generated,skip_special_tokens=True)},
                        "generated_output_kind":"text","generated_output_mime":"text/plain",
                        "generated_output_meta":{"tokens": i+1},
                    })
                if terminal: break

                if supported:
                    logits,cache=recurrent_model.decode_step(
                        next_token,cache,position=len(active_ids)-1
                    )
                else:
                    active_ids=active_ids[-context:]
                    x=torch.tensor([active_ids],dtype=torch.long,device=device)
                    logits=model(x)
                if isinstance(logits,(tuple,list)): logits=logits[0]
        return tokenizer.decode(generated,skip_special_tokens=True),len(generated)-len(ids)
    finally:
        if was_training: model.train()



_CLASSICAL_FIT_TYPES = {"knn_classifier", "decision_tree_classifier", "kmeans", "pca"}


def _classical_fit_node(model_entry, state):
    graph = copy.deepcopy((model_entry or {}).get("architecture") or _root_model(state))
    matches = [node for node in (graph.get("nodes") or []) if str(node.get("type") or "") in _CLASSICAL_FIT_TYPES]
    if not matches:
        return None
    if len(matches) != 1:
        raise ModelCompileError("A classical fit graph must contain exactly one KNN, Decision Tree, K-Means, or PCA fit component.")
    return matches[0]


def _classical_feature_columns(split, algorithm_type):
    columns = list(getattr(split, "column_names", []) or [])
    if not columns and isinstance(split, dict):
        columns = list(split.keys())
    ignored = {"label", "target", "class", "class_id", "target_image", "text", "caption"}
    feature_named = [c for c in columns if re.fullmatch(r"feature_\d+", str(c))]
    if feature_named:
        feature_named.sort(key=lambda c: int(str(c).split("_")[-1]))
        return feature_named
    if algorithm_type == "kmeans" and all(c in columns for c in ("x", "y")):
        return ["x", "y"]
    # Educational/custom tabular fallback: retain scalar numeric columns while
    # excluding common supervised target names. The conversion helper below
    # still validates that each selected value is numeric.
    return [c for c in columns if str(c).lower() not in ignored]


def _classical_tensor_column(split, name, *, dtype):
    try:
        values = split[name]
    except Exception:
        values = [row[name] for row in split]
    try:
        tensor = torch.as_tensor(values, dtype=dtype)
    except Exception as exc:
        raise ValueError(f"Classical fit column {name!r} is not a numeric tensor-compatible field.") from exc
    return tensor


def _classical_xy(split, algorithm_type):
    # DataLoader-like prepared outputs expose their original dataset here.
    if not hasattr(split, "column_names") and hasattr(split, "dataset"):
        split = split.dataset
    feature_columns = _classical_feature_columns(split, algorithm_type)
    if not feature_columns:
        raise ValueError("Classical fit could not find numeric feature columns in the selected dataset.")
    feature_tensors = []
    for name in feature_columns:
        col = _classical_tensor_column(split, name, dtype=torch.float32)
        if col.ndim == 1:
            col = col.unsqueeze(-1)
        if col.ndim != 2:
            raise ValueError(
                f"Classical fit feature {name!r} must be scalar or vector per sample; received shape {tuple(col.shape)}."
            )
        feature_tensors.append(col)
    x = torch.cat(feature_tensors, dim=-1)
    y = None
    if algorithm_type in {"knn_classifier", "decision_tree_classifier"}:
        columns = list(getattr(split, "column_names", []) or [])
        label = next((name for name in ("label", "target", "class", "class_id") if name in columns), None)
        if label is None:
            raise ValueError("KNN and Decision Tree training require a label/target column.")
        y = _classical_tensor_column(split, label, dtype=torch.long).reshape(-1)
        if y.numel() != x.size(0):
            raise ValueError("Classical fit feature and target sample counts do not match.")
    return x, y, feature_columns


def _classical_accuracy(module, x, y):
    if x is None or y is None or x.numel() == 0:
        return None
    with torch.no_grad():
        pred = module(x).reshape(-1).cpu()
    return float((pred == y.reshape(-1).cpu()).float().mean().item())


def _fit_classical_builder_model(*, state, model_entry, dataset, dataset_meta, config, progress, stop_event, resume_from=None, resume_mode=None):
    node = _classical_fit_node(model_entry, state)
    if node is None:
        raise ModelCompileError("No classical fit component was found in this graph.")
    algorithm_type = str(node.get("type") or "")
    label = str(node.get("name") or algorithm_type)
    if stop_event.is_set():
        raise TrainingStopped("Classical fit stopped.")

    progress({
        "status":"running", "runtime_kind":"train", "phase":"fit_prepare", "overall":2,
        "step":0, "max_steps":1, "message":f"Preparing data for {label} fit…",
    })

    eager_config = dict(config or {})
    eager_config["execution_mode"] = "eager"  # fitting mutates persisted buffers; compile only after fit if requested later
    compiled, _ = compile_builder_model(
        state, model_entry, dataset_meta, eager_config, progress=None, for_training=False
    )
    raw = compiled.raw_model
    node_id = str(node.get("id"))
    module = raw.mods[node_id] if node_id in raw.mods else None
    if module is None or not callable(getattr(module, "fit", None)):
        raise ModelCompileError(f"{label} does not expose the Studio classical fit contract.")

    train = dataset["train"] if isinstance(dataset, dict) or hasattr(dataset, "keys") else dataset
    validation_name = str(config.get("validation_split") or "validation")
    validation = dataset.get(validation_name) if hasattr(dataset, "get") else None
    x_train, y_train, feature_columns = _classical_xy(train, algorithm_type)
    x_val = y_val = None
    if validation is not None:
        try:
            x_val, y_val, _ = _classical_xy(validation, algorithm_type)
        except Exception:
            # Validation is optional for unsupervised fit and for custom datasets
            # whose validation schema intentionally differs.
            x_val = y_val = None

    progress({
        "status":"running", "runtime_kind":"train", "phase":"fit", "overall":25,
        "step":0, "max_steps":1, "message":f"Fitting {label} on {x_train.size(0):,} samples × {x_train.size(1):,} features…",
        "samples_seen":int(x_train.size(0)), "feature_count":int(x_train.size(1)),
    })
    started = time.perf_counter()
    if stop_event.is_set():
        raise TrainingStopped("Classical fit stopped.")
    module.fit(x_train, y_train)
    elapsed = time.perf_counter() - started
    raw.to(compiled.device)
    compiled.model = raw
    compiled.raw_model = raw
    compiled.parameter_count = sum(p.numel() for p in raw.parameters())

    metrics = {"fit_seconds": elapsed, "samples": int(x_train.size(0)), "features": int(x_train.size(1))}
    metric_text = ""
    if algorithm_type in {"knn_classifier", "decision_tree_classifier"}:
        train_acc = _classical_accuracy(module, x_train.to(module.feature_.device if algorithm_type == "decision_tree_classifier" else module.train_x.device), y_train)
        val_acc = _classical_accuracy(module, x_val.to(module.feature_.device if algorithm_type == "decision_tree_classifier" else module.train_x.device), y_val) if x_val is not None and y_val is not None else None
        metrics.update({"train_accuracy": train_acc, "validation_accuracy": val_acc})
        metric_text = f" · train accuracy {train_acc*100:.1f}%" if train_acc is not None else ""
        if val_acc is not None: metric_text += f" · validation accuracy {val_acc*100:.1f}%"
    elif algorithm_type == "kmeans":
        metrics.update({"inertia": float(module.inertia_.item()), "iterations": int(module.n_iter_.item())})
        metric_text = f" · inertia {metrics['inertia']:.4f} · {metrics['iterations']} iterations"
    elif algorithm_type == "pca":
        ratios = module.explained_variance_ratio_.detach().cpu()
        metrics.update({"explained_variance_ratio": [float(v) for v in ratios], "explained_variance_total": float(ratios.sum().item())})
        metric_text = f" · explained variance {metrics['explained_variance_total']*100:.1f}%"

    progress({
        "status":"running", "runtime_kind":"train", "phase":"fit_done", "overall":85,
        "step":1, "max_steps":1, "elapsed_seconds":elapsed, "fit_metrics":metrics,
        "message":f"{label} fit complete{metric_text}.",
    })

    output = Path(str(config.get("output_dir") or "mlbricks_workspace/models")) / _safe_name(model_entry.get("name", "model"))
    output.mkdir(parents=True, exist_ok=True)
    final = output / "last"
    architecture = copy.deepcopy(model_entry.get("architecture") or _root_model(state))
    custom_components = copy.deepcopy(state.get("custom_components") or {})
    custom_components.update(copy.deepcopy(model_entry.get("custom_components_snapshot") or {}))
    builder_package = {
        "format":"mlb-studio-model-v2", "builder_version":__version__,
        "project":copy.deepcopy(state.get("project") or {}),
        "model_component":architecture, "custom_components":custom_components,
        "model_entry":copy.deepcopy(model_entry), "dataset_meta":copy.deepcopy(dataset_meta or {}),
    }
    metadata = {
        "kind":"trained_model", "step":1, "training_mode":"classical_fit",
        "fit_algorithm":algorithm_type, "fit_metrics":metrics,
        "feature_columns":feature_columns, "training_config":copy.deepcopy(config or {}),
        "builder_package":builder_package,
    }
    progress({
        "status":"running", "runtime_kind":"train", "phase":"final_save", "overall":95,
        "step":1, "max_steps":1, "message":f"Saving fitted {label} artifact…",
    })
    mlbricks_save = IMPORT_POOL.resolve_api("lifecycle.save")
    mlbricks_save(raw, final, metadata=metadata)
    now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    update = {
        "training_status":"trained", "weights_ready":True, "path":str(final), "checkpoint_path":str(final),
        "trained_steps":1, "tokens_seen":0, "parameter_count":compiled.parameter_count,
        "fit_algorithm":algorithm_type, "fit_metrics":metrics, "feature_columns":feature_columns,
        "training_mode":"classical_fit", "execution_mode_used":"eager", "trained_at":now,
        "format":"MLBricks model artifact", "artifact_format":"mlbricks.model",
        "retrained_from":str(resume_from) if resume_from else None,
    }
    progress({
        "status":"done", "runtime_kind":"train", "phase":"done", "overall":100,
        "step":1, "max_steps":1, "elapsed_seconds":elapsed, "fit_metrics":metrics,
        "message":f"{label} fitted and saved{metric_text}.", "model_update":update,
    })
    return {"compiled":compiled, "tokenizer":None, "model_update":update, "last_sample":None}




_SUPERVISED_INPUT_TYPES = {"feature_input", "signal_input", "image_input"}
_SUPERVISED_TARGET_NAMES = ("label", "target", "class", "class_id", "target_image")


def _split_column_names(split):
    if split is None:
        return []
    if not hasattr(split, "column_names") and hasattr(split, "dataset"):
        split = split.dataset
    columns = list(getattr(split, "column_names", []) or [])
    if not columns and isinstance(split, dict):
        columns = list(split.keys())
    return columns


def _split_column_tensor(split, name, *, dtype=torch.float32):
    if not hasattr(split, "column_names") and hasattr(split, "dataset"):
        split = split.dataset
    try:
        values = split[name]
    except Exception:
        values = [row[name] for row in split]
    try:
        return torch.as_tensor(values, dtype=dtype)
    except Exception as exc:
        raise ValueError(f"Training field {name!r} is not numeric/tensor compatible.") from exc


def _supervised_graph_info(model_entry, state):
    graph = copy.deepcopy((model_entry or {}).get("architecture") or _root_model(state))
    nodes = list(graph.get("nodes") or [])
    inputs = [n for n in nodes if str(n.get("type") or "") in _SUPERVISED_INPUT_TYPES]
    if not inputs:
        return None
    if len(inputs) > 1:
        raise ModelCompileError(
            "Generic supervised training currently expects one Feature, Signal, or Image input. "
            "Use a fusion component before enabling multi-input supervised training."
        )
    output = next(
        (n for n in reversed(nodes) if str(n.get("type") or "") in {"classifier", "detection_head", "detection_pyramid_head", "tensor_output", "logits_output"}),
        None,
    )
    if output is None:
        return None
    input_node = inputs[0]
    input_type = str(input_node.get("type") or "")
    output_type = str(output.get("type") or "")
    project_task = str((state.get("project") or {}).get("task") or (model_entry or {}).get("task") or "").strip().lower()
    name = str((model_entry or {}).get("name") or (state.get("project") or {}).get("name") or "").strip().lower()
    types = {str(n.get("type") or "") for n in nodes}

    if output_type in {"detection_head", "detection_pyramid_head"} or "object detection" in project_task or "object_detection" in project_task or project_task == "detection":
        task = "object_detection"
    elif "autoencoder" in name or "reconstruction" in project_task:
        task = "reconstruction"
    elif output_type != "classifier" and ("logistic" in name or "binary" in project_task or "sigmoid" in types):
        task = "binary_classification"
    elif output_type == "classifier" or "classification" in project_task:
        task = "classification"
    else:
        task = "regression"

    expected_feature_dim = None
    if input_type == "feature_input":
        try:
            value = int((input_node.get("params") or {}).get("feature_dim") or 0)
            expected_feature_dim = value if value > 0 else None
        except (TypeError, ValueError):
            expected_feature_dim = None

    output_classes = None
    if output_type in {"classifier", "detection_head", "detection_pyramid_head"}:
        try:
            value = int((output.get("params") or {}).get("classes") or 0)
            output_classes = value if value > 0 else None
        except (TypeError, ValueError):
            output_classes = None

    return {
        "graph": graph,
        "nodes": nodes,
        "input_node": input_node,
        "input_type": input_type,
        "expected_feature_dim": expected_feature_dim,
        "output": output,
        "output_type": output_type,
        "output_classes": output_classes,
        "task": task,
    }


def _feature_columns_for_supervised(columns):
    named = [c for c in columns if re.fullmatch(r"feature_\d+", str(c))]
    if named:
        named.sort(key=lambda c: int(str(c).split("_")[-1]))
        return named
    ignored = {str(v).lower() for v in _SUPERVISED_TARGET_NAMES}
    return [c for c in columns if str(c).lower() not in ignored and c not in {"image", "sequence", "signal"}]


def _supervised_xy(split, info):
    if split is None:
        raise ValueError("Training split is missing.")
    columns = _split_column_names(split)
    input_type = info["input_type"]
    task = info["task"]
    feature_columns = []

    if input_type == "image_input":
        if "image" not in columns:
            raise ValueError("Image training requires an 'image' field in the selected dataset.")
        x = _split_column_tensor(split, "image", dtype=torch.float32)
        if x.ndim == 3:  # [N,H,W] -> [N,1,H,W]
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError(f"Image training expects [N,C,H,W] or [N,H,W], received {tuple(x.shape)}.")
        feature_columns = ["image"]
    else:
        if "sequence" in columns:
            x = _split_column_tensor(split, "sequence", dtype=torch.float32)
            feature_columns = ["sequence"]
        elif "signal" in columns:
            x = _split_column_tensor(split, "signal", dtype=torch.float32)
            feature_columns = ["signal"]
        else:
            feature_columns = _feature_columns_for_supervised(columns)
            if not feature_columns:
                raise ValueError("Supervised training could not find numeric feature columns in the selected dataset.")
            feature_tensors = []
            for name in feature_columns:
                col = _split_column_tensor(split, name, dtype=torch.float32)
                if col.ndim == 1:
                    col = col.unsqueeze(-1)
                if col.ndim != 2:
                    raise ValueError(
                        f"Feature {name!r} must be scalar/vector per sample; received shape {tuple(col.shape)}."
                    )
                feature_tensors.append(col)
            x = torch.cat(feature_tensors, dim=-1)

    if task == "object_detection":
        if "boxes" not in columns or "class_ids" not in columns:
            raise ValueError("Object detection training requires 'boxes' and 'class_ids' fields.")
        try:
            raw_boxes = split["boxes"]
            raw_classes = split["class_ids"]
        except Exception:
            raw_boxes = [row["boxes"] for row in split]
            raw_classes = [row["class_ids"] for row in split]
        image_h, image_w = int(x.shape[-2]), int(x.shape[-1])
        max_objects=max(1,max((min(len(boxes or []),len(classes or [])) for boxes,classes in zip(raw_boxes,raw_classes)),default=0))
        targets=[]
        for boxes, class_ids in zip(raw_boxes, raw_classes):
            rows=[]
            for box,class_id in list(zip(boxes or [],class_ids or []))[:max_objects]:
                if len(box) < 4:
                    raise ValueError("Detection boxes must use [x, y, width, height] format.")
                bx,by,bw,bh=[float(v) for v in box[:4]]
                cx=(bx+bw*0.5)/max(float(image_w),1.0)
                cy=(by+bh*0.5)/max(float(image_h),1.0)
                nw=bw/max(float(image_w),1.0)
                nh=bh/max(float(image_h),1.0)
                if nw <= 0.0 or nh <= 0.0:
                    continue
                rows.append([
                    max(0.0,min(0.999999,cx)),max(0.0,min(0.999999,cy)),
                    max(1e-6,min(1.0,nw)),max(1e-6,min(1.0,nh)),float(class_id),1.0,
                ])
            while len(rows)<max_objects:
                rows.append([0.0,0.0,0.0,0.0,-1.0,0.0])
            targets.append(rows)
        y=torch.tensor(targets,dtype=torch.float32)
    elif task == "reconstruction":
        if "target_image" in columns:
            y = _split_column_tensor(split, "target_image", dtype=torch.float32)
            if y.ndim == 3:
                y = y.unsqueeze(1)
        else:
            y = x.clone()
    elif task in {"classification", "binary_classification"}:
        target_name = next((name for name in ("label", "class", "class_id", "target") if name in columns), None)
        if target_name is None:
            raise ValueError("Classification training requires a label/target field.")
        if task == "binary_classification":
            y = _split_column_tensor(split, target_name, dtype=torch.float32).reshape(-1, 1)
        else:
            y = _split_column_tensor(split, target_name, dtype=torch.long).reshape(-1)
    else:
        target_name = next((name for name in ("target", "label") if name in columns), None)
        if target_name is None:
            raise ValueError("Regression training requires a 'target' field.")
        y = _split_column_tensor(split, target_name, dtype=torch.float32)
        if y.ndim == 1:
            y = y.unsqueeze(-1)

    if int(x.shape[0]) != int(y.shape[0]):
        raise ValueError("Training input and target sample counts do not match.")
    return x.contiguous(), y.contiguous(), feature_columns


class _SupervisedTensorBatcher:
    def __init__(self, x, y, *, seed=42, shuffle=True):
        self.x = x
        self.y = y
        self.shuffle = bool(shuffle)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))
        self.order = torch.arange(int(x.shape[0]), dtype=torch.long)
        self.position = 0
        self.epoch = 0
        if self.shuffle:
            self.order = self.order[torch.randperm(len(self.order), generator=self.generator)]

    def __len__(self):
        return int(self.x.shape[0])

    def batch(self, batch_size, device):
        n = len(self)
        if n < 1:
            raise ValueError("Training split contains no samples.")
        batch_size = max(1, int(batch_size))
        parts = []
        remaining = batch_size
        while remaining > 0:
            take = min(remaining, n - self.position)
            parts.append(self.order[self.position:self.position + take])
            self.position += take
            remaining -= take
            if self.position >= n:
                self.epoch += 1
                self.position = 0
                self.order = torch.arange(n, dtype=torch.long)
                if self.shuffle:
                    self.order = self.order[torch.randperm(n, generator=self.generator)]
        idx = torch.cat(parts, dim=0)
        return self.x.index_select(0, idx).to(device), self.y.index_select(0, idx).to(device), int(idx.numel())


class _SequentialSupervisedBatcher:
    def __init__(self, x, y):
        self.x=x; self.y=y; self.position=0
    def __len__(self): return int(self.x.shape[0])
    def batch(self,batch_size,device):
        n=len(self); batch_size=max(1,int(batch_size))
        if n < 1: raise ValueError("Validation split contains no samples.")
        idx=(torch.arange(batch_size,dtype=torch.long)+self.position)%n
        self.position=(self.position+batch_size)%n
        return self.x.index_select(0,idx).to(device),self.y.index_select(0,idx).to(device),int(idx.numel())



def _detection_prediction_scales(prediction):
    """Normalize one or many raw detector maps to [B,slots,5+C,H,W]."""
    if isinstance(prediction, torch.Tensor):
        prediction=(prediction,)
    if not isinstance(prediction,(tuple,list)) or not prediction:
        raise ValueError("Detection output must be a tensor or a non-empty tuple/list of scale tensors.")
    out=[]
    for pred in prediction:
        if not isinstance(pred,torch.Tensor):
            raise ValueError("Each detection scale must be a tensor.")
        if pred.ndim==4:
            if pred.size(1)<6:
                raise ValueError(f"Detection output must have at least 6 channels, received {tuple(pred.shape)}.")
            pred=pred.unsqueeze(1)
        elif pred.ndim!=5 or pred.size(2)<6:
            raise ValueError(f"Detection scale must be [B,5+C,H,W] or [B,slots,5+C,H,W], received {tuple(pred.shape)}.")
        out.append(pred)
    batches={int(p.size(0)) for p in out}
    class_widths={int(p.size(2)-5) for p in out}
    if len(batches)!=1 or len(class_widths)!=1:
        raise ValueError("Detection pyramid scales must agree on batch size and class count.")
    return out


def _xywh_to_xyxy(box):
    half=box[...,2:4]*0.5
    return torch.cat([box[...,:2]-half,box[...,:2]+half],dim=-1)


def _box_iou_xyxy(a,b,eps=1e-7):
    # Supports broadcasting [...,4] against [...,4].
    tl=torch.maximum(a[...,:2],b[...,:2])
    br=torch.minimum(a[...,2:],b[...,2:])
    inter=(br-tl).clamp(min=0).prod(dim=-1)
    area_a=(a[...,2:]-a[...,:2]).clamp(min=0).prod(dim=-1)
    area_b=(b[...,2:]-b[...,:2]).clamp(min=0).prod(dim=-1)
    return inter/(area_a+area_b-inter+eps)


def _complete_iou_xywh(pred,target,eps=1e-7):
    pxy=_xywh_to_xyxy(pred); txy=_xywh_to_xyxy(target)
    iou=_box_iou_xyxy(pxy,txy,eps=eps)
    pcenter=pred[...,:2]; tcenter=target[...,:2]
    rho2=((pcenter-tcenter)**2).sum(dim=-1)
    enc_tl=torch.minimum(pxy[...,:2],txy[...,:2]); enc_br=torch.maximum(pxy[...,2:],txy[...,2:])
    c2=((enc_br-enc_tl)**2).sum(dim=-1)+eps
    pw=pred[...,2].clamp(min=eps); ph=pred[...,3].clamp(min=eps)
    tw=target[...,2].clamp(min=eps); th=target[...,3].clamp(min=eps)
    v=(4.0/(math.pi**2))*torch.pow(torch.atan(tw/th)-torch.atan(pw/ph),2)
    with torch.no_grad():
        alpha=v/(1.0-iou+v+eps)
    return (iou-rho2/c2-alpha*v).clamp(min=-1.0,max=1.0)


def _detection_targets_3d(target):
    if target.ndim==2 and target.size(-1)>=5:
        valid=torch.ones((target.size(0),1,1),device=target.device,dtype=target.dtype)
        return torch.cat([target[:,:5].unsqueeze(1),valid],dim=-1)
    if target.ndim==3 and target.size(-1)>=6:
        return target[...,:6]
    raise ValueError(f"Detection target must be [B,5] or [B,M,6], received {tuple(target.shape)}.")


def _assign_detection_targets(scales,target):
    """Assign every valid GT object to the best pyramid scale/cell/free slot."""
    tgt=_detection_targets_3d(target).float()
    bsz=int(tgt.size(0)); device=scales[0].device
    assignments=[]
    for pred in scales:
        b,a,_,h,w=pred.shape
        assignments.append({
            "obj":torch.zeros((b,a,h,w),device=device,dtype=pred.dtype),
            "box":torch.zeros((b,a,h,w,4),device=device,dtype=pred.dtype),
            "cls":torch.full((b,a,h,w),-1,device=device,dtype=torch.long),
        })
    refs=[2.0/max(float(max(int(p.size(-2)),int(p.size(-1)))),1.0) for p in scales]
    positive=[]
    for bi in range(bsz):
        for oi in range(int(tgt.size(1))):
            row=tgt[bi,oi]
            if float(row[5].item())<=0.0 or float(row[2].item())<=0.0 or float(row[3].item())<=0.0:
                continue
            size=max(float(row[2].item()),float(row[3].item()),1e-6)
            scale_order=sorted(range(len(scales)),key=lambda si:abs(math.log(size/max(refs[si],1e-6))))
            placed=False
            for si in scale_order:
                pred=scales[si]; _,slots,_,gh,gw=pred.shape
                gx=min(max(float(row[0].item()),0.0),0.999999)*gw
                gy=min(max(float(row[1].item()),0.0),0.999999)*gh
                ix=min(max(int(math.floor(gx)),0),gw-1); iy=min(max(int(math.floor(gy)),0),gh-1)
                free=None
                for slot in range(slots):
                    if float(assignments[si]["obj"][bi,slot,iy,ix].item())==0.0:
                        free=slot;break
                if free is None:
                    continue
                assignments[si]["obj"][bi,free,iy,ix]=1.0
                assignments[si]["box"][bi,free,iy,ix]=row[:4].to(device=device,dtype=pred.dtype)
                assignments[si]["cls"][bi,free,iy,ix]=max(0,int(row[4].item()))
                positive.append((si,bi,free,iy,ix))
                placed=True
                break
            # Extremely dense data may exceed all slots for the same cells.
            # The object is skipped rather than silently overwriting another GT.
            if not placed:
                continue
    return assignments,positive


def _decode_positive_box(raw,iy,ix,gh,gw):
    xy=raw[...,0:2].sigmoid()
    wh=raw[...,2:4].sigmoid().clamp(min=1e-6,max=1.0)
    cx=(float(ix)+xy[...,0])/float(gw)
    cy=(float(iy)+xy[...,1])/float(gh)
    return torch.stack([cx,cy,wh[...,0],wh[...,1]],dim=-1)


def _nms_xyxy(boxes,scores,iou_threshold=0.5,max_detections=100):
    if boxes.numel()==0:
        return torch.empty((0,),dtype=torch.long,device=boxes.device)
    order=scores.argsort(descending=True)
    keep=[]
    while order.numel()>0 and len(keep)<int(max_detections):
        idx=order[0]; keep.append(idx)
        if order.numel()==1: break
        rest=order[1:]
        iou=_box_iou_xyxy(boxes[idx].unsqueeze(0),boxes[rest])
        order=rest[iou<=float(iou_threshold)]
    return torch.stack(keep) if keep else torch.empty((0,),dtype=torch.long,device=boxes.device)


def decode_detection_predictions(prediction,*,score_threshold=0.25,iou_threshold=0.5,max_detections=100):
    """Decode raw grid outputs and run class-aware NMS.

    Returns one ``[N,6]`` tensor per image: ``x1,y1,x2,y2,score,class`` using
    normalized image coordinates.
    """
    scales=_detection_prediction_scales(prediction)
    bsz=int(scales[0].size(0)); classes=int(scales[0].size(2)-5)
    per_image=[[] for _ in range(bsz)]
    for pred in scales:
        b,slots,_,gh,gw=pred.shape
        yy,xx=torch.meshgrid(torch.arange(gh,device=pred.device,dtype=pred.dtype),torch.arange(gw,device=pred.device,dtype=pred.dtype),indexing="ij")
        xy=pred[:,:,0:2].sigmoid()
        wh=pred[:,:,2:4].sigmoid().clamp(min=1e-6,max=1.0)
        cx=(xx.view(1,1,gh,gw)+xy[:,:,0])/float(gw)
        cy=(yy.view(1,1,gh,gw)+xy[:,:,1])/float(gh)
        boxes=_xywh_to_xyxy(torch.stack([cx,cy,wh[:,:,0],wh[:,:,1]],dim=-1)).clamp(0.0,1.0)
        obj=pred[:,:,4].sigmoid()
        cls_prob=pred[:,:,5:].softmax(dim=2)
        cls_score,cls_id=cls_prob.max(dim=2)
        score=obj*cls_score
        for bi in range(bsz):
            mask=score[bi]>=float(score_threshold)
            if not bool(mask.any()): continue
            per_image[bi].append(torch.cat([
                boxes[bi][mask],score[bi][mask].unsqueeze(-1),cls_id[bi][mask].to(pred.dtype).unsqueeze(-1)
            ],dim=-1))
    results=[]
    for chunks in per_image:
        if not chunks:
            results.append(torch.empty((0,6),device=scales[0].device,dtype=scales[0].dtype));continue
        det=torch.cat(chunks,dim=0)
        kept=[]
        for cls in range(classes):
            idx=torch.nonzero(det[:,5].long()==cls,as_tuple=False).flatten()
            if idx.numel()==0: continue
            local=_nms_xyxy(det[idx,:4],det[idx,4],iou_threshold,max_detections)
            if local.numel(): kept.append(idx[local])
        if kept:
            idx=torch.cat(kept); idx=idx[det[idx,4].argsort(descending=True)[:int(max_detections)]]; det=det[idx]
        else: det=det[:0]
        results.append(det)
    return results


def _average_precision(recalls,precisions):
    if recalls.numel()==0:return 0.0
    mrec=torch.cat([torch.tensor([0.0],device=recalls.device),recalls,torch.tensor([1.0],device=recalls.device)])
    mpre=torch.cat([torch.tensor([1.0],device=precisions.device),precisions,torch.tensor([0.0],device=precisions.device)])
    for i in range(mpre.numel()-2,-1,-1): mpre[i]=torch.maximum(mpre[i],mpre[i+1])
    idx=torch.nonzero(mrec[1:]!=mrec[:-1],as_tuple=False).flatten()
    return float(torch.sum((mrec[idx+1]-mrec[idx])*mpre[idx+1]).item())


def detection_map50(prediction,target,*,score_threshold=0.05,nms_iou=0.5):
    tgt=_detection_targets_3d(target).detach()
    detections=decode_detection_predictions(prediction,score_threshold=score_threshold,iou_threshold=nms_iou,max_detections=300)
    classes=int(_detection_prediction_scales(prediction)[0].size(2)-5)
    aps=[]; total_tp=0; total_fp=0; total_gt=0
    for cls in range(classes):
        gt_by_image={}
        for bi in range(int(tgt.size(0))):
            rows=tgt[bi]
            mask=(rows[:,5]>0)&(rows[:,4].long()==cls)
            boxes=_xywh_to_xyxy(rows[mask,:4]).clamp(0,1)
            gt_by_image[bi]={"boxes":boxes,"used":torch.zeros((boxes.size(0),),dtype=torch.bool,device=boxes.device)}
        n_gt=sum(int(v["boxes"].size(0)) for v in gt_by_image.values())
        if n_gt==0: continue
        total_gt+=n_gt
        preds=[]
        for bi,det in enumerate(detections):
            rows=det[det[:,5].long()==cls]
            for ri in range(int(rows.size(0))): preds.append((float(rows[ri,4].item()),bi,rows[ri,:4]))
        preds.sort(key=lambda item:item[0],reverse=True)
        tp=[];fp=[]
        for _,bi,box in preds:
            gt=gt_by_image[bi]
            if gt["boxes"].numel()==0:
                tp.append(0.0);fp.append(1.0);continue
            ious=_box_iou_xyxy(box.unsqueeze(0),gt["boxes"])
            best_iou,best_idx=ious.max(dim=0)
            idx=int(best_idx.item())
            if float(best_iou.item())>=0.5 and not bool(gt["used"][idx]):
                gt["used"][idx]=True;tp.append(1.0);fp.append(0.0)
            else: tp.append(0.0);fp.append(1.0)
        if preds:
            tp_t=torch.tensor(tp,device=tgt.device);fp_t=torch.tensor(fp,device=tgt.device)
            total_tp+=int(tp_t.sum().item()); total_fp+=int(fp_t.sum().item())
            tp_c=tp_t.cumsum(0);fp_c=fp_t.cumsum(0)
            recall=tp_c/max(float(n_gt),1.0);precision=tp_c/(tp_c+fp_c+1e-9)
            aps.append(_average_precision(recall,precision))
        else: aps.append(0.0)
    map50=sum(aps)/len(aps) if aps else 0.0
    precision=total_tp/max(float(total_tp+total_fp),1.0)
    recall=total_tp/max(float(total_gt),1.0)
    return float(map50),float(precision),float(recall)


def _detection_loss_and_metrics(prediction,target):
    scales=_detection_prediction_scales(prediction)
    tgt=_detection_targets_3d(target).to(device=scales[0].device)
    assignments,positive=_assign_detection_targets(scales,tgt)
    obj_losses=[];box_terms=[];cls_logits=[];cls_targets=[];decoded_boxes=[];target_boxes=[]
    for si,pred in enumerate(scales):
        ass=assignments[si]
        # Slightly down-weight the vast negative grid while retaining hard negatives.
        obj_raw=F.binary_cross_entropy_with_logits(pred[:,:,4],ass["obj"],reduction="none")
        obj_weight=torch.where(ass["obj"]>0,torch.ones_like(obj_raw),torch.full_like(obj_raw,0.35))
        obj_losses.append((obj_raw*obj_weight).mean())
    for si,bi,slot,iy,ix in positive:
        pred=scales[si];gh=int(pred.size(-2));gw=int(pred.size(-1));raw=pred[bi,slot,:,iy,ix]
        decoded=_decode_positive_box(raw,iy,ix,gh,gw)
        target_box=assignments[si]["box"][bi,slot,iy,ix]
        decoded_boxes.append(decoded);target_boxes.append(target_box)
        box_terms.append(1.0-_complete_iou_xywh(decoded.unsqueeze(0),target_box.unsqueeze(0)).mean())
        cls_logits.append(raw[5:]);cls_targets.append(assignments[si]["cls"][bi,slot,iy,ix])
    loss_obj=torch.stack(obj_losses).mean() if obj_losses else torch.tensor(0.0,device=scales[0].device)
    if box_terms:
        loss_box=torch.stack(box_terms).mean()
        logits=torch.stack(cls_logits);targets=torch.stack(cls_targets).long().clamp(0,logits.size(-1)-1)
        loss_cls=F.cross_entropy(logits,targets)
        pred_cls=logits.detach().argmax(dim=-1)
        accuracy=float((pred_cls==targets).float().mean().item())
        dec=torch.stack(decoded_boxes);tar=torch.stack(target_boxes)
        mae=float(F.l1_loss(dec.detach(),tar.detach()).item())
        mean_iou=float(_box_iou_xyxy(_xywh_to_xyxy(dec.detach()),_xywh_to_xyxy(tar.detach())).mean().item())
    else:
        loss_box=loss_obj*0.0;loss_cls=loss_obj*0.0;accuracy=0.0;mae=0.0;mean_iou=0.0
    loss=7.5*loss_box+loss_obj+1.5*loss_cls
    with torch.no_grad():
        map50,precision,recall=detection_map50(scales,tgt)
    metrics={"accuracy":accuracy,"mae":mae,"box_iou":mean_iou,"map50":map50,"precision":precision,"recall":recall,"objects":float(len(positive))}
    return loss,metrics


def _supervised_loss_and_metrics(prediction, target, task):
    metrics = {}
    if task == "object_detection":
        return _detection_loss_and_metrics(prediction, target)
    if isinstance(prediction, (tuple, list)):
        prediction = prediction[0]
    if not isinstance(prediction, torch.Tensor):
        raise ValueError(f"Supervised graph output must be a tensor, received {type(prediction)!r}.")
    elif task == "classification":
        if prediction.ndim != 2:
            raise ValueError(f"Classification output must be [B,classes], received {tuple(prediction.shape)}.")
        loss = F.cross_entropy(prediction, target.long().reshape(-1))
        metrics["accuracy"] = float((prediction.detach().argmax(dim=-1) == target.reshape(-1)).float().mean().item())
    elif task == "binary_classification":
        pred = prediction.reshape(-1, 1)
        tgt = target.float().reshape_as(pred)
        # Educational Logistic Regression templates expose probabilities after
        # Sigmoid. Clamp only for numerical stability; custom logits models can
        # instead end in a classifier head and use cross entropy above.
        prob = pred.clamp(1e-7, 1.0 - 1e-7)
        loss = F.binary_cross_entropy(prob, tgt)
        metrics["accuracy"] = float(((prob.detach() >= 0.5).to(tgt.dtype) == tgt).float().mean().item())
    else:
        tgt = target.to(dtype=prediction.dtype)
        if tuple(prediction.shape) != tuple(tgt.shape):
            try:
                tgt = tgt.reshape_as(prediction)
            except Exception as exc:
                raise ValueError(
                    f"{task.title()} target shape {tuple(target.shape)} does not match model output {tuple(prediction.shape)}."
                ) from exc
        loss = F.mse_loss(prediction, tgt)
        metrics["mae"] = float(F.l1_loss(prediction.detach(), tgt.detach()).item())
    return loss, metrics


@torch.no_grad()
def _evaluate_supervised(model, batcher, *, steps, batch_size, device, precision, task, stop_event=None):
    if batcher is None:
        return None
    model.eval()
    losses=[]; metric_values={}
    try:
        for _ in range(max(1,int(steps))):
            if stop_event is not None and stop_event.is_set():
                raise TrainingStopped("Training stopped.")
            x,y,_=batcher.batch(batch_size,device)
            with _autocast_context(device,precision):
                pred=model(x)
                loss,metrics=_supervised_loss_and_metrics(pred,y,task)
            losses.append(float(loss.detach().float().cpu()))
            for key,value in metrics.items(): metric_values.setdefault(key,[]).append(float(value))
    finally:
        model.train()
    out={"loss":sum(losses)/len(losses)}
    for key,values in metric_values.items(): out[key]=sum(values)/len(values)
    return out


def _train_supervised_builder_model(*, state, model_entry, dataset, dataset_meta, config, progress, stop_event, resume_from=None, resume_mode=None):
    info=_supervised_graph_info(model_entry,state)
    if info is None:
        raise ModelCompileError("No supported supervised Feature, Signal, or Image graph was found.")
    task=info["task"]
    dataset_class_names=[str(v) for v in ((dataset_meta or {}).get("class_names") or [])]
    dataset_num_classes=int((dataset_meta or {}).get("num_classes") or len(dataset_class_names) or 0)
    if task=="object_detection" and dataset_num_classes>0 and info.get("output_classes"):
        model_classes=int(info["output_classes"])
        if model_classes!=dataset_num_classes:
            raise ValueError(
                f"Detection-class mismatch: model head has {model_classes} classes, "
                f"but the selected dataset provides {dataset_num_classes}. "
                f"Set the Detection Head classes to {dataset_num_classes} before training."
            )
    if task=="classification" and dataset_num_classes>0 and info.get("output_classes"):
        model_classes=int(info["output_classes"])
        if model_classes!=dataset_num_classes:
            raise ValueError(
                f"Classification-class mismatch: model head has {model_classes} classes, "
                f"but the selected dataset provides {dataset_num_classes}. "
                f"Set the Classifier Head classes to {dataset_num_classes} before training."
            )
    seed=runtime_int(config.get("seed"),42,"Seed")
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    progress({"status":"running","runtime_kind":"train","phase":"prepare","overall":1,"step":0,
              "training_mode":"supervised","training_task":task,
              "message":f"Preparing {task.replace('_',' ')} training data…"})

    compiled, _ = compile_builder_model(
        state, model_entry, dataset_meta, config, progress=progress, for_training=True
    )
    device=compiled.device; precision=compiled.precision; raw=compiled.raw_model
    train_split=dataset["train"] if isinstance(dataset,dict) or hasattr(dataset,"keys") else dataset
    val_name=str(config.get("validation_split") or "validation")
    val_split=dataset.get(val_name) if hasattr(dataset,"get") else None
    x_train,y_train,feature_columns=_supervised_xy(train_split,info)
    expected_feature_dim=info.get("expected_feature_dim")
    if expected_feature_dim and x_train.ndim==2 and int(x_train.shape[-1]) != int(expected_feature_dim):
        raise ValueError(
            f"Feature-width mismatch: model input expects {int(expected_feature_dim)} features, "
            f"but the selected dataset provides {int(x_train.shape[-1])}. "
            "Update the Feature Input/model weights or choose a compatible dataset before training."
        )
    x_val=y_val=None
    if val_split is not None:
        try: x_val,y_val,_=_supervised_xy(val_split,info)
        except Exception: x_val=y_val=None

    batch=runtime_int(config.get("batch_size"),16,"Batch Size",minimum=1)
    accum=runtime_int(config.get("gradient_accumulation"),1,"Gradient Accumulation",minimum=1)
    budget=str(config.get("budget_type") or "steps").lower()
    max_steps=runtime_int(config.get("max_steps"),1000,"Training Steps",minimum=1)
    epochs=runtime_float(config.get("epochs"),1.0,"Epochs",minimum=0.000001)
    if budget not in {"steps","epochs","samples","tokens"}:
        raise ValueError(f"Budget By must be steps, epochs, or samples; received {budget!r}.")
    if budget=="epochs":
        max_steps=max(1,math.ceil(len(x_train)/(batch*accum)*epochs))
    max_samples=runtime_int(config.get("max_samples") or config.get("max_tokens"),1000000,"Sample Budget",minimum=1)

    train_batcher=_SupervisedTensorBatcher(x_train,y_train,seed=seed,shuffle=True)
    val_batcher=_SequentialSupervisedBatcher(x_val,y_val) if x_val is not None and y_val is not None else None
    forward_model=compiled.model
    opt=_optimizer(raw,config)
    scaler=torch.amp.GradScaler("cuda",enabled=(device.type=="cuda" and precision=="fp16")) if hasattr(torch,"amp") else None
    warm=runtime_int(config.get("warmup_steps"),0,"Warmup Steps",minimum=0)
    validate_every=runtime_int(config.get("validate_every"),100,"Validate Every N Steps",minimum=0)
    val_steps=runtime_int(config.get("validation_steps"),20,"Validation Steps",minimum=1)
    checkpoint_every=runtime_int(config.get("checkpoint_every"),500,"Checkpoint Every N Steps",minimum=0)

    output=Path(str(config.get("output_dir") or "mlbricks_workspace/models"))/_safe_name(model_entry.get("name","model"))
    output.mkdir(parents=True,exist_ok=True); (output/"checkpoints").mkdir(exist_ok=True)

    # Resume learned weights for generic supervised graphs. Optimizer state is
    # restored only for Studio checkpoints that include training_state.pt.
    resume_path=Path(str(resume_from)).expanduser() if resume_from else None
    resume_kind=str(resume_mode or "weights").lower() if resume_path is not None else None
    resume_step=0; samples_seen=0
    if resume_path is not None:
        progress({"status":"running","runtime_kind":"train","phase":"resume_weights","overall":1,"step":0,
                  "message":f"Loading existing trained parameters from {resume_path.name or resume_path}…"})
        try:
            if resume_path.is_dir() and (resume_path/"model.pt").exists():
                loaded=IMPORT_POOL.resolve_api("lifecycle.load")(resume_path,device=device,strict=True)
                if not isinstance(loaded,nn.Module): raise RuntimeError("loaded artifact is not a torch.nn.Module")
                raw.load_state_dict(loaded.state_dict(),strict=True); del loaded
                state_path=resume_path/"training_state.pt"
                if resume_kind=="checkpoint" and state_path.is_file():
                    payload=safe_torch_load(state_path,map_location="cpu",allow_unsafe_pickle=False)
                    if payload.get("optimizer"): opt.load_state_dict(payload["optimizer"])
                    if scaler is not None and payload.get("scaler"): scaler.load_state_dict(payload["scaler"])
                    resume_step=max(0,int(payload.get("step") or 0)); samples_seen=max(0,int(payload.get("samples_seen") or 0))
            elif resume_path.is_file():
                payload=safe_torch_load(resume_path,map_location="cpu",allow_unsafe_pickle=_bool(config.get("allow_unsafe_legacy_checkpoint",False)))
                raw.load_state_dict(payload["model_state"],strict=True)
            else: raise FileNotFoundError(f"model artifact was not found: {resume_path}")
        except Exception as exc:
            raise ExistingModelArtifactError(resume_path,f"Existing model parameters at {resume_path} could not be loaded safely: {type(exc).__name__}: {exc}") from exc

    raw.train(); forward_model.train()
    if device.type=="cuda":
        try: torch.cuda.reset_peak_memory_stats(device)
        except Exception: pass
    wall_start=time.perf_counter(); train_seconds=0.0; run_samples=0; step=resume_step
    last_loss=None; last_metrics={}; last_val=None
    progress({"status":"running","runtime_kind":"train","phase":"train","overall":2,"step":step,"max_steps":max_steps,
              "samples_seen":samples_seen,"training_mode":"supervised","training_task":task,
              "message":f"Training {model_entry.get('name','model')} on {device} · {compiled.parameter_count:,} parameters · {len(x_train):,} samples · {task.replace('_',' ')}"})

    while True:
        if stop_event.is_set(): raise TrainingStopped("Training stopped.")
        if budget in {"steps","epochs"} and step>=max_steps: break
        if budget in {"samples","tokens"} and samples_seen>=max_samples: break
        step+=1
        opt.zero_grad(set_to_none=True); detached=[]; metric_accum={}; step_samples=0
        started=time.perf_counter()
        for _ in range(accum):
            x,y,count=train_batcher.batch(batch,device); step_samples+=count
            with _autocast_context(device,precision):
                pred=forward_model(x)
                loss,metrics=_supervised_loss_and_metrics(pred,y,task)
                scaled_loss=loss/accum
            if scaler is not None and scaler.is_enabled(): scaler.scale(scaled_loss).backward()
            else: scaled_loss.backward()
            detached.append(float(loss.detach().float().cpu()))
            for key,value in metrics.items(): metric_accum.setdefault(key,[]).append(value)
        if scaler is not None and scaler.is_enabled():
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(raw.parameters(),1.0); scaler.step(opt); scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(raw.parameters(),1.0); opt.step()
        if warm>0 and step<=warm:
            factor=step/warm; base_lr=runtime_float(config.get("learning_rate"),5e-4,"Learning Rate",minimum=0.0)
            for group in opt.param_groups: group["lr"]=base_lr*factor
        _sync_device(device)
        elapsed=max(time.perf_counter()-started,1e-9); train_seconds+=elapsed
        step_sps=step_samples/elapsed; run_samples+=step_samples; samples_seen+=step_samples
        last_loss=sum(detached)/len(detached)
        last_metrics={key:sum(vals)/len(vals) for key,vals in metric_accum.items()}
        lr=float(opt.param_groups[0].get("lr",0.0)) if opt.param_groups else None
        if budget in {"samples","tokens"}: ratio=min(1.0,samples_seen/max(float(max_samples),1.0))
        else: ratio=min(1.0,step/max(float(max_steps),1.0))
        overall=min(95,max(2,round(ratio*95)))

        do_val=validate_every>0 and val_batcher is not None and (step%validate_every==0 or (budget in {"steps","epochs"} and step>=max_steps))
        if do_val:
            last_val=_evaluate_supervised(forward_model,val_batcher,steps=val_steps,batch_size=batch,device=device,precision=precision,task=task,stop_event=stop_event)
        event={"status":"running","runtime_kind":"train","phase":"train","overall":overall,"step":step,"max_steps":max_steps,
               "samples_seen":samples_seen,"samples_per_sec":step_sps,"avg_samples_per_sec":run_samples/max(train_seconds,1e-9),
               "loss":last_loss,"val_loss":last_val.get("loss") if last_val else None,"lr":lr,
               "training_mode":"supervised","training_task":task,"elapsed_seconds":time.perf_counter()-wall_start,**_memory_snapshot(device)}
        if "accuracy" in last_metrics: event["accuracy"]=last_metrics["accuracy"]
        if "mae" in last_metrics: event["mae"]=last_metrics["mae"]
        for key in ("box_iou","map50","precision","recall","objects"):
            if key in last_metrics: event[key]=last_metrics[key]
        if last_val:
            if "accuracy" in last_val: event["val_accuracy"]=last_val["accuracy"]
            if "mae" in last_val: event["val_mae"]=last_val["mae"]
            for key in ("box_iou","map50","precision","recall"):
                if key in last_val: event[f"val_{key}"]=last_val[key]
        metric_text=f"loss {last_loss:.4f}"
        if "accuracy" in last_metrics: metric_text+=f" · accuracy {last_metrics['accuracy']*100:.1f}%"
        if "mae" in last_metrics: metric_text+=f" · MAE {last_metrics['mae']:.4f}"
        if "map50" in last_metrics: metric_text+=f" · mAP@.50 {last_metrics['map50']:.3f} · IoU {last_metrics.get('box_iou',0.0):.3f}"
        if last_val:
            metric_text+=f" · val loss {last_val['loss']:.4f}"
            if "accuracy" in last_val: metric_text+=f" · val accuracy {last_val['accuracy']*100:.1f}%"
            if "mae" in last_val: metric_text+=f" · val MAE {last_val['mae']:.4f}"
            if "map50" in last_val: metric_text+=f" · val mAP@.50 {last_val['map50']:.3f}"
        event["message"]=f"Step {step} · {step_sps:,.0f} samples/s · {metric_text}"
        progress(event)

        if checkpoint_every>0 and step%checkpoint_every==0:
            cp_path=output/"checkpoints"/f"step_{step:06d}"
            architecture=copy.deepcopy(model_entry.get("architecture") or _root_model(state))
            custom_components=copy.deepcopy(state.get("custom_components") or {})
            custom_components.update(copy.deepcopy(model_entry.get("custom_components_snapshot") or {}))
            metadata={"kind":"training_checkpoint","step":step,"samples_seen":samples_seen,"training_mode":"supervised","training_task":task,
                      "feature_columns":feature_columns,"class_names":dataset_class_names or None,"num_classes":dataset_num_classes or None,
                      "training_config":copy.deepcopy(config or {}),
                      "builder_package":{"format":"mlb-studio-model-v2","builder_version":__version__,"project":copy.deepcopy(state.get("project") or {}),
                                         "model_component":architecture,"custom_components":custom_components,"model_entry":copy.deepcopy(model_entry),"dataset_meta":copy.deepcopy(dataset_meta or {})}}
            IMPORT_POOL.resolve_api("lifecycle.save")(raw,cp_path,metadata=metadata)
            torch.save({"optimizer":opt.state_dict(),"scaler":scaler.state_dict() if scaler is not None else None,"step":step,"samples_seen":samples_seen},cp_path/"training_state.pt")
            progress({**event,"phase":"checkpoint","checkpoint_path":str(cp_path),"message":f"Checkpoint saved · step {step}"})

    final=output/"last"
    architecture=copy.deepcopy(model_entry.get("architecture") or _root_model(state))
    custom_components=copy.deepcopy(state.get("custom_components") or {})
    custom_components.update(copy.deepcopy(model_entry.get("custom_components_snapshot") or {}))
    metrics={"loss":last_loss,"validation_loss":last_val.get("loss") if last_val else None,**last_metrics}
    if last_val:
        for key,value in last_val.items(): metrics[f"validation_{key}"]=value
    builder_package={"format":"mlb-studio-model-v2","builder_version":__version__,"project":copy.deepcopy(state.get("project") or {}),
                     "model_component":architecture,"custom_components":custom_components,"model_entry":copy.deepcopy(model_entry),"dataset_meta":copy.deepcopy(dataset_meta or {})}
    metadata={"kind":"trained_model","step":step,"samples_seen":samples_seen,"training_mode":"supervised","training_task":task,
              "supervised_metrics":metrics,"feature_columns":feature_columns,
              "class_names":dataset_class_names or None,"num_classes":dataset_num_classes or None,
              "training_config":copy.deepcopy(config or {}),"builder_package":builder_package}
    progress({"status":"running","runtime_kind":"train","phase":"final_save","overall":99,"step":step,"max_steps":max_steps,
              "samples_seen":samples_seen,"loss":last_loss,"val_loss":last_val.get("loss") if last_val else None,
              "training_mode":"supervised","training_task":task,"message":"Training complete · saving final MLBricks model artifact…"})
    IMPORT_POOL.resolve_api("lifecycle.save")(raw,final,metadata=metadata)
    final_mem=_memory_snapshot(device)
    update={"training_status":"trained","weights_ready":True,"path":str(final),"checkpoint_path":str(final),"trained_steps":step,
            "samples_seen":samples_seen,"tokens_seen":0,"last_loss":last_loss,"last_val_loss":last_val.get("loss") if last_val else None,
            "parameter_count":compiled.parameter_count,"training_mode":"supervised","training_task":task,"supervised_metrics":metrics,
            "feature_columns":feature_columns,"class_names":dataset_class_names or None,"num_classes":dataset_num_classes or None,
            "avg_samples_per_sec":run_samples/max(train_seconds,1e-9) if run_samples else None,
            "memory_peak_gb":final_mem.get("memory_peak_gb"),"execution_mode_used":"compiled" if compiled.compile_used else "eager",
            "trained_at":time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),"format":"MLBricks model artifact","artifact_format":"mlbricks.model",
            "retrained_from":str(resume_path) if resume_path is not None and resume_kind!="checkpoint" else None,
            "resumed_checkpoint":str(resume_path) if resume_path is not None and resume_kind=="checkpoint" else None}
    progress({"status":"done","runtime_kind":"train","phase":"done","overall":100,"step":step,"max_steps":max_steps,
              "samples_seen":samples_seen,"loss":last_loss,"val_loss":last_val.get("loss") if last_val else None,
              "accuracy":last_metrics.get("accuracy"),"val_accuracy":last_val.get("accuracy") if last_val else None,
              "mae":last_metrics.get("mae"),"val_mae":last_val.get("mae") if last_val else None,
              "map50":last_metrics.get("map50"),"val_map50":last_val.get("map50") if last_val else None,
              "box_iou":last_metrics.get("box_iou"),"val_box_iou":last_val.get("box_iou") if last_val else None,
              "samples_per_sec":run_samples/max(train_seconds,1e-9) if run_samples else None,
              "training_mode":"supervised","training_task":task,"elapsed_seconds":time.perf_counter()-wall_start,
              "message":f"{model_entry.get('name','Model')} training complete and saved.","model_update":update})
    return {"compiled":compiled,"tokenizer":None,"model_update":update,"last_sample":None}



def _audio_graph_info(model_entry, state):
    graph=copy.deepcopy((model_entry or {}).get("architecture") or _root_model(state))
    nodes=list(graph.get("nodes") or [])
    output=next((n for n in reversed(nodes) if str(n.get("type") or "")=="audio_output"),None)
    if output is None:
        return None
    name=str((model_entry or {}).get("name") or (state.get("project") or {}).get("name") or "").lower()
    task=str((model_entry or {}).get("task") or (state.get("project") or {}).get("task") or "").lower()
    if "clone" in name or "clone" in task: kind="voice_clone"
    elif "voice-conditioned" in name or "speaker" in task: kind="voice_tts"
    elif "music" in name or "music" in task: kind="music_generation"
    elif "sound" in name or "sound" in task: kind="sound_generation"
    else: kind="tts"
    return {"graph":graph,"nodes":nodes,"output":output,"task":kind}


def _audio_text_tensor(values, *, length=64):
    rows=[]
    for value in list(values):
        raw=list(str(value or "").encode("utf-8",errors="replace"))[:length]
        ids=[b+1 for b in raw]+[0]*(length-len(raw))
        rows.append(ids)
    return torch.tensor(rows,dtype=torch.long)


def _audio_dataset_tensors(split, info, *, text_length=64):
    if split is None: raise ValueError("Audio training split is missing.")
    columns=_split_column_names(split)
    text_col="text" if "text" in columns else ("caption" if "caption" in columns else None)
    if text_col is None: raise ValueError("Audio generation training requires a 'text' or 'caption' field.")
    if "audio" not in columns: raise ValueError("Audio generation training requires an 'audio' waveform field.")
    try: texts=split[text_col]; audios=split["audio"]
    except Exception:
        rows=list(split); texts=[r[text_col] for r in rows]; audios=[r["audio"] for r in rows]
    tokens=_audio_text_tensor(texts,length=text_length)
    waveform=[]
    for value in audios:
        if isinstance(value,dict) and "array" in value: value=value["array"]
        waveform.append(torch.as_tensor(value,dtype=torch.float32).reshape(-1))
    target_len=min(int(v.numel()) for v in waveform) if waveform else 0
    if target_len<16: raise ValueError("Audio demo waveforms are too short.")
    target=torch.stack([v[:target_len] for v in waveform],dim=0)
    named={}
    if "speaker_id" in columns:
        named["speaker_id"]=_split_column_tensor(split,"speaker_id",dtype=torch.long).reshape(-1)
    if "reference_audio" in columns:
        try: refs=split["reference_audio"]
        except Exception: refs=[r["reference_audio"] for r in split]
        ref_tensors=[]
        for value in refs:
            if isinstance(value,dict) and "array" in value: value=value["array"]
            ref_tensors.append(torch.as_tensor(value,dtype=torch.float32).reshape(-1)[:target_len])
        named["reference_audio"]=torch.stack(ref_tensors,dim=0)
    return tokens,target,named,text_col


class _AudioBatcher:
    def __init__(self,tokens,target,named=None,*,seed=42,shuffle=True):
        self.tokens=tokens; self.target=target; self.named=dict(named or {}); self.shuffle=bool(shuffle)
        self.g=torch.Generator(device="cpu"); self.g.manual_seed(int(seed)); self.pos=0
        self.order=torch.arange(int(tokens.shape[0]),dtype=torch.long)
        if self.shuffle: self.order=self.order[torch.randperm(len(self.order),generator=self.g)]
    def __len__(self): return int(self.tokens.shape[0])
    def batch(self,batch_size,device):
        n=len(self); batch_size=max(1,int(batch_size)); parts=[]; left=batch_size
        while left>0:
            take=min(left,n-self.pos); parts.append(self.order[self.pos:self.pos+take]); self.pos+=take; left-=take
            if self.pos>=n:
                self.pos=0; self.order=torch.arange(n,dtype=torch.long)
                if self.shuffle: self.order=self.order[torch.randperm(n,generator=self.g)]
        idx=torch.cat(parts)
        named={k:v.index_select(0,idx).to(device) for k,v in self.named.items()}
        return self.tokens.index_select(0,idx).to(device),self.target.index_select(0,idx).to(device),named,int(idx.numel())


def _audio_loss(pred,target):
    if isinstance(pred,(tuple,list)): pred=pred[0]
    if pred.ndim>2: pred=pred.reshape(pred.shape[0],-1)
    target=target.to(dtype=pred.dtype)
    if pred.shape[-1]!=target.shape[-1]:
        n=min(pred.shape[-1],target.shape[-1]); pred=pred[...,:n]; target=target[...,:n]
    wave=F.mse_loss(pred,target)
    spectral=torch.zeros((),device=pred.device,dtype=pred.dtype)
    if pred.shape[-1]>=32:
        n_fft=min(64,int(pred.shape[-1])); hop=max(4,n_fft//4)
        window=torch.hann_window(n_fft,device=pred.device,dtype=torch.float32)
        p=torch.stft(pred.float(),n_fft=n_fft,hop_length=hop,window=window,return_complex=True).abs()
        t=torch.stft(target.float(),n_fft=n_fft,hop_length=hop,window=window,return_complex=True).abs()
        spectral=F.l1_loss(p,t).to(dtype=pred.dtype)
    loss=wave+0.05*spectral
    return loss,{"mae":float(F.l1_loss(pred.detach(),target.detach()).item()),"spectral_loss":float(spectral.detach().float().item())}


@torch.no_grad()
def _evaluate_audio(model,batcher,*,steps,batch_size,device,precision,stop_event=None):
    if batcher is None:return None
    model.eval(); losses=[]; maes=[]; specs=[]
    try:
        for _ in range(max(1,int(steps))):
            if stop_event is not None and stop_event.is_set(): raise TrainingStopped("Training stopped.")
            x,y,named,_=batcher.batch(batch_size,device)
            with _autocast_context(device,precision):
                pred=model(x,graph_named=named); loss,m=_audio_loss(pred,y)
            losses.append(float(loss.detach().float().cpu())); maes.append(m["mae"]); specs.append(m["spectral_loss"])
    finally:model.train()
    return {"loss":sum(losses)/len(losses),"mae":sum(maes)/len(maes),"spectral_loss":sum(specs)/len(specs)}


def _train_audio_builder_model(*,state,model_entry,dataset,dataset_meta,config,progress,stop_event,resume_from=None,resume_mode=None):
    info=_audio_graph_info(model_entry,state)
    if info is None: raise ModelCompileError("No Audio Output graph was found.")
    seed=runtime_int(config.get("seed"),42,"Seed"); random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    progress({"status":"running","runtime_kind":"train","phase":"prepare","overall":1,"step":0,"training_mode":"audio_generation","training_task":info["task"],"message":"Preparing text/audio conditioning pairs…"})
    compiled,_=compile_builder_model(state,model_entry,dataset_meta,config,progress=progress,for_training=True)
    device=compiled.device; precision=compiled.precision; raw=compiled.raw_model; model=compiled.model
    train_split=dataset["train"] if isinstance(dataset,dict) or hasattr(dataset,"keys") else dataset
    val_split=dataset.get(str(config.get("validation_split") or "validation")) if hasattr(dataset,"get") else None
    text_length=runtime_int(config.get("audio_text_length"),64,"Audio Text Length",minimum=8,maximum=512)
    x,y,named,text_col=_audio_dataset_tensors(train_split,info,text_length=text_length)
    vx=vy=vnamed=None
    if val_split is not None:
        try:vx,vy,vnamed,_=_audio_dataset_tensors(val_split,info,text_length=text_length)
        except Exception:vx=vy=vnamed=None
    batch=runtime_int(config.get("batch_size"),16,"Batch Size",minimum=1); accum=runtime_int(config.get("gradient_accumulation"),1,"Gradient Accumulation",minimum=1)
    max_steps=runtime_int(config.get("max_steps"),1000,"Training Steps",minimum=1); budget=str(config.get("budget_type") or "steps").lower(); max_samples=runtime_int(config.get("max_samples") or config.get("max_tokens"),1000000,"Sample Budget",minimum=1)
    if budget=="epochs": max_steps=max(1,math.ceil(len(x)/(batch*accum)*runtime_float(config.get("epochs"),1.0,"Epochs",minimum=0.000001)))
    train=_AudioBatcher(x,y,named,seed=seed,shuffle=True); val=_AudioBatcher(vx,vy,vnamed,seed=seed,shuffle=False) if vx is not None else None
    opt=_optimizer(raw,config); scaler=torch.amp.GradScaler("cuda",enabled=(device.type=="cuda" and precision=="fp16")) if hasattr(torch,"amp") else None
    validate_every=runtime_int(config.get("validate_every"),100,"Validate Every N Steps",minimum=0); val_steps=runtime_int(config.get("validation_steps"),10,"Validation Steps",minimum=1); checkpoint_every=runtime_int(config.get("checkpoint_every"),500,"Checkpoint Every N Steps",minimum=0)
    output=Path(str(config.get("output_dir") or "mlbricks_workspace/models"))/_safe_name(model_entry.get("name","audio-model")); output.mkdir(parents=True,exist_ok=True); (output/"checkpoints").mkdir(exist_ok=True)
    raw.train(); model.train(); step=0; samples_seen=0; wall=time.perf_counter(); train_seconds=0.; run_samples=0; last_loss=None; last_metrics={}; last_val=None
    progress({"status":"running","runtime_kind":"train","phase":"train","overall":2,"step":0,"max_steps":max_steps,"training_mode":"audio_generation","training_task":info["task"],"message":f"Training {model_entry.get('name','Audio Model')} on {device} · {compiled.parameter_count:,} parameters · {len(x):,} audio pairs"})
    while True:
        if stop_event.is_set(): raise TrainingStopped("Training stopped.")
        if budget in {"steps","epochs"} and step>=max_steps: break
        if budget in {"samples","tokens"} and samples_seen>=max_samples: break
        step+=1; opt.zero_grad(set_to_none=True); losses=[]; metric_acc={}; step_samples=0; started=time.perf_counter()
        for _ in range(accum):
            bx,by,bnamed,count=train.batch(batch,device); step_samples+=count
            with _autocast_context(device,precision): pred=model(bx,graph_named=bnamed); loss,m=_audio_loss(pred,by); sl=loss/accum
            if scaler is not None and scaler.is_enabled(): scaler.scale(sl).backward()
            else: sl.backward()
            losses.append(float(loss.detach().float().cpu()))
            for k,v in m.items(): metric_acc.setdefault(k,[]).append(float(v))
        if scaler is not None and scaler.is_enabled(): scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(raw.parameters(),1.0); scaler.step(opt); scaler.update()
        else: torch.nn.utils.clip_grad_norm_(raw.parameters(),1.0); opt.step()
        _sync_device(device); elapsed=max(time.perf_counter()-started,1e-9); train_seconds+=elapsed; run_samples+=step_samples; samples_seen+=step_samples
        last_loss=sum(losses)/len(losses); last_metrics={k:sum(v)/len(v) for k,v in metric_acc.items()}; ratio=min(1.,(samples_seen/max_samples) if budget in {"samples","tokens"} else (step/max_steps)); overall=min(95,max(2,round(ratio*95)))
        if validate_every>0 and val is not None and (step%validate_every==0 or step>=max_steps): last_val=_evaluate_audio(model,val,steps=val_steps,batch_size=batch,device=device,precision=precision,stop_event=stop_event)
        event={"status":"running","runtime_kind":"train","phase":"train","overall":overall,"step":step,"max_steps":max_steps,"samples_seen":samples_seen,"samples_per_sec":step_samples/elapsed,"loss":last_loss,"mae":last_metrics.get("mae"),"spectral_loss":last_metrics.get("spectral_loss"),"val_loss":last_val.get("loss") if last_val else None,"training_mode":"audio_generation","training_task":info["task"],"message":f"Step {step} · audio loss {last_loss:.5f} · MAE {last_metrics.get('mae',0):.5f}"}
        progress(event)
        if checkpoint_every>0 and step%checkpoint_every==0:
            cp=output/"checkpoints"/f"step_{step:06d}"; IMPORT_POOL.resolve_api("lifecycle.save")(raw,cp,metadata={"kind":"training_checkpoint","step":step,"training_mode":"audio_generation","training_task":info["task"]}); torch.save({"optimizer":opt.state_dict(),"step":step,"samples_seen":samples_seen},cp/"training_state.pt")
    final=output/"last"; architecture=copy.deepcopy(model_entry.get("architecture") or _root_model(state)); custom_components=copy.deepcopy(state.get("custom_components") or {}); custom_components.update(copy.deepcopy(model_entry.get("custom_components_snapshot") or {})); builder_package={"format":"mlb-studio-model-v2","builder_version":__version__,"project":copy.deepcopy(state.get("project") or {}),"model_component":architecture,"custom_components":custom_components,"model_entry":copy.deepcopy(model_entry),"dataset_meta":copy.deepcopy(dataset_meta or {})}
    metrics={"loss":last_loss,**last_metrics,"validation_loss":last_val.get("loss") if last_val else None}; metadata={"kind":"trained_model","step":step,"samples_seen":samples_seen,"training_mode":"audio_generation","training_task":info["task"],"audio_metrics":metrics,"text_column":text_col,"training_config":copy.deepcopy(config or {}),"builder_package":builder_package}
    progress({"status":"running","runtime_kind":"train","phase":"final_save","overall":99,"step":step,"loss":last_loss,"training_mode":"audio_generation","training_task":info["task"],"message":"Audio training complete · saving final MLBricks model artifact…"}); IMPORT_POOL.resolve_api("lifecycle.save")(raw,final,metadata=metadata)
    update={"training_status":"trained","weights_ready":True,"path":str(final),"checkpoint_path":str(final),"trained_steps":step,"samples_seen":samples_seen,"last_loss":last_loss,"last_val_loss":last_val.get("loss") if last_val else None,"parameter_count":compiled.parameter_count,"training_mode":"audio_generation","training_task":info["task"],"audio_metrics":metrics,"avg_samples_per_sec":run_samples/max(train_seconds,1e-9) if run_samples else None,"trained_at":time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),"format":"MLBricks model artifact","artifact_format":"mlbricks.model"}
    progress({"status":"done","runtime_kind":"train","phase":"done","overall":100,"step":step,"loss":last_loss,"training_mode":"audio_generation","training_task":info["task"],"message":f"{model_entry.get('name','Audio Model')} training complete and saved.","model_update":update}); return {"compiled":compiled,"tokenizer":None,"model_update":update,"last_sample":None}




def _multimodal_graph_info(model_entry, state):
    requirements = dict((model_entry or {}).get("requirements") or {})
    mode = str(requirements.get("training_mode") or "").strip().lower()
    task = str(requirements.get("training_task") or "").strip().lower()
    graph = copy.deepcopy((model_entry or {}).get("architecture") or _root_model(state))
    nodes = list(graph.get("nodes") or [])
    input_types = [str(n.get("type") or "") for n in nodes if str(n.get("type") or "") in {"text_input","image_input","audio_input","video_input","signal_input","feature_input"}]
    if mode != "multimodal" and len(set(input_types)) < 2:
        return None
    if not task:
        if any(str(n.get("type") or "") == "jepa_latent_loss" for n in nodes):
            task = "multimodal_jepa"
        elif "image_input" in input_types and "signal_input" in input_types:
            task = "sensor_vision_fusion"
        else:
            return None
    if task not in {"multimodal_jepa", "sensor_vision_fusion"}:
        return None
    return {"graph":graph,"nodes":nodes,"task":task}


def _multimodal_text_tensor(values, *, length=32):
    rows=[]
    width=max(4,int(length or 32))
    for value in list(values):
        raw=list(str(value or "").encode("utf-8",errors="replace"))[:width]
        ids=[int(v)+1 for v in raw]+[0]*(width-len(raw))
        rows.append(ids)
    return torch.tensor(rows,dtype=torch.long)


def _multimodal_dataset_tensors(split, task, *, text_length=32):
    if split is None:
        raise ValueError("Multimodal training split is missing.")
    columns=_split_column_names(split)
    if "image" not in columns:
        raise ValueError("Multimodal training requires an 'image' field.")
    image=_split_column_tensor(split,"image",dtype=torch.float32)
    if image.ndim==3: image=image.unsqueeze(1)
    if image.ndim!=4:
        raise ValueError(f"Multimodal image input expects [N,C,H,W] or [N,H,W], received {tuple(image.shape)}.")
    if task=="multimodal_jepa":
        if "text" not in columns:
            raise ValueError("Multimodal JEPA requires aligned 'image' and 'text' fields.")
        try: texts=split["text"]
        except Exception: texts=[row["text"] for row in split]
        text=_multimodal_text_tensor(texts,length=text_length)
        if int(text.shape[0])!=int(image.shape[0]): raise ValueError("Aligned image/text sample counts do not match.")
        return image.contiguous(), {"text":text.contiguous()}, None
    if "sensor" not in columns or "label" not in columns:
        raise ValueError("Sensor + Vision Fusion requires 'image', 'sensor', and 'label' fields.")
    sensor=_split_column_tensor(split,"sensor",dtype=torch.float32)
    if sensor.ndim!=2:
        raise ValueError(f"Sensor input expects [N,T], received {tuple(sensor.shape)}.")
    label=_split_column_tensor(split,"label",dtype=torch.long).reshape(-1)
    if not (int(image.shape[0])==int(sensor.shape[0])==int(label.shape[0])):
        raise ValueError("Aligned image/sensor/label sample counts do not match.")
    return image.contiguous(), {"sensor":sensor.contiguous()}, label.contiguous()


class _MultimodalBatcher:
    def __init__(self, main, named, target=None, *, seed=42, shuffle=True):
        self.main=main; self.named=dict(named or {}); self.target=target; self.shuffle=bool(shuffle)
        self.generator=torch.Generator(device="cpu"); self.generator.manual_seed(int(seed))
        self.order=torch.arange(int(main.shape[0]),dtype=torch.long); self.position=0
        if self.shuffle and len(self.order)>1: self.order=self.order[torch.randperm(len(self.order),generator=self.generator)]
    def __len__(self): return int(self.main.shape[0])
    def batch(self,batch_size,device):
        n=len(self); batch_size=max(1,int(batch_size)); parts=[]; remaining=batch_size
        if n<1: raise ValueError("Multimodal split contains no samples.")
        while remaining>0:
            take=min(remaining,n-self.position); parts.append(self.order[self.position:self.position+take]); self.position+=take; remaining-=take
            if self.position>=n:
                self.position=0; self.order=torch.arange(n,dtype=torch.long)
                if self.shuffle and n>1: self.order=self.order[torch.randperm(n,generator=self.generator)]
        idx=torch.cat(parts,dim=0)
        main=self.main.index_select(0,idx).to(device)
        named={k:v.index_select(0,idx).to(device) for k,v in self.named.items()}
        target=self.target.index_select(0,idx).to(device) if self.target is not None else None
        return main,named,target,int(idx.numel())


def _multimodal_forward_loss(model, main, named, target, task):
    pred=model(main,graph_named=named)
    if isinstance(pred,(tuple,list)): pred=pred[0]
    if not isinstance(pred,torch.Tensor): raise ValueError("Multimodal graph output must be a tensor.")
    if task=="multimodal_jepa":
        loss=pred.mean() if pred.numel()!=1 else pred.reshape(())
        return loss,{"latent_loss":float(loss.detach().float().cpu())}
    loss=F.cross_entropy(pred,target.long())
    acc=float((pred.detach().argmax(dim=-1)==target).float().mean().cpu())
    return loss,{"accuracy":acc}


def _evaluate_multimodal(model,batcher,*,steps,batch_size,device,precision,task,stop_event=None):
    if batcher is None: return None
    was_training=model.training; model.eval(); losses=[]; metrics={}
    try:
        with torch.inference_mode():
            for _ in range(max(1,int(steps))):
                if stop_event is not None and stop_event.is_set(): break
                main,named,target,_=batcher.batch(batch_size,device)
                with _autocast_context(device,precision): loss,m=_multimodal_forward_loss(model,main,named,target,task)
                losses.append(float(loss.detach().float().cpu()))
                for k,v in m.items(): metrics.setdefault(k,[]).append(float(v))
    finally:
        if was_training: model.train()
    if not losses: return None
    out={"loss":sum(losses)/len(losses)}
    for k,vals in metrics.items(): out[k]=sum(vals)/len(vals)
    return out


def _train_multimodal_builder_model(*,state,model_entry,dataset,dataset_meta,config,progress,stop_event,resume_from=None,resume_mode=None):
    info=_multimodal_graph_info(model_entry,state)
    if info is None: raise ModelCompileError("No supported multimodal graph was found.")
    task=info["task"]
    seed=runtime_int(config.get("seed"),42,"Seed"); random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    progress({"status":"running","runtime_kind":"train","phase":"prepare","overall":1,"step":0,"training_mode":"multimodal","training_task":task,"message":f"Preparing {task.replace('_',' ')} aligned data…"})
    compiled,_=compile_builder_model(state,model_entry,dataset_meta,config,progress=progress,for_training=True)
    device=compiled.device; precision=compiled.precision; raw=compiled.raw_model; model=compiled.model
    train_split=dataset["train"] if isinstance(dataset,dict) or hasattr(dataset,"keys") else dataset
    val_name=str(config.get("validation_split") or "validation"); val_split=dataset.get(val_name) if hasattr(dataset,"get") else None
    text_len=int((state.get("project") or {}).get("context_length") or 32)
    main,named,target=_multimodal_dataset_tensors(train_split,task,text_length=text_len)
    vmain=vnamed=vtarget=None
    if val_split is not None:
        try: vmain,vnamed,vtarget=_multimodal_dataset_tensors(val_split,task,text_length=text_len)
        except Exception: vmain=vnamed=vtarget=None
    batch=runtime_int(config.get("batch_size"),16,"Batch Size",minimum=1); accum=runtime_int(config.get("gradient_accumulation"),1,"Gradient Accumulation",minimum=1)
    budget=str(config.get("budget_type") or "steps").lower(); max_steps=runtime_int(config.get("max_steps"),200,"Training Steps",minimum=1); epochs=runtime_float(config.get("epochs"),1.0,"Epochs",minimum=0.000001)
    if budget=="epochs": max_steps=max(1,math.ceil(len(main)/(batch*accum)*epochs))
    max_samples=runtime_int(config.get("max_samples") or config.get("max_tokens"),100000,"Sample Budget",minimum=1)
    train_batcher=_MultimodalBatcher(main,named,target,seed=seed,shuffle=True); val_batcher=_MultimodalBatcher(vmain,vnamed,vtarget,seed=seed,shuffle=False) if vmain is not None else None
    opt=_optimizer(raw,config); scaler=torch.amp.GradScaler("cuda",enabled=(device.type=="cuda" and precision=="fp16")) if hasattr(torch,"amp") else None
    validate_every=runtime_int(config.get("validate_every"),20,"Validate Every N Steps",minimum=0); val_steps=runtime_int(config.get("validation_steps"),5,"Validation Steps",minimum=1); checkpoint_every=runtime_int(config.get("checkpoint_every"),500,"Checkpoint Every N Steps",minimum=0)
    output=Path(str(config.get("output_dir") or "mlbricks_workspace/models"))/_safe_name(model_entry.get("name","multimodal_model")); output.mkdir(parents=True,exist_ok=True); (output/"checkpoints").mkdir(exist_ok=True)
    # Restore weights/checkpoint when requested, matching other Studio training lifecycles.
    resume_path=Path(str(resume_from)).expanduser() if resume_from else None; resume_kind=str(resume_mode or "weights").lower() if resume_path is not None else None; step=0; samples_seen=0
    if resume_path is not None:
        progress({"status":"running","runtime_kind":"train","phase":"resume_weights","overall":1,"step":0,"training_mode":"multimodal","training_task":task,"message":f"Loading existing parameters from {resume_path.name or resume_path}…"})
        if resume_path.is_dir() and (resume_path/"model.pt").exists():
            loaded=IMPORT_POOL.resolve_api("lifecycle.load")(resume_path,device=device,strict=True); raw.load_state_dict(loaded.state_dict(),strict=True); del loaded
            if resume_kind=="checkpoint" and (resume_path/"training_state.pt").exists():
                payload=safe_torch_load(resume_path/"training_state.pt",map_location="cpu",allow_unsafe_pickle=False); opt.load_state_dict(payload.get("optimizer") or {}); step=int(payload.get("step") or 0); samples_seen=int(payload.get("samples_seen") or 0)
        else: raise FileNotFoundError(f"model artifact was not found: {resume_path}")
    progress({"status":"running","runtime_kind":"train","phase":"train","overall":2,"step":step,"max_steps":max_steps,"training_mode":"multimodal","training_task":task,"message":f"Training {model_entry.get('name','Multimodal Model')} on {device} · {compiled.parameter_count:,} parameters"})
    last_loss=None; last_metrics={}; last_val=None; wall_start=time.perf_counter(); run_samples=0; train_seconds=0.0
    while True:
        if stop_event.is_set(): raise TrainingStopped("Training stopped.")
        if budget in {"steps","epochs"} and step>=max_steps: break
        if budget in {"samples","tokens"} and samples_seen>=max_samples: break
        step+=1; opt.zero_grad(set_to_none=True); losses=[]; metric_accum={}; step_samples=0; started=time.perf_counter()
        for _ in range(accum):
            bm,bn,bt,count=train_batcher.batch(batch,device); step_samples+=count
            with _autocast_context(device,precision): loss,m=_multimodal_forward_loss(model,bm,bn,bt,task); scaled=loss/accum
            if scaler is not None and scaler.is_enabled(): scaler.scale(scaled).backward()
            else: scaled.backward()
            losses.append(float(loss.detach().float().cpu()))
            for k,v in m.items(): metric_accum.setdefault(k,[]).append(float(v))
        if scaler is not None and scaler.is_enabled(): scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(raw.parameters(),1.0); scaler.step(opt); scaler.update()
        else: torch.nn.utils.clip_grad_norm_(raw.parameters(),1.0); opt.step()
        _sync_device(device); elapsed=max(time.perf_counter()-started,1e-9); train_seconds+=elapsed; run_samples+=step_samples; samples_seen+=step_samples
        last_loss=sum(losses)/len(losses); last_metrics={k:sum(v)/len(v) for k,v in metric_accum.items()}; ratio=min(1.0,(samples_seen/max(float(max_samples),1.0)) if budget in {"samples","tokens"} else (step/max(float(max_steps),1.0))); overall=min(95,max(2,round(ratio*95)))
        if validate_every>0 and val_batcher is not None and (step%validate_every==0 or (budget in {"steps","epochs"} and step>=max_steps)):
            last_val=_evaluate_multimodal(model,val_batcher,steps=val_steps,batch_size=batch,device=device,precision=precision,task=task,stop_event=stop_event)
        sps=step_samples/elapsed; msg=f"Step {step} · {sps:,.0f} samples/s · loss {last_loss:.5f}"
        if "accuracy" in last_metrics: msg+=f" · accuracy {last_metrics['accuracy']*100:.1f}%"
        if last_val: msg+=f" · val {last_val['loss']:.5f}"
        event={"status":"running","runtime_kind":"train","phase":"train","overall":overall,"step":step,"max_steps":max_steps,"samples_seen":samples_seen,"samples_per_sec":sps,"avg_samples_per_sec":run_samples/max(train_seconds,1e-9),"loss":last_loss,"val_loss":last_val.get("loss") if last_val else None,"accuracy":last_metrics.get("accuracy"),"val_accuracy":last_val.get("accuracy") if last_val else None,"training_mode":"multimodal","training_task":task,"elapsed_seconds":time.perf_counter()-wall_start,**_memory_snapshot(device),"message":msg}; progress(event)
        if checkpoint_every>0 and step%checkpoint_every==0:
            cp=output/"checkpoints"/f"step_{step:06d}"; IMPORT_POOL.resolve_api("lifecycle.save")(raw,cp,metadata={"kind":"training_checkpoint","step":step,"training_mode":"multimodal","training_task":task}); torch.save({"optimizer":opt.state_dict(),"step":step,"samples_seen":samples_seen},cp/"training_state.pt")
    final=output/"last"; architecture=copy.deepcopy(model_entry.get("architecture") or _root_model(state)); custom_components=copy.deepcopy(state.get("custom_components") or {}); custom_components.update(copy.deepcopy(model_entry.get("custom_components_snapshot") or {})); builder_package={"format":"mlb-studio-model-v2","builder_version":__version__,"project":copy.deepcopy(state.get("project") or {}),"model_component":architecture,"custom_components":custom_components,"model_entry":copy.deepcopy(model_entry),"dataset_meta":copy.deepcopy(dataset_meta or {})}
    metrics={"loss":last_loss,"validation_loss":last_val.get("loss") if last_val else None,**last_metrics}; metadata={"kind":"trained_model","step":step,"samples_seen":samples_seen,"training_mode":"multimodal","training_task":task,"multimodal_metrics":metrics,"training_config":copy.deepcopy(config or {}),"builder_package":builder_package}; progress({"status":"running","runtime_kind":"train","phase":"final_save","overall":99,"step":step,"loss":last_loss,"training_mode":"multimodal","training_task":task,"message":"Multimodal training complete · saving final MLBricks model artifact…"}); IMPORT_POOL.resolve_api("lifecycle.save")(raw,final,metadata=metadata)
    update={"training_status":"trained","weights_ready":True,"path":str(final),"checkpoint_path":str(final),"trained_steps":step,"samples_seen":samples_seen,"last_loss":last_loss,"last_val_loss":last_val.get("loss") if last_val else None,"parameter_count":compiled.parameter_count,"training_mode":"multimodal","training_task":task,"multimodal_metrics":metrics,"avg_samples_per_sec":run_samples/max(train_seconds,1e-9) if run_samples else None,"trained_at":time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),"format":"MLBricks model artifact","artifact_format":"mlbricks.model"}
    progress({"status":"done","runtime_kind":"train","phase":"done","overall":100,"step":step,"loss":last_loss,"training_mode":"multimodal","training_task":task,"message":f"{model_entry.get('name','Multimodal Model')} training complete and saved.","model_update":update}); return {"compiled":compiled,"tokenizer":None,"model_update":update,"last_sample":None}

def _jepa_graph_info(model_entry, state):
    graph = copy.deepcopy((model_entry or {}).get("architecture") or _root_model(state))
    nodes = list(graph.get("nodes") or [])
    loss_node = next((n for n in nodes if str(n.get("type") or "") == "jepa_latent_loss"), None)
    mask_node = next((n for n in nodes if str(n.get("type") or "") == "jepa_mask"), None)
    encoders = [n for n in nodes if str(n.get("type") or "") == "jepa_encoder"]
    context = next((n for n in encoders if str((n.get("params") or {}).get("role") or "context").lower() == "context"), None)
    target = next((n for n in encoders if str((n.get("params") or {}).get("role") or "context").lower() == "target"), None)
    predictor = next((n for n in nodes if str(n.get("type") or "") == "jepa_predictor"), None)
    if not loss_node:
        return None
    if not all((mask_node, context, target, predictor)):
        raise ModelCompileError("JEPA training requires JEPA Mask, context JEPA Encoder, target JEPA Encoder, JEPA Predictor, and JEPA Latent Loss.")
    modality = str((context.get("params") or {}).get("modality") or "image").strip().lower()
    if modality != str((target.get("params") or {}).get("modality") or modality).strip().lower():
        raise ModelCompileError("JEPA context and target encoders must use the same modality.")
    return {"graph": graph, "nodes": nodes, "mask": mask_node, "context": context, "target": target, "predictor": predictor, "loss": loss_node, "modality": modality}


def _jepa_split_tensor(split, info, *, sequence_length=64):
    if split is None:
        raise ValueError("JEPA training split is missing.")
    columns = _split_column_names(split)
    modality = str(info.get("modality") or "image").lower()
    if "jepa_input" in columns:
        dtype = torch.long if modality == "text" else torch.float32
        x = _split_column_tensor(split, "jepa_input", dtype=dtype)
    elif modality == "text":
        if "input_ids" in columns:
            x = _split_column_tensor(split, "input_ids", dtype=torch.long)
        elif "text" in columns:
            try:
                texts = split["text"]
            except Exception:
                texts = [row["text"] for row in split]
            width = max(4, int(sequence_length or 64))
            rows=[]
            for text in texts:
                raw=str(text).encode("utf-8",errors="replace")[:width]
                ids=[int(v)+1 for v in raw]
                ids += [0]*(width-len(ids))
                rows.append(ids)
            x=torch.tensor(rows,dtype=torch.long)
        else:
            raise ValueError("Text JEPA requires 'jepa_input', 'input_ids', or 'text'.")
    else:
        field={"image":"image","video":"video","audio":"audio","signal":"signal"}.get(modality)
        if not field or field not in columns:
            raise ValueError(f"{modality.title()} JEPA requires a 'jepa_input' or {field!r} field.")
        x=_split_column_tensor(split,field,dtype=torch.float32)

    if modality == "image":
        if x.ndim == 3: x=x.unsqueeze(1)
        if x.ndim != 4: raise ValueError(f"Image JEPA expects [N,C,H,W], received {tuple(x.shape)}.")
    elif modality == "video":
        if x.ndim == 4: x=x.unsqueeze(2)
        if x.ndim != 5: raise ValueError(f"Video JEPA expects [N,T,C,H,W], received {tuple(x.shape)}.")
    elif modality in {"audio","signal"}:
        if x.ndim == 3 and x.size(1) == 1: pass
        elif x.ndim != 2: raise ValueError(f"{modality.title()} JEPA expects [N,T] or [N,1,T], received {tuple(x.shape)}.")
    elif modality == "text":
        if x.ndim != 2: raise ValueError(f"Text JEPA expects token ids [N,T], received {tuple(x.shape)}.")
        x=x.long()
    return x.contiguous()


class _JEPATensorBatcher:
    def __init__(self, x, *, seed=42, shuffle=True):
        self.x=x; self.shuffle=bool(shuffle); self.generator=torch.Generator(device="cpu"); self.generator.manual_seed(int(seed))
        self.order=torch.arange(int(x.shape[0]),dtype=torch.long); self.position=0
        if self.shuffle: self.order=self.order[torch.randperm(len(self.order),generator=self.generator)]
    def __len__(self): return int(self.x.shape[0])
    def batch(self,batch_size,device):
        n=len(self); batch_size=max(1,int(batch_size))
        if n < 1: raise ValueError("JEPA split contains no samples.")
        parts=[]; remaining=batch_size
        while remaining>0:
            take=min(remaining,n-self.position); parts.append(self.order[self.position:self.position+take]); self.position+=take; remaining-=take
            if self.position>=n:
                self.position=0; self.order=torch.arange(n,dtype=torch.long)
                if self.shuffle: self.order=self.order[torch.randperm(n,generator=self.generator)]
        idx=torch.cat(parts,dim=0)
        return self.x.index_select(0,idx).to(device),int(idx.numel())


def _jepa_modules(raw, info):
    if not isinstance(raw, TensorGraph):
        raise ModelCompileError("JEPA EMA target update requires the editable TensorGraph runtime.")
    context = raw.mods[info["context"]["id"]] if info["context"]["id"] in raw.mods else None
    target = raw.mods[info["target"]["id"]] if info["target"]["id"] in raw.mods else None
    if context is None or target is None:
        raise ModelCompileError("JEPA context/target encoder modules were not compiled.")
    return context,target


@torch.no_grad()
def _jepa_sync_target(context, target, momentum=0.0):
    c_params=list(context.parameters()); t_params=list(target.parameters())
    if len(c_params)!=len(t_params) or any(a.shape!=b.shape for a,b in zip(c_params,t_params)):
        raise ModelCompileError("JEPA context and target encoder architectures must match for EMA updates.")
    tau=float(momentum)
    for cp,tp in zip(c_params,t_params):
        if tau <= 0.0: tp.copy_(cp.detach())
        else: tp.mul_(tau).add_(cp.detach(),alpha=1.0-tau)
    for cb,tb in zip(context.buffers(),target.buffers()):
        if cb.shape==tb.shape: tb.copy_(cb.detach())


@torch.no_grad()
def _evaluate_jepa(model,batcher,*,steps,batch_size,device,precision,stop_event=None):
    if batcher is None: return None
    model.eval(); losses=[]
    try:
        for _ in range(max(1,int(steps))):
            if stop_event is not None and stop_event.is_set(): raise TrainingStopped("Training stopped.")
            x,_=batcher.batch(batch_size,device)
            with _autocast_context(device,precision): loss=model(x)
            if not isinstance(loss,torch.Tensor) or loss.numel()!=1: raise ValueError("JEPA graph must end in a scalar JEPA Latent Loss.")
            losses.append(float(loss.detach().float().cpu()))
    finally:
        model.train()
    return sum(losses)/len(losses)


def _train_jepa_builder_model(*,state,model_entry,dataset,dataset_meta,config,progress,stop_event,resume_from=None,resume_mode=None):
    info=_jepa_graph_info(model_entry,state)
    if info is None: raise ModelCompileError("No JEPA Latent Loss graph was found.")
    seed=runtime_int(config.get("seed"),42,"Seed"); random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    progress({"status":"running","runtime_kind":"train","phase":"prepare","overall":1,"step":0,"training_mode":"jepa","training_task":"jepa","message":f"Preparing {info['modality']} JEPA training data…"})
    compiled,_=compile_builder_model(state,model_entry,dataset_meta,config,progress=progress,for_training=True)
    device=compiled.device; precision=compiled.precision; raw=compiled.raw_model; forward_model=compiled.model
    train_split=dataset["train"] if isinstance(dataset,dict) or hasattr(dataset,"keys") else dataset
    val_name=str(config.get("validation_split") or "validation"); val_split=dataset.get(val_name) if hasattr(dataset,"get") else None
    seq_len=runtime_int((model_entry.get("context_length") or (state.get("project") or {}).get("context_length")),64,"JEPA Sequence Length",minimum=4)
    x_train=_jepa_split_tensor(train_split,info,sequence_length=seq_len)
    x_val=None
    if val_split is not None:
        try: x_val=_jepa_split_tensor(val_split,info,sequence_length=seq_len)
        except Exception: x_val=None

    batch=runtime_int(config.get("batch_size"),16,"Batch Size",minimum=1); accum=runtime_int(config.get("gradient_accumulation"),1,"Gradient Accumulation",minimum=1)
    budget=str(config.get("budget_type") or "steps").lower(); max_steps=runtime_int(config.get("max_steps"),200,"Training Steps",minimum=1)
    epochs=runtime_float(config.get("epochs"),1.0,"Epochs",minimum=0.000001); max_samples=runtime_int(config.get("max_samples") or config.get("max_tokens"),100000,"Sample Budget",minimum=1)
    if budget not in {"steps","epochs","samples","tokens"}: raise ValueError("JEPA Budget By must be steps, samples, or epochs.")
    if budget=="epochs": max_steps=max(1,math.ceil(len(x_train)/(batch*accum)*epochs))
    train_batcher=_JEPATensorBatcher(x_train,seed=seed,shuffle=True); val_batcher=_JEPATensorBatcher(x_val,seed=seed^0xA51E,shuffle=False) if x_val is not None else None
    context_encoder,target_encoder=_jepa_modules(raw,info)
    opt=_optimizer(raw,config); scaler=torch.amp.GradScaler("cuda",enabled=(device.type=="cuda" and precision=="fp16")) if hasattr(torch,"amp") else None
    target_momentum=runtime_float(config.get("jepa_target_momentum"),0.996,"JEPA Target Momentum",minimum=0.0,maximum=0.999999)
    validate_every=runtime_int(config.get("validate_every"),20,"Validate Every N Steps",minimum=0); val_steps=runtime_int(config.get("validation_steps"),5,"Validation Steps",minimum=1)
    checkpoint_every=runtime_int(config.get("checkpoint_every"),100,"Checkpoint Every N Steps",minimum=0)
    output=Path(str(config.get("output_dir") or "mlbricks_workspace/models"))/_safe_name(model_entry.get("name","jepa")); output.mkdir(parents=True,exist_ok=True); (output/"checkpoints").mkdir(exist_ok=True)

    resume_path=Path(str(resume_from)).expanduser() if resume_from else None; resume_kind=str(resume_mode or "weights").lower() if resume_path is not None else None
    resume_step=0; samples_seen=0
    if resume_path is not None:
        try:
            if resume_path.is_dir() and (resume_path/"model.pt").exists():
                loaded=IMPORT_POOL.resolve_api("lifecycle.load")(resume_path,device=device,strict=True); raw.load_state_dict(loaded.state_dict(),strict=True); del loaded
                state_path=resume_path/"training_state.pt"
                if resume_kind=="checkpoint" and state_path.is_file():
                    payload=safe_torch_load(state_path,map_location="cpu",allow_unsafe_pickle=False)
                    if payload.get("optimizer"): opt.load_state_dict(payload["optimizer"])
                    if scaler is not None and payload.get("scaler"): scaler.load_state_dict(payload["scaler"])
                    resume_step=max(0,int(payload.get("step") or 0)); samples_seen=max(0,int(payload.get("samples_seen") or 0))
            elif resume_path.is_file():
                payload=safe_torch_load(resume_path,map_location="cpu",allow_unsafe_pickle=_bool(config.get("allow_unsafe_legacy_checkpoint",False))); raw.load_state_dict(payload["model_state"],strict=True)
            else: raise FileNotFoundError(f"model artifact was not found: {resume_path}")
        except Exception as exc:
            raise ExistingModelArtifactError(resume_path,f"Existing JEPA parameters could not be loaded safely: {type(exc).__name__}: {exc}") from exc
    else:
        _jepa_sync_target(context_encoder,target_encoder,momentum=0.0)

    raw.train(); forward_model.train(); wall_start=time.perf_counter(); train_seconds=0.0; run_samples=0; step=resume_step; last_loss=None; last_val=None
    progress({"status":"running","runtime_kind":"train","phase":"train","overall":2,"step":step,"max_steps":max_steps,"samples_seen":samples_seen,"training_mode":"jepa","training_task":"jepa","message":f"Training {model_entry.get('name','JEPA')} on {device} · {compiled.parameter_count:,} parameters · EMA target {target_momentum:.4f}"})
    while True:
        if stop_event.is_set(): raise TrainingStopped("Training stopped.")
        if budget in {"steps","epochs"} and step>=max_steps: break
        if budget in {"samples","tokens"} and samples_seen>=max_samples: break
        step+=1; opt.zero_grad(set_to_none=True); detached=[]; step_samples=0; started=time.perf_counter()
        for _ in range(accum):
            x,count=train_batcher.batch(batch,device); step_samples+=count
            with _autocast_context(device,precision):
                loss=forward_model(x)
                if not isinstance(loss,torch.Tensor) or loss.numel()!=1: raise ValueError("JEPA graph must end in a scalar JEPA Latent Loss.")
                scaled_loss=loss/accum
            if scaler is not None and scaler.is_enabled(): scaler.scale(scaled_loss).backward()
            else: scaled_loss.backward()
            detached.append(float(loss.detach().float().cpu()))
        trainable=[p for p in raw.parameters() if p.requires_grad]
        if scaler is not None and scaler.is_enabled():
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(trainable,1.0); scaler.step(opt); scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(trainable,1.0); opt.step()
        _jepa_sync_target(context_encoder,target_encoder,momentum=target_momentum)
        _sync_device(device); elapsed=max(time.perf_counter()-started,1e-9); train_seconds+=elapsed; run_samples+=step_samples; samples_seen+=step_samples; last_loss=sum(detached)/len(detached)
        ratio=min(1.0,(samples_seen/max(float(max_samples),1.0)) if budget in {"samples","tokens"} else (step/max(float(max_steps),1.0))); overall=min(95,max(2,round(ratio*95)))
        if validate_every>0 and val_batcher is not None and (step%validate_every==0 or (budget in {"steps","epochs"} and step>=max_steps)):
            last_val=_evaluate_jepa(forward_model,val_batcher,steps=val_steps,batch_size=batch,device=device,precision=precision,stop_event=stop_event)
        sps=step_samples/elapsed
        progress({"status":"running","runtime_kind":"train","phase":"train","overall":overall,"step":step,"max_steps":max_steps,"samples_seen":samples_seen,"samples_per_sec":sps,"avg_samples_per_sec":run_samples/max(train_seconds,1e-9),"loss":last_loss,"val_loss":last_val,"training_mode":"jepa","training_task":"jepa","jepa_target_momentum":target_momentum,"elapsed_seconds":time.perf_counter()-wall_start,**_memory_snapshot(device),"message":f"Step {step} · {sps:,.0f} samples/s · JEPA latent loss {last_loss:.5f}"+(f" · val {last_val:.5f}" if last_val is not None else "")})
        if checkpoint_every>0 and step%checkpoint_every==0:
            cp_path=output/"checkpoints"/f"step_{step:06d}"; metadata={"kind":"training_checkpoint","step":step,"samples_seen":samples_seen,"training_mode":"jepa","training_task":"jepa","jepa_target_momentum":target_momentum,"training_config":copy.deepcopy(config or {})}
            IMPORT_POOL.resolve_api("lifecycle.save")(raw,cp_path,metadata=metadata); torch.save({"optimizer":opt.state_dict(),"scaler":scaler.state_dict() if scaler is not None else None,"step":step,"samples_seen":samples_seen},cp_path/"training_state.pt")

    final=output/"last"; architecture=copy.deepcopy(model_entry.get("architecture") or _root_model(state)); custom_components=copy.deepcopy(state.get("custom_components") or {}); custom_components.update(copy.deepcopy(model_entry.get("custom_components_snapshot") or {}))
    builder_package={"format":"mlb-studio-model-v2","builder_version":__version__,"project":copy.deepcopy(state.get("project") or {}),"model_component":architecture,"custom_components":custom_components,"model_entry":copy.deepcopy(model_entry),"dataset_meta":copy.deepcopy(dataset_meta or {})}
    metadata={"kind":"trained_model","step":step,"samples_seen":samples_seen,"training_mode":"jepa","training_task":"jepa","jepa_modality":info["modality"],"jepa_target_momentum":target_momentum,"latent_loss":last_loss,"validation_latent_loss":last_val,"training_config":copy.deepcopy(config or {}),"builder_package":builder_package}
    progress({"status":"running","runtime_kind":"train","phase":"final_save","overall":99,"step":step,"max_steps":max_steps,"samples_seen":samples_seen,"loss":last_loss,"val_loss":last_val,"training_mode":"jepa","training_task":"jepa","message":"JEPA training complete · saving final MLBricks model artifact…"})
    IMPORT_POOL.resolve_api("lifecycle.save")(raw,final,metadata=metadata); final_mem=_memory_snapshot(device)
    update={"training_status":"trained","weights_ready":True,"path":str(final),"checkpoint_path":str(final),"trained_steps":step,"samples_seen":samples_seen,"tokens_seen":0,"last_loss":last_loss,"last_val_loss":last_val,"parameter_count":compiled.parameter_count,"training_mode":"jepa","training_task":"jepa","jepa_modality":info["modality"],"jepa_target_momentum":target_momentum,"avg_samples_per_sec":run_samples/max(train_seconds,1e-9) if run_samples else None,"memory_peak_gb":final_mem.get("memory_peak_gb"),"execution_mode_used":"compiled" if compiled.compile_used else "eager","trained_at":time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),"format":"MLBricks model artifact","artifact_format":"mlbricks.model"}
    progress({"status":"done","runtime_kind":"train","phase":"done","overall":100,"step":step,"max_steps":max_steps,"samples_seen":samples_seen,"loss":last_loss,"val_loss":last_val,"samples_per_sec":run_samples/max(train_seconds,1e-9) if run_samples else None,"training_mode":"jepa","training_task":"jepa","message":f"{model_entry.get('name','JEPA')} training complete and saved.","model_update":update})
    return {"compiled":compiled,"tokenizer":None,"model_update":update,"last_sample":None}

def train_builder_model(*,state,model_entry,dataset,dataset_meta,config,progress,stop_event,resume_from=None,resume_mode=None):
    """Train any current Builder model through the correct public runtime.

    JEPA graphs use the latent-prediction trainer with a stop-gradient EMA target
    encoder. Classical KNN/Tree/K-Means/PCA graphs use one-shot fit. Non-text
    Feature, Signal and Image graphs use the generic supervised tensor trainer,
    including the educational object-detection head. Text language models retain
    the packed causal-LM/token training path.
    """
    multimodal_info = _multimodal_graph_info(model_entry, state)
    if multimodal_info is not None:
        return _train_multimodal_builder_model(
            state=state, model_entry=model_entry, dataset=dataset, dataset_meta=dataset_meta,
            config=config, progress=progress, stop_event=stop_event,
            resume_from=resume_from, resume_mode=resume_mode,
        )
    jepa_info = _jepa_graph_info(model_entry, state)
    if jepa_info is not None:
        return _train_jepa_builder_model(
            state=state, model_entry=model_entry, dataset=dataset, dataset_meta=dataset_meta,
            config=config, progress=progress, stop_event=stop_event,
            resume_from=resume_from, resume_mode=resume_mode,
        )
    audio_info = _audio_graph_info(model_entry, state)
    if audio_info is not None:
        return _train_audio_builder_model(
            state=state, model_entry=model_entry, dataset=dataset, dataset_meta=dataset_meta,
            config=config, progress=progress, stop_event=stop_event,
            resume_from=resume_from, resume_mode=resume_mode,
        )
    classical_node = _classical_fit_node(model_entry, state)
    if classical_node is not None:
        return _fit_classical_builder_model(
            state=state, model_entry=model_entry, dataset=dataset, dataset_meta=dataset_meta,
            config=config, progress=progress, stop_event=stop_event,
            resume_from=resume_from, resume_mode=resume_mode,
        )
    supervised_info = _supervised_graph_info(model_entry, state)
    if supervised_info is not None and not _model_requires_tokenizer(model_entry):
        return _train_supervised_builder_model(
            state=state, model_entry=model_entry, dataset=dataset, dataset_meta=dataset_meta,
            config=config, progress=progress, stop_event=stop_event,
            resume_from=resume_from, resume_mode=resume_mode,
        )
    seed=runtime_int(config.get("seed"),42,"Seed")
    random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

    compiled,tokenizer=compile_builder_model(
        state,model_entry,dataset_meta,config,progress=progress,for_training=True
    )
    device=compiled.device
    model=compiled.model
    raw=compiled.raw_model
    loss_model=(compiled.training_model if compiled.training_model is not None else _CausalLMTrainingGraph(raw))
    precision=compiled.precision

    resume_path=Path(str(resume_from)).expanduser() if resume_from else None
    resume_kind=str(resume_mode or "weights").lower() if resume_path is not None else None
    resume_payload=None
    resume_step=0
    resume_tokens=0
    if resume_path is not None:
        progress({
            "status":"running","runtime_kind":"train","phase":"resume_weights","overall":0,
            "step":0,"tokens_seen":0,
            "message":f"Loading existing trained parameters from {resume_path.name or resume_path}…",
            "resume_from":str(resume_path),"resume_mode":resume_kind,
        })
        try:
            if resume_path.is_dir() and (resume_path/"model.pt").exists():
                mlbricks_load=IMPORT_POOL.resolve_api("lifecycle.load")
                loaded=mlbricks_load(resume_path,device=device,strict=True)
                if not isinstance(loaded,nn.Module):
                    raise RuntimeError(f"loaded artifact is not a torch.nn.Module: {type(loaded)!r}")
                raw.load_state_dict(loaded.state_dict(),strict=True)
                del loaded
                if resume_kind=="checkpoint" and (resume_path/"training_state.pt").is_file():
                    resume_payload=safe_torch_load(resume_path/"training_state.pt",map_location="cpu",allow_unsafe_pickle=False)
            elif resume_path.is_file():
                payload=safe_torch_load(resume_path,map_location="cpu",allow_unsafe_pickle=_bool(config.get("allow_unsafe_legacy_checkpoint",False)))
                if not isinstance(payload,dict) or "model_state" not in payload:
                    raise RuntimeError("legacy checkpoint does not contain model_state")
                raw.load_state_dict(payload["model_state"],strict=True)
                resume_payload=payload if resume_kind=="checkpoint" else None
            else:
                raise FileNotFoundError(f"model artifact was not found: {resume_path}")
        except Exception as exc:
            raise ExistingModelArtifactError(
                resume_path,
                f"Existing model parameters at {resume_path} could not be loaded safely: {type(exc).__name__}: {exc}",
            ) from exc

    train=dataset["train"] if isinstance(dataset,dict) or hasattr(dataset,"keys") else dataset
    val_name=str(config.get("validation_split") or "validation")
    val=dataset.get(val_name) if hasattr(dataset,"get") else None

    context=runtime_int(
        model_entry.get("context_length") or state.get("project",{}).get("context_length"),
        512,"Model Context",minimum=2,
    )
    batch=runtime_int(config.get("batch_size"),16,"Batch Size",minimum=1)
    accum=runtime_int(config.get("gradient_accumulation"),1,"Gradient Accumulation",minimum=1)
    pad=runtime_int(tokenizer.pad_token_id,0,"Tokenizer Pad Token ID",minimum=0)
    separator=int(tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad)
    opt=_optimizer(raw,config)
    warm=runtime_int(config.get("warmup_steps"),0,"Warmup Steps",minimum=0)
    scaler=torch.amp.GradScaler("cuda",enabled=(device.type=="cuda" and precision=="fp16")) if hasattr(torch,"amp") else None
    if resume_payload and resume_kind=="checkpoint":
        resume_step=max(0,int(resume_payload.get("step") or 0))
        resume_tokens=max(0,int(resume_payload.get("tokens_seen") or 0))
        try:
            if resume_payload.get("optimizer"):
                opt.load_state_dict(resume_payload["optimizer"])
            if scaler is not None and resume_payload.get("scaler"):
                scaler.load_state_dict(resume_payload["scaler"])
        except Exception as exc:
            # Optimizer/scaler state is optional. The learned model weights are
            # still valuable, so an optimizer mismatch must not force a fresh
            # untrained model. Continue from checkpoint parameters with a new
            # optimizer and a new run counter instead.
            opt=_optimizer(raw,config)
            scaler=torch.amp.GradScaler("cuda",enabled=(device.type=="cuda" and precision=="fp16")) if hasattr(torch,"amp") else None
            resume_step=0
            resume_tokens=0
            progress({
                "status":"running","runtime_kind":"train","phase":"resume_optimizer_reset","overall":0,
                "step":0,"tokens_seen":0,"resume_from":str(resume_path),
                "message":f"Checkpoint weights loaded, but optimizer state is incompatible ({type(exc).__name__}). Continuing retraining with a fresh optimizer…",
            })

    budget=str(config.get("budget_type") or "steps").lower()
    max_steps=runtime_int(config.get("max_steps"),1000,"Training Steps",minimum=1)
    max_tokens=runtime_int(config.get("max_tokens"),1000000,"Token Budget",minimum=1)
    epochs=runtime_float(config.get("epochs"),1.0,"Epochs",minimum=0.000001)
    if budget not in {"steps","tokens","epochs"}:
        raise ValueError(f"Budget By must be steps, tokens, or epochs; received {budget!r}.")
    if budget=="epochs":
        max_steps=max(1,math.ceil(len(train)/batch*epochs))

    # A checkpoint saved exactly at the old budget has no remaining work. In
    # that case treat its weights as the starting point for a new retraining run
    # instead of immediately finishing with no loss value.
    exhausted_resume=(
        resume_kind=="checkpoint" and (
            (budget in {"steps","epochs"} and resume_step>=max_steps)
            or (budget=="tokens" and resume_tokens>=max_tokens)
        )
    )
    if exhausted_resume:
        resume_kind="weights"
        resume_step=0
        resume_tokens=0
        opt=_optimizer(raw,config)
        scaler=torch.amp.GradScaler("cuda",enabled=(device.type=="cuda" and precision=="fp16")) if hasattr(torch,"amp") else None
        progress({
            "status":"running","runtime_kind":"train","phase":"resume_budget_reset","overall":0,
            "step":0,"tokens_seen":0,"resume_from":str(resume_path),
            "message":"Existing checkpoint already reached the configured budget. Starting a new retraining run from its learned weights…",
        })

    validate_every=runtime_int(config.get("validate_every"),100,"Validate Every N Steps",minimum=0)
    val_steps=runtime_int(config.get("validation_steps"),20,"Validation Steps",minimum=1)
    checkpoint_every=runtime_int(config.get("checkpoint_every"),500,"Checkpoint Every N Steps",minimum=0)

    def _train_overall(current_step, current_tokens):
        """Reserve the last 5% for validation, save and artifact commit.

        The old runtime reported 99% as soon as the last optimizer step ended,
        even though final validation, sample generation and model serialization
        were still pending. That made healthy retrains look permanently stuck.
        """
        if budget=="tokens":
            ratio=min(1.0,max(0.0,float(current_tokens)/max(float(max_tokens),1.0)))
        else:
            ratio=min(1.0,max(0.0,float(current_step)/max(float(max_steps),1.0)))
        return min(95,max(2,round(ratio*95)))

    def _is_final_training_point(current_step,current_tokens):
        if budget=="tokens":
            return current_tokens>=max_tokens
        return current_step>=max_steps
    output=Path(str(config.get("output_dir") or "mlbricks_workspace/models"))/_safe_name(model_entry.get("name","model"))
    output.mkdir(parents=True,exist_ok=True)
    (output/'checkpoints').mkdir(exist_ok=True)
    saved_training_config=copy.deepcopy(config)
    logical_output_dir=saved_training_config.pop("_studio_logical_output_dir",None)
    if logical_output_dir:
        saved_training_config["output_dir"]=str(logical_output_dir)

    architecture=copy.deepcopy(model_entry.get("architecture") or _root_model(state))
    custom_components=copy.deepcopy(state.get("custom_components") or {})
    custom_components.update(copy.deepcopy(model_entry.get("custom_components_snapshot") or {}))
    builder_package={
        "format":"mlb-studio-model-v2",
        "builder_version":__version__,
        "project":copy.deepcopy(state.get("project") or {}),
        "model_component":architecture,
        "custom_components":custom_components,
        "model_entry":copy.deepcopy(model_entry),
        "dataset_meta":copy.deepcopy(dataset_meta or {}),
    }

    train_batcher=_PackedLMBatcher(
        train,context=context,separator_id=separator,rng=random.Random(seed ^ 0x54524149)
    )
    val_batcher=(
        _PackedLMBatcher(val,context=context,separator_id=separator,rng=random.Random(seed ^ 0x56414C49))
        if val is not None else None
    )
    # Validation uses the same raw weights but does not need another compiled
    # training graph variant under no_grad().
    eval_loss_model=_CausalLMTrainingGraph(raw)

    tokens_seen=resume_tokens;run_tokens_seen=0;best_val=float('inf');last_val=None;last_val_ppl=None
    model.train();raw.train();loss_model.train()

    # torch.compile is lazy. Prepare two exact-shape batches first, then force
    # two forward+backward passes so compilation and first-use autotuning are
    # excluded from the throughput timer. No optimizer step is taken.
    compile_seconds=0.0
    if compiled.compile_used:
        progress({
            "status":"running","runtime_kind":"train","phase":"compile_warmup","overall":1,
            "step":resume_step,"max_steps":max_steps,"tokens_seen":tokens_seen,"tokens_per_sec":None,
            "avg_tokens_per_sec":None,"end_to_end_tokens_per_sec":None,
            "avg_end_to_end_tokens_per_sec":None,"loss":None,"ppl":None,
            "val_loss":None,"val_ppl":None,**_memory_snapshot(device),
            "message":f"Compiling one whole training graph on {device} · fixed shape [{batch}, {context}] · 2 warm-up passes…",
        })
        warmup_batcher=_PackedLMBatcher(
            train,context=context,separator_id=separator,rng=random.Random(seed ^ 0x4D4C4252)
        )
        warm_batches=[warmup_batcher.batch(batch,device) for _ in range(2)]
        _sync_device(device)
        if device.type=="cuda":
            try: torch.cuda.reset_peak_memory_stats(device)
            except Exception: pass
        compile_started=time.perf_counter()
        for xw,yw,_ in warm_batches:
            opt.zero_grad(set_to_none=True)
            with _autocast_context(device,precision):
                warm_loss=loss_model(xw,yw)/accum
            if scaler is not None and scaler.is_enabled(): scaler.scale(warm_loss).backward()
            else: warm_loss.backward()
        _sync_device(device)
        compile_seconds=max(time.perf_counter()-compile_started,0.0)
        opt.zero_grad(set_to_none=True)
        progress({
            "status":"running","runtime_kind":"train","phase":"compile_done","overall":2,
            "step":resume_step,"max_steps":max_steps,"tokens_seen":tokens_seen,"tokens_per_sec":None,
            "avg_tokens_per_sec":None,"end_to_end_tokens_per_sec":None,
            "avg_end_to_end_tokens_per_sec":None,"loss":None,"ppl":None,
            "val_loss":None,"val_ppl":None,**_memory_snapshot(device),
            "compile_seconds":compile_seconds,
            "message":f"Whole-model compilation complete · {compile_seconds:.1f}s · throughput timer starts now",
        })

    if device.type=="cuda":
        try: torch.cuda.reset_peak_memory_stats(device)
        except Exception: pass
    wall_start=time.perf_counter()
    gpu_train_seconds=0.0
    e2e_train_seconds=0.0
    progress({
        "status":"running","runtime_kind":"train","phase":"train","overall":2,
        "step":resume_step,"max_steps":max_steps,"tokens_seen":tokens_seen,"tokens_per_sec":None,
        "avg_tokens_per_sec":None,"end_to_end_tokens_per_sec":None,
        "avg_end_to_end_tokens_per_sec":None,"loss":None,"ppl":None,
        "val_loss":None,"val_ppl":None,**_memory_snapshot(device),
        "compile_seconds":compile_seconds,
        "message":f"Training started on {device} · {compiled.parameter_count:,} parameters · packed [{batch}, {context}]"+
                  (f" · retraining existing weights from {resume_path.name}" if resume_path is not None and resume_kind!="checkpoint" else "")+
                  (f" · resumed checkpoint step {resume_step}" if resume_path is not None and resume_kind=="checkpoint" else "")+
                  (f" · whole-model compiled ({compile_seconds:.1f}s warm-up)" if compiled.compile_used else " · eager"),
        "compile_warning":compiled.compile_error,
    })

    step=resume_step;loss_value=None;sample=None
    while True:
        if stop_event.is_set(): raise TrainingStopped("Training stopped.")
        if budget=="steps" and step>=max_steps:break
        if budget=="tokens" and tokens_seen>=max_tokens:break
        if budget=="epochs" and step>=max_steps:break
        step+=1

        # Prepare packed CPU batches and finish H2D transfer before GPU timing.
        # This makes Tok/s a model-training metric, while E2E Tok/s separately
        # reports data preparation + transfer + model training.
        e2e_started=time.perf_counter()
        microbatches=[train_batcher.batch(batch,device) for _ in range(accum)]
        _sync_device(device)
        step_started=time.perf_counter()

        opt.zero_grad(set_to_none=True)
        step_tokens=0
        detached_losses=[]
        for x,y,toks in microbatches:
            step_tokens+=toks
            with _autocast_context(device,precision):
                loss=loss_model(x,y)/accum
            if scaler is not None and scaler.is_enabled():scaler.scale(loss).backward()
            else:loss.backward()
            detached_losses.append(loss.detach())

        if scaler is not None and scaler.is_enabled():
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(raw.parameters(),1.0)
            scaler.step(opt);scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(raw.parameters(),1.0)
            opt.step()

        if warm>0 and step<=warm:
            factor=step/warm
            base_lr=runtime_float(config.get('learning_rate'),5e-4,'Learning Rate',minimum=0.0)
            for group in opt.param_groups:group['lr']=base_lr*factor

        _sync_device(device)
        gpu_elapsed=max(time.perf_counter()-step_started,1e-9)
        e2e_elapsed=max(time.perf_counter()-e2e_started,1e-9)
        gpu_train_seconds+=gpu_elapsed
        e2e_train_seconds+=e2e_elapsed
        tokens_seen+=step_tokens
        run_tokens_seen+=step_tokens
        loss_value=sum(float(v.float().cpu()) for v in detached_losses)
        tokens_per_sec=float(step_tokens)/gpu_elapsed
        avg_tokens_per_sec=float(run_tokens_seen)/max(gpu_train_seconds,1e-9)
        end_to_end_tokens_per_sec=float(step_tokens)/e2e_elapsed
        avg_end_to_end_tokens_per_sec=float(run_tokens_seen)/max(e2e_train_seconds,1e-9)
        ppl=_perplexity(loss_value)
        mem=_memory_snapshot(device)
        lr=float(opt.param_groups[0].get('lr',0.0)) if opt.param_groups else None
        do_val=validate_every>0 and (step%validate_every==0 or (budget=="steps" and step==max_steps))
        sample=None

        base_event={
            "step":step,"max_steps":max_steps,"tokens_seen":tokens_seen,
            "tokens_per_sec":tokens_per_sec,"avg_tokens_per_sec":avg_tokens_per_sec,
            "end_to_end_tokens_per_sec":end_to_end_tokens_per_sec,
            "avg_end_to_end_tokens_per_sec":avg_end_to_end_tokens_per_sec,
            "loss":loss_value,"ppl":ppl,"val_loss":last_val,"val_ppl":last_val_ppl,
            "lr":lr,"compile_seconds":compile_seconds,
        }

        if do_val and val_batcher is not None:
            final_validation=_is_final_training_point(step,tokens_seen)
            validation_base_overall=96 if final_validation else _train_overall(step,tokens_seen)
            progress({
                "status":"running","runtime_kind":"train","phase":"validation",
                "overall":validation_base_overall,
                **base_event,**mem,
                "validation_step":0,"validation_steps":val_steps,
                "message":f"Validating at step {step} · 0/{val_steps} batches…",
            })

            def emit_validation_batch(done,total,running_val):
                if stop_event.is_set():
                    raise TrainingStopped("Training stopped.")
                if final_validation:
                    # Final validation owns 96-98%. Keep 99% for sample/save.
                    validation_overall=min(98,96+round(2*done/max(total,1)))
                else:
                    validation_overall=validation_base_overall
                progress({
                    "status":"running","runtime_kind":"train","phase":"validation",
                    "overall":validation_overall,
                    **base_event,**_memory_snapshot(device),
                    "validation_step":done,"validation_steps":total,
                    "validation_running_loss":running_val,
                    "message":f"Validating at step {step} · {done}/{total} batches…",
                })

            last_val=_evaluate(
                eval_loss_model,raw,val_batcher,steps=val_steps,batch_size=batch,
                device=device,precision=precision,
                progress_callback=emit_validation_batch,stop_event=stop_event,
            )
            best_val=min(best_val,last_val)
            last_val_ppl=_perplexity(last_val)
            base_event.update({"val_loss":last_val,"val_ppl":last_val_ppl})

            if _bool(config.get("generate_on_validation",True)):
                sample_tokens=runtime_int(
                    config.get("validation_generate_tokens"),64,
                    "Validation Sample Tokens",minimum=1,
                )
                progress({
                    "status":"running","runtime_kind":"train","phase":"validation_generation",
                    "overall":98 if final_validation else validation_base_overall,
                    **base_event,**_memory_snapshot(device),
                    "validation_step":val_steps,"validation_steps":val_steps,
                    "validation_generated_tokens":0,
                    "validation_generate_tokens":sample_tokens,
                    "message":f"Validation complete · generating sample 0/{sample_tokens} tokens…",
                })

                def emit_validation_generation(event):
                    generated_count=int((event or {}).get("generated_tokens") or 0)
                    progress({
                        "status":"running","runtime_kind":"train","phase":"validation_generation",
                        "overall":98 if final_validation else validation_base_overall,
                        **base_event,**_memory_snapshot(device),
                        "validation_step":val_steps,"validation_steps":val_steps,
                        "validation_generated_tokens":generated_count,
                        "validation_generate_tokens":sample_tokens,
                        "message":f"Validation complete · generating sample {generated_count}/{sample_tokens} tokens…",
                    })

                try:
                    sample,_=generate_text(
                        model,tokenizer,config.get("validation_prompt","Once upon a time"),
                        max_new_tokens=sample_tokens,
                        context=context,device=device,precision=precision,temperature=.8,top_k=50,top_p=.95,
                        seed=seed+step,stop_event=stop_event,progress=emit_validation_generation,
                    )
                except TrainingStopped:
                    raise
                except Exception as exc:
                    sample=f"[sample generation skipped: {exc}]"
            mem=_memory_snapshot(device)
            progress({
                "status":"running","runtime_kind":"train","phase":"validation_done",
                "overall":98 if final_validation else validation_base_overall,
                **base_event,
                "best_val_loss":None if best_val==float('inf') else best_val,
                "sample_text":sample,"elapsed_seconds":time.perf_counter()-wall_start,**mem,
                "validation_step":val_steps,"validation_steps":val_steps,
                "message":f"Validation complete · val loss {last_val:.4f} · val ppl {last_val_ppl:.2f}",
            })

        if checkpoint_every>0 and step%checkpoint_every==0:
            checkpoint_path=output/'checkpoints'/f'step_{step:06d}'
            metadata={
                "kind":"training_checkpoint","step":step,"tokens_seen":tokens_seen,
                "vocab_size":compiled.vocab_size,"training_config":copy.deepcopy(saved_training_config),
                "builder_package":builder_package,
            }
            mlbricks_save = IMPORT_POOL.resolve_api("lifecycle.save")
            mlbricks_save(raw,checkpoint_path,metadata=metadata)
            # Optimizer/scaler state is supplemental training state; the model
            # itself is always stored through the public MLBricks lifecycle API.
            torch.save({
                "optimizer":opt.state_dict(),"scaler":scaler.state_dict() if scaler is not None else None,
                "step":step,"tokens_seen":tokens_seen,
            },checkpoint_path/'training_state.pt')
            progress({
                "status":"running","runtime_kind":"train","phase":"checkpoint",
                "overall":_train_overall(step,tokens_seen),
                **base_event,"best_val_loss":None if best_val==float('inf') else best_val,
                "sample_text":sample,"checkpoint_path":str(checkpoint_path),
                "elapsed_seconds":time.perf_counter()-wall_start,**_memory_snapshot(device),
                "message":f"MLBricks checkpoint saved · step {step}",
            })

        overall=(
            98 if (do_val and val_batcher is not None and _is_final_training_point(step,tokens_seen))
            else _train_overall(step,tokens_seen)
        )
        mem=_memory_snapshot(device)
        mem_text=(f" · mem {mem['memory_allocated_gb']:.2f} GB" if mem.get('memory_allocated_gb') is not None else "")
        val_text=(f" · val {last_val:.4f} · val ppl {last_val_ppl:.2f}" if last_val is not None else "")
        progress({
            "status":"running","runtime_kind":"train","phase":"train","overall":overall,
            **base_event,"best_val_loss":None if best_val==float('inf') else best_val,
            "sample_text":sample,"elapsed_seconds":time.perf_counter()-wall_start,**mem,
            "message":f"Step {step} · {tokens_per_sec:,.0f} GPU tok/s · {end_to_end_tokens_per_sec:,.0f} E2E tok/s · loss {loss_value:.4f} · ppl {ppl:.2f}"+val_text+mem_text,
        })

    final=output/'last'
    mlbricks_save = IMPORT_POOL.resolve_api("lifecycle.save")
    tok_cfg=((dataset_meta or {}).get("pipeline") or {}).get("tokenizer") or {}
    progress({
        "status":"running","runtime_kind":"train","phase":"final_save","overall":99,
        "step":step,"max_steps":max_steps,"tokens_seen":tokens_seen,
        "tokens_per_sec":tokens_per_sec if 'tokens_per_sec' in locals() else None,
        "avg_tokens_per_sec":float(run_tokens_seen)/max(gpu_train_seconds,1e-9) if run_tokens_seen else None,
        "end_to_end_tokens_per_sec":end_to_end_tokens_per_sec if 'end_to_end_tokens_per_sec' in locals() else None,
        "avg_end_to_end_tokens_per_sec":float(run_tokens_seen)/max(e2e_train_seconds,1e-9) if run_tokens_seen else None,
        "loss":loss_value,"ppl":_perplexity(loss_value) if loss_value is not None else None,
        "val_loss":last_val,"val_ppl":last_val_ppl,
        "best_val_loss":None if best_val==float('inf') else best_val,
        "compile_seconds":compile_seconds,"elapsed_seconds":time.perf_counter()-wall_start,
        **_memory_snapshot(device),
        "message":"Training steps complete · saving final MLBricks model artifact…",
    })
    final_metadata={
        "kind":"trained_model","step":step,"tokens_seen":tokens_seen,
        "best_val_loss":None if best_val==float('inf') else best_val,
        "training_config":copy.deepcopy(saved_training_config),"builder_package":builder_package,
        "tokenizer_name":tok_cfg.get("tokenizer_name") or "gpt2",
        "execution":"whole-model compiled" if compiled.compile_used else "eager",
        "compile_mode":str(config.get("compile_mode") or "default") if compiled.compile_used else None,
        "compile_fullgraph":True if compiled.compile_used else None,
        "compile_dynamic":False if compiled.compile_used else None,
    }
    mlbricks_save(raw,final,metadata=final_metadata)
    progress({
        "status":"running","runtime_kind":"train","phase":"finalize","overall":99,
        "step":step,"max_steps":max_steps,"tokens_seen":tokens_seen,
        "loss":loss_value,"ppl":_perplexity(loss_value) if loss_value is not None else None,
        "val_loss":last_val,"val_ppl":last_val_ppl,
        "best_val_loss":None if best_val==float('inf') else best_val,
        "compile_seconds":compile_seconds,"elapsed_seconds":time.perf_counter()-wall_start,
        **_memory_snapshot(device),
        "message":"Model weights saved · finalizing tokenizer and artifact metadata…",
    })
    tokenizer_dir=final/'tokenizer'
    try:
        tokenizer.save_pretrained(str(tokenizer_dir))
    except Exception:
        tokenizer_dir=None

    final_mem=_memory_snapshot(device)
    update={
        "training_status":"trained","weights_ready":True,"path":str(final),"checkpoint_path":str(final),
        "trained_steps":step,"tokens_seen":tokens_seen,"last_loss":loss_value,"last_ppl":_perplexity(loss_value),
        "best_val_loss":None if best_val==float('inf') else best_val,"last_val_loss":last_val,"last_val_ppl":last_val_ppl,
        "avg_tokens_per_sec":float(run_tokens_seen)/max(gpu_train_seconds,1e-9),
        "avg_end_to_end_tokens_per_sec":float(run_tokens_seen)/max(e2e_train_seconds,1e-9),
        "memory_peak_gb":final_mem.get("memory_peak_gb"),"parameter_count":compiled.parameter_count,
        "effective_vocab_size":compiled.vocab_size,"execution_mode_used":"compiled" if compiled.compile_used else "eager",
        "compile_warning":compiled.compile_error,"compile_seconds":compile_seconds,
        "trained_at":time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        "format":"MLBricks model artifact","artifact_format":"mlbricks.model",
        "tokenizer_path":str(tokenizer_dir) if tokenizer_dir is not None else None,
        "retrained_from":str(resume_path) if resume_path is not None and resume_kind!="checkpoint" else None,
        "resumed_checkpoint":str(resume_path) if resume_path is not None and resume_kind=="checkpoint" else None,
    }
    return {"compiled":compiled,"tokenizer":tokenizer,"model_update":update,"last_sample":sample}


def load_trained_for_generation(*,state,model_entry,dataset_meta,config,checkpoint_path=None,progress=None):
    """Load a trained runtime with a fast path for unified MLBricks artifacts.

    Studio training saves ``raw`` TensorGraph itself with ``mlbricks.save``.  The
    older generation path rebuilt the entire 50M+ graph, then called
    ``mlbricks.load`` (which loaded a second full graph), and finally copied the
    second graph's state_dict into the first.  On Windows/CPU this can dominate
    first-token latency by tens of seconds.

    Unified artifacts already contain the complete trained module graph, so the
    normal eager/auto generation path can load that graph directly, cast/move it
    once, and reuse it in Builder's runtime cache.  Explicit backend overrides
    still use the reconstruction path because they intentionally re-instantiate
    components with a different backend.
    """
    path=Path(str(checkpoint_path or model_entry.get("checkpoint_path") or model_entry.get("path") or ""))
    if not path.exists():
        raise RuntimeError("Trained model artifact was not found. Train the model in this session or select a valid MLBricks model artifact.")

    backend=str(config.get("backend") or "auto").lower()
    unified=path.is_dir() and (path/"model.pt").exists()
    direct_ok=unified and backend in {"", "auto"}

    if direct_ok:
        device=resolve_device(config.get("device","auto"))
        precision,dtype=resolve_precision(config.get("precision","auto"),device)
        tokenizer=None
        if _model_requires_tokenizer(model_entry):
            if progress:
                progress({
                    "status":"running","runtime_kind":"generate","phase":"tokenizer","overall":0,
                    "message":f"Loading tokenizer for {device}…",
                })
            tokenizer=_tokenizer_for(
                dataset_meta, tokenizer_path=(model_entry or {}).get("tokenizer_path")
            )
        if progress:
            progress({
                "status":"running","runtime_kind":"generate","phase":"weights","overall":0,
                "message":f"Loading trained model directly from {path.name or path}…",
            })
        try:
            mlbricks_load=IMPORT_POOL.resolve_api("lifecycle.load")
        except ImportError as exc:
            raise RuntimeError("Current MLBricks installation does not expose mlbricks.load().") from exc
        loaded=mlbricks_load(path,device=device,strict=True)
        if not isinstance(loaded,nn.Module):
            raise RuntimeError(f"Loaded MLBricks artifact is not a torch.nn.Module: {type(loaded)!r}")
        # mlbricks.load already places the model on the requested device. Cast
        # floating tensors once to the generation precision instead of creating
        # and moving a second graph first.
        loaded.to(device=device,dtype=dtype)
        graph=copy.deepcopy((model_entry or {}).get("architecture") or _root_model(state))
        graph_vocab=_graph_vocab(graph)
        effective_vocab=max(graph_vocab,len(tokenizer)) if tokenizer is not None else graph_vocab
        params=sum(p.numel() for p in loaded.parameters())
        inference_model=loaded
        compile_used=False
        if str(config.get("execution_mode") or "eager")=="compiled":
            if not hasattr(torch,"compile"):
                raise RuntimeError("Compiled execution was selected, but torch.compile is unavailable in this PyTorch build.")
            inference_model=torch.compile(
                loaded,mode=str(config.get("compile_mode") or "default"),dynamic=None,fullgraph=False
            )
            compile_used=True
        return CompiledModel(
            inference_model,loaded,None,device,precision,effective_vocab,
            params,compile_used,None,
        ),tokenizer

    # Compatibility path for legacy checkpoints and explicit backend overrides.
    compiled,tokenizer=compile_builder_model(
        state,model_entry,dataset_meta,config,progress=progress,for_training=False
    )

    if progress:
        progress({
            "status":"running","runtime_kind":"generate","phase":"weights",
            "overall":0,
            "message":f"Loading trained weights from {path.name or path}…",
        })

    if path.is_dir() and (path/"model.pt").exists():
        try:
            mlbricks_load = IMPORT_POOL.resolve_api("lifecycle.load")
        except ImportError as exc:
            raise RuntimeError("Current MLBricks installation does not expose mlbricks.load().") from exc
        loaded=mlbricks_load(path,device=compiled.device,strict=True)
        compiled.raw_model.load_state_dict(loaded.state_dict(),strict=True)
        del loaded
    else:
        # Legacy MLB Studio checkpoint format.
        payload=safe_torch_load(path,map_location="cpu",allow_unsafe_pickle=_bool(config.get("allow_unsafe_legacy_checkpoint",False)))
        if not isinstance(payload,dict) or "model_state" not in payload:
            raise RuntimeError("Selected file is neither an MLBricks model artifact nor a legacy Builder checkpoint.")
        compiled.raw_model.load_state_dict(payload["model_state"],strict=True)

    compiled.raw_model.to(compiled.device)
    return compiled,tokenizer
