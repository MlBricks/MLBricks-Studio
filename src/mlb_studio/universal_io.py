from __future__ import annotations

"""Universal runtime input helpers for MLBricks Studio.

This module deliberately separates *what* is arriving (text/image/audio/video/
signal/file/multimodal) from *how* it arrives (single/sequence/live/continuous)
and from *what the user wants to do* (generate/edit/analyze/monitor/etc.).

The browser sends a compact input envelope.  Python owns source ingestion so the
same Studio project can run in a notebook, on a workstation, or against a remote
sensor/CCTV URL without baking modality-specific logic into the UI.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
import base64
import io
import json
import queue
import socket
import time
import urllib.request
import wave


_INPUT_KINDS = {"text", "image", "audio", "video", "signal", "file", "multimodal"}
_LIVE_MODES = {"live", "continuous", "cctv", "stream", "monitor"}


@dataclass(frozen=True)
class InputEnvelope:
    kind: str
    mode: str
    task: str
    source_type: str
    source: str
    data: Any
    mime: str
    prompt: str
    sample_rate: int
    fps: float
    buffer_size: int
    channel: str
    metadata: dict[str, Any]

    @property
    def continuous(self) -> bool:
        return self.mode in _LIVE_MODES

    def public_dict(self) -> dict[str, Any]:
        # Do not echo potentially huge inline payloads through telemetry.
        data_summary: Any = None
        if self.data not in (None, ""):
            if isinstance(self.data, str):
                data_summary = f"inline:{len(self.data)} chars"
            elif isinstance(self.data, (list, tuple)):
                data_summary = f"inline:{len(self.data)} items"
            else:
                data_summary = type(self.data).__name__
        return {
            "kind": self.kind,
            "mode": self.mode,
            "task": self.task,
            "source_type": self.source_type,
            "source": self.source,
            "mime": self.mime,
            "prompt": self.prompt,
            "sample_rate": self.sample_rate,
            "fps": self.fps,
            "buffer_size": self.buffer_size,
            "channel": self.channel,
            "metadata": dict(self.metadata),
            "data_summary": data_summary,
            "continuous": self.continuous,
        }


def _to_int(value: Any, default: int, minimum: int = 1) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def _to_float(value: Any, default: float, minimum: float = 0.01) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def normalize_input_config(config: dict[str, Any] | None, model_entry: dict[str, Any] | None = None) -> InputEnvelope:
    config = dict(config or {})
    requirements = dict((model_entry or {}).get("requirements") or {})
    model_kind = str(requirements.get("modality") or "text").strip().lower()
    kind = str(config.get("input_kind") or model_kind or "text").strip().lower()
    if kind not in _INPUT_KINDS:
        kind = "text"

    default_mode = {
        "text": "single",
        "image": "single",
        "audio": "file",
        "video": "file",
        "signal": "static",
        "file": "single",
        "multimodal": "single",
    }[kind]
    mode = str(config.get("input_mode") or default_mode).strip().lower()

    default_task = {
        "text": "generate",
        "image": "analyze",
        "audio": "analyze",
        "video": "analyze",
        "signal": "analyze",
        "file": "process",
        "multimodal": "run",
    }[kind]
    task = str(config.get("task_type") or default_task).strip().lower()
    source_type = str(config.get("input_source_type") or ("inline" if kind == "text" else "path_or_url")).strip().lower()
    source = str(config.get("input_source") or "").strip()
    mime = str(config.get("input_mime") or "").strip()
    prompt = str(config.get("prompt") or "")
    channel = str(config.get("input_channel") or "").strip()
    metadata = dict(config.get("input_metadata") or {}) if isinstance(config.get("input_metadata"), dict) else {}

    return InputEnvelope(
        kind=kind,
        mode=mode,
        task=task,
        source_type=source_type,
        source=source,
        data=config.get("input_data"),
        mime=mime,
        prompt=prompt,
        sample_rate=_to_int(config.get("input_sample_rate"), 16000),
        fps=_to_float(config.get("input_fps"), 5.0),
        buffer_size=_to_int(config.get("input_buffer_size"), 256),
        channel=channel,
        metadata=metadata,
    )


def _read_bytes(source: str) -> bytes:
    source = str(source or "").strip()
    if not source:
        raise ValueError("An input source is required.")
    if source.startswith("data:"):
        head, _, payload = source.partition(",")
        if not payload:
            raise ValueError("Invalid data URL input.")
        return base64.b64decode(payload) if ";base64" in head else payload.encode("utf-8")
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=15) as response:  # nosec - explicit user source
            return response.read()
    path = Path(source).expanduser()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Input source was not found: {source}")
    return path.read_bytes()


def _numeric_values(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: list[float] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                out.extend(_numeric_values(item))
            else:
                try:
                    out.append(float(item))
                except (TypeError, ValueError):
                    pass
        return out
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if parsed is not value:
                values = _numeric_values(parsed)
                if values:
                    return values
        except Exception:
            pass
        out = []
        for part in text.replace(";", ",").replace("\n", ",").split(","):
            try:
                out.append(float(part.strip()))
            except (TypeError, ValueError):
                pass
        return out
    try:
        return [float(value)]
    except (TypeError, ValueError):
        return []


def _signal_from_file(source: str) -> list[float]:
    path = Path(source).expanduser()
    suffix = path.suffix.lower()
    if suffix == ".npy":
        import numpy as np
        return [float(x) for x in np.load(path).reshape(-1).tolist()]
    text = path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".json":
        return _numeric_values(json.loads(text))
    return _numeric_values(text)


def _image_tensor(source: str, *, image_size: int | None = None):
    import numpy as np
    import torch
    from PIL import Image

    raw = _read_bytes(source)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    if image_size and image_size > 0:
        image = image.resize((int(image_size), int(image_size)))
    arr = np.asarray(image, dtype="float32") / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor, {"width": image.width, "height": image.height, "channels": 3}


def _audio_tensor(envelope: InputEnvelope):
    import numpy as np
    import torch

    if envelope.data not in (None, ""):
        values = _numeric_values(envelope.data)
        arr = np.asarray(values, dtype="float32")
        return torch.from_numpy(arr).view(1, -1, 1), {"samples": len(values), "sample_rate": envelope.sample_rate}

    source = envelope.source
    suffix = Path(source.split("?",1)[0]).suffix.lower()
    if suffix != ".wav" and not source.startswith("data:audio/wav"):
        raise ValueError("Core Studio audio input currently reads WAV files/URLs or inline numeric samples. Other codecs can be supplied by a custom input adapter.")
    raw = _read_bytes(source)
    with wave.open(io.BytesIO(raw), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
    if dtype is None:
        raise ValueError(f"Unsupported WAV sample width: {width}")
    arr = np.frombuffer(frames, dtype=dtype).astype("float32")
    if width == 1:
        arr = (arr - 128.0) / 128.0
    else:
        arr /= float(2 ** (width * 8 - 1))
    if channels > 1:
        arr = arr.reshape(-1, channels)
    else:
        arr = arr.reshape(-1, 1)
    return torch.from_numpy(arr).unsqueeze(0), {"samples": int(arr.shape[0]), "sample_rate": rate, "channels": channels}


def load_single_input(envelope: InputEnvelope, *, image_size: int | None = None):
    """Load one envelope into a model-ready Python value plus metadata."""
    import numpy as np
    import torch

    if envelope.kind == "text":
        return envelope.prompt, {"characters": len(envelope.prompt)}
    if envelope.kind == "image":
        source = envelope.source or (envelope.data if isinstance(envelope.data, str) else "")
        return _image_tensor(source, image_size=image_size)
    if envelope.kind == "audio":
        return _audio_tensor(envelope)
    if envelope.kind == "signal":
        values = _numeric_values(envelope.data)
        if not values and envelope.source:
            values = _signal_from_file(envelope.source)
        if not values:
            raise ValueError("Signal input needs numeric samples or a signal file.")
        arr = np.asarray(values, dtype="float32")
        return torch.from_numpy(arr).view(1, -1, 1), {"samples": len(values), "sample_rate": envelope.sample_rate}
    if envelope.kind == "file":
        raw = _read_bytes(envelope.source)
        return raw, {"bytes": len(raw), "name": Path(envelope.source).name}
    if envelope.kind == "multimodal":
        data = envelope.data
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                data = {"text": data}
        return data, {"parts": len(data) if isinstance(data, dict) else 1}
    if envelope.kind == "video":
        # A file-mode video is a finite frame stream; the runtime loop owns it.
        iterator = iter_input_stream(envelope, image_size=image_size)
        try:
            return next(iterator)
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
    raise ValueError(f"Unsupported input kind: {envelope.kind}")


def _cv2_frame_tensor(frame, image_size: int | None = None):
    import numpy as np
    import torch
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Video/CCTV input needs OpenCV. Install `opencv-python-headless` in the runtime environment.") from exc
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if image_size and image_size > 0:
        frame = cv2.resize(frame, (int(image_size), int(image_size)))
    arr = np.asarray(frame, dtype="float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous(), {
        "width": int(arr.shape[1]), "height": int(arr.shape[0]), "channels": int(arr.shape[2])
    }


def _video_stream(envelope: InputEnvelope, image_size: int | None = None, stop_event=None) -> Iterator[tuple[Any, dict[str, Any]]]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Video/CCTV input needs OpenCV. Install `opencv-python-headless` in the runtime environment.") from exc
    source: Any = envelope.source
    if envelope.source_type in {"camera", "webcam"}:
        try:
            source = int(source or 0)
        except ValueError:
            source = 0
    if not source and envelope.source_type not in {"camera", "webcam"}:
        raise ValueError("Video/CCTV input needs a file path, stream URL, or camera index.")
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source!r}")
    frame_index = 0
    delay = 1.0 / max(envelope.fps, 0.01)
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            started = time.monotonic()
            ok, frame = capture.read()
            if not ok:
                break
            tensor, meta = _cv2_frame_tensor(frame, image_size=image_size)
            frame_index += 1
            meta.update({"frame": frame_index, "fps": envelope.fps})
            yield tensor, meta
            if envelope.continuous:
                remaining = delay - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        capture.release()


def _image_sequence(envelope: InputEnvelope, image_size: int | None = None, stop_event=None) -> Iterator[tuple[Any, dict[str, Any]]]:
    root = Path(envelope.source).expanduser()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Image sequence directory was not found: {root}")
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
    paths = sorted({p for pattern in patterns for p in root.glob(pattern)})
    if not paths:
        raise ValueError(f"No image frames were found in {root}")
    for index, path in enumerate(paths, 1):
        if stop_event is not None and stop_event.is_set():
            break
        tensor, meta = _image_tensor(str(path), image_size=image_size)
        meta.update({"frame": index, "path": str(path)})
        yield tensor, meta


def _serial_signal_stream(envelope: InputEnvelope, stop_event=None) -> Iterator[tuple[Any, dict[str, Any]]]:
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("Serial sensor/antenna input needs pyserial. Install `pyserial` in the runtime environment.") from exc
    import numpy as np
    import torch

    baud = int(envelope.metadata.get("baud_rate") or 115200)
    port = envelope.source or envelope.channel
    if not port:
        raise ValueError("Serial signal input needs a port such as COM3 or /dev/ttyUSB0.")
    ser = serial.Serial(port, baudrate=baud, timeout=0.5)
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            line = ser.readline().decode("utf-8", errors="replace")
            values = _numeric_values(line)
            if not values:
                continue
            arr = np.asarray(values[: envelope.buffer_size], dtype="float32")
            yield torch.from_numpy(arr).view(1, -1, 1), {"samples": len(arr), "sample_rate": envelope.sample_rate, "port": port}
    finally:
        ser.close()


def _tcp_signal_stream(envelope: InputEnvelope, stop_event=None) -> Iterator[tuple[Any, dict[str, Any]]]:
    import numpy as np
    import torch

    host, sep, port_text = envelope.source.rpartition(":")
    if not sep or not host:
        raise ValueError("TCP signal source must look like host:port.")
    sock = socket.create_connection((host, int(port_text)), timeout=5)
    sock.settimeout(0.5)
    buffer = ""
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                break
            buffer += chunk.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                values = _numeric_values(line)
                if not values:
                    continue
                arr = np.asarray(values[: envelope.buffer_size], dtype="float32")
                yield torch.from_numpy(arr).view(1, -1, 1), {"samples": len(arr), "sample_rate": envelope.sample_rate, "source": envelope.source}
    finally:
        sock.close()


def _tail_signal_file(envelope: InputEnvelope, stop_event=None) -> Iterator[tuple[Any, dict[str, Any]]]:
    import numpy as np
    import torch

    path = Path(envelope.source).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Signal stream file was not found: {path}")
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(0, 2)
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            line = handle.readline()
            if not line:
                time.sleep(0.05)
                continue
            values = _numeric_values(line)
            if not values:
                continue
            arr = np.asarray(values[: envelope.buffer_size], dtype="float32")
            yield torch.from_numpy(arr).view(1, -1, 1), {"samples": len(arr), "sample_rate": envelope.sample_rate, "path": str(path)}


def _live_audio_stream(envelope: InputEnvelope, stop_event=None) -> Iterator[tuple[Any, dict[str, Any]]]:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("Live microphone input needs sounddevice. Install `sounddevice` or provide a custom audio source adapter.") from exc
    import numpy as np
    import torch

    chunks: queue.Queue[Any] = queue.Queue(maxsize=8)
    channels = int(envelope.metadata.get("channels") or 1)

    def callback(indata, frames, time_info, status):
        del frames, time_info, status
        try:
            chunks.put_nowait(indata.copy())
        except queue.Full:
            # Monitoring favors the newest sensor window over blocking the audio callback.
            try:
                chunks.get_nowait()
            except queue.Empty:
                pass
            try:
                chunks.put_nowait(indata.copy())
            except queue.Full:
                pass

    with sd.InputStream(
        samplerate=envelope.sample_rate,
        channels=channels,
        blocksize=envelope.buffer_size,
        callback=callback,
    ):
        while stop_event is None or not stop_event.is_set():
            try:
                arr=chunks.get(timeout=0.25)
            except queue.Empty:
                continue
            arr=np.asarray(arr,dtype="float32")
            yield torch.from_numpy(arr).unsqueeze(0), {
                "samples": int(arr.shape[0]), "sample_rate": envelope.sample_rate,
                "channels": int(arr.shape[1]) if arr.ndim>1 else 1,
            }


def iter_input_stream(envelope: InputEnvelope, *, image_size: int | None = None, stop_event=None) -> Iterator[tuple[Any, dict[str, Any]]]:
    """Yield model-ready samples for finite sequences or continuous sources."""
    if envelope.kind == "image" and envelope.mode in {"sequence", "batch"}:
        yield from _image_sequence(envelope, image_size=image_size, stop_event=stop_event)
        return
    if envelope.kind == "image" and envelope.mode == "live":
        yield from _video_stream(envelope, image_size=image_size, stop_event=stop_event)
        return
    if envelope.kind == "video":
        yield from _video_stream(envelope, image_size=image_size, stop_event=stop_event)
        return
    if envelope.kind == "signal" and envelope.mode in _LIVE_MODES:
        source_type = envelope.source_type
        if source_type in {"serial", "sensor", "antenna"}:
            yield from _serial_signal_stream(envelope, stop_event=stop_event)
            return
        if source_type == "tcp":
            yield from _tcp_signal_stream(envelope, stop_event=stop_event)
            return
        if source_type in {"file_tail", "file"}:
            yield from _tail_signal_file(envelope, stop_event=stop_event)
            return
        raise ValueError("Continuous signal input needs Serial/Sensor/Antenna, TCP, or File Tail as its source type.")
    if envelope.kind == "audio" and envelope.mode in _LIVE_MODES:
        yield from _live_audio_stream(envelope, stop_event=stop_event)
        return
    value, meta = load_single_input(envelope, image_size=image_size)
    yield value, meta
