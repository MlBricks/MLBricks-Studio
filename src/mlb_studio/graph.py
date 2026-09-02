from __future__ import annotations

from datetime import datetime, timezone
import uuid


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def primitive_catalog():
    return [
        {
            "type": "text_input",
            "builder_utility": True,
            "builder_python_api": False,
            "name": "Text Input",
            "icon": "TXT",
            "category": "Inputs",
            "description": "Text input for prompts or prepared dataset samples.",
            "accent": "green",
            "api": [
                {"key": "input_mode", "label": "Input Source", "type": "select", "value": "prompt",
                 "options": ["prompt", "prepared_dataset"]},
                {"key": "prompt", "label": "Prompt / Text", "type": "textarea", "value": "Once upon a time",
                 "show_when": {"input_mode": "prompt"}},
                {"key": "dataset_id", "label": "Available Dataset", "type": "dataset_select", "value": "",
                 "show_when": {"input_mode": "prepared_dataset"}},
                {"key": "dataset_split", "label": "Use Split", "type": "dataset_split_select", "value": "train",
                 "show_when": {"input_mode": "prepared_dataset"}}
            ],
        },
        {
            "type": "image_input",
            "builder_utility": True,
            "name": "Image Input",
            "icon": "IMG",
            "category": "Inputs",
            "description": "Image input for vision and multimodal models.",
            "accent": "green",
            "api": [
                {"key": "channels", "label": "Channels", "type": "number", "value": 3},
                {"key": "image_size", "label": "Image Size", "type": "number", "value": 224},
            ],
        },
        {
            "type": "audio_input",
            "builder_utility": True,
            "name": "Audio Input",
            "icon": "AUD",
            "category": "Inputs",
            "description": "Audio input for speech and audio models.",
            "accent": "green",
            "api": [
                {"key": "sample_rate", "label": "Sample Rate", "type": "number", "value": 16000},
            ],
        },

        {
            "type": "hf_dataset",
            "builder_utility": True,
            "builder_python_api": True,
            "name": "Hugging Face Dataset",
            "icon": "HF",
            "category": "Data Source",
            "description": "Dataset source for loading data from the Hugging Face Hub.",
            "accent": "cyan",
            "api": [
                {"key": "dataset_id", "label": "Dataset ID", "type": "text", "value": "roneneldan/TinyStories"},
                {"key": "config", "label": "Config", "type": "text", "value": ""},
                {"key": "split", "label": "Hub Source Split", "type": "text", "value": "train", "help": "Which split is downloaded from Hugging Face. Use the Train / Validation / Test Split step for percentages."},
                {"key": "text_column", "label": "Text Column", "type": "text", "value": "text"},
                {"key": "streaming", "label": "Streaming", "type": "select", "value": "false", "options": ["false", "true"]},
                {"key": "max_rows", "label": "Max Rows (0 = All)", "type": "number", "value": 0}
            ],
        },
        {
            "type": "kaggle_dataset",
            "builder_utility": True,
            "builder_python_api": True,
            "name": "Kaggle Dataset",
            "icon": "KG",
            "category": "Data Source",
            "description": "Dataset source for loading data from Kaggle.",
            "accent": "blue",
            "api": [
                {"key": "dataset_handle", "label": "Dataset Handle", "type": "text", "value": "owner/dataset-name"},
                {"key": "file_pattern", "label": "File Pattern", "type": "text", "value": "*.csv"},
                {"key": "format", "label": "Format", "type": "select", "value": "auto", "options": ["auto", "txt", "csv", "json", "jsonl", "parquet"]},
                {"key": "text_column", "label": "Text Column", "type": "text", "value": "text"},
                {"key": "max_rows", "label": "Max Rows (0 = All)", "type": "number", "value": 0}
            ],
        },
        {
            "type": "url_dataset",
            "builder_utility": True,
            "builder_python_api": True,
            "name": "URL Dataset",
            "icon": "URL",
            "category": "Data Source",
            "description": "Dataset source for loading data from a web URL.",
            "accent": "green",
            "api": [
                {"key": "url", "label": "Dataset URL", "type": "text", "value": "https://example.com/data.txt"},
                {"key": "format", "label": "Format", "type": "select", "value": "auto", "options": ["auto", "txt", "csv", "json", "jsonl", "parquet"]},
                {"key": "text_column", "label": "Text Column", "type": "text", "value": "text"},
                {"key": "max_rows", "label": "Max Rows (0 = All)", "type": "number", "value": 0}
            ],
        },
        {
            "type": "local_dataset",
            "builder_utility": True,
            "builder_python_api": True,
            "name": "Local Dataset",
            "icon": "FILE",
            "category": "Data Source",
            "description": "Dataset source for loading files from the current environment.",
            "accent": "green",
            "api": [
                {"key": "path", "label": "Path", "type": "text", "value": "."},
                {"key": "format", "label": "Format", "type": "select", "value": "auto", "options": ["auto", "txt", "csv", "json", "jsonl", "parquet"]},
                {"key": "text_column", "label": "Text Column", "type": "text", "value": "text"},
                {"key": "max_rows", "label": "Max Rows (0 = All)", "type": "number", "value": 0}
            ],
        },
        {
            "type": "text_process",
            "builder_utility": True,
            "builder_python_api": True,
            "name": "Text Processing",
            "icon": "TXT+",
            "category": "Text",
            "description": "Text processing step for cleaning and normalizing dataset text.",
            "accent": "orange",
            "api": [
                {"key": "text_column", "label": "Text Column", "type": "text", "value": "text"},
                {"key": "lowercase", "label": "Lowercase", "type": "select", "value": "false", "options": ["false", "true"]},
                {"key": "strip", "label": "Strip Spaces", "type": "select", "value": "true", "options": ["false", "true"]},
                {"key": "normalize_whitespace", "label": "Normalize Whitespace", "type": "select", "value": "true", "options": ["false", "true"]},
                {"key": "unicode_nfkc", "label": "Unicode NFKC", "type": "select", "value": "true", "options": ["false", "true"]},
                {"key": "remove_empty", "label": "Remove Empty", "type": "select", "value": "true", "options": ["false", "true"]},
                {"key": "min_chars", "label": "Min Characters", "type": "number", "value": 1},
                {"key": "max_chars", "label": "Max Characters (0 = All)", "type": "number", "value": 0}
            ],
        },
        {
            "type": "train_test_split",
            "builder_utility": True,
            "builder_python_api": True,
            "name": "Train / Validation / Test Split",
            "icon": "SPLT",
            "category": "Splitting",
            "description": "Dataset split step for training, validation, and test sets.",
            "accent": "purple",
            "api": [
                {"key": "train_size", "label": "Training", "type": "percent", "value": 90, "min": 0, "max": 100, "step": 1,
                 "help": "Percentage used to train the model."},
                {"key": "validation_size", "label": "Validation", "type": "percent", "value": 5, "min": 0, "max": 100, "step": 1,
                 "help": "Percentage used to check the model during training."},
                {"key": "test_size", "label": "Testing", "type": "percent", "value": 5, "min": 0, "max": 100, "step": 1,
                 "help": "Percentage kept for final evaluation."},
                {"key": "seed", "label": "Random Seed", "type": "number", "value": 42,
                 "help": "Use the same seed to reproduce the same split."},
                {"key": "shuffle", "label": "Shuffle Before Split", "type": "select", "value": "true", "options": ["true", "false"],
                 "help": "Mix examples before dividing them."}
            ],
        },
        {
            "type": "tokenize_text",
            "builder_utility": True,
            "builder_python_api": True,
            "name": "Tokenize Text",
            "icon": "TOK",
            "category": "Text",
            "description": "Tokenization step that converts text into model-ready token IDs.",
            "accent": "blue",
            "api": [
                {"key": "tokenizer_name", "label": "Tokenizer", "type": "text", "value": "gpt2"},
                {"key": "text_column", "label": "Text Column", "type": "text", "value": "text"},
                {"key": "context_length", "label": "Tokenizer Max Length", "type": "number", "value": 512},
                {"key": "truncation", "label": "Truncation", "type": "select", "value": "true", "options": ["false", "true"]},
                {"key": "padding", "label": "Padding", "type": "select", "value": "false", "options": ["false", "true", "max_length"]},
                {"key": "add_special_tokens", "label": "Add Special Tokens", "type": "select", "value": "true", "options": ["false", "true"]}
            ],
        },
        {
            "type": "manual_dataset",
            "builder_utility": True,
            "builder_python_api": True,
            "name": "Manual Text Data",
            "icon": "TXT",
            "category": "Data Source",
            "description": "Text dataset source for entering samples directly in Studio.",
            "accent": "green",
            "api": [
                {"key": "text", "label": "Text Data", "type": "textarea", "value": "Once upon a time"},
                {"key": "text_column", "label": "Column Name", "type": "text", "value": "text"},
                {"key": "one_line_per_sample", "label": "One Line = One Sample", "type": "select", "value": "true", "options": ["true", "false"]}
            ],
        },
        {
            "type": "image_process",
            "builder_utility": True,
            "builder_python_api": True,
            "name": "Image Processing",
            "icon": "IMG+",
            "category": "Image",
            "description": "Image processing step for resizing and preparing visual samples.",
            "accent": "orange",
            "api": [
                {"key": "image_column", "label": "Image Column", "type": "text", "value": "image"},
                {"key": "width", "label": "Width", "type": "number", "value": 224},
                {"key": "height", "label": "Height", "type": "number", "value": 224},
                {"key": "mode", "label": "Color Mode", "type": "select", "value": "RGB", "options": ["RGB", "L"]},
                {"key": "center_crop", "label": "Center Crop", "type": "select", "value": "false", "options": ["false", "true"]}
            ],
        },
        {
            "type": "audio_process",
            "builder_utility": True,
            "builder_python_api": True,
            "name": "Audio Processing",
            "icon": "AUD+",
            "category": "Audio",
            "description": "Audio processing step for resampling and preparing audio samples.",
            "accent": "orange",
            "api": [
                {"key": "audio_column", "label": "Audio Column", "type": "text", "value": "audio"},
                {"key": "sample_rate", "label": "Sample Rate", "type": "number", "value": 16000},
                {"key": "normalize", "label": "Normalize", "type": "select", "value": "true", "options": ["true", "false"]},
                {"key": "trim_silence", "label": "Trim Silence", "type": "select", "value": "false", "options": ["false", "true"]},
                {"key": "silence_threshold", "label": "Silence Threshold", "type": "number", "value": 0.01}
            ],
        },
        {
            "type": "batch_data",
            "builder_utility": True,
            "builder_python_api": True,
            "name": "Batch / DataLoader",
            "icon": "BTC",
            "category": "Dataset",
            "description": "Batching step for grouping prepared samples for training.",
            "accent": "blue",
            "api": [
                {"key": "batch_size", "label": "Batch Size", "type": "number", "value": 16},
                {"key": "shuffle", "label": "Shuffle", "type": "select", "value": "true", "options": ["true", "false"]},
                {"key": "num_workers", "label": "Workers", "type": "number", "value": 2},
                {"key": "drop_last", "label": "Drop Last", "type": "select", "value": "false", "options": ["false", "true"]}
            ],
        },
        {
            "type": "prepared_dataset",
            "builder_utility": True,
            "builder_python_api": True,
            "name": "Prepared Dataset",
            "icon": "DATA",
            "category": "Output",
            "description": "Dataset output that registers processed data for model training.",
            "accent": "green",
            "api": [
                {"key": "dataset_name", "label": "Dataset Name", "type": "text", "value": "Prepared Dataset",
                 "help": "Use different names to keep multiple prepared datasets."},
                {"key": "save_to_disk", "label": "Save To Disk", "type": "select", "value": "false", "options": ["false", "true"]},
                {"key": "path", "label": "Save Path", "type": "text", "value": "mlbricks_workspace/data/prepared_dataset"}
            ],
        },
        {
            "type": "embedding",
            "name": "Embedding",
            "icon": "EMB",
            "category": "Core Components",
            "description": "Token embedding layer that maps token IDs into dense vector representations.",
            "accent": "blue",
            "api": [
                {"key": "dim", "label": "Hidden Dim", "type": "number", "value": 384},
                {"key": "vocab_size", "label": "Vocab Size", "type": "number", "value": 32000},
                {"key": "dtype", "label": "DType", "type": "select", "value": "float16",
                 "options": ["float32", "float16", "bfloat16"]},
                {"key": "device", "label": "Device", "type": "select", "value": "auto",
                 "options": ["auto", "cpu", "cuda"]},
            ],
        },
        {
            "type": "esa",
            "name": "ESA",
            "icon": "ESA",
            "category": "Core Components",
            "description": "Entangled State Attention sequence-mixing layer.",
            "accent": "purple",
            "api": [
                {"key": "dim", "label": "Hidden Dim", "type": "number", "value": 384},
                {"key": "state_dim", "label": "State Dim", "type": "number", "value": 192},
                {"key": "heads", "label": "Heads", "type": "number", "value": 6},
                {"key": "chunk_size", "label": "Chunk Size", "type": "number", "value": 16},
                {"key": "kernel", "label": "Kernel", "type": "select", "value": "auto",
                 "options": ["auto", "native", "pytorch"]},
                {"key": "dtype", "label": "DType", "type": "select", "value": "float16",
                 "options": ["float32", "float16", "bfloat16"]},
                {"key": "device", "label": "Device", "type": "select", "value": "auto",
                 "options": ["auto", "cpu", "cuda"]},
            ],
        },
        {
            "type": "soup",
            "name": "SOUP",
            "icon": "SUP",
            "category": "Core Components",
            "description": "State-Oriented Unified Processing architecture with mixers, FFNs, state, memory, and fusion.",
            "accent": "purple",
            "api": [],
        },
        {
            "type": "stateaware_esa_stack",
            "name": "StateAware ESA Stack",
            "icon": "ESA",
            "category": "Core Components",
            "description": "State-Aware ESA stack that carries recurrent feature state across model depth.",
            "accent": "purple",
            "library_hidden": True,
            "api": [
                {"key": "dim", "label": "Model Dim", "type": "number", "value": 384},
                {"key": "state_dim", "label": "State Dim", "type": "number", "value": 2749},
                {"key": "layers", "label": "Physical Layers", "type": "number", "value": 8},
                {"key": "heads", "label": "ESA Heads", "type": "number", "value": 6},
                {"key": "block", "label": "Block Size", "type": "number", "value": 256},
                {"key": "batch", "label": "ESA Batch", "type": "number", "value": 16},
                {"key": "depth_dim", "label": "Depth Embedding Dim", "type": "number", "value": 64},
                {"key": "compass", "label": "Compass", "type": "number", "value": 16},
                {"key": "update_ratio_start", "label": "Update Ratio Start", "type": "number", "value": 0.20},
                {"key": "update_ratio_end", "label": "Update Ratio End", "type": "number", "value": 0.14},
                {"key": "stream_ratio", "label": "Stream Ratio", "type": "number", "value": 1.08},
            ],
        },
        {
            "type": "vesa",
            "name": "VESA",
            "icon": "VES",
            "category": "Core Components",
            "description": "Vision Entangled State Attention for image and vision processing.",
            "accent": "lime",
            "api": [
                {"key": "dim", "label": "Hidden Dim", "type": "number", "value": 384},
                {"key": "heads", "label": "Heads", "type": "number", "value": 6},
                {"key": "kernel", "label": "Kernel", "type": "select", "value": "auto",
                 "options": ["auto", "native", "pytorch"]},
            ],
        },
        {
            "type": "rmsnorm",
            "name": "RMSNorm",
            "icon": "RMS",
            "category": "Core Components",
            "description": "Root Mean Square Normalization layer for stabilizing activations.",
            "accent": "orange",
            "api": [
                {"key": "dim", "label": "Hidden Dim", "type": "number", "value": 384},
                {"key": "eps", "label": "Epsilon", "type": "number", "value": 0.00001},
            ],
        },
        {
            "type": "ffn",
            "name": "FFN",
            "icon": "FFN",
            "category": "Core Components",
            "description": "Feed-Forward Network for transforming features within each layer.",
            "accent": "pink",
            "api": [
                {"key": "dim", "label": "Hidden Dim", "type": "number", "value": 384},
                {"key": "ffn_dim", "label": "FFN Hidden Dim", "type": "number", "value": 1536},
                {"key": "activation", "label": "Activation", "type": "select", "value": "silu",
                 "options": ["silu", "gelu", "relu"]},
                {"key": "dropout", "label": "Dropout", "type": "number", "value": 0.1},
                {"key": "bias", "label": "Use Bias", "type": "select", "value": "true",
                 "options": ["true", "false"]},
            ],
        },
        {
            "type": "saffn",
            "name": "SAFFN",
            "icon": "SAF",
            "category": "Core Components",
            "description": "State-Aware Feed-Forward Network conditioned across physical depth.",
            "accent": "pink",
            # SAFFN's runtime contract is not a simple y=module(x) call.  Named
            # ports mirror the original MLBricks API exactly and are rendered
            # by Studio instead of the generic Main/Skip/Extra lane buttons.
            "runtime_ports": {
                "inputs": [
                    {"id": "x", "name": "x"},
                    {"id": "esa_update", "name": "ESA Update"},
                    {"id": "previous_esa", "name": "Previous ESA"},
                    {"id": "previous_state", "name": "Previous State"},
                ],
                "outputs": [
                    {"id": "main", "name": "Main"},
                    {"id": "state", "name": "State"},
                ],
            },
            "api": [
                {"key": "dim", "label": "Hidden Dim", "type": "number", "value": 384},
                {"key": "ffn_dim", "label": "FFN Hidden Dim", "type": "number", "value": 1536},
                {"key": "activation", "label": "Activation", "type": "select", "value": "silu",
                 "options": ["silu", "gelu", "relu"]},
            ],
        },
        {
            "type": "residual",
            "name": "Residual Add",
            "icon": "ADD",
            "category": "Core Components",
            "description": "Residual connection block that adds the skip path to the main path.",
            "accent": "cyan",
            "inputs": ["main", "skip"],
            "api": [
                {"key": "enabled", "label": "Use Residual", "type": "select", "value": "true",
                 "options": ["true", "false"]},
                {"key": "scale", "label": "Scaling", "type": "number", "value": 1.0},
                {"key": "pre_norm", "label": "Pre-Norm", "type": "select", "value": "RMSNorm",
                 "options": ["None", "RMSNorm", "LayerNorm"]},
            ],
        },
        {
            "type": "dropout",
            "builder_utility": True,
            "name": "Dropout",
            "icon": "DRP",
            "category": "Core Components",
            "description": "Regularization layer that randomly drops activations during training.",
            "accent": "purple",
            "api": [
                {"key": "p", "label": "Probability", "type": "number", "value": 0.1},
            ],
        },
        {
            "type": "bolt",
            "name": "BOLT",
            "icon": "BLT",
            "category": "Core Components",
            "description": "BOLT sequence-mixing layer for routed latent feature processing.",
            "accent": "blue",
            "api": [
                {"key": "dim", "label": "Hidden Dim", "type": "number", "value": 384},
                {"key": "kernel", "label": "Kernel", "type": "select", "value": "auto",
                 "options": ["auto", "native", "pytorch"]},
            ],
        },
        {
            "type": "visualbolt",
            "name": "VisualBOLT",
            "icon": "VBL",
            "category": "Core Components",
            "description": "Vision BOLT layer for image and visual feature processing.",
            "accent": "cyan",
            "api": [
                {"key": "dim", "label": "Hidden Dim", "type": "number", "value": 384},
                {"key": "kernel", "label": "Kernel", "type": "select", "value": "auto",
                 "options": ["auto", "native", "pytorch"]},
            ],
        },
        {"type":"linear","name":"Linear","icon":"LIN","category":"Core Blocks","description":"Linear projection layer for mapping features between dimensions.","accent":"blue","api":[]},
        {"type":"layernorm","name":"LayerNorm","icon":"LN","category":"Core Blocks","description":"Layer Normalization for stabilizing activations across features.","accent":"orange","api":[]},
        {"type":"rescontroller","name":"ResController","icon":"RSC","category":"Core Blocks","description":"Residual Controller for regulating residual update strength.","accent":"cyan","api":[]},
        {"type":"micro_ffn","name":"MicroVirtualFFN","icon":"MVF","category":"Core Blocks","description":"Micro Feed-Forward Network for lightweight virtual refinement.","accent":"pink","api":[]},
        {"type":"virtual_saffn","name":"VirtualStateAwareFFN","icon":"VSF","category":"Core Blocks","description":"Virtual State-Aware Feed-Forward Network for recurrent refinement.","accent":"pink","api":[]},

        {"type":"elasticbit_runtime","name":"ElasticBit","icon":"EB","category":"Advanced","description":"Adaptive precision runtime for selecting efficient 4–32-bit storage.","accent":"blue","api":[]},
        {"type":"rope","name":"RoPE","icon":"RP","category":"Position","description":"Rotary Positional Embedding for encoding token position through rotation.","accent":"purple","api":[]},
        {"type":"learned_position","name":"Learned Position","icon":"LP","category":"Position","description":"Learned positional embedding for trainable sequence position information.","accent":"purple","api":[]},
        {"type":"sinusoidal_position","name":"Sinusoidal Position","icon":"SP","category":"Position","description":"Sinusoidal positional encoding for deterministic sequence positions.","accent":"purple","api":[]},
        {
            "type": "lm_head",
            "name": "LM Head",
            "icon": "LM",
            "category": "Heads",
            "description": "Language Modeling Head that projects hidden features into vocabulary logits.",
            "accent": "purple",
            "api": [
                {"key": "dim", "label": "Hidden Dim", "type": "number", "value": 384},
                {"key": "vocab_size", "label": "Vocab Size", "type": "number", "value": 32000},
                {"key": "bias", "label": "Use Bias", "type": "select", "value": "false",
                 "options": ["true", "false"]},
            ],
        },
        {
            "type": "classifier",
            "builder_utility": True,
            "name": "Classifier Head",
            "icon": "CLS",
            "category": "Heads",
            "description": "Classification Head that maps model features to class scores.",
            "accent": "orange",
            "api": [
                {"key": "dim", "label": "Hidden Dim", "type": "number", "value": 384},
                {"key": "classes", "label": "Classes", "type": "number", "value": 10},
            ],
        },
        {
            "type": "text_output",
            "builder_utility": True,
            "name": "Text Output",
            "icon": "OUT",
            "category": "Outputs",
            "description": "Text output block for decoding generated tokens into readable text.",
            "accent": "green",
            "api": [
                {"key": "max_new_tokens", "label": "Max New Tokens", "type": "number", "value": 64},
                {"key": "temperature", "label": "Temperature", "type": "number", "value": 0.8},
                {"key": "top_p", "label": "Top P", "type": "number", "value": 0.95},
            ],
        },
        {
            "type": "logits_output",
            "builder_utility": True,
            "name": "Logits Output",
            "icon": "LOG",
            "category": "Outputs",
            "description": "Output block for exposing model logits and prediction scores.",
            "accent": "blue",
            "api": [],
        },
    ]


def _node(type_, name, params=None, *, definition_id=None, x=0, y=0):
    return {
        "id": _id("node"),
        "type": type_,
        "name": name,
        "definition_id": definition_id,
        "repeat": 1,
        "params": params or {},
        "position": {"x": x, "y": y},
    }


def _edge(source, target, source_port="out", target_port="in", kind="main"):
    return {
        "id": _id("edge"),
        "source": source,
        "target": target,
        "source_port": source_port,
        "target_port": target_port,
        "kind": kind,
    }


def _default_data_processing_graph():
    """Beginner-ready, executable text pipeline shown in every new project."""
    source = _node("hf_dataset", "Hugging Face Dataset", {
        "dataset_id": "roneneldan/TinyStories",
        "config": "",
        "split": "train",
        "text_column": "text",
        "streaming": "false",
        "max_rows": 10000,
    })
    clean = _node("text_process", "Text Processing", {
        "text_column": "text",
        "lowercase": "false",
        "strip": "true",
        "normalize_whitespace": "true",
        "unicode_nfkc": "true",
        "remove_empty": "true",
        "min_chars": 1,
        "max_chars": 0,
    })
    split = _node("train_test_split", "Train / Validation / Test Split", {
        "train_size": 90,
        "validation_size": 5,
        "test_size": 5,
        "seed": 42,
        "shuffle": "true",
    })
    tokenize = _node("tokenize_text", "Tokenize Text", {
        "tokenizer_name": "gpt2",
        "text_column": "text",
        "context_length": 512,
        "truncation": "true",
        "padding": "false",
        "add_special_tokens": "true",
    })
    output = _node("prepared_dataset", "Prepared Dataset", {
        "dataset_name": "TinyStories Prepared",
        "save_to_disk": "false",
        "path": "mlbricks_workspace/data/prepared_dataset",
    })
    nodes = [source, clean, split, tokenize, output]
    edges = [
        _edge(left["id"], right["id"], "main_out", "main_in", "main")
        for left, right in zip(nodes[:-1], nodes[1:])
    ]
    return nodes, edges


def new_project(name: str = "Untitled Model"):
    root_id = _id("component")
    data_root_id = _id("component")
    data_nodes, data_edges = _default_data_processing_graph()
    now = datetime.now(timezone.utc).isoformat()
    return {
        "format": "mlb-studio",
        "format_version": "1.0.0",
        "project": {
            "name": name,
            "created_at": now,
            "updated_at": now,
            "context_length": 512,
            "batch_size": 16,
            "model_settings": {
                "embedding_size": 384,
                "heads": 6,
                "block": 512,
                "default_batch": 16,
                "vocab_size": 32000,
                "precision": "fp16",
            },
            "dataset": None,
            "estimated_parameters": None,
        },
        "root_component_id": root_id,
        "components": {
            root_id: {
                "id": root_id,
                "name": name,
                "kind": "model",
                "revision": 1,
                "nodes": [],
                "edges": [],
            },
            data_root_id: {
                "id": data_root_id,
                "name": "Data Processing",
                "kind": "data",
                "revision": 1,
                "nodes": data_nodes,
                "edges": data_edges,
            },
        },
        "workspaces": {
            "model": {
                "name": "Model Builder",
                "root_component_id": root_id,
                "view_component_id": root_id,
                "breadcrumbs": [{"id": root_id, "name": name}],
            },
            "data": {
                "name": "Data Processing",
                "root_component_id": data_root_id,
                "view_component_id": data_root_id,
                "breadcrumbs": [{"id": data_root_id, "name": "Data Processing"}],
            },
        },
        "active_workspace": "model",
        "prepared_datasets": [],
        "model_outputs": [],
        "project_files": [],
        "custom_components": {},
        "view_component_id": root_id,
        "breadcrumbs": [{"id": root_id, "name": name}],
        "auto_connect": True,
    }


def tinystories_30m_project():
    """Notebook-matched TinyStories ~30M ESA starter.

    This preset intentionally mirrors the validated eager-vs-whole-model-compile
    benchmark: 10 layers, width 330, six ESA heads, context 512, standard 4x
    FFN, learned positions, two pre-norm residuals, final LayerNorm, and tied
    token-embedding/LM-head weights.
    """
    project = new_project("TinyStories 30M ESA")
    project["project"].update({
        "context_length": 512,
        "batch_size": 16,
        "dataset": "TinyStories",
        "estimated_parameters": "~29.85M",
        "description": "10-layer ESA causal LM matched to the validated whole-model compile benchmark",
        "model_settings": {
            "embedding_size": 330,
            "heads": 6,
            "block": 512,
            "default_batch": 16,
            "vocab_size": 50257,
            "precision": "fp16",
        },
    })

    # Match the benchmark tokenizer/data semantics. The runtime packer joins
    # tokenized stories with EOS and emits exact [batch, 512] training tensors.
    data_ws = (project.get("workspaces") or {}).get("data") or {}
    data_root = data_ws.get("root_component_id")
    for node in (project.get("components") or {}).get(data_root, {}).get("nodes", []):
        if node.get("type") == "tokenize_text":
            node.setdefault("params", {}).update({
                "tokenizer_name": "EleutherAI/gpt-neo-125M",
                "context_length": 512,
                "truncation": "false",
                "padding": "false",
                "add_special_tokens": "false",
            })

    root_id = project["root_component_id"]

    # Reusable block exactly matching ESAModel's standard block:
    # x -> LN -> ESA -> +x -> LN -> 4x GELU FFN -> +residual.
    layer_def_id = _id("custom")
    block_input = _node("dropout", "Block Input", {"p": 0.0})
    ln1 = _node("layernorm", "LayerNorm 1", {
        "normalized_shape": 330, "eps": 1e-5,
        "elementwise_affine": True, "bias": True,
        "device": None, "dtype": None,
    })
    esa = _node("esa", "ESA", {
        "embd": 330, "head": 6, "batch": 16, "block": 512,
        "backend": "pytorch", "precision": "fp16", "compass": 16,
        "dropout": 0.0, "gate_min": 0.8, "gate_max": 0.995,
        "eps": 1e-5, "device": "auto", "auto_compile": False,
        "compile_mode": "default", "auto_move_input": True,
        "strict_checks": False,
    })
    res1 = _node("residual", "ESA Residual", {"dropout": 0.0})
    ln2 = _node("layernorm", "LayerNorm 2", {
        "normalized_shape": 330, "eps": 1e-5,
        "elementwise_affine": True, "bias": True,
        "device": None, "dtype": None,
    })
    ffn = _node("ffn", "FFN", {
        "hidden_size": 330, "intermediate_size": 1320,
        "activation": "gelu", "dropout": 0.0, "bias": True,
        "gated": False, "device": None, "dtype": None,
    })
    res2 = _node("residual", "FFN Residual", {"dropout": 0.0})

    project["custom_components"][layer_def_id] = {
        "id": layer_def_id,
        "name": "TinyStories ESA Layer",
        "description": "Pre-LN ESA + residual → Pre-LN FFN + residual",
        "revision": 2,
        "nodes": [block_input, ln1, esa, res1, ln2, ffn, res2],
        "edges": [
            _edge(block_input["id"], ln1["id"]),
            _edge(ln1["id"], esa["id"]),
            _edge(esa["id"], res1["id"]),
            _edge(block_input["id"], res1["id"], kind="residual"),
            _edge(res1["id"], ln2["id"]),
            _edge(ln2["id"], ffn["id"]),
            _edge(ffn["id"], res2["id"]),
            _edge(res1["id"], res2["id"], kind="residual"),
        ],
        "exposed_api": [
            {"source_node": esa["id"], "key": "embd", "label": "Embedding Dim"},
            {"source_node": esa["id"], "key": "head", "label": "ESA Heads"},
            {"source_node": esa["id"], "key": "compass", "label": "Compass"},
            {"source_node": ffn["id"], "key": "intermediate_size", "label": "FFN Hidden Dim"},
        ],
    }

    text = _node("text_input", "Text Input", {"prompt": "Once upon a time"})
    emb = _node("embedding", "Token Embedding", {
        "vocab_size": 50257, "embedding_dim": 330,
    })
    pos = _node("learned_position", "Learned Position", {
        "dim": 330, "max_seq_len": 512,
    })
    drop = _node("dropout", "Embedding Dropout", {"p": 0.0})
    nodes = [text, emb, pos, drop]

    for i in range(1, 11):
        nodes.append(_node(
            "custom",
            f"Layer {i}",
            {"embd": 330, "head": 6, "compass": 16, "intermediate_size": 1320},
            definition_id=layer_def_id,
        ))

    final_norm = _node("layernorm", "Final LayerNorm", {
        "normalized_shape": 330, "eps": 1e-5,
        "elementwise_affine": True, "bias": True,
        "device": None, "dtype": None,
    })
    head = _node("lm_head", "LM Head", {
        "hidden_size": 330, "vocab_size": 50257, "bias": False,
        "tie_embeddings": True, "device": None, "dtype": None,
    })
    out = _node("text_output", "Text Output", {
        "max_new_tokens": 64, "temperature": 0.8, "top_p": 0.95,
    })
    nodes.extend([final_norm, head, out])

    edges = [
        _edge(left["id"], right["id"])
        for left, right in zip(nodes[:-1], nodes[1:])
    ]
    project["components"][root_id]["nodes"] = nodes
    project["components"][root_id]["edges"] = edges
    return project




def stateaware_esa_200m_project():
    """Notebook-matched StateAware ESA 200M starter (199,982,344 params)."""
    project = new_project("StateAware ESA 200M")
    project["project"].update({
        "context_length": 256, "batch_size": 16, "dataset": None,
        "estimated_parameters": "199,982,344",
        "description": "Notebook-matched 8-layer StateAware ESA causal LM",
        "model_settings": {"embedding_size": 384, "heads": 6, "block": 256,
                           "default_batch": 16, "vocab_size": 50257, "precision": "fp16"},
    })
    root_id = project["root_component_id"]
    nodes = [
        _node("text_input", "Text Input", {"prompt": "Once upon a time"}),
        _node("embedding", "Token Embedding", {"vocab_size": 50257, "embedding_dim": 384}),
        _node("stateaware_esa_stack", "StateAware ESA ×8", {
            "dim": 384, "state_dim": 2749, "layers": 8, "heads": 6,
            "block": 256, "batch": 16, "depth_dim": 64, "compass": 16,
            "update_ratio_start": 0.20, "update_ratio_end": 0.14, "stream_ratio": 1.08,
        }),
        _node("rmsnorm", "Final RMSNorm", {"normalized_shape": 384, "eps": 1e-6, "elementwise_affine": True}),
        _node("lm_head", "LM Head", {"hidden_size": 384, "vocab_size": 50257, "bias": False, "tie_embeddings": True}),
        _node("text_output", "Text Output", {"max_new_tokens": 64, "temperature": 0.8, "top_p": 0.95}),
    ]
    project["components"][root_id]["nodes"] = nodes
    project["components"][root_id]["edges"] = [_edge(a["id"], b["id"]) for a,b in zip(nodes[:-1],nodes[1:])]
    return project


def soup_200m_project():
    """Exact supplied-notebook SOUP 200M starter (199,916,160 params)."""
    project = new_project("SOUP 200M")
    project["project"].update({
        "context_length": 256, "batch_size": 16, "dataset": None,
        "estimated_parameters": "199,916,160",
        "description": "Notebook-matched SOUP 200M causal LM with three physical layers",
        "model_settings": {"embedding_size": 1152, "heads": 18, "block": 256,
                           "default_batch": 16, "vocab_size": 50257, "precision": "fp16"},
    })
    root_id = project["root_component_id"]
    nodes = [
        _node("text_input", "Text Input", {"prompt": "Once upon a time"}),
        _node("embedding", "Token Embedding", {"vocab_size": 50257, "embedding_dim": 1152}),
        _node("soup", "SOUP ×3", {
            "dim": 1152, "width": 2864, "depth": 3, "mixer": "esa", "ffn": "saffn",
            "mixer_config": {"head": 18, "batch": 16, "block": 256, "compass": 16, "auto_compile": False},
            "ffn_config": {"depth_dim": 128}, "memory_dim": 256, "fusion_hidden": 1728,
        }),
        _node("rmsnorm", "Final RMSNorm", {"normalized_shape": 1152, "eps": 1e-6, "elementwise_affine": True}),
        _node("lm_head", "LM Head", {"hidden_size": 1152, "vocab_size": 50257, "bias": False, "tie_embeddings": True}),
        _node("text_output", "Text Output", {"max_new_tokens": 64, "temperature": 0.8, "top_p": 0.95}),
    ]
    project["components"][root_id]["nodes"] = nodes
    project["components"][root_id]["edges"] = [_edge(a["id"], b["id"]) for a,b in zip(nodes[:-1],nodes[1:])]
    return project


def soup_30m_1l_project():
    """One-layer SOUP ~30M starter (30,003,528 params)."""
    project = new_project("SOUP 30M 1L")
    project["project"].update({
        "context_length": 512, "batch_size": 16, "dataset": "TinyStories",
        "estimated_parameters": "30,003,528",
        "description": "One-layer SOUP causal LM at ~30M parameters",
        "model_settings": {"embedding_size": 384, "heads": 6, "block": 512,
                           "default_batch": 16, "vocab_size": 50257, "precision": "fp16"},
    })
    root_id = project["root_component_id"]
    nodes = [
        _node("text_input", "Text Input", {"prompt": "Once upon a time"}),
        _node("embedding", "Token Embedding", {"vocab_size": 50257, "embedding_dim": 384}),
        _node("soup", "SOUP ×1", {
            "dim": 384, "width": 1408, "depth": 1, "mixer": "esa", "ffn": "saffn",
            "mixer_config": {"head": 6, "batch": 16, "block": 512, "compass": 16, "auto_compile": False},
            "ffn_config": {"depth_dim": 64}, "memory_dim": 128, "fusion_hidden": 928,
        }),
        _node("rmsnorm", "Final RMSNorm", {"normalized_shape": 384, "eps": 1e-6, "elementwise_affine": True}),
        _node("lm_head", "LM Head", {"hidden_size": 384, "vocab_size": 50257, "bias": False, "tie_embeddings": True}),
        _node("text_output", "Text Output", {"max_new_tokens": 64, "temperature": 0.8, "top_p": 0.95}),
    ]
    project["components"][root_id]["nodes"] = nodes
    project["components"][root_id]["edges"] = [_edge(a["id"], b["id"]) for a,b in zip(nodes[:-1],nodes[1:])]
    return project
