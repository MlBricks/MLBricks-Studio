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
            "description": "Image, image-sequence, or live-frame input for vision and multimodal models.",
            "accent": "green",
            "api": [
                {"key": "input_mode", "label": "Input Mode", "type": "select", "value": "single",
                 "options": ["single", "sequence", "live"]},
                {"key": "source_type", "label": "Source Type", "type": "select", "value": "path_or_url",
                 "options": ["path_or_url", "directory", "camera", "cctv"]},
                {"key": "channels", "label": "Channels", "type": "number", "value": 3},
                {"key": "image_size", "label": "Image Size", "type": "number", "value": 224},
                {"key": "fps", "label": "Live FPS", "type": "number", "value": 5},
            ],
        },
        {
            "type": "audio_input",
            "builder_utility": True,
            "name": "Audio Input",
            "icon": "AUD",
            "category": "Inputs",
            "description": "Audio file, sample-array, or continuous audio input.",
            "accent": "green",
            "api": [
                {"key": "input_mode", "label": "Input Mode", "type": "select", "value": "file",
                 "options": ["file", "live", "continuous"]},
                {"key": "source_type", "label": "Source Type", "type": "select", "value": "path_or_url",
                 "options": ["path_or_url", "microphone", "sensor"]},
                {"key": "sample_rate", "label": "Sample Rate", "type": "number", "value": 16000},
                {"key": "buffer_size", "label": "Buffer Size", "type": "number", "value": 1024},
            ],
        },
        {
            "type": "video_input",
            "builder_utility": True,
            "name": "Video Input",
            "icon": "VID",
            "category": "Inputs",
            "description": "Video file, camera, or CCTV stream input with frame-rate control.",
            "accent": "green",
            "api": [
                {"key": "input_mode", "label": "Input Mode", "type": "select", "value": "file",
                 "options": ["file", "live", "cctv"]},
                {"key": "source_type", "label": "Source Type", "type": "select", "value": "path_or_url",
                 "options": ["path_or_url", "camera", "cctv"]},
                {"key": "image_size", "label": "Frame Size", "type": "number", "value": 224},
                {"key": "fps", "label": "Process FPS", "type": "number", "value": 5},
            ],
        },
        {
            "type": "signal_input",
            "builder_utility": True,
            "name": "Signal Input",
            "icon": "SIG",
            "category": "Inputs",
            "description": "Static or continuous numeric signal input for sensors, telemetry, antenna, serial, or TCP sources.",
            "accent": "green",
            "api": [
                {"key": "input_mode", "label": "Input Mode", "type": "select", "value": "static",
                 "options": ["static", "continuous"]},
                {"key": "source_type", "label": "Source Type", "type": "select", "value": "inline",
                 "options": ["inline", "file", "file_tail", "serial", "tcp", "sensor", "antenna"]},
                {"key": "sample_rate", "label": "Sample Rate", "type": "number", "value": 16000},
                {"key": "buffer_size", "label": "Buffer Size", "type": "number", "value": 256},
                {"key": "channel", "label": "Channel / Port", "type": "text", "value": ""},
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
                {"key": "credential_profile", "label": "Credential Profile", "type": "text", "value": "Default", "help": "Optional saved Hugging Face credential from Cloud & Repositories. Public datasets work without a token."},
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
                # Studio keeps a universal six-socket card: one input/output
                # socket on the top, back/front, and bottom surfaces.  Logical
                # API arguments may share a physical socket (a hover/click hub)
                # and one logical output may be exposed on multiple sockets.
                # The ids below remain the exact MLBricks forward() keywords;
                # only the visual names are architecture-neutral.
                "inputs": [
                    # Universal top/back sockets carry the ordinary current
                    # layer inputs.  Previous-depth signals get their own
                    # dedicated surface sockets between the universal top/
                    # bottom input/output pair, matching the Studio card
                    # layout rather than hiding them in a hub.
                    {"id": "x", "name": "Input Signal", "socket": "top"},
                    {"id": "esa_update", "name": "Current Signal", "socket": "back"},
                    {"id": "previous_esa", "name": "Previous Signal", "socket": "top_aux"},
                    {"id": "previous_state", "name": "Previous State", "socket": "bottom_aux"},
                ],
                "outputs": [
                    {"id": "main", "name": "Main Output", "socket": "front"},
                    # The same state tensor is exposed from dedicated top and
                    # bottom surface sockets so downstream nodes can take the
                    # cleanest route without duplicating computation/state.
                    {"id": "state", "name": "State Out", "sockets": ["top_aux", "bottom_aux"]},
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
            "type": "abstract_layer",
            "builder_utility": True,
            "builder_python_api": False,
            "name": "Abstract Layer",
            "icon": "ABS",
            "category": "Core Blocks",
            "description": "Editable layer shell with 3 fixed inputs/outputs and up to 10 custom inputs/outputs.",
            "accent": "purple",
            "api": [],
        },
        {
            "type": "abstract_input",
            "builder_utility": True,
            "builder_python_api": False,
            "library_hidden": True,
            "name": "Layer Inputs",
            "icon": "IN",
            "category": "Core Blocks",
            "description": "Abstract Layer boundary: exposes fixed and custom external inputs to the internal graph.",
            "accent": "green",
            "api": [],
        },
        {
            "type": "abstract_output",
            "builder_utility": True,
            "builder_python_api": False,
            "library_hidden": True,
            "name": "Layer Outputs",
            "icon": "OUT",
            "category": "Core Blocks",
            "description": "Abstract Layer boundary: maps the internal graph back to fixed and custom external outputs.",
            "accent": "cyan",
            "api": [],
        },
        {
            "type": "layer_block",
            "builder_utility": True,
            "builder_python_api": False,
            "library_hidden": True,
            "name": "Layer Block",
            "icon": "LYR",
            "category": "Core Blocks",
            "description": "Legacy fixed Pre-LN ESA layer block. New designs should use Abstract Layer.",
            "accent": "purple",
            "runtime_ports": {
                "inputs": [
                    {"id": "signal", "name": "Signal In", "socket": "back"},
                    {"id": "residual", "name": "Residual In", "socket": "bottom"},
                ],
                "outputs": [
                    {"id": "signal", "name": "Signal Out", "socket": "front"},
                    {"id": "residual", "name": "Residual Out", "socket": "bottom"},
                ],
            },
            "api": [
                {"key": "dim", "label": "Hidden Dim", "type": "number", "value": 480},
                {"key": "heads", "label": "ESA Heads", "type": "number", "value": 6},
                {"key": "ffn_dim", "label": "FFN Hidden Dim", "type": "number", "value": 1920},
                {"key": "block", "label": "Context / Block", "type": "number", "value": 512},
                {"key": "batch", "label": "Batch", "type": "number", "value": 16},
                {"key": "compass", "label": "ESA Compass", "type": "number", "value": 16},
                {"key": "activation", "label": "Activation", "type": "select", "value": "gelu",
                 "options": ["gelu", "silu", "relu"]},
                {"key": "dropout", "label": "Dropout", "type": "number", "value": 0.0},
                {"key": "norm_eps", "label": "Norm Epsilon", "type": "number", "value": 0.00001},
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
        {
            "type": "value_buffer",
            "builder_utility": True,
            "name": "Previous Value Buffer",
            "icon": "BUF",
            "category": "Core Blocks",
            "description": "Carries a previous physical-depth value, or creates a zero-initialized previous vector for the first depth.",
            "accent": "cyan",
            "api": [
                {"key": "mode", "label": "Mode", "type": "select", "value": "hold",
                 "options": ["hold", "zero_init"]},
                {"key": "width", "label": "Output Width (0 = Auto)", "type": "number", "value": 0},
            ],
        },
        {"type":"linear","name":"Linear","icon":"LIN","category":"Core Blocks","description":"Linear projection layer for mapping features between dimensions.","accent":"blue","api":[]},
        {"type":"layernorm","name":"LayerNorm","icon":"LN","category":"Core Blocks","description":"Layer Normalization for stabilizing activations across features.","accent":"orange","api":[]},
        {"type":"rescontroller","name":"ResController","icon":"RSC","category":"Core Blocks","description":"Residual Controller for regulating residual update strength.","accent":"cyan","api":[]},
        {"type":"micro_ffn","name":"MicroVirtualFFN","icon":"MVF","category":"Core Blocks","description":"Micro Feed-Forward Network for lightweight virtual refinement.","accent":"pink","api":[]},
        {
            "type":"virtual_saffn",
            "name":"VirtualStateAwareFFN",
            "icon":"VSF",
            "category":"Core Blocks",
            "description":"Virtual State-Aware Feed-Forward Network for recurrent refinement.",
            "accent":"pink",
            # VirtualStateAwareFFN has the same original forward signature and
            # state outputs as SAFFN, so it uses the same physical signal map.
            "runtime_ports": {
                "inputs": [
                    {"id": "x", "name": "Input Signal", "socket": "top"},
                    {"id": "esa_update", "name": "Current Signal", "socket": "back"},
                    {"id": "previous_esa", "name": "Previous Signal", "socket": "top_aux"},
                    {"id": "previous_state", "name": "Previous State", "socket": "bottom_aux"},
                ],
                "outputs": [
                    {"id": "main", "name": "Main Output", "socket": "front"},
                    {"id": "state", "name": "State Out", "sockets": ["top_aux", "bottom_aux"]},
                ],
            },
            "api":[],
        },

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
        "format_version": "1.0.0b2",
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


def _abstract_layer_definition(*, name, dim, heads, ffn_dim, block, batch=16):
    """Editable Pre-LN ESA layer expressed as an Abstract Layer internal graph."""
    definition_id = _id("abs")
    layer_in = _node("abstract_input", "Layer Inputs")
    norm1 = _node("layernorm", "Pre-ESA LayerNorm", {
        "normalized_shape": dim, "eps": 1e-5,
        "elementwise_affine": True, "bias": True,
        "device": None, "dtype": None,
    })
    esa = _node("esa", "ESA", {
        "embd": dim, "head": heads, "batch": batch, "block": block,
        "backend": "auto", "precision": "auto", "compass": 16,
        "dropout": 0.0, "gate_min": 0.8, "gate_max": 0.995,
        "eps": 1e-5, "strict_checks": False,
    })
    residual1 = _node("residual", "ESA Residual", {"dropout": 0.0})
    norm2 = _node("layernorm", "Pre-FFN LayerNorm", {
        "normalized_shape": dim, "eps": 1e-5,
        "elementwise_affine": True, "bias": True,
        "device": None, "dtype": None,
    })
    ffn = _node("ffn", "FFN", {
        "hidden_size": dim, "intermediate_size": ffn_dim,
        "activation": "gelu", "dropout": 0.0, "bias": True, "gated": False,
    })
    residual2 = _node("residual", "FFN Residual", {"dropout": 0.0})
    layer_out = _node("abstract_output", "Layer Outputs")
    nodes = [layer_in, norm1, esa, residual1, norm2, ffn, residual2, layer_out]
    edges = [
        _edge(layer_in["id"], norm1["id"], source_port="main_out", target_port="main_in", kind="main"),
        _edge(norm1["id"], esa["id"]),
        _edge(esa["id"], residual1["id"], source_port="main_out", target_port="main_in", kind="main"),
        _edge(layer_in["id"], residual1["id"], source_port="skip_out", target_port="skip_in", kind="residual"),
        _edge(residual1["id"], norm2["id"]),
        _edge(norm2["id"], ffn["id"]),
        _edge(ffn["id"], residual2["id"], source_port="main_out", target_port="main_in", kind="main"),
        _edge(residual1["id"], residual2["id"], source_port="main_out", target_port="skip_in", kind="residual"),
        _edge(residual2["id"], layer_out["id"], source_port="main_out", target_port="main_in", kind="main"),
        _edge(residual2["id"], layer_out["id"], source_port="main_out", target_port="skip_in", kind="residual"),
    ]
    return {
        "id": definition_id,
        "local_id": _id("component"),
        "name": name,
        "description": "Editable Pre-LN ESA Abstract Layer",
        "revision": 1,
        "implementation": "abstract_layer",
        "nodes": nodes,
        "edges": edges,
        "input_count": 3,
        "output_count": 3,
        "interface": {"input_ports": [], "output_ports": []},
        "palette_hidden": True,
        "palette_installed": False,
        "gallery_entry_id": None,
    }


def _standard_esa_layer_block_project(*, name, dim, heads, layers, ffn_dim, block, batch=16, dataset=None):
    """Build a standard ESA SLM from editable Abstract Layer instances."""
    project = new_project(name)
    project["project"].update({
        "context_length": block,
        "batch_size": batch,
        "dataset": dataset,
        "estimated_parameters": "~50M" if dim == 480 else "~200M",
        "description": f"{layers}-layer ESA SLM with editable Abstract Layer blocks",
        "model_settings": {
            "embedding_size": dim,
            "heads": heads,
            "block": block,
            "default_batch": batch,
            "vocab_size": 50257,
            "precision": "fp16",
        },
    })

    if dataset == "TinyStories":
        data_ws = (project.get("workspaces") or {}).get("data") or {}
        data_root = data_ws.get("root_component_id")
        for node in (project.get("components") or {}).get(data_root, {}).get("nodes", []):
            if node.get("type") == "tokenize_text":
                node.setdefault("params", {}).update({
                    "tokenizer_name": "EleutherAI/gpt-neo-125M",
                    "context_length": block,
                    "truncation": "false",
                    "padding": "false",
                    "add_special_tokens": "false",
                })

    root_id = project["root_component_id"]
    text_input = _node("text_input", "Text Input", {"prompt": "Once upon a time"})
    emb = _node("embedding", "Token Embedding", {
        "vocab_size": 50257, "embedding_dim": dim,
    })
    pos = _node("learned_position", "Learned Position", {
        "dim": dim, "max_seq_len": block,
    })
    drop = _node("dropout", "Embedding Dropout", {"p": 0.0})

    # One reusable editable definition is instantiated independently at every
    # physical layer. Editing the Abstract Layer updates the architecture of
    # every instance while each runtime instance owns its own parameters/state.
    abstract_def = _abstract_layer_definition(
        name=f"{name} · ESA Layer", dim=dim, heads=heads,
        ffn_dim=ffn_dim, block=block, batch=batch,
    )
    project["custom_components"][abstract_def["id"]] = abstract_def
    layer_nodes = [
        _node("custom", f"Layer {i}", {}, definition_id=abstract_def["id"])
        for i in range(1, layers + 1)
    ]
    final_norm = _node("layernorm", "Final LayerNorm", {
        "normalized_shape": dim, "eps": 1e-5,
        "elementwise_affine": True, "bias": True,
        "device": None, "dtype": None,
    })
    head = _node("lm_head", "LM Head", {
        "hidden_size": dim, "vocab_size": 50257, "bias": False,
        "tie_embeddings": True, "device": None, "dtype": None,
    })
    out = _node("text_output", "Text Output", {
        "max_new_tokens": 64, "temperature": 0.8, "top_p": 0.95,
    })

    nodes = [text_input, emb, pos, drop, *layer_nodes, final_norm, head, out]
    edges = [
        _edge(text_input["id"], emb["id"]),
        _edge(emb["id"], pos["id"]),
        _edge(pos["id"], drop["id"]),
    ]

    first = layer_nodes[0]
    edges.extend([
        _edge(drop["id"], first["id"], source_port="main_out", target_port="main_in", kind="main"),
        _edge(drop["id"], first["id"], source_port="main_out", target_port="skip_in", kind="residual"),
    ])

    for left, right in zip(layer_nodes[:-1], layer_nodes[1:]):
        edges.extend([
            _edge(left["id"], right["id"], source_port="main_out", target_port="main_in", kind="main"),
            _edge(left["id"], right["id"], source_port="skip_out", target_port="skip_in", kind="residual"),
        ])

    edges.extend([
        _edge(layer_nodes[-1]["id"], final_norm["id"], source_port="main_out", target_port="main_in", kind="main"),
        _edge(final_norm["id"], head["id"]),
        _edge(head["id"], out["id"]),
    ])

    project["components"][root_id]["nodes"] = nodes
    project["components"][root_id]["edges"] = edges
    return project


def tinystories_30m_project():
    """Compatibility entry point for the 50M SLM ESA starter."""
    return _standard_esa_layer_block_project(
        name="50M SLM", dim=480, heads=6, layers=10, ffn_dim=1920,
        block=512, batch=16, dataset="TinyStories",
    )


def slm_50m_project():
    """Preferred public name for the 50M SLM preset."""
    return tinystories_30m_project()


def tinystories_50m_project():
    """Dataset-oriented alias for the 50M SLM preset."""
    return tinystories_30m_project()


def esa_200m_project():
    """12-layer standard ESA preset using explicit Signal/Residual Layer Blocks."""
    return _standard_esa_layer_block_project(
        name="200M SLM", dim=1024, heads=16, layers=12, ffn_dim=4096,
        block=256, batch=16, dataset=None,
    )


def stateaware_esa_200m_project():
    """12-layer StateAware ESA preset targeting the 200M SLM class."""
    project = new_project("200M SLM · StateAware")
    project["project"].update({
        "context_length": 256, "batch_size": 16, "dataset": None,
        "estimated_parameters": "~200M",
        "description": "Legacy 12-layer StateAware ESA variant targeting the ~200M class",
        "model_settings": {"embedding_size": 384, "heads": 6, "block": 256,
                           "default_batch": 16, "vocab_size": 50257, "precision": "fp16"},
    })
    root_id = project["root_component_id"]
    nodes = [
        _node("text_input", "Text Input", {"prompt": "Once upon a time"}),
        _node("embedding", "Token Embedding", {"vocab_size": 50257, "embedding_dim": 384}),
        _node("stateaware_esa_stack", "StateAware ESA ×12", {
            "dim": 384, "state_dim": 1824, "layers": 12, "heads": 6,
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


def slm_200m_project():
    """Preferred public name for the standard 200M ESA SLM preset."""
    return esa_200m_project()


def soup_200m_project():
    """Three-layer SOUP 200M SLM preset."""
    project = new_project("200M SLM · SOUP")
    project["project"].update({
        "context_length": 256, "batch_size": 16, "dataset": None,
        "estimated_parameters": "199,916,160",
        "description": "SOUP 200M SLM with three physical layers",
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


def soup_200m_3l_project():
    """Explicit-depth alias for the 200M SOUP SLM preset."""
    return soup_200m_project()


def soup_30m_1l_project():
    """Compatibility entry point for the two-layer ~50M SOUP SLM preset."""
    project = new_project("50M SLM · SOUP")
    project["project"].update({
        "context_length": 512, "batch_size": 16, "dataset": "TinyStories",
        "estimated_parameters": "~50M",
        "description": "Two-layer SOUP small language model targeting ~50M parameters",
        "model_settings": {"embedding_size": 448, "heads": 8, "block": 512,
                           "default_batch": 16, "vocab_size": 50257, "precision": "fp16"},
    })
    root_id = project["root_component_id"]
    nodes = [
        _node("text_input", "Text Input", {"prompt": "Once upon a time"}),
        _node("embedding", "Token Embedding", {"vocab_size": 50257, "embedding_dim": 448}),
        _node("soup", "SOUP ×2", {
            "dim": 448, "width": 1600, "depth": 2, "mixer": "esa", "ffn": "saffn",
            "mixer_config": {"head": 8, "batch": 16, "block": 512, "compass": 16, "auto_compile": False},
            "ffn_config": {"depth_dim": 64}, "memory_dim": 160, "fusion_hidden": 1088,
        }),
        _node("rmsnorm", "Final RMSNorm", {"normalized_shape": 448, "eps": 1e-6, "elementwise_affine": True}),
        _node("lm_head", "LM Head", {"hidden_size": 448, "vocab_size": 50257, "bias": False, "tie_embeddings": True}),
        _node("text_output", "Text Output", {"max_new_tokens": 64, "temperature": 0.8, "top_p": 0.95}),
    ]
    project["components"][root_id]["nodes"] = nodes
    project["components"][root_id]["edges"] = [_edge(a["id"], b["id"]) for a,b in zip(nodes[:-1],nodes[1:])]
    return project


def soup_50m_2l_project():
    """Preferred public name for the two-layer 50M SOUP SLM preset."""
    return soup_30m_1l_project()

