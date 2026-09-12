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
                {"key": "input_key", "label": "Runtime Input Key", "type": "text", "value": "",
                 "help": "Optional named runtime input for multi-input custom/audio models."},
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
                {"key": "input_key", "label": "Runtime Input Key", "type": "text", "value": "",
                 "help": "Optional named runtime input for multimodal graphs, for example image."},
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
                {"key": "input_key", "label": "Runtime Input Key", "type": "text", "value": "",
                 "help": "Optional named runtime input, for example reference_audio."},
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
                {"key": "input_key", "label": "Runtime Input Key", "type": "text", "value": "",
                 "help": "Optional named runtime input for multimodal graphs, for example video."},
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
                {"key": "input_key", "label": "Runtime Input Key", "type": "text", "value": "",
                 "help": "Optional named runtime input for multimodal graphs, for example sensor."},
                {"key": "sample_rate", "label": "Sample Rate", "type": "number", "value": 16000},
                {"key": "buffer_size", "label": "Buffer Size", "type": "number", "value": 256},
                {"key": "channel", "label": "Channel / Port", "type": "text", "value": ""},
            ],
        },

        {
            "type": "demo_dataset",
            "builder_utility": True,
            "builder_python_api": True,
            "name": "Studio Demo Dataset",
            "icon": "DEMO",
            "category": "Data Source",
            "description": "Deterministic offline demo data for ML, DL, JEPA, vision, audio, signal, and multimodal examples.",
            "accent": "green",
            "api": [
                {"key": "demo_type", "label": "Demo Type", "type": "select", "value": "tabular_regression",
                 "options": [
                    "tabular_regression", "binary_classification", "multiclass_classification", "clustering", "high_dimensional",
                    "neuron_regression", "tabular_classification", "image_classification", "sequence_classification", "image_reconstruction",
                    "text_corpus", "image_jepa", "video_jepa", "text_jepa", "audio_jepa", "signal_jepa", "object_detection",
                    "speech_transcript", "multispeaker_speech", "music_caption", "sound_caption", "timeseries_forecast",
                    "signal_classification", "anomaly_detection", "signal_denoise", "sensor_fusion", "spectral_signal", "rf_iq",
                    "long_signal", "multimodal_image_text", "sensor_vision"
                 ]},
                {"key": "samples", "label": "Samples", "type": "number", "value": 512},
                {"key": "seed", "label": "Random Seed", "type": "number", "value": 42},
                {"key": "sequence_length", "label": "Sequence Length", "type": "number", "value": 32},
                {"key": "feature_count", "label": "Feature Count", "type": "number", "value": 8},
                {"key": "classes", "label": "Classes", "type": "number", "value": 3}
            ],
        },
        {
            "type": "coco128_cloud",
            "builder_utility": True,
            "builder_python_api": True,
            "name": "COCO128 Cloud",
            "icon": "COCO",
            "category": "Data Source",
            "description": "Fetch COCO128 once from the cloud as a reusable vision source for detection, primary-object classification, reconstruction and image JEPA; nothing is bundled or persistently cached; files use temporary session storage.",
            "accent": "cyan",
            "api": [
                {"key": "download_url", "label": "Cloud URL", "type": "text", "value": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"},
                {"key": "max_images", "label": "Max Images (0 = All 128)", "type": "number", "value": 0},
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
                {"key": "center_crop", "label": "Center Crop", "type": "select", "value": "false", "options": ["false", "true"]},
                {"key": "tensor_ready", "label": "Tensor Ready", "type": "select", "value": "false", "options": ["false", "true"]},
                {"key": "normalize", "label": "Normalize 0–1", "type": "select", "value": "true", "options": ["true", "false"]}
            ],
        },
        {
            "type": "detection_process",
            "builder_utility": True,
            "builder_python_api": True,
            "name": "Detection Processing",
            "icon": "BOX",
            "category": "Vision",
            "description": "Resize images and boxes together, preserve all detection labels, and expose the largest-object class as an image-level label for shared vision training.",
            "accent": "orange",
            "api": [
                {"key": "image_column", "label": "Image Column", "type": "text", "value": "image"},
                {"key": "boxes_column", "label": "Boxes Column", "type": "text", "value": "boxes"},
                {"key": "classes_column", "label": "Classes Column", "type": "text", "value": "class_ids"},
                {"key": "width", "label": "Width", "type": "number", "value": 16},
                {"key": "height", "label": "Height", "type": "number", "value": 16},
                {"key": "mode", "label": "Color Mode", "type": "select", "value": "L", "options": ["RGB", "L"]},
                {"key": "box_format", "label": "Box Format", "type": "select", "value": "xywh", "options": ["xywh"]},
                {"key": "normalize_images", "label": "Normalize Images", "type": "select", "value": "true", "options": ["true", "false"]}
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
            "type": "signal_process",
            "builder_utility": True,
            "builder_python_api": True,
            "name": "Signal Schema Mapper",
            "icon": "SIG+",
            "category": "Signal",
            "description": "Map one or more signal columns into Studio's canonical signal tensor and optionally expose a training target.",
            "accent": "cyan",
            "api": [
                {"key": "signal_columns", "label": "Signal Columns", "type": "text", "value": "signal",
                 "help": "Comma-separated columns. Multiple columns are stacked as channels."},
                {"key": "output_column", "label": "Output Signal Column", "type": "text", "value": "signal"},
                {"key": "target_column", "label": "Target Column (Optional)", "type": "text", "value": ""},
                {"key": "target_output_column", "label": "Output Target Column", "type": "text", "value": "target"},
                {"key": "normalize", "label": "Normalize Signal", "type": "select", "value": "false", "options": ["true", "false"]},
                {"key": "pad_length", "label": "Pad / Trim Length (0 = Keep)", "type": "number", "value": 0}
            ],
        },
        {
            "type": "jepa_prepare",
            "builder_utility": True,
            "builder_python_api": True,
            "name": "JEPA Preparation",
            "icon": "JDP",
            "category": "Data Processing",
            "description": "Prepare image, video, text, audio, or signal samples into a common numeric JEPA input tensor field.",
            "accent": "purple",
            "api": [
                {"key": "modality", "label": "Modality", "type": "select", "value": "image",
                 "options": ["image", "video", "text", "audio", "signal"]},
                {"key": "input_column", "label": "Input Column", "type": "text", "value": "image"},
                {"key": "output_column", "label": "Output Column", "type": "text", "value": "jepa_input"},
                {"key": "sequence_length", "label": "Sequence Length", "type": "number", "value": 64},
                {"key": "image_size", "label": "Image Size", "type": "number", "value": 16},
                {"key": "normalize", "label": "Normalize", "type": "select", "value": "true", "options": ["true", "false"]}
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
        # ------------------------------------------------------------------
        # Educational foundations: classical ML, deep-learning primitives,
        # and low-level tensor/math operations. These are intentionally
        # Builder-native PyTorch utilities so students can compose models from
        # first principles without depending on opaque prebuilt architecture
        # nodes. Higher-level Gallery templates should be made from these same
        # public components.
        # ------------------------------------------------------------------
        {
            "type": "feature_input",
            "builder_utility": True,
            "builder_python_api": False,
            "name": "Feature Input",
            "icon": "X",
            "category": "ML Core",
            "description": "Numeric feature tensor input for regression, classification, and tabular ML experiments.",
            "accent": "green",
            "api": [
                {"key": "feature_dim", "label": "Feature Count", "type": "number", "value": 4,
                 "help": "Documentation/validation hint for the expected final feature dimension."},
                {"key": "input_key", "label": "Runtime Input Key", "type": "text", "value": "",
                 "help": "Optional named runtime input, for example speaker_id."},
            ],
        },
        {
            "type": "linear_regression",
            "builder_utility": True,
            "builder_python_api": False,
            "name": "Linear Regression",
            "icon": "LR",
            "category": "ML Core",
            "description": "Trainable y = XW + b regression layer for learning and small supervised models.",
            "accent": "blue",
            "api": [
                {"key": "in_features", "label": "Input Features", "type": "number", "value": 4},
                {"key": "out_features", "label": "Outputs", "type": "number", "value": 1},
                {"key": "bias", "label": "Use Bias", "type": "select", "value": "true", "options": ["true", "false"]},
            ],
        },
        {
            "type": "logistic_regression",
            "builder_utility": True,
            "builder_python_api": False,
            "name": "Logistic Regression",
            "icon": "LOGR",
            "category": "ML Core",
            "description": "Linear classifier with an optional sigmoid probability output.",
            "accent": "purple",
            "api": [
                {"key": "in_features", "label": "Input Features", "type": "number", "value": 4},
                {"key": "out_features", "label": "Outputs", "type": "number", "value": 1},
                {"key": "bias", "label": "Use Bias", "type": "select", "value": "true", "options": ["true", "false"]},
                {"key": "output", "label": "Output", "type": "select", "value": "probability", "options": ["probability", "logits"],
                 "help": "Probability applies sigmoid. Logits is preferred when a BCE-with-logits loss is used."},
            ],
        },
        {
            "type": "polynomial_features",
            "builder_utility": True,
            "builder_python_api": False,
            "name": "Polynomial Features",
            "icon": "POLY",
            "category": "ML Core",
            "description": "Expand numeric features with polynomial and interaction terms for classical regression experiments.",
            "accent": "cyan",
            "api": [
                {"key": "degree", "label": "Degree", "type": "number", "value": 2},
                {"key": "include_bias", "label": "Include Constant 1", "type": "select", "value": "false", "options": ["false", "true"]},
                {"key": "interaction_only", "label": "Interaction Only", "type": "select", "value": "false", "options": ["false", "true"]},
            ],
        },
        {
            "type": "knn_classifier",
            "builder_utility": True,
            "builder_python_api": False,
            "name": "KNN Classifier",
            "icon": "KNN",
            "category": "ML Core",
            "description": "Fit-based K-nearest-neighbours classifier. Stores the fitted reference samples inside the Studio model artifact.",
            "accent": "purple",
            "api": [
                {"key": "neighbors", "label": "Neighbors (K)", "type": "number", "value": 5},
                {"key": "weights", "label": "Voting", "type": "select", "value": "uniform", "options": ["uniform", "distance"]},
                {"key": "p", "label": "Distance P", "type": "number", "value": 2},
            ],
        },
        {
            "type": "decision_tree_classifier",
            "builder_utility": True,
            "builder_python_api": False,
            "name": "Decision Tree",
            "icon": "TREE",
            "category": "ML Core",
            "description": "Educational CART-style fit-based decision tree classifier with inspectable depth and split settings.",
            "accent": "green",
            "api": [
                {"key": "max_depth", "label": "Max Depth", "type": "number", "value": 5},
                {"key": "min_samples_split", "label": "Min Samples Split", "type": "number", "value": 2},
                {"key": "min_samples_leaf", "label": "Min Samples Leaf", "type": "number", "value": 1},
                {"key": "criterion", "label": "Criterion", "type": "select", "value": "gini", "options": ["gini", "entropy"]},
            ],
        },
        {
            "type": "kmeans",
            "builder_utility": True,
            "builder_python_api": False,
            "name": "K-Means",
            "icon": "KM",
            "category": "ML Core",
            "description": "Fit-based centroid clustering with K-Means++ initialization and persisted fitted centroids.",
            "accent": "blue",
            "api": [
                {"key": "clusters", "label": "Clusters (K)", "type": "number", "value": 3},
                {"key": "max_iter", "label": "Max Iterations", "type": "number", "value": 100},
                {"key": "tolerance", "label": "Tolerance", "type": "number", "value": 0.0001},
                {"key": "seed", "label": "Seed", "type": "number", "value": 42},
            ],
        },
        {
            "type": "pca",
            "builder_utility": True,
            "builder_python_api": False,
            "name": "PCA",
            "icon": "PCA",
            "category": "ML Core",
            "description": "Fit-based principal component analysis using SVD, with optional whitening and persisted components.",
            "accent": "cyan",
            "api": [
                {"key": "components", "label": "Components", "type": "number", "value": 2},
                {"key": "center", "label": "Center Data", "type": "select", "value": "true", "options": ["true", "false"]},
                {"key": "whiten", "label": "Whiten", "type": "select", "value": "false", "options": ["false", "true"]},
            ],
        },

        # Deep Learning Core -------------------------------------------------
        {"type":"relu","builder_utility":True,"builder_python_api":False,"name":"ReLU","icon":"RELU","category":"Deep Learning Core","description":"Rectified Linear Unit activation: max(0, x).","accent":"orange","api":[]},
        {"type":"leaky_relu","builder_utility":True,"builder_python_api":False,"name":"Leaky ReLU","icon":"LRELU","category":"Deep Learning Core","description":"ReLU variant that keeps a small negative slope.","accent":"orange","api":[{"key":"negative_slope","label":"Negative Slope","type":"number","value":0.01}]},
        {"type":"gelu","builder_utility":True,"builder_python_api":False,"name":"GELU","icon":"GELU","category":"Deep Learning Core","description":"Gaussian Error Linear Unit activation commonly used in modern neural networks.","accent":"orange","api":[{"key":"approximate","label":"Approximation","type":"select","value":"none","options":["none","tanh"]}]},
        {"type":"silu","builder_utility":True,"builder_python_api":False,"name":"SiLU","icon":"SILU","category":"Deep Learning Core","description":"Sigmoid Linear Unit (Swish) activation.","accent":"orange","api":[]},
        {"type":"sigmoid","builder_utility":True,"builder_python_api":False,"name":"Sigmoid","icon":"SIGM","category":"Deep Learning Core","description":"Maps values to the 0–1 interval; useful for binary probabilities and gates.","accent":"orange","api":[]},
        {"type":"tanh","builder_utility":True,"builder_python_api":False,"name":"Tanh","icon":"TANH","category":"Deep Learning Core","description":"Hyperbolic tangent activation with output in the -1 to 1 range.","accent":"orange","api":[]},
        {"type":"softmax","builder_utility":True,"builder_python_api":False,"name":"Softmax","icon":"SMAX","category":"Deep Learning Core","description":"Normalize scores into a probability distribution along one dimension.","accent":"orange","api":[{"key":"dim","label":"Dimension","type":"number","value":-1}]},
        {
            "type":"batchnorm1d","builder_utility":True,"builder_python_api":False,"name":"BatchNorm 1D","icon":"BN1","category":"Deep Learning Core",
            "description":"Batch normalization for feature vectors, sequences, or 1D channel data.","accent":"orange",
            "api":[
                {"key":"num_features","label":"Features / Channels","type":"number","value":128},
                {"key":"eps","label":"Epsilon","type":"number","value":0.00001},
                {"key":"momentum","label":"Momentum","type":"number","value":0.1},
                {"key":"layout","label":"Layout","type":"select","value":"features_last","options":["features_last","channels_first"]},
            ],
        },
        {
            "type":"batchnorm2d","builder_utility":True,"builder_python_api":False,"name":"BatchNorm 2D","icon":"BN2","category":"Deep Learning Core",
            "description":"Batch normalization for image tensors in [B,C,H,W] layout.","accent":"orange",
            "api":[
                {"key":"num_features","label":"Channels","type":"number","value":32},
                {"key":"eps","label":"Epsilon","type":"number","value":0.00001},
                {"key":"momentum","label":"Momentum","type":"number","value":0.1},
            ],
        },
        {
            "type":"conv1d","builder_utility":True,"builder_python_api":False,"name":"Conv1D","icon":"C1D","category":"Deep Learning Core",
            "description":"One-dimensional convolution for sequences, audio features, and signals.","accent":"blue",
            "api":[
                {"key":"in_channels","label":"Input Channels","type":"number","value":1},
                {"key":"out_channels","label":"Output Channels","type":"number","value":16},
                {"key":"kernel_size","label":"Kernel Size","type":"number","value":3},
                {"key":"stride","label":"Stride","type":"number","value":1},
                {"key":"padding","label":"Padding","type":"number","value":1},
                {"key":"dilation","label":"Dilation","type":"number","value":1},
                {"key":"groups","label":"Groups","type":"number","value":1},
                {"key":"bias","label":"Use Bias","type":"select","value":"true","options":["true","false"]},
            ],
        },
        {
            "type":"conv2d","builder_utility":True,"builder_python_api":False,"name":"Conv2D","icon":"C2D","category":"Deep Learning Core",
            "description":"Two-dimensional convolution for CNNs and image feature extraction.","accent":"blue",
            "api":[
                {"key":"in_channels","label":"Input Channels","type":"number","value":3},
                {"key":"out_channels","label":"Output Channels","type":"number","value":32},
                {"key":"kernel_size","label":"Kernel Size","type":"number","value":3},
                {"key":"stride","label":"Stride","type":"number","value":1},
                {"key":"padding","label":"Padding","type":"number","value":1},
                {"key":"dilation","label":"Dilation","type":"number","value":1},
                {"key":"groups","label":"Groups","type":"number","value":1},
                {"key":"bias","label":"Use Bias","type":"select","value":"true","options":["true","false"]},
            ],
        },
        {
            "type":"conv3d","builder_utility":True,"builder_python_api":False,"name":"Conv3D","icon":"C3D","category":"Deep Learning Core",
            "description":"Three-dimensional convolution for volumetric and spatiotemporal data.","accent":"blue",
            "api":[
                {"key":"in_channels","label":"Input Channels","type":"number","value":3},
                {"key":"out_channels","label":"Output Channels","type":"number","value":16},
                {"key":"kernel_size","label":"Kernel Size","type":"number","value":3},
                {"key":"stride","label":"Stride","type":"number","value":1},
                {"key":"padding","label":"Padding","type":"number","value":1},
                {"key":"bias","label":"Use Bias","type":"select","value":"true","options":["true","false"]},
            ],
        },
        {"type":"maxpool1d","builder_utility":True,"builder_python_api":False,"name":"MaxPool 1D","icon":"MP1","category":"Deep Learning Core","description":"Max pooling over one-dimensional features.","accent":"cyan","api":[{"key":"kernel_size","label":"Kernel Size","type":"number","value":2},{"key":"stride","label":"Stride (0 = Kernel)","type":"number","value":0},{"key":"padding","label":"Padding","type":"number","value":0}]},
        {"type":"maxpool2d","builder_utility":True,"builder_python_api":False,"name":"MaxPool 2D","icon":"MP2","category":"Deep Learning Core","description":"Max pooling for CNN image feature maps.","accent":"cyan","api":[{"key":"kernel_size","label":"Kernel Size","type":"number","value":2},{"key":"stride","label":"Stride (0 = Kernel)","type":"number","value":0},{"key":"padding","label":"Padding","type":"number","value":0}]},
        {"type":"avgpool1d","builder_utility":True,"builder_python_api":False,"name":"AvgPool 1D","icon":"AP1","category":"Deep Learning Core","description":"Average pooling over one-dimensional features.","accent":"cyan","api":[{"key":"kernel_size","label":"Kernel Size","type":"number","value":2},{"key":"stride","label":"Stride (0 = Kernel)","type":"number","value":0},{"key":"padding","label":"Padding","type":"number","value":0}]},
        {"type":"avgpool2d","builder_utility":True,"builder_python_api":False,"name":"AvgPool 2D","icon":"AP2","category":"Deep Learning Core","description":"Average pooling for CNN image feature maps.","accent":"cyan","api":[{"key":"kernel_size","label":"Kernel Size","type":"number","value":2},{"key":"stride","label":"Stride (0 = Kernel)","type":"number","value":0},{"key":"padding","label":"Padding","type":"number","value":0}]},
        {"type":"adaptive_avgpool1d","builder_utility":True,"builder_python_api":False,"name":"Adaptive AvgPool 1D","icon":"AA1","category":"Deep Learning Core","description":"Adaptive 1D average pooling to a fixed output length.","accent":"cyan","api":[{"key":"output_size","label":"Output Size","type":"number","value":1}]},
        {"type":"adaptive_avgpool2d","builder_utility":True,"builder_python_api":False,"name":"Adaptive AvgPool 2D","icon":"AA2","category":"Deep Learning Core","description":"Adaptive 2D average pooling to a fixed square output size.","accent":"cyan","api":[{"key":"output_size","label":"Output H/W","type":"number","value":1}]},
        {
            "type":"rnn","builder_utility":True,"builder_python_api":False,"name":"RNN","icon":"RNN","category":"Deep Learning Core",
            "description":"Vanilla recurrent neural network with batch-first sequence input [B,T,D].","accent":"purple",
            "api":[
                {"key":"input_size","label":"Input Size","type":"number","value":128},
                {"key":"hidden_size","label":"Hidden Size","type":"number","value":128},
                {"key":"num_layers","label":"Layers","type":"number","value":1},
                {"key":"nonlinearity","label":"Activation","type":"select","value":"tanh","options":["tanh","relu"]},
                {"key":"dropout","label":"Dropout","type":"number","value":0.0},
                {"key":"bidirectional","label":"Bidirectional","type":"select","value":"false","options":["false","true"]},
                {"key":"output","label":"Output","type":"select","value":"sequence","options":["sequence","last"]},
            ],
        },
        {
            "type":"lstm","builder_utility":True,"builder_python_api":False,"name":"LSTM","icon":"LSTM","category":"Deep Learning Core",
            "description":"Long Short-Term Memory recurrent layer with batch-first sequence input [B,T,D].","accent":"purple",
            "api":[
                {"key":"input_size","label":"Input Size","type":"number","value":128},
                {"key":"hidden_size","label":"Hidden Size","type":"number","value":128},
                {"key":"num_layers","label":"Layers","type":"number","value":1},
                {"key":"dropout","label":"Dropout","type":"number","value":0.0},
                {"key":"bidirectional","label":"Bidirectional","type":"select","value":"false","options":["false","true"]},
                {"key":"output","label":"Output","type":"select","value":"sequence","options":["sequence","last"]},
            ],
        },
        {
            "type":"gru","builder_utility":True,"builder_python_api":False,"name":"GRU","icon":"GRU","category":"Deep Learning Core",
            "description":"Gated Recurrent Unit layer with batch-first sequence input [B,T,D].","accent":"purple",
            "api":[
                {"key":"input_size","label":"Input Size","type":"number","value":128},
                {"key":"hidden_size","label":"Hidden Size","type":"number","value":128},
                {"key":"num_layers","label":"Layers","type":"number","value":1},
                {"key":"dropout","label":"Dropout","type":"number","value":0.0},
                {"key":"bidirectional","label":"Bidirectional","type":"select","value":"false","options":["false","true"]},
                {"key":"output","label":"Output","type":"select","value":"sequence","options":["sequence","last"]},
            ],
        },
        {
            "type":"self_attention","builder_utility":True,"builder_python_api":False,"name":"Self Attention","icon":"ATTN","category":"Deep Learning Core",
            "description":"Multi-head self-attention for [B,T,D] tensors.","accent":"purple",
            "api":[
                {"key":"dim","label":"Embedding Dim","type":"number","value":128},
                {"key":"heads","label":"Heads","type":"number","value":4},
                {"key":"dropout","label":"Dropout","type":"number","value":0.0},
                {"key":"causal","label":"Causal Mask","type":"select","value":"false","options":["false","true"]},
                {"key":"bias","label":"Use Bias","type":"select","value":"true","options":["true","false"]},
            ],
        },

        # Math & Tensor Ops --------------------------------------------------
        {
            "type":"learnable_parameter","builder_utility":True,"builder_python_api":False,"name":"Learnable Parameter","icon":"PAR","category":"Math & Tensor Ops",
            "description":"Create a trainable tensor parameter; combine with MatMul/Add to build layers from scratch.","accent":"pink",
            "runtime_ports":{"inputs":[],"outputs":[{"id":"main","name":"Parameter","socket":"front"}]},
            "api":[
                {"key":"shape","label":"Shape","type":"text","value":"4,1","help":"Comma-separated tensor shape, for example 4,1."},
                {"key":"init","label":"Initialization","type":"select","value":"normal","options":["zeros","ones","normal","uniform"]},
                {"key":"scale","label":"Init Scale","type":"number","value":0.02},
            ],
        },
        {
            "type":"constant","builder_utility":True,"builder_python_api":False,"name":"Constant","icon":"CONST","category":"Math & Tensor Ops",
            "description":"Create a fixed tensor constant for arithmetic and educational graphs.","accent":"cyan",
            "runtime_ports":{"inputs":[],"outputs":[{"id":"main","name":"Constant","socket":"front"}]},
            "api":[
                {"key":"shape","label":"Shape","type":"text","value":"1"},
                {"key":"value","label":"Value","type":"number","value":0.0},
            ],
        },
        {
            "type":"matmul","builder_utility":True,"builder_python_api":False,"name":"MatMul","icon":"MM","category":"Math & Tensor Ops",
            "description":"Matrix/tensor multiplication. Use A × B to construct linear layers from first principles.","accent":"blue",
            "runtime_ports":{"inputs":[{"id":"a","name":"A","socket":"back"},{"id":"b","name":"B","socket":"bottom"}],"outputs":[{"id":"main","name":"A × B","socket":"front"}]},
            "api":[],
        },
        {
            "type":"tensor_add","builder_utility":True,"builder_python_api":False,"name":"Add","icon":"ADD","category":"Math & Tensor Ops",
            "description":"Elementwise tensor addition with broadcasting.","accent":"cyan",
            "runtime_ports":{"inputs":[{"id":"a","name":"A","socket":"back"},{"id":"b","name":"B","socket":"bottom"}],"outputs":[{"id":"main","name":"A + B","socket":"front"}]},"api":[],
        },
        {
            "type":"tensor_subtract","builder_utility":True,"builder_python_api":False,"name":"Subtract","icon":"SUB","category":"Math & Tensor Ops",
            "description":"Elementwise tensor subtraction A - B with broadcasting.","accent":"cyan",
            "runtime_ports":{"inputs":[{"id":"a","name":"A","socket":"back"},{"id":"b","name":"B","socket":"bottom"}],"outputs":[{"id":"main","name":"A - B","socket":"front"}]},"api":[],
        },
        {
            "type":"tensor_multiply","builder_utility":True,"builder_python_api":False,"name":"Multiply","icon":"MUL","category":"Math & Tensor Ops",
            "description":"Elementwise tensor multiplication with broadcasting.","accent":"cyan",
            "runtime_ports":{"inputs":[{"id":"a","name":"A","socket":"back"},{"id":"b","name":"B","socket":"bottom"}],"outputs":[{"id":"main","name":"A × B","socket":"front"}]},"api":[],
        },
        {
            "type":"tensor_divide","builder_utility":True,"builder_python_api":False,"name":"Divide","icon":"DIV","category":"Math & Tensor Ops",
            "description":"Elementwise tensor division A / B with optional numerical epsilon.","accent":"cyan",
            "runtime_ports":{"inputs":[{"id":"a","name":"A","socket":"back"},{"id":"b","name":"B","socket":"bottom"}],"outputs":[{"id":"main","name":"A / B","socket":"front"}]},
            "api":[{"key":"epsilon","label":"Epsilon","type":"number","value":0.0}],
        },
        {
            "type":"concat","builder_utility":True,"builder_python_api":False,"name":"Concatenate","icon":"CAT","category":"Math & Tensor Ops",
            "description":"Concatenate two tensors along a selected dimension.","accent":"purple",
            "runtime_ports":{"inputs":[{"id":"a","name":"A","socket":"back"},{"id":"b","name":"B","socket":"bottom"}],"outputs":[{"id":"main","name":"Concatenated","socket":"front"}]},
            "api":[{"key":"dim","label":"Dimension","type":"number","value":-1}],
        },
        {"type":"reduce_mean","builder_utility":True,"builder_python_api":False,"name":"Mean","icon":"MEAN","category":"Math & Tensor Ops","description":"Reduce a tensor by its mean along a dimension.","accent":"green","api":[{"key":"dim","label":"Dimension","type":"number","value":-1},{"key":"keepdim","label":"Keep Dimension","type":"select","value":"false","options":["false","true"]}]},
        {"type":"reduce_sum","builder_utility":True,"builder_python_api":False,"name":"Sum","icon":"SUM","category":"Math & Tensor Ops","description":"Reduce a tensor by summing along a dimension.","accent":"green","api":[{"key":"dim","label":"Dimension","type":"number","value":-1},{"key":"keepdim","label":"Keep Dimension","type":"select","value":"false","options":["false","true"]}]},
        {"type":"reduce_max","builder_utility":True,"builder_python_api":False,"name":"Max","icon":"MAX","category":"Math & Tensor Ops","description":"Reduce a tensor by its maximum along a dimension.","accent":"green","api":[{"key":"dim","label":"Dimension","type":"number","value":-1},{"key":"keepdim","label":"Keep Dimension","type":"select","value":"false","options":["false","true"]}]},
        {"type":"reduce_min","builder_utility":True,"builder_python_api":False,"name":"Min","icon":"MIN","category":"Math & Tensor Ops","description":"Reduce a tensor by its minimum along a dimension.","accent":"green","api":[{"key":"dim","label":"Dimension","type":"number","value":-1},{"key":"keepdim","label":"Keep Dimension","type":"select","value":"false","options":["false","true"]}]},
        {"type":"tensor_exp","builder_utility":True,"builder_python_api":False,"name":"Exp","icon":"EXP","category":"Math & Tensor Ops","description":"Elementwise exponential e^x.","accent":"green","api":[]},
        {"type":"tensor_log","builder_utility":True,"builder_python_api":False,"name":"Log","icon":"LOG","category":"Math & Tensor Ops","description":"Elementwise natural logarithm with optional minimum clamp.","accent":"green","api":[{"key":"epsilon","label":"Minimum Clamp","type":"number","value":1e-12}]},
        {"type":"tensor_sqrt","builder_utility":True,"builder_python_api":False,"name":"Sqrt","icon":"SQRT","category":"Math & Tensor Ops","description":"Elementwise square root with optional minimum clamp.","accent":"green","api":[{"key":"epsilon","label":"Minimum Clamp","type":"number","value":0.0}]},
        {"type":"transpose","builder_utility":True,"builder_python_api":False,"name":"Transpose","icon":"TR","category":"Math & Tensor Ops","description":"Swap two tensor dimensions.","accent":"purple","api":[{"key":"dim0","label":"Dimension A","type":"number","value":-2},{"key":"dim1","label":"Dimension B","type":"number","value":-1}]},
        {"type":"reshape","builder_utility":True,"builder_python_api":False,"name":"Reshape","icon":"RSH","category":"Math & Tensor Ops","description":"Reshape a tensor. Use 0 to copy the corresponding input dimension and -1 to infer one dimension.","accent":"purple","api":[{"key":"shape","label":"New Shape","type":"text","value":"0,-1"}]},
        {"type":"flatten","builder_utility":True,"builder_python_api":False,"name":"Flatten","icon":"FLAT","category":"Math & Tensor Ops","description":"Flatten a range of tensor dimensions.","accent":"purple","api":[{"key":"start_dim","label":"Start Dimension","type":"number","value":1},{"key":"end_dim","label":"End Dimension","type":"number","value":-1}]},
        {"type":"unsqueeze","builder_utility":True,"builder_python_api":False,"name":"Unsqueeze","icon":"UNSQ","category":"Math & Tensor Ops","description":"Insert a size-one tensor dimension.","accent":"purple","api":[{"key":"dim","label":"Dimension","type":"number","value":1}]},
        {"type":"squeeze","builder_utility":True,"builder_python_api":False,"name":"Squeeze","icon":"SQZ","category":"Math & Tensor Ops","description":"Remove a size-one tensor dimension, or all size-one dimensions when Dimension is blank.","accent":"purple","api":[{"key":"dim","label":"Dimension (blank = all)","type":"text","value":""}]},

        {
            "type": "signal_fft",
            "builder_utility": True,
            "builder_python_api": False,
            "name": "FFT Magnitude",
            "icon": "FFT",
            "category": "Signal",
            "description": "Real FFT magnitude features for waveform/sensor tensors. Accepts [B,T] or [B,C,T].",
            "accent": "cyan",
            "api": [
                {"key": "bins", "label": "Output Bins (0 = All)", "type": "number", "value": 16},
                {"key": "log_scale", "label": "Log Magnitude", "type": "select", "value": "false", "options": ["false", "true"]},
                {"key": "remove_dc", "label": "Remove DC Bin", "type": "select", "value": "false", "options": ["false", "true"]},
                {"key": "flatten_channels", "label": "Flatten Channels", "type": "select", "value": "true", "options": ["true", "false"]}
            ],
        },
        {
            "type": "embedding",
            "name": "Embedding",
            "icon": "EMB",
            "category": "Deep Learning Core",
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
            "type": "jepa_mask",
            "builder_utility": True,
            "builder_python_api": False,
            "name": "JEPA Mask",
            "icon": "MSK",
            "category": "JEPA & Predictive",
            "description": "Create a masked context view and an unmasked target view for joint-embedding predictive learning.",
            "accent": "purple",
            "runtime_ports": {
                "inputs": [{"id": "main", "name": "Input", "socket": "back"}],
                "outputs": [
                    {"id": "context", "name": "Context", "socket": "front"},
                    {"id": "target", "name": "Target", "socket": "bottom"}
                ]
            },
            "api": [
                {"key": "mask_ratio", "label": "Mask Ratio", "type": "number", "value": 0.35},
                {"key": "mask_value", "label": "Mask Value", "type": "number", "value": 0.0},
                {"key": "mode", "label": "Mask Mode", "type": "select", "value": "auto",
                 "options": ["auto", "random", "contiguous", "spatiotemporal"]}
            ],
        },
        {
            "type": "jepa_encoder",
            "builder_utility": True,
            "builder_python_api": False,
            "name": "JEPA Encoder",
            "icon": "JEN",
            "category": "JEPA & Predictive",
            "description": "Modality-aware context or target encoder that maps prepared inputs into a compact latent representation.",
            "accent": "cyan",
            "api": [
                {"key": "modality", "label": "Modality", "type": "select", "value": "image",
                 "options": ["image", "video", "text", "audio", "signal"]},
                {"key": "role", "label": "Encoder Role", "type": "select", "value": "context",
                 "options": ["context", "target"]},
                {"key": "latent_dim", "label": "Latent Dim", "type": "number", "value": 64},
                {"key": "hidden_dim", "label": "Hidden Dim", "type": "number", "value": 64},
                {"key": "vocab_size", "label": "Text Vocab Size", "type": "number", "value": 257},
                {"key": "in_channels", "label": "Image Channels", "type": "number", "value": 1}
            ],
        },
        {
            "type": "jepa_predictor",
            "builder_utility": True,
            "builder_python_api": False,
            "name": "JEPA Predictor",
            "icon": "JEP",
            "category": "JEPA & Predictive",
            "description": "Predict the target latent representation from the context latent representation.",
            "accent": "orange",
            "api": [
                {"key": "latent_dim", "label": "Latent Dim", "type": "number", "value": 64},
                {"key": "hidden_dim", "label": "Predictor Hidden Dim", "type": "number", "value": 128},
                {"key": "dropout", "label": "Dropout", "type": "number", "value": 0.0}
            ],
        },
        {
            "type": "jepa_latent_loss",
            "builder_utility": True,
            "builder_python_api": False,
            "name": "JEPA Latent Loss",
            "icon": "JLS",
            "category": "Losses",
            "description": "Compare predicted and stop-gradient target embeddings using MSE, Smooth-L1, or cosine distance.",
            "accent": "pink",
            "runtime_ports": {
                "inputs": [
                    {"id": "prediction", "name": "Prediction", "socket": "back"},
                    {"id": "target", "name": "Target", "socket": "bottom"}
                ],
                "outputs": [{"id": "main", "name": "Loss", "socket": "front"}]
            },
            "api": [
                {"key": "loss", "label": "Loss", "type": "select", "value": "mse",
                 "options": ["mse", "smooth_l1", "cosine"]},
                {"key": "normalize", "label": "Normalize Embeddings", "type": "select", "value": "true",
                 "options": ["true", "false"]}
            ],
        },
        {
            "type": "fpn_fusion",
            "builder_utility": True,
            "builder_python_api": False,
            "name": "FPN Fusion",
            "icon": "FPN",
            "category": "Vision",
            "description": "Top-down feature-pyramid fusion: upsample a high-level feature and merge it with a lateral feature.",
            "accent": "cyan",
            "runtime_ports": {
                "inputs": [
                    {"id": "high", "name": "High-Level Feature", "socket": "back"},
                    {"id": "lateral", "name": "Lateral Feature", "socket": "bottom"}
                ],
                "outputs": [{"id": "main", "name": "Fused Feature", "socket": "front"}]
            },
            "api": [
                {"key": "high_channels", "label": "High Channels", "type": "number", "value": 64},
                {"key": "lateral_channels", "label": "Lateral Channels", "type": "number", "value": 32},
                {"key": "out_channels", "label": "Output Channels", "type": "number", "value": 32},
                {"key": "fusion", "label": "Fusion", "type": "select", "value": "add", "options": ["add", "concat"]}
            ],
        },
        {
            "type": "pan_fusion",
            "builder_utility": True,
            "builder_python_api": False,
            "name": "PAN Fusion",
            "icon": "PAN",
            "category": "Vision",
            "description": "Bottom-up path-aggregation fusion: downsample a fine feature and merge it with a coarser feature.",
            "accent": "cyan",
            "runtime_ports": {
                "inputs": [
                    {"id": "fine", "name": "Fine Feature", "socket": "back"},
                    {"id": "coarse", "name": "Coarse Feature", "socket": "bottom"}
                ],
                "outputs": [{"id": "main", "name": "Aggregated Feature", "socket": "front"}]
            },
            "api": [
                {"key": "fine_channels", "label": "Fine Channels", "type": "number", "value": 32},
                {"key": "coarse_channels", "label": "Coarse Channels", "type": "number", "value": 64},
                {"key": "out_channels", "label": "Output Channels", "type": "number", "value": 64},
                {"key": "fusion", "label": "Fusion", "type": "select", "value": "concat", "options": ["add", "concat"]}
            ],
        },
        {
            "type": "detection_head",
            "builder_utility": True,
            "builder_python_api": False,
            "name": "Detection Head",
            "icon": "DET",
            "category": "Vision",
            "description": "Anchor-free detection head producing box, objectness, and class logits with configurable prediction slots per cell.",
            "accent": "lime",
            "api": [
                {"key": "in_channels", "label": "Input Channels", "type": "number", "value": 64},
                {"key": "classes", "label": "Classes", "type": "number", "value": 3},
                {"key": "anchors", "label": "Prediction Slots / Cell", "type": "number", "value": 1}
            ],
        },
        {
            "type": "detection_pyramid_head",
            "builder_utility": True,
            "builder_python_api": False,
            "name": "Detection Pyramid Head",
            "icon": "PYR",
            "category": "Vision",
            "description": "Three-scale P3/P4/P5 anchor-free head with multiple prediction slots per cell for dense multi-object detection.",
            "accent": "lime",
            "runtime_ports": {
                "inputs": [
                    {"id": "p3", "name": "P3 Fine", "socket": "back"},
                    {"id": "p4", "name": "P4 Medium", "socket": "bottom"},
                    {"id": "p5", "name": "P5 Coarse", "socket": "top"}
                ],
                "outputs": [{"id": "main", "name": "Raw Pyramid Predictions", "socket": "front"}]
            },
            "api": [
                {"key": "p3_channels", "label": "P3 Channels", "type": "number", "value": 32},
                {"key": "p4_channels", "label": "P4 Channels", "type": "number", "value": 64},
                {"key": "p5_channels", "label": "P5 Channels", "type": "number", "value": 96},
                {"key": "classes", "label": "Classes", "type": "number", "value": 3},
                {"key": "slots", "label": "Prediction Slots / Cell", "type": "number", "value": 3}
            ],
        },
        {
            "type": "detection_nms",
            "builder_utility": True,
            "builder_python_api": False,
            "name": "Detection NMS",
            "icon": "NMS",
            "category": "Heads & Decoders",
            "description": "Class-aware non-maximum suppression for raw single-scale or multi-scale detection predictions.",
            "accent": "orange",
            "api": [
                {"key": "score_threshold", "label": "Score Threshold", "type": "number", "value": 0.40},
                {"key": "iou_threshold", "label": "NMS IoU Threshold", "type": "number", "value": 0.45},
                {"key": "max_detections", "label": "Max Detections", "type": "number", "value": 20}
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
                {"key": "image_size", "label": "Image Size", "type": "number", "value": 32},
                {"key": "patch_size", "label": "Patch Size", "type": "number", "value": 4},
                {"key": "in_channels", "label": "Input Channels", "type": "number", "value": 3},
                {"key": "num_classes", "label": "Output / Classes", "type": "number", "value": 10},
                {"key": "dim", "label": "Hidden Dim", "type": "number", "value": 384},
                {"key": "depth", "label": "Depth", "type": "number", "value": 6},
                {"key": "heads", "label": "Heads", "type": "number", "value": 6},
                {"key": "engine", "label": "Vision Engine", "type": "select", "value": "Serpentine",
                 "options": ["Serpentine", "ViT", "VisionTransformer", "CNN"]},
                {"key": "kernel", "label": "Kernel", "type": "select", "value": "auto",
                 "options": ["auto", "native", "pytorch"]},
            ],
        },
        {
            "type": "rmsnorm",
            "name": "RMSNorm",
            "icon": "RMS",
            "category": "Deep Learning Core",
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
            "category": "Deep Learning Core",
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
            "category": "Deep Learning Core",
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
            "category": "Deep Learning Core",
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
        {"type":"linear","name":"Linear / Dense","icon":"LIN","category":"Deep Learning Core","description":"Linear projection layer for mapping features between dimensions.","accent":"blue","api":[]},
        {"type":"layernorm","name":"LayerNorm","icon":"LN","category":"Deep Learning Core","description":"Layer Normalization for stabilizing activations across features.","accent":"orange","api":[]},
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
        # Audio generation / representation primitives -----------------------
        {
            "type": "speaker_embedding",
            "builder_utility": True,
            "builder_python_api": False,
            "name": "Speaker Embedding",
            "icon": "SPK",
            "category": "Audio",
            "description": "Learn a speaker-conditioning vector from integer speaker IDs.",
            "accent": "purple",
            "api": [
                {"key": "num_speakers", "label": "Speakers", "type": "number", "value": 4},
                {"key": "embedding_dim", "label": "Embedding Dim", "type": "number", "value": 16}
            ],
        },
        {
            "type": "audio_codec_encoder",
            "builder_utility": True,
            "builder_python_api": False,
            "name": "Audio Codec Encoder",
            "icon": "ACE",
            "category": "Audio",
            "description": "Educational neural audio encoder that compresses a fixed waveform window into a latent vector.",
            "accent": "cyan",
            "api": [
                {"key": "latent_dim", "label": "Latent Dim", "type": "number", "value": 32},
                {"key": "hidden_channels", "label": "Hidden Channels", "type": "number", "value": 16}
            ],
        },
        {
            "type": "audio_token_predictor",
            "builder_utility": True,
            "builder_python_api": False,
            "name": "Audio Token Predictor",
            "icon": "ATP",
            "category": "Audio",
            "description": "Predict an audio latent/token state from text, style, speaker, or multimodal conditioning.",
            "accent": "blue",
            "api": [
                {"key": "in_features", "label": "Input Features", "type": "number", "value": 32},
                {"key": "latent_dim", "label": "Audio Latent Dim", "type": "number", "value": 64},
                {"key": "hidden_dim", "label": "Hidden Dim", "type": "number", "value": 64}
            ],
        },
        {
            "type": "audio_codec_decoder",
            "builder_utility": True,
            "builder_python_api": False,
            "name": "Audio Codec Decoder",
            "icon": "ACD",
            "category": "Audio",
            "description": "Educational neural decoder that converts an audio latent state into a fixed waveform window.",
            "accent": "orange",
            "api": [
                {"key": "latent_dim", "label": "Latent Dim", "type": "number", "value": 64},
                {"key": "output_samples", "label": "Output Samples", "type": "number", "value": 256},
                {"key": "hidden_dim", "label": "Hidden Dim", "type": "number", "value": 128}
            ],
        },
        {
            "type": "audio_output",
            "builder_utility": True,
            "builder_python_api": False,
            "name": "Audio Output",
            "icon": "WAV",
            "category": "Outputs",
            "description": "Waveform output for TTS, voice, music, and sound-generation graphs.",
            "accent": "green",
            "api": [
                {"key": "sample_rate", "label": "Sample Rate", "type": "number", "value": 16000}
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
            "type": "tensor_output",
            "builder_utility": True,
            "builder_python_api": False,
            "name": "Tensor Output",
            "icon": "TEN",
            "category": "Outputs",
            "description": "Generic numeric tensor output for ML, DL, math, vision, and signal graphs.",
            "accent": "green",
            "api": [],
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

