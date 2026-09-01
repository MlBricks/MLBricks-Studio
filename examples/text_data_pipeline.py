from mlb_studio.data import (
    load_huggingface_dataset,
    process_text_dataset,
    train_test_split,
    tokenize_text_dataset,
)

dataset = load_huggingface_dataset(
    "roneneldan/TinyStories",
    split="train",
    text_column="text",
    max_rows=10000,
)

dataset = process_text_dataset(
    dataset,
    text_column="text",
    normalize_whitespace=True,
    remove_empty=True,
)

splits = train_test_split(
    dataset,
    train_size=0.9,
    test_size=0.1,
    seed=42,
)

tokenized = tokenize_text_dataset(
    splits,
    tokenizer_name="gpt2",
    text_column="text",
    context_length=512,
)

print(tokenized)
