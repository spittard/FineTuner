"""
model.py — Thin utility module.

The original FineTuner / HuggingFace fine-tuning code has been removed.
This module now only exposes the dataset helper that is still referenced
by the CLI entry-point in the project root.
"""
import json
import os


def load_dataset_from_file(file_path: str) -> list:
    """Load a company-name dataset from a JSON file.

    Expected format: [{"Company Name": "..."}, ...]
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Dataset must be a list of dictionaries")

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} must be a dictionary")
        if "Company Name" not in item:
            raise ValueError(f"Item {i} missing 'Company Name' field")
        if not isinstance(item["Company Name"], str):
            raise ValueError(f"Item {i} 'Company Name' must be a string")

    print(f"OK: Loaded {len(data)} company name entries")
    return data
