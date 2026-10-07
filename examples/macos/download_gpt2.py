#!/usr/bin/env python3
"""Download the pinned public GPT-2 checkpoint used by the M1 macOS runner."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import snapshot_download

MODEL = "openai-community/gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
WEIGHT_SHA256 = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2]/"models/macos/gpt2")
    args = parser.parse_args()
    path = Path(snapshot_download(MODEL, revision=REVISION,
        allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"], local_dir=args.output))
    with (path/"model.safetensors").open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    if digest != WEIGHT_SHA256: raise ValueError("GPT-2 checkpoint weight hash mismatch")
    (path/"checkpoint.json").write_text(json.dumps(dict(model_id=MODEL, revision=REVISION), indent=2)+"\n")
    print(path)
