#!/usr/bin/env python3
"""Download the exact public checkpoint used by the native Asahi validation."""
import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2]/"models/asahi/qwen2.5-0.5b")
    args = parser.parse_args()
    path = Path(snapshot_download(MODEL, revision=REVISION,
        allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"], local_dir=args.output))
    (path/"checkpoint.json").write_text(json.dumps(dict(model_id=MODEL, revision=REVISION), indent=2)+"\n")
    print(path)
