#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a CLIP vision model for GaussianWAM Stage 1.")
    parser.add_argument("--model", default="openai/clip-vit-base-patch16")
    parser.add_argument("--output-dir", default="/data/zijianzhang/clip-vit-base-patch16")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--endpoint", default=None, help="Optional Hugging Face endpoint mirror, e.g. https://hf-mirror.com")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint

    try:
        from transformers import CLIPImageProcessor, CLIPVisionModel
    except ImportError as exc:
        raise SystemExit(
            "transformers is not installed in this environment. Run this with the FastWAM environment, "
            "or install transformers there first."
        ) from exc

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = CLIPImageProcessor.from_pretrained(args.model, revision=args.revision, force_download=args.force)
    model = CLIPVisionModel.from_pretrained(args.model, revision=args.revision, force_download=args.force)
    processor.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

    CLIPImageProcessor.from_pretrained(str(output_dir), local_files_only=True)
    CLIPVisionModel.from_pretrained(str(output_dir), local_files_only=True)

    print(f"Saved {args.model} to {output_dir}")
    print(f"Use config override: --override clip.model_name={output_dir}")


if __name__ == "__main__":
    main()
