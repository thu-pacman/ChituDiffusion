#!/usr/bin/env python3
import argparse
from pathlib import Path


def write_local_config(source_config: Path, output_config: Path, model_path: Path) -> None:
    text = source_config.read_text(encoding="utf-8")
    lines = []
    replaced = False
    for line in text.splitlines():
        if line.startswith("model_name_or_path:"):
            lines.append(f'model_name_or_path: "{model_path.resolve()}"')
            replaced = True
        else:
            lines.append(line)
    if not replaced:
        lines.append(f'model_name_or_path: "{model_path.resolve()}"')
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert_checkpoint(source: Path, output: Path) -> None:
    from safetensors.torch import load_file, save_file

    state = load_file(str(source), device="cpu")
    converted = {}
    for key, value in state.items():
        if key.startswith("model.embed_tokens."):
            new_key = "model.language_model.embed_tokens." + key.removeprefix("model.embed_tokens.")
        elif key.startswith("model.layers."):
            new_key = "model.language_model.layers." + key.removeprefix("model.layers.")
        elif key.startswith("model.norm."):
            new_key = "model.language_model.norm." + key.removeprefix("model.norm.")
        elif key.startswith("visual."):
            new_key = "model.visual." + key.removeprefix("visual.")
        else:
            new_key = key
        converted[new_key] = value
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(output))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    local_config = output_dir / "HPSv3_7B.local.yaml"
    compat_checkpoint = output_dir / "HPSv3.chitu_compat.safetensors"
    write_local_config(Path(args.source_config).resolve(), local_config, Path(args.model_path).resolve())
    convert_checkpoint(Path(args.source_checkpoint).resolve(), compat_checkpoint)
    print(local_config)
    print(compat_checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
