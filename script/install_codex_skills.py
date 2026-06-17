#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SKILLS_DIR = REPO_ROOT / ".codex" / "skills"


def codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")).expanduser()


def install_skill(source: Path, target_dir: Path, *, copy: bool, force: bool) -> str:
    target = target_dir / source.name
    if target.exists() or target.is_symlink():
        if not force:
            return f"skip {source.name}: already exists"
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)

    if copy:
        shutil.copytree(source, target)
        return f"copy {source.name} -> {target}"

    target.symlink_to(source.resolve(), target_is_directory=True)
    return f"link {source.name} -> {target}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Install repository Codex skills into CODEX_HOME.")
    parser.add_argument("--skills-dir", type=Path, default=codex_home() / "skills")
    parser.add_argument("--copy", action="store_true", help="Copy skills instead of symlinking them.")
    parser.add_argument("--force", action="store_true", help="Replace existing installed skills with the same names.")
    args = parser.parse_args()

    if not REPO_SKILLS_DIR.exists():
        raise SystemExit(f"Repository skills directory not found: {REPO_SKILLS_DIR}")

    skills = sorted(path for path in REPO_SKILLS_DIR.iterdir() if (path / "SKILL.md").exists())
    if not skills:
        raise SystemExit(f"No skills found under {REPO_SKILLS_DIR}")

    target_dir = args.skills_dir.expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)

    for skill in skills:
        print(install_skill(skill, target_dir, copy=args.copy, force=args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
