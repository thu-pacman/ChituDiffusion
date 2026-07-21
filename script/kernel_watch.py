#!/usr/bin/env python3
"""Track and materialize upstream kernel repositories used for ChituDiffusion."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "kernels" / "upstreams.toml"
DEFAULT_LOCK = REPO_ROOT / "kernels" / "upstreams.lock.json"
HISTORY_DEPTH = "100"


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    branch: str
    priority: str
    paths: tuple[str, ...]
    topics: tuple[str, ...]


def run_git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def load_manifest(path: Path) -> tuple[Path, list[Source]]:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    if raw.get("schema_version") != 1:
        raise ValueError(f"Unsupported manifest schema in {path}")
    reference_root = REPO_ROOT / raw.get("reference_root", "refs/kernels")
    sources = [
        Source(
            name=item["name"],
            url=item["url"],
            branch=item.get("branch", "main"),
            priority=item.get("priority", "medium"),
            paths=tuple(item.get("paths", ())),
            topics=tuple(item.get("topics", ())),
        )
        for item in raw.get("source", ())
    ]
    names = [source.name for source in sources]
    if len(names) != len(set(names)):
        raise ValueError("Source names must be unique")
    return reference_root, sources


def load_lock(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "sources": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def remote_head(source: Source) -> str:
    output = run_git("ls-remote", source.url, f"refs/heads/{source.branch}")
    if not output:
        raise RuntimeError(f"No branch {source.branch!r} found for {source.name}")
    return output.split()[0]


def local_head(checkout: Path) -> str | None:
    if not (checkout / ".git").exists():
        return None
    return run_git("rev-parse", "HEAD", cwd=checkout)


def select_sources(sources: list[Source], names: list[str]) -> list[Source]:
    if not names:
        return sources
    by_name = {source.name: source for source in sources}
    unknown = sorted(set(names) - by_name.keys())
    if unknown:
        raise ValueError(f"Unknown sources: {', '.join(unknown)}")
    return [by_name[name] for name in names]


def scan(sources: list[Source], lock_path: Path, as_json: bool) -> int:
    locked = load_lock(lock_path).get("sources", {})
    rows = []
    for source in sources:
        head = remote_head(source)
        previous = locked.get(source.name, {}).get("commit")
        rows.append(
            {
                "name": source.name,
                "priority": source.priority,
                "branch": source.branch,
                "locked": previous,
                "remote": head,
                "changed": previous != head,
                "topics": list(source.topics),
            }
        )
    if as_json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        for row in rows:
            state = "NEW" if row["locked"] is None else ("UPDATE" if row["changed"] else "current")
            print(
                f"{row['name']:<18} {state:<7} {row['remote'][:12]} "
                f"priority={row['priority']} topics={','.join(row['topics'])}"
            )
    return int(any(row["changed"] for row in rows))


def sync(reference_root: Path, sources: list[Source]) -> None:
    reference_root.mkdir(parents=True, exist_ok=True)
    for source in sources:
        checkout = reference_root / source.name
        if not checkout.exists():
            print(f"clone {source.name} -> {checkout.relative_to(REPO_ROOT)}")
            run_git(
                "clone",
                "--filter=blob:none",
                f"--depth={HISTORY_DEPTH}",
                "--branch",
                source.branch,
                source.url,
                str(checkout),
            )
        else:
            print(f"update {source.name}")
            run_git(
                "fetch",
                "--filter=blob:none",
                f"--depth={HISTORY_DEPTH}",
                "origin",
                source.branch,
                cwd=checkout,
            )
            run_git("checkout", "--detach", "FETCH_HEAD", cwd=checkout)


def write_lock(reference_root: Path, sources: list[Source], lock_path: Path) -> None:
    old = load_lock(lock_path)
    entries = dict(old.get("sources", {}))
    now = datetime.now(timezone.utc).isoformat()
    for source in sources:
        checkout = reference_root / source.name
        commit = local_head(checkout)
        if commit is None:
            raise RuntimeError(f"{source.name} is not synced at {checkout}")
        entries[source.name] = {
            "url": source.url,
            "branch": source.branch,
            "commit": commit,
            "reviewed_at": now,
        }
    payload = {"schema_version": 1, "sources": dict(sorted(entries.items()))}
    lock_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {lock_path.relative_to(REPO_ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="compare remote heads with reviewed commits")
    scan_parser.add_argument("sources", nargs="*")
    scan_parser.add_argument("--json", action="store_true")

    sync_parser = subparsers.add_parser("sync", help="clone or update ignored reference checkouts")
    sync_parser.add_argument("sources", nargs="*")
    sync_parser.add_argument("--update-lock", action="store_true")

    lock_parser = subparsers.add_parser("lock", help="record synced commits as reviewed")
    lock_parser.add_argument("sources", nargs="*")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reference_root, all_sources = load_manifest(args.manifest.resolve())
    chosen = select_sources(all_sources, args.sources)
    if args.command == "scan":
        return scan(chosen, args.lock.resolve(), args.json)
    if args.command == "sync":
        sync(reference_root, chosen)
        if args.update_lock:
            write_lock(reference_root, chosen, args.lock.resolve())
        return 0
    if args.command == "lock":
        write_lock(reference_root, chosen, args.lock.resolve())
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"kernel-watch: {error}", file=sys.stderr)
        sys.exit(2)
