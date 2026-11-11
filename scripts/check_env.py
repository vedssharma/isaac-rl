#!/usr/bin/env python3
"""Check that the API keys in .env are present and actually work.

Usage:
    pip install -r requirements-local.txt
    python scripts/check_env.py            # check everything
    python scripts/check_env.py --offline  # only check that the variables are set

Each service gets one cheap, read-only authenticated call. Secrets are never printed.
Exit code is 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED = {
    "Daytona": ["DAYTONA_API_KEY"],
    "W&B": ["WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT"],
    "Hugging Face": ["HF_TOKEN", "HF_REPO_ID"],
}


def _mask(value: str) -> str:
    return f"{value[:4]}…({len(value)} chars)" if len(value) > 8 else "(set)"


def check_vars() -> list[str]:
    missing = []
    for service, names in REQUIRED.items():
        for name in names:
            value = os.environ.get(name, "").strip()
            if value:
                print(f"  ok      {name:<16} {_mask(value) if 'KEY' in name or 'TOKEN' in name else value}")
            else:
                print(f"  MISSING {name:<16} ({service})")
                missing.append(name)
    return missing


def check_daytona() -> str:
    from daytona import Daytona, ListSandboxesQuery

    client = Daytona()  # reads DAYTONA_API_KEY / DAYTONA_API_URL / DAYTONA_TARGET
    sandboxes = list(client.list(ListSandboxesQuery(limit=50)))
    return f"authenticated; {len(sandboxes)} sandbox(es) currently exist"


def check_wandb() -> str:
    import wandb

    api = wandb.Api(timeout=30)
    user = api.viewer
    entity = os.environ["WANDB_ENTITY"]
    teams = set(getattr(user, "teams", None) or [])
    note = "" if entity == user.username or entity in teams else f" (warning: '{entity}' is not your user or one of your teams)"
    return f"logged in as {user.username}{note}"


def check_hf() -> str:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    api = HfApi(token=os.environ["HF_TOKEN"])
    who = api.whoami()
    repo_id = os.environ["HF_REPO_ID"]
    try:
        api.repo_info(repo_id, repo_type="model")
        repo_note = f"repo {repo_id} exists"
    except RepositoryNotFoundError:
        repo_note = f"repo {repo_id} not found yet (create it with: hf repos create {repo_id} --private)"
    return f"logged in as {who.get('name')}; {repo_note}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-file", default=str(REPO_ROOT / ".env"))
    parser.add_argument("--offline", action="store_true", help="only check that variables are set")
    args = parser.parse_args()

    env_file = Path(args.env_file)
    if env_file.exists():
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)
        print(f"Loaded {env_file}")
    else:
        print(f"No {env_file} found; using the shell environment. (cp .env.example .env to start one.)")

    print("\nVariables:")
    missing = check_vars()
    failed = bool(missing)

    if not args.offline:
        print("\nLive checks:")
        checks = {"Daytona": check_daytona, "W&B": check_wandb, "Hugging Face": check_hf}
        for service, fn in checks.items():
            if any(name in missing for name in REQUIRED[service]):
                print(f"  skip    {service:<13} (missing variables)")
                continue
            try:
                print(f"  ok      {service:<13} {fn()}")
            except Exception as exc:  # report every failure, keep going
                failed = True
                print(f"  FAIL    {service:<13} {type(exc).__name__}: {exc}")

    print("\nAll good." if not failed else "\nSome checks failed; see above.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
