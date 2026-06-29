import json
import subprocess
import sys
from pathlib import Path

import yaml

from .engine import Iteron, EXPERIMENTS_DIR
from .safety import atomic_write


def cmd_init(args):
    if not args:
        print("Usage: iteron init <name>")
        sys.exit(1)
    name = args[0]
    exp_dir = EXPERIMENTS_DIR / name

    if exp_dir.exists():
        print(f"Error: experiment '{name}' already exists at {exp_dir}")
        sys.exit(1)

    exp_dir.mkdir(parents=True)

    config = {
        "name": name,
        "problem": "TODO: describe your problem",
        "budget": 50.0,
        "model": {
            "fast": "claude-3-5-haiku-latest",
            "smart": "claude-3-opus-latest",
            "provider": "anthropic",
        },
        "plateau_window": 5,
        "plateau_threshold": 0.01,
    }
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    state = {
        "phase": "cold_start",
        "round": -3,
        "best_score": 0.0,
        "best_round": -1,
        "total_cost": 0.0,
        "cpg": 0.0,
        "budget_remaining": config["budget"],
        "last_scores": [],
        "per_round_gains": [],
        "meta_suggestions": [],
        "skills_prefix": "",
    }
    atomic_write(exp_dir / "state.json", json.dumps(state, indent=2))

    (exp_dir / "journal.jsonl").touch()

    eval_sh = exp_dir / "eval.sh"
    eval_sh.write_text(
        "#!/bin/bash\n"
        "# Evaluate a solution directory\n"
        "# Usage: eval.sh <solution_dir>\n"
        "# Must output JSON: {\"score\": 0.85, \"cost\": 0.0}\n"
        'echo \'{"score": 0.0, "cost": 0.0}\'\n'
    )
    eval_sh.chmod(0o755)

    for d in ["archive/solution", "archive/refinement", "archive/execution"]:
        (exp_dir / d).mkdir(parents=True)

    try:
        subprocess.run(
            ["git", "init"], cwd=exp_dir, capture_output=True, timeout=10
        )
        (exp_dir / ".gitignore").write_text("__pycache__/\n.chronicle/\n")
        subprocess.run(
            ["git", "add", "-A"], cwd=exp_dir, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "init experiment", "--allow-empty"],
            cwd=exp_dir, capture_output=True,
        )
    except Exception as e:
        print(f"  (warning: git init failed: {e})")

    print(f"Created experiment '{name}' at {exp_dir}")
    print("  - Edit config.yaml with your problem description")
    print("  - Edit eval.sh with your evaluation script")
    print("  - Run: iteron run", name)


def cmd_run(args):
    if not args:
        print("Usage: iteron run <name>")
        sys.exit(1)
    name = args[0]
    iteron = Iteron(name)
    iteron.run()
    print(json.dumps(iteron.status(), indent=2))


def cmd_status(args):
    if not args:
        print("Usage: iteron status <name>")
        sys.exit(1)
    name = args[0]
    try:
        iteron = Iteron(name)
        print(json.dumps(iteron.status(), indent=2))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_ls(args):
    if not EXPERIMENTS_DIR.exists():
        print("No experiments found.")
        return
    dirs = [d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir()]
    if not dirs:
        print("No experiments found.")
        return
    print("Experiments:")
    for d in sorted(dirs):
        state_file = d / "state.json"
        if state_file.exists():
            try:
                state = json.loads(state_file.read_text())
                phase = state.get("phase", "?")
                score = state.get("best_score", "?")
                print(f"  {d.name:20s} phase={phase} best_score={score}")
            except (json.JSONDecodeError, OSError):
                print(f"  {d.name:20s} (corrupted state)")
        else:
            print(f"  {d.name}")


def main():
    if len(sys.argv) < 2:
        print("Usage: iteron <init|run|status|ls> [name]")
        sys.exit(1)
    cmd = sys.argv[1]
    args = sys.argv[2:]

    commands = {
        "init": cmd_init,
        "run": cmd_run,
        "status": cmd_status,
        "ls": cmd_ls,
    }
    fn = commands.get(cmd)
    if not fn:
        print(f"Unknown command: {cmd}")
        print("Usage: iteron <init|run|status|ls> [name]")
        sys.exit(1)
    fn(args)


if __name__ == "__main__":
    main()
