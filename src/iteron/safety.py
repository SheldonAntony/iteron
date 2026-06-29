import hashlib
import os
import shutil
from pathlib import Path


class EvalContractViolation(RuntimeError):
    pass


class EvalContract:
    def __init__(self, exp_dir: Path):
        self.exp_dir = exp_dir
        self.contract_path = exp_dir / "eval.sh"
        self._hash_path = exp_dir / ".eval_hash"
        self.expected_hash = self._load_or_create_hash()

    def _compute_hash(self) -> str:
        return hashlib.sha256(self.contract_path.read_bytes()).hexdigest()

    def _load_or_create_hash(self) -> str:
        if self._hash_path.exists():
            return self._hash_path.read_text().strip()
        h = self._compute_hash()
        atomic_write(self._hash_path, h)
        return h

    def verify(self):
        current = self._compute_hash()
        if current != self.expected_hash:
            raise EvalContractViolation(
                f"eval.sh modified since init. "
                f"Re-run `iteron init` or revert changes."
            )


class Chronicle:
    def __init__(self, exp_dir: Path):
        self.exp_dir = exp_dir
        self.snapshot_dir = exp_dir / ".chronicle"
        self.snapshot_dir.mkdir(exist_ok=True)

    def snapshot(self, name: str):
        target = self.snapshot_dir / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(
            self.exp_dir,
            target,
            symlinks=True,
            ignore=shutil.ignore_patterns(".chronicle", ".git", "__pycache__"),
        )

    def restore(self, name: str) -> bool:
        target = self.snapshot_dir / name
        if not target.exists():
            return False
        keep = {".chronicle", ".git"}
        for item in list(self.exp_dir.iterdir()):
            if item.name not in keep:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        for item in target.iterdir():
            if item.name not in keep:
                dst = self.exp_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dst, symlinks=True)
                else:
                    shutil.copy2(item, dst, follow_symlinks=False)
        return True


def atomic_write(path: Path, content: str):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content)
    tmp.rename(path)
