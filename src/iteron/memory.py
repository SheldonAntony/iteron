import json
from pathlib import Path
from typing import Optional


def atomic_write(path: Path, content: str):
    # ponytail: duplicated from safety.py; DMF is standalone (extractable library)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content)
    tmp.rename(path)


class DMFTier:
    def __init__(self, path: Path):
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self):
        for f in self.path.iterdir():
            if f.suffix == ".json":
                self._data[f.stem] = json.loads(f.read_text())

    def put(self, key: str, value: dict):
        self._data[key] = value
        atomic_write(self.path / f"{key}.json", json.dumps(value, indent=2))

    def get(self, key: str, default: Optional[dict] = None) -> Optional[dict]:
        return self._data.get(key, default)

    def all(self) -> dict[str, dict]:
        return dict(self._data)

    def keys(self) -> list[str]:
        return list(self._data.keys())

    def __len__(self):
        return len(self._data)


class DMF:
    def __init__(self, archive_dir: Path):
        self.solution = DMFTier(archive_dir / "solution")
        self.refinement = DMFTier(archive_dir / "refinement")
        self.execution = DMFTier(archive_dir / "execution")
