from __future__ import annotations

import fnmatch
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


EXCLUDE_DIRS = {
    ".git",
    ".cursor",
    ".pytest_cache",
    "__pycache__",
    "build",
    "build-headless",
    "dist",
    "results",
    "logs",
    "figures",
    "data",
    "temp",
}
EXCLUDE_GLOBS = {"*.pyc", "*.pyo", "*.o", "*.so", "*.a", "*.tmp", "*.log"}
EXCLUDE_FILES = {".gitignore", ".cursorignore", ".rayignore"}


def _should_skip(name: str) -> bool:
    if name in EXCLUDE_FILES:
        return True
    if name in EXCLUDE_DIRS:
        return True
    return any(fnmatch.fnmatch(name, pattern) for pattern in EXCLUDE_GLOBS)


@dataclass
class RuntimeWorkingDir:
    path: Path
    _temp_dir: tempfile.TemporaryDirectory | None = field(repr=False, default=None)

    def cleanup(self) -> None:
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None


def stage_runtime_working_dir(
    repo_root: str | os.PathLike[str],
    *,
    binary_path: str | os.PathLike[str] | None = None,
) -> RuntimeWorkingDir:
    repo_root = Path(repo_root).expanduser().resolve()
    binary = Path(binary_path).expanduser().resolve() if binary_path else repo_root / "build" / "dragonchess"

    if not binary.is_file():
        raise FileNotFoundError(
            f"Headless binary not found at {binary}. Build it locally before starting Ray."
        )

    temp_dir = tempfile.TemporaryDirectory(prefix="dragonchess-ray-")
    staged_root = Path(temp_dir.name)

    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [directory for directory in dirs if not _should_skip(directory)]
        rel_path = Path(root).resolve().relative_to(repo_root)
        target_dir = staged_root / rel_path
        target_dir.mkdir(parents=True, exist_ok=True)

        for file_name in files:
            if _should_skip(file_name):
                continue
            src = Path(root) / file_name
            dst = target_dir / file_name
            shutil.copy2(src, dst)

    staged_build_dir = staged_root / "build"
    staged_build_dir.mkdir(parents=True, exist_ok=True)
    staged_binary = staged_build_dir / "dragonchess"
    shutil.copy2(binary, staged_binary)
    staged_binary.chmod(0o755)

    # Ray honors ignore files when packaging working_dir. The source repo's
    # .gitignore excludes build/, which would silently drop the staged binary.
    # Emit an empty .rayignore so the already-pruned temp tree is packaged as-is.
    (staged_root / ".rayignore").write_text("", encoding="utf-8")

    return RuntimeWorkingDir(path=staged_root, _temp_dir=temp_dir)
