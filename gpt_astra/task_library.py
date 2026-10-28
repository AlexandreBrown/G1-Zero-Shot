"""Load named task prompts and optional simulation code from the task library."""
from pathlib import Path
import importlib.util
import re
import shutil

TASKS = Path(__file__).resolve().parent / "tasks"


def task_directory(name):
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", name):
        raise ValueError("Task name must use lowercase letters, numbers, '-' or '_'")
    source = TASKS / name
    prompt_file = source / "task.txt"
    if not prompt_file.is_file():
        raise ValueError(f"Unknown task {name!r}; add tasks/{name}/task.txt")
    return source


def snapshot_task(name, override, destination):
    source = task_directory(name)
    prompt_file = source / "task.txt"
    prompt = (override if override is not None else prompt_file.read_text(encoding="utf-8")).strip()
    if not prompt or len(prompt) > 10000:
        raise ValueError("Task prompt must contain 1–10000 characters")
    destination.mkdir()
    (destination / "task.txt").write_text(prompt + "\n", encoding="utf-8")
    for name in ("sim.py", "motions.py"):
        if (source / name).is_file():
            shutil.copyfile(source / name, destination / name)
    return prompt


def task_generator(directory):
    path = Path(directory) / "motions.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("gpt_astra_task_motions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_library
