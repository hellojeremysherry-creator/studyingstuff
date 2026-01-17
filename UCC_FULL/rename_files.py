import os
import subprocess
import re

HERE = os.path.dirname(os.path.abspath(__file__))

def git_tracked(path: str) -> bool:
    r = subprocess.run(
        ["git", "ls-files", "--error-unmatch", path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return r.returncode == 0

existing_names = set(os.listdir(HERE))

for name in list(existing_names):
    if not name.endswith(".txt"):
        continue
    if name == "rename_files.py":
        continue

    base, ext = os.path.splitext(name)

    cleaned = re.sub(r"[ \-.,;()]+", "_", base)
    cleaned = re.sub(r"_+", "_A_", cleaned).strip("_")
    new_name = cleaned + ext

    if new_name == name:
        continue

    if new_name in existing_names:
        print(f"⚠️  COLLISION SKIPPED:\n  {name}\n  → {new_name} already exists\n")
        continue

    old_path = os.path.join(HERE, name)
    new_path = os.path.join(HERE, new_name)

    rel_old = os.path.relpath(old_path, os.getcwd())
    rel_new = os.path.relpath(new_path, os.getcwd())

    print(f"Renaming:\n  {name}\n→ {new_name}\n")

    if git_tracked(rel_old):
        subprocess.run(["git", "mv", "--", rel_old, rel_new], check=True)
    else:
        os.rename(old_path, new_path)

    existing_names.remove(name)
    existing_names.add(new_name)
