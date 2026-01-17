import os
import subprocess
import re

HERE = os.path.dirname(os.path.abspath(__file__))

for name in os.listdir(HERE):
    if not name.endswith(".txt"):
        continue

    if name == "rename_files.py":
        continue

    base, ext = os.path.splitext(name)

    # Replace spaces, dashes, periods, semicolons with underscores
    cleaned = re.sub(r"[ \-.;]+", "_", base)

    # Collapse multiple underscores
    cleaned = re.sub(r"_+", "_", cleaned)

    # Strip leading/trailing underscores (optional but nice)
    cleaned = cleaned.strip("_")

    new_name = cleaned + ext

    if new_name != name:
        old_path = os.path.join(HERE, name)
        new_path = os.path.join(HERE, new_name)

        print(f"Renaming:\n  {name}\n→ {new_name}\n")
        subprocess.run(["git", "mv", "--", old_path, new_path], check=True)
