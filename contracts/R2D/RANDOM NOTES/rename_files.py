import os
import subprocess
import re

HERE = os.path.dirname(os.path.abspath(__file__))

for name in os.listdir(HERE):
    if not name.endswith(".txt"):
        continue

    # Skip the script itself just in case
    if name == "rename_files.py":
        continue

    # Replace one or more spaces with a single underscore
    new_name = re.sub(r"\s+", "_", name)

    if new_name != name:
        old_path = os.path.join(HERE, name)
        new_path = os.path.join(HERE, new_name)

        print(f"Renaming:\n  {name}\n→ {new_name}\n")
        subprocess.run(["git", "mv", "--", old_path, new_path], check=True)
