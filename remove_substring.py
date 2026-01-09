#!/usr/bin/env python3
# remove_substring.py
import argparse
import re
from pathlib import Path

def remove_substring_from_name(name: str, target: str, case_sensitive: bool, cleanup: bool) -> str:
    if case_sensitive:
        new = name.replace(target, "")
    else:
        new = re.sub(re.escape(target), "", name, flags=re.IGNORECASE)
    if cleanup:
        # replace runs of spaces/underscores/dashes with single underscore
        new = re.sub(r'[_\-\s]+', '_', new)
        # strip leading/trailing separators/spaces/dots
        new = new.strip('_ .-')
    return new

def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    parent = path.parent
    stem = path.stem
    suffix = path.suffix
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1

def gather_paths(folder: Path, recursive: bool, include_dirs: bool):
    if recursive:
        items = list(folder.rglob('*'))
    else:
        items = list(folder.iterdir())
    if not include_dirs:
        items = [p for p in items if p.is_file()]
    # when renaming directories, rename deeper ones first to avoid problems:
    items = sorted(items, key=lambda p: len(p.parts), reverse=True)
    return items

def main():
    p = argparse.ArgumentParser(description="Remove a substring from filenames.")
    p.add_argument("folder", type=Path, help="Target folder")
    p.add_argument("target", help="Substring to remove from filenames")
    p.add_argument("--recursive", "-r", action="store_true", help="Process subfolders")
    p.add_argument("--include-dirs", action="store_true", help="Also rename directories")
    p.add_argument("--case-insensitive", "-i", action="store_true", help="Match target ignoring case")
    p.add_argument("--cleanup", action="store_true", help="Collapse multiple separators and trim edges")
    p.add_argument("--execute", action="store_true", help="Actually perform renames (default: dry-run)")
    p.add_argument("--yes", "-y", action="store_true", help="Skip confirmation when --execute")
    args = p.parse_args()

    folder = args.folder
    if not folder.exists() or not folder.is_dir():
        print("Error: folder does not exist or is not a directory.")
        return

    items = gather_paths(folder, args.recursive, args.include_dirs)
    planned = []
    for item in items:
        new_name = remove_substring_from_name(item.name, args.target, not args.case_insensitive, args.cleanup)
        if new_name != item.name:
            new_path = item.with_name(new_name)
            if new_path.exists():
                new_path = unique_path(new_path)
            planned.append((item, new_path))

    if not planned:
        print("No filenames would change.")
        return

    print(f"Found {len(planned)} items to rename (dry-run):")
    for old, new in planned:
        print(f"{old}  ->  {new}")

    if not args.execute:
        print("\nDry-run mode (no changes made). Re-run with --execute to perform renames.")
        return

    if not args.yes:
        ans = input("\nProceed with these changes? [y/N] ").strip().lower()
        if ans not in ('y', 'yes'):
            print("Aborted by user.")
            return

    renamed = 0
    for old, new in planned:
        old.rename(new)
        renamed += 1

    print(f"\n✅ Done. Renamed {renamed} items.")

if __name__ == "__main__":
    main()

    "C:\Users\veeti\miniconda3\envs\pytorchenv\python.exe" "C:\Koodit\Gaze-Tracking-\remove_substring.py" "C:\Koodit\Gaze-Tracking-\data/validation/straight2/images" "straight2_" --execute