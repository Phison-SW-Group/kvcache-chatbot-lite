#!/usr/bin/env python3
import argparse
from pathlib import Path

def clean_blank_lines(content: str) -> str:
    """
    Remove tabs/spaces from lines that are otherwise blank.
    """
    lines = content.split("\n")
    cleaned_lines = [
        "" if line.strip() == "" else line.rstrip()  # Only remove trailing whitespace
        for line in lines
    ]
    result = "\n".join(cleaned_lines)

    # Ensure file ends with exactly one newline
    if result and not result.endswith("\n"):
        result += "\n"

    return result

def process_file(file_path: Path):
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return

    # Skip if not a file
    if not file_path.is_file():
        return

    try:
        original_content = file_path.read_text(encoding="utf-8")
        cleaned_content = clean_blank_lines(original_content)

        file_path.write_text(cleaned_content, encoding="utf-8")
        print(f"✅ Cleaned and saved: {file_path}")
    except UnicodeDecodeError:
        print(f"⚠️  Skipped (binary file): {file_path}")
    except Exception as e:
        print(f"⚠️  Error processing {file_path}: {e}")

def collect_files(paths, pattern=None, exclude_dirs=None):
    """
    Collect all files from given paths (can be files or directories).

    Args:
        paths: List of file or directory paths
        pattern: File pattern to match (e.g., "*.py")
        exclude_dirs: List of directory names to exclude (e.g., ["__pycache__", ".git"])
    """
    if exclude_dirs is None:
        exclude_dirs = ["__pycache__", ".git", ".venv", "venv", "node_modules"]

    files = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"⚠️  Path not found: {p}")
            continue

        if p.is_file():
            files.append(p)
        elif p.is_dir():
            # Recursively find all files in directory
            if pattern:
                found_files = p.rglob(pattern)
            else:
                found_files = p.rglob("*")

            for f in found_files:
                if f.is_file() and not any(excl in f.parts for excl in exclude_dirs):
                    files.append(f)

    return files

def main():
    parser = argparse.ArgumentParser(
        description="Remove tabs/spaces from otherwise blank lines in files or directories."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more file or directory paths to process"
    )
    parser.add_argument(
        "-p", "--pattern",
        help="File pattern to match (e.g., '*.py'). Only used for directories.",
        default=None
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        help="Directory names to exclude (default: __pycache__, .git, .venv, venv, node_modules)",
        default=None
    )

    args = parser.parse_args()

    files = collect_files(args.paths, args.pattern, args.exclude)

    if not files:
        print("❌ No files found to process")
        return

    print(f"📝 Found {len(files)} file(s) to process\n")

    for file in files:
        process_file(file)

if __name__ == "__main__":
    main()
