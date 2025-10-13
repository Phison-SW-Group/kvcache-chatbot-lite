# Clean Blank Lines Script

A utility script to remove trailing whitespace from files.

## What It Does

This script cleans up two types of whitespace issues:

1. **Blank lines with whitespace** - Lines that contain only spaces/tabs
   ```
   Before: "    \n"  (4 spaces + newline)
   After:  "\n"      (pure newline)
   ```

2. **Trailing whitespace** - Spaces/tabs at the end of code lines
   ```
   Before: "def hello():    \n"  (code + 4 spaces + newline)
   After:  "def hello():\n"      (code + newline)
   ```

## Basic Usage

### Process Individual Files

```bash
# Single file
python scripts/clean_blank_lines.py file.py

# Multiple files
python scripts/clean_blank_lines.py file1.py file2.py file3.txt
```

### Process Directories

```bash
# Process all files in a directory
python scripts/clean_blank_lines.py src/api

# Process multiple directories
python scripts/clean_blank_lines.py src/api src/web
```

### Filter by File Pattern

```bash
# Only process Python files
python scripts/clean_blank_lines.py src/api -p "*.py"

# Only process text files
python scripts/clean_blank_lines.py testcases/docs -p "*.txt"

# Process specific file types
python scripts/clean_blank_lines.py src -p "*.md"
```

### Exclude Directories

```bash
# Exclude specific directories (default: __pycache__, .git, .venv, venv, node_modules)
python scripts/clean_blank_lines.py src --exclude __pycache__ logs uploads

# Custom exclusions
python scripts/clean_blank_lines.py . -p "*.py" --exclude .venv __pycache__ build dist
```

## Advanced Usage

### Combine with Git Commands

```bash
# Process all files in git status (modified + untracked)
python scripts/clean_blank_lines.py $(git status --short | awk '{print $2}')

# Process only modified files
python scripts/clean_blank_lines.py $(git diff --name-only)

# Process only untracked files
python scripts/clean_blank_lines.py $(git ls-files --others --exclude-standard)

# Process all changed Python files
python scripts/clean_blank_lines.py $(git status --short | awk '{print $2}' | grep '\.py$')
```

### Combine with Find Command

```bash
# Find all Python files in a directory
python scripts/clean_blank_lines.py $(find src/api -type f -name "*.py")

# Exclude __pycache__ directories
python scripts/clean_blank_lines.py $(find src -type f -name "*.py" ! -path "*/__pycache__/*")

# Multiple exclusions
python scripts/clean_blank_lines.py $(find src -type f -name "*.py" \
  ! -path "*/__pycache__/*" \
  ! -path "*/logs/*" \
  ! -path "*/.venv/*")

# Find files in specific subdirectories
python scripts/clean_blank_lines.py $(find src/api/routers src/api/services -type f -name "*.py")
```

### Mix Files and Directories

```bash
# Process specific files and entire directories together
python scripts/clean_blank_lines.py README.md src/api/main.py src/web
```

## Command-Line Options

```
usage: clean_blank_lines.py [-h] [-p PATTERN] [--exclude [EXCLUDE ...]] paths [paths ...]

positional arguments:
  paths                 One or more file or directory paths to process

options:
  -h, --help            Show this help message and exit
  -p PATTERN, --pattern PATTERN
                        File pattern to match (e.g., '*.py'). Only used for directories.
  --exclude [EXCLUDE ...]
                        Directory names to exclude (default: __pycache__, .git, .venv, venv, node_modules)
```

## Output Examples

### Successful Processing

```
📝 Found 15 file(s) to process

✅ Cleaned and saved: src/api/main.py
✅ Cleaned and saved: src/api/config.py
✅ Cleaned and saved: src/api/routers/upload.py
...
```

### Binary File Detection

```
⚠️  Skipped (binary file): src/api/__pycache__/main.cpython-313.pyc
```

### Error Handling

```
⚠️  Path not found: invalid/path.py
⚠️  Error processing file.txt: Permission denied
```

## Default Exclusions

The script automatically excludes these directories when processing:

- `__pycache__` - Python bytecode cache
- `.git` - Git repository data
- `.venv` / `venv` - Python virtual environments
- `node_modules` - Node.js dependencies

## Tips

1. **Preview before processing**: Use `echo` to preview which files will be processed
   ```bash
   echo $(find src -type f -name "*.py" ! -path "*/__pycache__/*")
   ```

2. **Process only text files**: The script automatically skips binary files (like `.pyc`, images, etc.)

3. **Recursive by default**: When given a directory, the script processes all matching files recursively

4. **Safe to run multiple times**: Running the script on already-cleaned files is safe and has no effect

## Why Remove Trailing Whitespace?

1. **Version control clarity** - Git shows invisible whitespace changes as diffs
2. **Code standards** - Most style guides require no trailing whitespace
3. **Editor warnings** - Many IDEs highlight trailing whitespace as issues
4. **Cleaner codebase** - Reduces unnecessary characters

## Related Git Workflow

After cleaning files, commit with:

```bash
git add .
git commit -m "[STYLE] Clean trailing whitespace"
```

## Error Handling

The script handles various error scenarios gracefully:

- **Binary files**: Automatically skipped with warning
- **Missing files**: Displays error message and continues
- **Permission errors**: Shows error and continues with remaining files
- **Non-file paths**: Automatically skipped (e.g., directories passed as files)

