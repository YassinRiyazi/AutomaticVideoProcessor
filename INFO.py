import os

def count_python_lines(root_dir: str) -> int:
    """
    Count the total number of lines in all Python (.py) files 
    under the given root directory (including sub-directories).

    Args:
        root_dir (str): Path to start searching for .py files.

    Returns:
        int: Total number of lines across all Python files.
    """
    total_lines = 0
    file_count = 0

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".py"):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        line_count = sum(1 for _ in f)
                        total_lines += line_count
                        file_count += 1
                        print(f"{file_path}: {line_count} lines")
                except Exception as e:
                    print(f"⚠️ Could not read {file_path}: {e}")

    print(f"\n📊 Total Python files: {file_count}")
    print(f"📏 Total lines of Python code: {total_lines}")
    return total_lines


if __name__ == "__main__":
    # Change "." to the path of your project root
    project_root = "."
    count_python_lines(project_root)
