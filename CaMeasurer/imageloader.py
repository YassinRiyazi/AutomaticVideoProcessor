import os
def load_files(ad,_natsort=False):
    """
    Load and return a naturally sorted list of image file names from a directory.

    This function scans the specified directory and collects the names of all files
    with valid image file extensions. It then returns the list in natural (human-friendly) order.

    Parameters:
        ad (str): The path to the directory containing image files.

    Returns:
        List[str]: A naturally sorted list of image file names with valid extensions.

    Example:
        >>> load_files("/path/to/images")
        ['image1.png', 'image2.png', 'image10.png']

    Caution:
        Files without an extension or with an unsupported extension are ignored silently.

    Notes:
        - Supported extensions include: 'tiff', 'tif', 'png', 'jpg', 'jpeg', 'bmp', 'gif', 'webp'.
        - Sorting is done using `natsort.natsorted` to ensure human-friendly ordering (e.g., 1, 2, 10).

    See Also:
        - `os.listdir`: For listing files in a directory.
        - `natsort.natsorted`: For performing natural sorting.

    Warning:
        Only file names are returned (not full paths). You must join with `ad` if you need full paths.

    Aurhor:
        - Sajjad Shumaly
    """
    valid_extensions = {"tiff", "tif", "png", "jpg", "jpeg", "bmp", "gif", "webp"}  # Common image formats
    FileNames = []
    for file in sorted(os.listdir(ad)):
        try:
            if file.split(".")[-1].lower() in valid_extensions:
                FileNames.append(file)
        except IndexError:
            pass
    if _natsort:
        import natsort
        return natsort.natsorted(FileNames)
    else:
        return sorted(FileNames)

def read_four_integers(file_path):
    """
    Read exactly four integers from the first line of a text file.

    This function opens the specified file, reads the first line, splits it by whitespace,
    and attempts to parse exactly four integers. If the line does not contain four integers,
    a ValueError is raised.

    Parameters:
        file_path (str): Path to the input text file containing integers.

    Returns:
        List[int]: A list of four integers read from the file.

    Raises:
        ValueError: If the file does not contain exactly four integers on the first line.
        IOError: If the file cannot be opened.

    Example:
        File content:
        12 34 56 78

        >>> read_four_integers("input.txt")
        [12, 34, 56, 78]

    Caution:
        Only the first line is read; remaining lines are ignored.

    Notes:
        - Extra spaces or tabs between numbers are allowed.
        - Non-integer values will raise a ValueError during conversion.
    """
    with open(file_path, 'r') as file:
        line = file.readline()
        numbers = list(map(int, line.strip().split()))
        if len(numbers) != 4:
            raise ValueError("The file does not contain exactly four integers.")
        return numbers
