import os
import cv2
import shutil

def check_and_save_color_images(folder_path:str):
    """
    Check images in a folder and ensure they are saved as 3-channel color images.

    This function scans a given folder for image files and verifies if each image 
    is a 3-channel color image (BGR). If an image has fewer channels (e.g., grayscale), 
    it is converted to a 3-channel BGR format and saved in place.

    Parameters:
        folder_path (str): The path to the folder containing image files.

    Raises:
        None

    Example:
        >>> check_and_save_color_images("images/")

    Caution:
        Only image files with extensions '.png', '.jpg', '.jpeg', or '.bmp' 
        will be processed. Non-image files are ignored.

    Notes:
        - Images that are already in color format will be left unmodified.
        - The function prints messages for missing folders, failed reads, and conversions.

    See Also:
        - `cv2.imread`: Used for reading images.
        - `cv2.cvtColor`: Used to convert grayscale to BGR.
        - `cv2.imwrite`: Used to overwrite the original image with the new one.

    Warning:
        This function overwrites original grayscale images with their color-converted versions.
    """
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    # Iterate over each file in the specified folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Process only image files with specific extensions
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # Read the image using OpenCV
            img = cv2.imread(file_path)

            # Skip the file if the image couldn't be read
            if img is None:
                print(f"Failed to read image: {filename}")
                continue

            # Check if the image has 3 color channels (i.e., is a color image)
            if img.shape[2] != 3:
                print(f"Image {filename} does not have 3 channels, converting...")
                # Convert grayscale or single-channel image to a 3-channel BGR color image
                img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                # Overwrite the original file with the color version
                cv2.imwrite(file_path, img_color)
                print(f"Saved color image: {filename}")
