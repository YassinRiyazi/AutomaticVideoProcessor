"""
    Date    : 2025-09-30
    Author  : Yassin Riyazi
    Project : Automatic Video Processor (AVP)
    File    : BaseLine/__init__.py
    Version : 1.0.0
    License : GNU GENERAL PUBLIC LICENSE Version 3

    Components:
        + fit_and_rotate_image ( 
                         image_path: os.PathLike,
                         experiment: str = None,
                         results: bool = True,
                         focus_ratio: float = 0.3 ): tuple[float, tuple, np.ndarray]

        + Forward () [Rotating and saving the result of an folder] 

    Assumptions:
        FrameExtractor major assumptions:
            frames are extracted and existing in the folder ( if not, use FrameExtractor/FrameGenerator )
            frames are in grayscale (if not, use FrameExtractor/FrameBWMaker)
            frames are not corrupted (if not, use FrameExtractor/FrameHealthChecker)
            frames are in order and there is no missing frames (if not, use FrameExtractor/FrameHealthChecker)
"""
if __name__ == "__main__":
    from BaseNormalizer import folderBaseLineNormalizer
else:
    from .BaseNormalizer import folderBaseLineNormalizer

class BaseLine:
    """
    TODO:
        - Check 5 linearly spread images for calculating the angle
        
    Description:
        BaseLine class provides methods to fit and rotate images based on detected lines. 
    """
    def __init__(self) -> None:
        pass

    def Forward(self, experiment: str
                             ) -> None:
        folderBaseLineNormalizer(experiment = experiment)