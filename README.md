# AutomaticVideoProcessor
Automatic Video Processor: Supposed to be the ultimate tool for processing high speed footage for drop detection.

Rules:
    Each Module is functional for single image/folder
    All assumption and exception handling in passed inside the \__init__



>[!IMPORTANT]
> Change scaleDown value for drop detection from 5 to 1.



Changelog:
    25.04 :
        [] In the "BaseUtils/Detection/DropDetection_Sum.py:Main()" the detection was failing for images for no reason, I decided to ditch fast detectionb for now, Used YOLO heavily. 
        [] Saving the drop boundry in a np array
