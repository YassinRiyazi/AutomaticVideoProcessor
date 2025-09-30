import cv2
import os
import numpy as np
import tqdm
import glob
images = glob.glob(r"D:\Videos\S1_30per_T1_C001H001S0001\frames_rotated\*.png")

for address in images:
    image = cv2.imread(address, cv2.IMREAD_UNCHANGED)
    image = image[:,40:-40]
    image = cv2.resize(image, (1245,130))  # Resize to match YOLO input size
    # cv2.imshow("a",image)
    # cv2.waitKey(1)
    cv2.imwrite(address,image)