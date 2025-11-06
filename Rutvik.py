import  BaseLine
import  Utilities
import  CaMeasurer
import os

if __name__ == "__main__":
    # bld = BaseLine.BaseLine()

    _folder = r"D:\Temp\tif"

    # bld.Forward(experiment = _folder)

    # YOLO = Utilities.YoloWalker(num_workers=1)
    S4 = CaMeasurer.processes_mp_shared(num_workers=1)

    Utilities.crop_Save(image_folder=_folder)


    os.makedirs(os.path.join(_folder, 'SR_edge'), exist_ok=True)

    # Phase 4: 4S-SROF
    # TODO: Share resources [Done] CaMeasurer.processes_mp(_folder, num_workers=10)
    S4.run(_folder)