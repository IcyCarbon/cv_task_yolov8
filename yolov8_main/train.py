import warnings

import torch.cuda

from ultralytics import YOLO
warnings.filterwarnings('ignore')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


if __name__ == '__main__':

    model = YOLO(r'./ultralytics/cfg/models/v8/yolov8m.yaml').load('yolov8m.pt')
    torch.cuda.empty_cache()
    model.train(
                data=r'../datasets/VOC.yaml',
                cache=True,
                imgsz=640,
                epochs=200,
                #lr0=0.001,
                batch=-1,
                # close_mosaic=10,
                # workers=0,
                device='0',
                optimizer='SGD',
                )