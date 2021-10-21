import os
import time

import cv2 as cv

import detect
from utils.datasets import LoadImages

opt = detect.parse_opt()
opt.view_img = True
opt.nosave = True
opt.save_txt = False
yolo = detect.Yolov5(**vars(opt))
ds = LoadImages('D:/pythonproject/yolov5/resource/test1/', img_size=yolo.imgsz, stride=yolo.stride,
                auto=yolo.pt)
lines = yolo.getObjectAix(ds)
