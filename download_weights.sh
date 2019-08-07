#!/bin/bash
# Download weights for pose_hrnet_w48_384x288
[ -f hrnet_weights.pth ] || gdown --id 1UoJhTtjHNByZSm96W3yFTfU5upJnsKiS -O hrnet_weights.pth
# Download weights for vanilla YOLOv3
wget -nc -P models/detectors/yolo/weights/ https://pjreddie.com/media/files/yolov3.weights
# Download weights for backbone network
wget -nc -P models/detectors/yolo/weights/ https://pjreddie.com/media/files/darknet53.conv.74
