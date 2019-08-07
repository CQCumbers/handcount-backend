# Hand Count Backend
> Count raised hands using deep learning pose estimation

Based on [simple-HRNet](https://github.com/stefanopini/simple-HRNet), this flask app takes uploaded images and runs yolo for person detection and hrnet for pose estimation, then returns the position of each detected head and each raised hand.
