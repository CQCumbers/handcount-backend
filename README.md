# Hand Count Backend
> Count raised hands using pose estimation

Based on [simple-HRNet](https://github.com/stefanopini/simple-HRNet), this flask app takes uploaded images and runs yolo for person detection and hrnet for pose estimation, then returns the position of each detected head and each raised hand. Uploaded images are also saved to help create a more representative dataset for further training. It works surprisingly well given no task-specific training, but is very much a proof of concept and should not be relied upon where accuracy is important.
