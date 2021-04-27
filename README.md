# Car detection with yolo v4
This repository contains code for car detection using YOLOv4 algorithm, that was pre-trained on COCO dataset by AlexeyAB.

The yolov4 python package was implemented in TensotFlow2: https://github.com/sicara/tf2-yolov4.

Detected cars are marked by bounding boxes.

There are two main functions to preform object detection - one for video and one for still images.

The 'Photos' folder contains the photos and videos before and after car detection. It also contains a JSON file with the bounding box coordinates.




### Download weights

Download YOLOv4 weights from AlexeyAB/darknet repository 


https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

Using pip package manager install tensorflow and tf2-yolov4 from the command line:

        pip install tensorflow
        pip install tf2-yolov4

The tf2-yolov4 package includes convert-darknet-weights command which allows to convert Darknet weights to TensorFlow weights:

        convert-darknet-weights yolov4.weights -o yolov4.h5
