# By Lihi Gur-Arie, 27.4.21
# https://lindevs.com/yolov4-object-detection-using-tensorflow-2/
# https://github.com/sicara/tf2-yolov4/blob/master/notebooks/YoloV4_Dectection_Example.ipynb

import tensorflow as tf
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4
import cv2
import numpy as np

#################################################

def load_image_to_tensor(file):
    # load image
    image = tf.io.read_file(file)
    # detect format (JPEG, PNG, BMP, or GIF) and converts to Tensor:
    image = tf.io.decode_image(image)
    return image

def resize_image(image):
    # Resize the output_image:
    image = tf.image.resize(image, (HEIGHT, WIDTH))
    # Add a batch dim:
    images = tf.expand_dims(image, axis=0)/255
    return images

def get_image_from_plot():
    # crates a numpy array from the output_image of the plot\figure
    canvas = FigureCanvasAgg(Figure())
    canvas.draw()
    return np.fromstring(canvas.tostring_rgb(), dtype='uint8')

def trained_yolov4_model():
    # load trained yolov4 model
    model = YOLOv4(
        input_shape=(HEIGHT, WIDTH, 3),
        anchors=YOLOV4_ANCHORS,
        num_classes=80,
        training=False,
        yolo_max_boxes=20,
        yolo_iou_threshold=0.5,
        yolo_score_threshold=0.73,
    )
    model.load_weights('yolov4.h5')
    return model

def detected_photo(boxes, scores, classes, detections,image):
    boxes = (boxes[0] * [WIDTH, HEIGHT, WIDTH, HEIGHT]).astype(int)
    scores = scores[0]
    classes = classes[0].astype(int)
    detections = detections[0]

    CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
        'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
        'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    ########################################################################


    image_cv = image.numpy()

    for (xmin, ymin, xmax, ymax), score, class_idx in zip(boxes, scores, classes):

        if score > 0:
            if class_idx == 2:         # show bounding box only to the "car" class

                #### Draw a rectangle ##################
                # convert from tf.Tensor to numpy
                cv2.rectangle(image_cv, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,255,0), thickness= 2)
                # Add detection text to the prediction
                text = CLASSES[class_idx] + ': {0:.2f}'.format(score)
                cv2.putText(image_cv, text, (int(xmin), int(ymin) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    return image_cv

def proccess_frame(photo, model):
    images = resize_image(photo)
    boxes, scores, classes, detections = model.predict(images)
    result_img = detected_photo(boxes, scores, classes, detections,images[0])
    return result_img

def Car_detection_single_photo(input_photo):
    my_image = load_image_to_tensor(input_photo)
    yolo_model = trained_yolov4_model()
    image = proccess_frame(my_image, yolo_model)
    image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert from BGR to RGB
    return image

def Car_detection_video(input_video_name, output_video_name, frames_to_save = 50):

    # load trained yolov4 model
    model = trained_yolov4_model()

    # load video
    my_video = cv2.VideoCapture(input_video_name)

    # write resulted frames to file
    out = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'MJPG'), 30, (WIDTH ,HEIGHT))

    success = 1
    i = 0
    while success and i < frames_to_save:                                 # While there are more frames in the video
        # function extract frames
        success, image = my_video.read()                                  # extract a frame
        if success:
            result_img = proccess_frame(tf.convert_to_tensor(image), model)   # tag cars on the frame

            out.write((result_img*255).astype('uint8'))                                             # write resulted frame to the video file
            i = i + 1
            print(i)
    out.release()                                                         # Close the video writer

#######   main   ####################################################################

if __name__ == "__main__":

    WIDTH, HEIGHT = (1024, 768)

    ####    Detect Cars on a single photo ####

    output_image = Car_detection_single_photo(input_photo ='photos/test3.jpg')

    # Show resulted photo
    cv2.imshow('output_image', output_image)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

    # Save photo
    cv2.imwrite('photos/test3_object_detected.jpg', output_image*255)

####   Detect Cars on a video and save ######
    Car_detection_video(input_video_name='photos/car_chase_01.mp4', output_video_name ='photos/delete.avi', frames_to_save = 20)
