import os
import cv2
import time
import datetime
import ctypes
import logging
import multiprocessing as mp
from contextlib import closing
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from flask import Flask, Response, make_response

RTSP_URL = os.getenv('RTSP_URL')

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PB_TEXT = '/label_map.pbtext'

# Pretrained classes in the model
classNames = {0: 'background',
              1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
              7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
              13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
              18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
              24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
              32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
              37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
              41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
              46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
              51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
              56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
              61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
              67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
              75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
              80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
              86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}


def id_class_name(class_id, classes):
    for key, value in classes.items():
        if class_id == key:
            return value

model = cv2.dnn.readNetFromTensorflow(PATH_TO_CKPT, PB_TEXT)

def detect_objects(image_np, sess, detection_graph):
    image_height, image_width, _ = image_np.shape

    model.setInput(cv2.dnn.blobFromImage(image_np, size=(300, 300), swapRB=True))

    # Actual detection.
    output = model.forward()

    objects = []
    for detection in output[0, 0, :, :]:
        confidence = detection[2]
        if confidence > .5:
            object_dict = {}
            class_id = detection[1]
            class_name=id_class_name(class_id,classNames)
            object_dict[class_name] = confidence
            objects.append(object_dict)
            # print(str(str(class_id) + " " + str(detection[2])  + " " + class_name))
            box_x = detection[3] * image_width
            box_y = detection[4] * image_height
            box_width = detection[5] * image_width
            box_height = detection[6] * image_height
            cv2.rectangle(image_np, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (23, 230, 210), thickness=4)
            cv2.putText(image_np,class_name ,(int(box_x), int(box_y+.05*image_height)),cv2.FONT_HERSHEY_SIMPLEX,(.005*image_width),(0, 0, 255))

    return objects, image_np

def main():
    # capture a single frame and check the frame shape so the correct array
    # size can be allocated in memory
    video = cv2.VideoCapture(RTSP_URL)
    ret, frame = video.read()
    if ret:
        frame_shape = frame.shape
    else:
        print("Unable to capture video stream")
        exit(1)
    video.release()

    # create shared value for storing the time the frame was captured
    # note: this must be a double even though the value you are storing
    #       is a float. otherwise it stops updating the value in shared
    #       memory. probably something to do with the size of the memory block
    shared_frame_time = mp.Value('d', 0.0)
    # compute the flattened array length from the array shape
    flat_array_length = frame_shape[0] * frame_shape[1] * frame_shape[2]
    # create shared array for passing the image data from capture to detect_objects
    shared_arr = mp.Array(ctypes.c_uint16, flat_array_length)
    # create shared array for passing the image data from detect_objects to flask
    shared_output_arr = mp.Array(ctypes.c_uint16, flat_array_length)
    # create a numpy array with the image shape from the shared memory array
    # this is used by flask to output an mjpeg stream
    frame_output_arr = tonumpyarray(shared_output_arr).reshape(frame_shape)

    capture_process = mp.Process(target=fetch_frames, args=(shared_arr, shared_frame_time, frame_shape))
    capture_process.daemon = True

    detection_process = mp.Process(target=process_frames, args=(shared_arr, shared_output_arr, shared_frame_time, frame_shape))
    detection_process.daemon = True

    capture_process.start()
    print("capture_process pid ", capture_process.pid)
    detection_process.start()
    print("detection_process pid ", detection_process.pid)

    app = Flask(__name__)

    @app.route('/')
    def index():
        # return a multipart response
        return Response(imagestream(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    def imagestream():
        while True:
            # max out at 5 FPS
            time.sleep(0.2)
            # convert back to BGR
            frame_bgr = cv2.cvtColor(frame_output_arr, cv2.COLOR_RGB2BGR)
            # encode the image into a jpg
            ret, jpg = cv2.imencode('.jpg', frame_bgr)
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n\r\n')

    app.run(host='0.0.0.0', debug=False)

    capture_process.join()
    detection_process.join()

# convert shared memory array into numpy array
def tonumpyarray(mp_arr):
    return np.frombuffer(mp_arr.get_obj(), dtype=np.uint16)

# fetch the frames as fast a possible, only decoding the frames when the
# detection_process has consumed the current frame
def fetch_frames(shared_arr, shared_frame_time, frame_shape):
    # convert shared memory array into numpy and shape into image array
    arr = tonumpyarray(shared_arr).reshape(frame_shape)

    # start the video capture
    video = cv2.VideoCapture(RTSP_URL)
    # keep the buffer small so we minimize old data
    video.set(cv2.CAP_PROP_BUFFERSIZE,1)

    while True:
        # grab the frame, but dont decode it yet
        ret = video.grab()
        # snapshot the time the frame was grabbed
        frame_time = datetime.datetime.now()
        if ret:
            # if the detection_process is ready for the next frame decode it
            # otherwise skip this frame and move onto the next one
            if shared_frame_time.value == 0.0:
                # go ahead and decode the current frame
                ret, frame = video.retrieve()
                if ret:
                    # copy the frame into the numpy array
                    arr[:] = frame
                    # signal to the detection_process by setting the shared_frame_time
                    shared_frame_time.value = frame_time.timestamp()
    
    video.release()

# do the actual object detection
def process_frames(shared_arr, shared_output_arr, shared_frame_time, frame_shape):
    # shape shared input array into frame for processing
    arr = tonumpyarray(shared_arr).reshape(frame_shape)
    # shape shared output array into frame so it can be copied into
    output_arr = tonumpyarray(shared_output_arr).reshape(frame_shape)

    no_frames_available = -1
    while True:
        # if there isnt a frame ready for processing
        if shared_frame_time.value == 0.0:
            # save the first time there were no frames available
            if no_frames_available == -1:
                no_frames_available = datetime.datetime.now().timestamp()
            # if there havent been any frames available in 30 seconds, 
            # sleep to avoid using so much cpu if the camera feed is down
            if no_frames_available > 0 and (datetime.datetime.now().timestamp() - no_frames_available) > 30:
                time.sleep(1)
                print("sleeping because no frames have been available in a while")
            continue
        
        # we got a valid frame, so reset the timer
        no_frames_available = -1

        # if the frame is more than 0.5 second old, discard it
        if (datetime.datetime.now().timestamp() - shared_frame_time.value) > 0.5:
            # signal that we need a new frame
            shared_frame_time.value = 0.0
            continue
        
        # make a copy of the frame
        frame = arr.copy()
        frame_time = shared_frame_time.value
        # signal that the frame has been used so a new one will be ready
        shared_frame_time.value = 0.0

        # do the object detection
        objects, frame_overlay = detect_objects(frame, sess, detection_graph)
        # copy the output frame with the bounding boxes to the output array
        output_arr[:] = frame_overlay
        if(len(objects) > 0):
            print(objects)

if __name__ == '__main__':
    mp.freeze_support()
    main()