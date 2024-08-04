from platesdetection import *
import os


model_name = "my_ssd_resnet152_v1_fpn_640x640_coco17_tpu-8_11"

URL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz"

video_path = "TensorFlow/workspace/training_demo/videos/demo.mp4"
path_to_images_dir = "TensorFlow/workspace/training_demo/images/new_data/test"

#model.download_model(URL)
detector = detection(model_name, path_to_images_dir)
detector.detect_all_images(ocr=False, save=True, showImages = False)
#detector.detect_video(video_path, False)
