import tensorflow as tf
import time
from datetime import datetime
import os
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from tensorflow.python.keras.utils.data_utils import get_file
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import easyocr


class ModelLoader:
    def __init__(self):
        self.start_time = datetime.now()
        self.save_log = True
        self.path_to_model_dir = "TensorFlow/workspace/training_demo/models/"
        self.downloaded_models_dir = "TensorFlow/workspace/training_demo/"
        self.exported_model_dir = "TensorFlow/workspace/training_demo/exported-models/"
        self.ckpt_dict = {"my_ssd_resnet50_v1_fpn":'/ckpt-31',
                          "my_ssd_resnet101_v1_fpn_640x640_coco17_tpu-8":"/ckpt-28",
                          "my_ssd_resnet152_v1_fpn_640x640_coco17_tpu-8":"/ckpt-26",
                          "my_ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8":"/ckpt-26",
                          "my_ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8": "/ckpt-51"}
  

    def log(self, message):
        if self.save_log == True:
            log = open("logs/log_" + str(self.start_time) + ".txt", "a")
            log.write(message)
            log.close()

    def print_message(self, message):
        print(message, end = '')
        self.log(message)

    def load_from_saved_model(self, model_name):
        self.print_message("\n[Loading TF2 Saved Model]\n")
        self.print_message(f'Loading model - {model_name}...\n')
        start_time = time.time()

        detection_model = tf.saved_model.load(self.exported_model_dir + f"{model_name}/saved_model")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'The model has been loaded - {elapsed_time:.2f}s\n')

        return detection_model


    def load_from_ckpt(self, model_name):
        self.path_to_model_dir += model_name
        tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

        # Enable GPU dynamic memory allocation
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
       
        self.print_message("\n[Loading TF2 Checkpoint]\n")
        self.print_message(f'Loading model - {model_name}... \n')
        start_time = time.time()

        # Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(self.path_to_model_dir + "/pipeline.config")
        model_config = configs['model']
        
        detection_model = model_builder.build(model_config=model_config, is_training=False)

        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(self.path_to_model_dir + self.ckpt_dict[model_name]).expect_partial()

        end_time = time.time()
        elapsed_time = end_time - start_time
        self.print_message(f'The model has been loaded - {elapsed_time:.2f}s\n\n')

        return detection_model
    
    def load_model(self, model_name):
        if model_name[-9:] == "_exported":
            detection_model = self.load_from_saved_model(model_name)
        else:
            detection_model = self.load_from_ckpt(model_name)

        return detection_model

    def download_model(self, URL):
        file_name = URL.split("/")[-1]
        model_name = file_name.split(".")[0]
        get_file(fname=file_name, origin=URL, cache_subdir = "pre-trained_models", cache_dir=self.downloaded_models_dir, extract=True)
        self.print_message(f"Model {model_name} downloaded successfully!\n")




class detection:
    def __init__(self, model_name):
        self.model = ModelLoader()
        self.detection_model = self.model.load_model(model_name)
        self.model_name =  model_name

        self.label_filename = "label_map.pbtxt"

        #self.path_to_images_dir = "TensorFlow/workspace/training_demo/images/test"
        #self.path_to_images_dir = "TensorFlow/workspace/training_demo/images/wedding"
        self.path_to_images_dir = "TensorFlow/workspace/training_demo/images/fb"
        #self.path_to_images_dir = "TensorFlow/workspace/training_demo/images/other_objects"
        self.path_to_labels = f"TensorFlow/workspace/training_demo/annotations/{self.label_filename}"

        self.image_names = self.find_images_names()

        self.region_threshold = 0.3
        self.detection_threshold = 0.3

        

    def find_images_names(self):
        images_names = []
        
        for image_name in os.listdir(self.path_to_images_dir):
            if image_name.endswith(('.bmp', '.jpg', '.png', '.jpeg')):
                images_names.append(image_name)
        
        return images_names

    def filter_text(self, region, ocr_result, region_threshold):
        rectangle_size = region.shape[0]*region.shape[1]
        
        plate = [] 
        for result in ocr_result:
            length = np.sum(np.subtract(result[0][1], result[0][0]))
            height = np.sum(np.subtract(result[0][2], result[0][1]))
            
            if length*height / rectangle_size > region_threshold:
                plate.append(result[1])
        return plate

    def ocr(self, image, detections, detection_threshold, region_threshold):
        
        # Scores, boxes and classes above threhold
        scores = list(filter(lambda x: x> detection_threshold, detections['detection_scores']))
        boxes = detections['detection_boxes'][:len(scores)]
        classes = detections['detection_classes'][:len(scores)]
        
        # Full image dimensions
        width = image.shape[1]
        height = image.shape[0]
        
        # Apply ROI filtering and OCR
        if len(scores) > 0:
            for idx, box in enumerate(boxes):
                roi = box*[height, width, height, width]
                region = image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]

                reader = easyocr.Reader(['en'])
                ocr_result = reader.readtext(region)
                
                text = self.filter_text(region, ocr_result, region_threshold)
                #plt.imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
                #plt.show()
                #self.model.print_message(f"text = {text}\n")
                return text, region
        else:
            return [""], None
    
    #****************************************
    # OBJECT DETECTION
    #****************************************

    @tf.function
    def detect_fn(self, image):
        """Detect objects in image."""

        image, shapes = self.detection_model.preprocess(image)
        prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)

        return detections

    def detect_object(self, image_np):
        category_index = label_map_util.create_category_index_from_labelmap(self.path_to_labels, use_display_name=True)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

        if model_name[-9:] == "_exported":
            # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
            input_tensor = tf.convert_to_tensor(image_np)
            # The model expects a batch of images, so add an axis with `tf.newaxis`.
            input_tensor = input_tensor[tf.newaxis, ...]
            detections = self.detection_model(input_tensor)
        else:
            detections = self.detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.30,
                agnostic_mode=False)

        return image_np_with_detections, detections

    def find_plate(self, image_name, ocr = True):
        start_detection = time.time()
        image_path = self.path_to_images_dir + '/' + image_name
        self.model.print_message(f'{image_name}... ')
        image, detections = self.detect_object(np.array(cv2.imread(image_path)))
        end_detection = time.time()
        self.model.print_message(f' || Detection time: {(end_detection-start_detection):.2f}s')

        if ocr == True:
        #OCR
            start_ocr = time.time()
            text, region = self.ocr(image, detections, self.detection_threshold, self.region_threshold)
            end_ocr = time.time()
            self.model.print_message(f' || OCR time: {(end_ocr - start_ocr):.2f}s || {text}\n')

            imageText = ""
            if len(text) > 0: 
                for word in text:
                    imageText += word
            cv2.putText(image, imageText, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            imageText = image_name + " || " + imageText
        else:
            imageText = image_name
       
        return image, imageText

    def detect_image(self, image_name, ocr = True):
        image, imageText = self.find_plate(image_name, ocr)
        plt.figure(num=imageText)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()

    def detect_all_images(self, ocr = True):
        plt.rcParams['figure.max_open_warning'] = len(self.image_names) + 1
        start_time = time.time()
        for image_name in self.image_names:
            image, imageText = self.find_plate(image_name, ocr)
            plt.figure(num=imageText)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        end_time = time.time()
        self.model.print_message(f"Number of images: {len(self.image_names)} || Total time: {(end_time-start_time):.2f}s\n")
        plt.show()          

    def detect_video(self, video_path, ocr=True):
        cap = cv2.VideoCapture(video_path)
        category_index = label_map_util.create_category_index_from_labelmap(self.path_to_labels, use_display_name=True)
        if not cap.isOpened():
            self.mode.print_message("Error opening video file")
            return
        
        (success, image) = cap.read()

        startTime = 0

        while success:
            currentTime = time.time()

            fps = 1/(currentTime - startTime)
            startTime = currentTime
            image_with_detections, detections = self.detect_object(image)
            if ocr == True:
                text, region = self.ocr(image, detections, self.detection_threshold, self.region_threshold)
                imageText = ""
                if len(text) > 0: 
                    for word in text:
                        imageText += word
                cv2.putText(image_with_detections, imageText, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image_with_detections, "FPS: " +  str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
            cv2.imshow("Result", image_with_detections)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            (success, image) = cap.read()       



#SSD MobileNet V2 320x320
#URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz'
#SSD MobileNet V1 FPN 640x640
#URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz'
#SSD MobileNet V2 FPNLite 640x640
#URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz'
#CenterNet MobileNet V2 FPN
#URL = 'http://download.tensorflow.org/models/object_detection/tf2/20210210/centernet_mobilenetv2fpn_512x512_coco17_od.tar.gz'

#EfficientDet D4 1024x1024
#URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz'
#EfficientDet D1 640x640
#URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz'

#Faster R-CNN ResNet101 640x640
#URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_640x640_coco17_tpu-8.tar.gz'
#CenterNet ResNet50 V1 FPN 512x512
#URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v1_fpn_512x512_coco17_tpu-8.tar.gz'
#CenterNet ResNet50 V2 512x512

#SSD ResNet101 640x640
#URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz'

#URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v2_512x512_coco17_tpu-8.tar.gz'
#SSD ResNet152 V1 FPN 640x640
#URL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_640x640_coco17_tpu-8.tar.gz"
#Faster R-CNN Inception ResNet V2 640x640
#URL = "http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_14_10_2017.tar.gz"


#model = ModelLoader()
model_name = "my_ssd_resnet50_v1_fpn"
#model_name = "my_ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8"
#model_name = "my_ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8"
#model_name = "my_ssd_resnet152_v1_fpn_640x640_coco17_tpu-8"
#model_name = "my_ssd_resnet101_v1_fpn_640x640_coco17_tpu-8"
#model_name = "my_ssd_resnet50_v1_fpn_exported"

video_path = "TensorFlow/workspace/training_demo/videos/demo.mp4"


#model.download_model(URL)
detector = detection(model_name)
detector.detect_all_images()
#detector.detect_video(video_path, False)