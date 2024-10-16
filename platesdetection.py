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
        self.downloaded_models_dir = "TensorFlow/workspace/training_demo/pre-trained_models/"
        self.exported_model_dir = "TensorFlow/workspace/training_demo/exported-models/"
        self.distilled_model_path = "TensorFlow/workspace/training_demo/distil_models/"

    def log(self, message):
        if self.save_log == True:
            log = open("logs/detection/log_" + str(self.start_time).split(".")[0].replace(":", "-") + ".txt", "a")
            log.write(message)
            log.close()
        #pass

    def print_message(self, message):
        print(message, end = '')
        self.log(message)

    def find_last_checkpoint(self, model_path):
        ckpt_files = os.listdir(model_path)
        ckpt_numbers = []
        for ckpt in ckpt_files:
            if ckpt.startswith("ckpt-"):
                ckpt_numbers.append(int(ckpt.split(".")[0].split("-")[1]))
        last_ckpt_number = str(max(ckpt_numbers))
        last_ckpt = "ckpt-" + last_ckpt_number
        return last_ckpt

    def load_from_saved_model(self, model_name):
        self.print_message("\n[Loading TF2 Saved Model]\n")
        self.print_message(f'Loading model - {model_name}...\n')
        start_time = time.time()
        if model_name[-9:] == "_exported":
            model_path = self.exported_model_dir + f"{model_name}/saved_model"
        else:
            model_path = self.exported_model_dir + f"{model_name}"

        detection_model = tf.saved_model.load(model_path)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'The model has been loaded - {elapsed_time:.2f}s\n')

        return detection_model

    def load_from_saved_model_keras(self, model_name):
        self.print_message("\n[Loading TF2 Saved Model]\n")
        self.print_message(f'Loading model - {model_name}...\n')
        start_time = time.time()

        detection_model = tf.saved_model.load(self.distilled_model_path + f"{model_name}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'The model has been loaded - {elapsed_time:.2f}s\n')

        return detection_model

    def load_from_ckpt(self, model_name):
        if "distilled" in model_name:
            model_path = self.distilled_model_path
        else:
            model_path = self.path_to_model_dir

        model_path += model_name
        tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

        # Enable GPU dynamic memory allocation
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
       
        self.print_message("\n[Loading TF2 Checkpoint]\n")
        self.print_message(f'Loading model - {model_name}... \n')
        start_time = time.time()

        # Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(model_path + "/pipeline.config")
        model_config = configs['model']
        
        detection_model = model_builder.build(model_config=model_config, is_training=False)

        # Restore checkpoint
        last_ckpt = self.find_last_checkpoint(model_path)
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(model_path + "/" + last_ckpt).expect_partial()

        end_time = time.time()
        elapsed_time = end_time - start_time
        self.print_message(f'The model has been loaded ({last_ckpt}) - {elapsed_time:.2f}s\n\n')

        return detection_model
    
    def load_model(self, model_name):
        if model_name[-9:] == "_exported" or model_name[-6:] == "_keras":
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
    def __init__(self, model_name, path_to_images):
        self.model = ModelLoader()
        self.detection_model = self.model.load_model(model_name)
        self.model_name =  model_name

        self.label_filename = "label_map.pbtxt"
        self.path_to_images_dir = path_to_images

        self.path_to_labels = f"TensorFlow/workspace/training_demo/annotations/{self.label_filename}"

        self.image_names = self.find_images_names()

        self.region_threshold = 0.3
        self.detection_threshold = 0.3

    def extract_number(self, filename):
        return int(filename.split("Cars")[1].split(".")[0])

    def find_images_names(self):
        '''
        images_names = []
        
        for image_name in os.listdir(self.path_to_images_dir):
            if image_name.endswith(('.bmp', '.jpg', '.png', '.jpeg')):
                images_names.append(image_name)
        sorted_names = sorted(images_names, key=self.extract_number)
        return sorted_names
        '''
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

        if self.model_name[-9:] == "_exported" or self.model_name[-6:] == "_keras":
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
            self.model.print_message(f' || OCR time: {(end_ocr - start_ocr):.2f}s || {text}')

            imageText = ""
            if len(text) > 0: 
                for word in text:
                    imageText += word
            cv2.putText(image, imageText, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            imageText = image_name.split(".")[0] + " || " + imageText + "." + image_name.split(".")[1]
        else:
            imageText = image_name
       
        return image, imageText

    def detect_image(self, image_name, ocr = True, save = False):
        self.model.print_message("\n")
        image, imageText = self.find_plate(image_name, ocr)
        plt.figure(num=imageText)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()
        if save:
            results_dir = os.path.join("results", self.model_name)
            if not os.path.exists(results_dir):
                os.makedirs(results_dir, exist_ok=True)
            output_path = os.path.join(results_dir, imageText + ".jpg")
            cv2.imwrite(output_path, image)

    def detect_all_images(self, ocr = True, save = False, showImages = False):
        plt.rcParams['figure.max_open_warning'] = len(self.image_names) + 1
        start_time = time.time()
        i = 0
        for image_name in self.image_names:
            i += 1
            self.model.print_message(f"\n[{i}/{len(self.image_names)}] - ")
            image, imageText = self.find_plate(image_name, ocr)
            if showImages:
                plt.figure(num=imageText)
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if save:
                results_dir = os.path.join("results", self.model_name)
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir, exist_ok=True)
                output_path = os.path.join(results_dir, imageText)
                cv2.imwrite(output_path, image)
        end_time = time.time()
        self.model.print_message(f"\nNumber of images: {len(self.image_names)} || Total time: {(end_time-start_time):.2f}s\n")
        if showImages:
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

