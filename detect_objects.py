from platesdetection import *

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


#model_name = "my_ssd_resnet152_v1_fpn_640x640_coco17_tpu-8"
#model_name = "my_ssd_resnet101_v1_fpn_640x640_coco17_tpu-8"
#model_name = "my_ssd_resnet50_v1_fpn_exported_keras"
#model_name = "MobileNet_distilled"
#model_name = "ssd_mobilenet_v1_fpn_640x640_distilled_8"
#model_name = "my_ssd_resnet152_v1_fpn_640x640_coco17_tpu-8_4"
#model_name = "my_ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8_distilled_7"
#model_name = "my_ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8_distilled_15"
model_name = "my_ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8_distilled_27"


#model_name = "my_ssd_resnet50_v1_fpn"
#model_name = "my_ssd_resnet50_v1_fpn_exported"
#model_name = "my_ssd_resnet101_v1_fpn_640x640_coco17_tpu-8_3"
#model_name = "my_ssd_resnet152_v1_fpn_640x640_coco17_tpu-8_2"
#model_name = "my_ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8"
#model_name = "my_ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8"
#model_name = "my_efficientdet_d1_coco17_tpu-32"

video_path = "TensorFlow/workspace/training_demo/videos/demo.mp4"
path_to_images_dir = "TensorFlow/workspace/training_demo/images/test"
#path_to_images_dir = "TensorFlow/workspace/training_demo/images/wedding"
#path_to_images_dir = "TensorFlow/workspace/training_demo/images/fb"
#path_to_images_dir = "TensorFlow/workspace/training_demo/images/other_objects"

#model.download_model(URL)
detector = detection(model_name, path_to_images_dir)
detector.detect_all_images(ocr=False, save=True)
#detector.detect_video(video_path, False)