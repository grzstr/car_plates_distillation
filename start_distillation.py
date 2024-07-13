from tf_od_dist_with_api_losses import *


teacher_model_name = "my_ssd_resnet152_v1_fpn_640x640_coco17_tpu-8_6"
student_model_name = "my_ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8"
epoch = 100

distillation = distiller()
distillation.distillation_init(teacher_model_name, student_model_name, epoch)