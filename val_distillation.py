from tf_od_dist_with_api_losses import *
import numpy as np

teacher_model_name = "my_ssd_resnet152_v1_fpn_640x640_coco17_tpu-8_6"
student_model_name = "my_ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8"
val_records_path = "TensorFlow/workspace/training_demo/annotations/val.record"
epoch = 100


distillation = distiller()

alpha = start_alpha = 0.1
beta = start_beta = 0.1
gamma = start_gamma = 0.1
temperature = start_temperature = 10



for alpha in np.arange(start_alpha, 1.1, 0.1):
    for beta in np.arange(start_beta, 1.1, 0.1):
        for gamma in np.arange(start_gamma, 1.1, 0.1):
                distillation.distillation_init(teacher_model_name, student_model_name, epoch, val_records_path, alpha, beta, gamma, temperature)


