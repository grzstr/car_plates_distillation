
from tf_od_dist import *


teacher_model_name = "my_ssd_resnet152_v1_fpn_640x640_coco17_tpu-8_11"
student_model_name = "my_ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8"

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

distillation = distiller(teacher_model_name, student_model_name, batch_size = 1)


distillation.distillation_init(epoch = 25,
                               alpha_n = 0.7,
                               beta_n = 0.3,
                               gamma_n = 0.1,
                               delta_n = 0.1,
                               temperature_n = 10,
                               learning_rate_n = 0.001)

                               