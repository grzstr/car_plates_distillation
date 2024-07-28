#from tf_od_dist_with_api_losses_attention import *
#from tf_od_dist_with_attention import *
from tf_od_dist_warm_up2 import *
#import matplotlib.pyplot as plt

teacher_model_name = "my_ssd_resnet152_v1_fpn_640x640_coco17_tpu-8_11"
student_model_name = "my_ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8"

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

'''
learning_rate_schedule = WarmUpCosine(learning_rate_base = 0.03999999910593033,
                                      total_steps = 25000,
                                      warmup_learning_rate =  0.013333000242710114,
                                      warmup_steps = 2000)

                                      '''
"""
lrs = [learning_rate_schedule(step) for step in range(25000)]
plt.plot(lrs)
plt.xlabel("Step", fontsize=14)
plt.ylabel("LR", fontsize=14)
plt.show()
"""

distillation = distiller(teacher_model_name, student_model_name)


distillation.distillation_init(epoch = 25,
                               alpha_n = 0.7,
                               beta_n = 0.3,
                               gamma_n = 0.1,
                               delta_n = 0.1,
                               temperature_n = 10,
                               learning_rate_n = 0.001)

                               
'''
distillation.distillation_init(epoch = 100,
                               alpha_n = 0.7,
                               beta_n = 0.3,
                               gamma_n = 0.1,
                               temperature_n = 10)
'''