from tf_od_dist_with_api_losses import *
from platesdetection import *
import numpy as np
import os
import argparse
from datetime import datetime


def main(start_alpha, start_beta, start_gamma, start_temperature, epoch):
    teacher_model_name = "my_ssd_resnet152_v1_fpn_640x640_coco17_tpu-8_6"
    student_model_name = "my_ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8"
    val_records_path = "TensorFlow/workspace/training_demo/annotations/val.record"
    path_to_images_dir = "TensorFlow/workspace/training_demo/images/new_data/test"
    time = datetime.now()

    alpha = start_alpha
    beta = start_beta
    gamma = start_gamma
    temperature = start_temperature
    distill_columns = ['DISTILLED MODEL', 
                       'TEACHER MODEL', 
                       'STUDENT MODEL', 
                       'ALPHA', 
                       'BETA', 
                       'GAMMA', 
                       'TEMPERATURE', 
                       "DISTILLATION LOSS",
                       "DISTILLATION LOSS VALUE", 
                       "LOCALIZATION LOSS", 
                       "LOCALIZATION LOSS VALUE",
                       "CLASSIFICATION LOSS", 
                       "CLASSIFICATION LOSS VALUE",
                       "EVALUATION METRICS",
                       "TOTAL LOSS"]

    distillation_data = []
    for alpha in np.arange(start_alpha, 1.1, 0.1):
        for beta in np.arange(start_beta, 0.6, 0.1):
            for gamma in np.arange(start_gamma, 1.1, 0.1):
                    distillation = distiller(teacher_model_name, student_model_name, val_records_path, True, False)
                    distill_data_path = "logs/distillation/distillation_data" + str(time).split(".")[0].replace(":", "-") + ".csv"
                    print(distill_data_path)
                    all_losses = distillation.distillation_init(epoch, alpha, beta, gamma, temperature)  
                    distillation_data.append([distillation.distilled_model_name, 
                                            distillation.teacher_model_name, 
                                            distillation.student_model_name, 
                                            alpha, 
                                            beta, 
                                            gamma, 
                                            temperature, 
                                            distillation.distillation_loss_name,
                                            all_losses[-1][0], 
                                            distillation.localization_loss_name,
                                            all_losses[-1][1], 
                                            distillation.classification_loss_name, 
                                            all_losses[-1][2],  
                                            distillation.evaluation_metrics_name,
                                            all_losses[-1][3]])
                    
                    pd.DataFrame(distillation_data, columns=distill_columns).to_csv(distill_data_path, index=False)

                    '''
                    if not os.path.exists(distill_data_path):
                        pd.DataFrame(distillation_data, columns=distill_columns).to_csv(distill_data_path, index=False)
                    else:
                        data = pd.read_csv(distill_data_path)
                        new_data = pd.DataFrame(distillation_data, columns=distill_columns)
                        updated_data = pd.concat([data, new_data], ignore_index=True).to_csv(distill_data_path, index=False)
                    '''
                        
                    #detector = detection(distillation.distilled_model_name, path_to_images_dir)
                    #detector.detect_all_images(ocr=False, save=True, showImages = False)


start_alpha = 1.0
start_beta = 0.1
start_gamma = 0.1
start_temperature = 10
epoch = 5
main(start_alpha, start_beta, start_gamma, start_temperature, epoch)



'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distillation parameters')
    parser.add_argument('--start_alpha', type=float, help='Start alpha value', required=True)
    parser.add_argument('--start_beta', type=float, help='Start beta value', required=True)
    parser.add_argument('--start_gamma', type=float, help='Start gamma value', required=True)
    parser.add_argument('--start_temperature', type=float, help='Start temperature value', required=True)
    parser.add_argument('--epoch', type=float, help='Epoch number', required=True)
    args = parser.parse_args()

    main(args.start_alpha, args.start_beta, args.start_gamma, args.start_temperature, args.epoch)
'''