import tensorflow as tf
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from datetime import datetime
import numpy as np
import time
import pandas as pd
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Change '0' to the GPU ID you want to use
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tqdm import tqdm

class distiller():
    def __init__(self, teacher_model_name, student_model_name, tf_records_path = "TensorFlow/workspace/training_demo/annotations/train.record", save_log = True, save_ckpt = True):
        self.start_time = time.time()
        self.datetime = datetime.now()

        self.save_log = save_log
        self.save_ckpt = save_ckpt

        self.label_filename = "label_map.pbtxt"

        self.tf_records_path = tf_records_path
        self.distilled_model_path = "TensorFlow/workspace/training_demo/distil_models/"
        self.models_path = "TensorFlow/workspace/training_demo/models/"
        self.path_to_labels = f"TensorFlow/workspace/training_demo/annotations/{self.label_filename}"
        self.category_index = label_map_util.create_category_index_from_labelmap(self.path_to_labels, use_display_name=True)

        self.teacher_model_name = teacher_model_name
        self.student_model_name = student_model_name

        self.alpha = 1
        self.beta = 1
        self.gamma = 1
        self.temperature = 10
        
        self.localization_loss_name = None
        self.classification_loss_name = None
        self.distillation_loss_name = None

        self.localization_loss = None
        self.classification_loss = None
        self.evaluation_metrics_name = None

        self.teacher_model = self.load_model(self.teacher_model_name, self.models_path)
        self.student_model, self.distilled_model_name = self.get_student_model(self.student_model_name, self.models_path, self.distilled_model_path, distilled_model_name = None) # Change distilled_model_name to continue training model from a specific checkpoint

        self.dataset = tf.compat.v1.data.TFRecordDataset(self.tf_records_path)
        self.dataset = self.dataset.map(self.parse_function).batch(1)
        self.dataset_name = tf_records_path.split("/")[-1]

    # Log and print messages
    def log(self,message):
        if self.save_log == True:
            log = open("logs/distillation/log_" + str(self.datetime).split(".")[0].replace(":", "-") + ".txt", "a")
            log.write(message)
            log.close()
        #pass

    def print_message(self,message):
        print(message, end = '')
        self.log(message)

    # Distillation functions

    #@tf.function
    def detect_fn(self, image, detection_model):
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)

        return prediction_dict, shapes

    @tf.function
    def reshape(self, boxes, classes):
        boxes = tf.squeeze(boxes, axis=0)  # Usunięcie zbędnego wymiaru, jeśli jest dodany
        if boxes.shape != (1, 4):
            #print_message(f"Nieoczekiwany kształt `boxes`: {boxes.shape}")
            boxes = tf.reshape(boxes[0, :2], (1, 2))
            boxes = tf.tile(boxes, [1, 2])

            classes = tf.constant([1,2], dtype=tf.int32)
            classes = tf.reshape(classes, [-1])
            classes = tf.reshape(classes[0], (1, 1))

        return boxes, classes
                    
    def distill(self, epoch_num, dataset, teacher_model, teacher_name, student_model, student_name, optimizer, distillation_loss_fn, save_every_n_epochs=10, checkpoint_path=None, alpha = 0.5, beta = 0.2, gamma = 0.2, temperature = 10):
        # Enable GPU dynamic memory allocation
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        self.print_message(f"\nDistillation started [{datetime.now().day}.{datetime.now().month}.{datetime.now().year}r. | {datetime.now().hour}:{datetime.now().minute}:{datetime.now().second}]...\n")
        self.print_message(f"Teacher model: {teacher_name}  || student model: {student_name}\n")
        self.print_message(f"Distilled model: {self.distilled_model_name}\n")
        start_distill = time.time()

        image_count = 0
        for image, (boxes, classes) in dataset:
            image_count += 1

        start_step = 0

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.distillation_loss_name = distillation_loss_fn.name

        # Initialize checkpoint manager
        if self.save_ckpt:
            if checkpoint_path:
                ckpt = tf.compat.v2.train.Checkpoint(model=student_model, optimizer=optimizer)
                ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
                # Restore the latest checkpoint if it exists
                if ckpt_manager.latest_checkpoint:
                    ckpt.restore(ckpt_manager.latest_checkpoint)
                    self.print_message(f"Restored from {ckpt_manager.latest_checkpoint}\n")
                    x = 1
                    while(True):
                        if ckpt_manager.latest_checkpoint[-(x+1)] == "-":
                            start_step = int(ckpt_manager.latest_checkpoint[-x:])
                            break
                        x += 1
                else:
                    self.print_message("Starting training from scratch.\n")
                    start_step = 0


        all_epoch_losses = []
        for epoch in range(start_step, epoch_num):
            start_time = time.time()

            is_training = True
            student_model._is_training = is_training  # pylint: disable=protected-access
            tf.keras.backend.set_learning_phase(is_training)

            losses_dict = {
                "distill": [],
                "localization": [],
                "classification": [],
                "total": [],
                "distill_avg": 0,
                "localization_avg": 0,
                "classification_avg": 0,
                "total_avg": 0
            }
            i = 0
            current_time = datetime.now()
            for image, (boxes, classes) in tqdm(dataset, total=image_count, desc=f"{datetime.now().hour}:{datetime.now().minute}:{datetime.now().second} | Epoch {epoch + 1}/{epoch_num}"):
                teacher_prediction_dict, _ = self.detect_fn(image, teacher_model)
                i+=1
                with tf.GradientTape() as tape:
                    student_prediction_dict, shapes= self.detect_fn(image, student_model)

                    reshaped_boxes, reshaped_classes = self.reshape(boxes, classes)
                    student_model.provide_groundtruth(groundtruth_boxes_list=[reshaped_boxes], groundtruth_classes_list=[tf.cast(reshaped_classes, tf.float32)])
                    
                    losses = student_model.loss(student_prediction_dict, shapes)

                    localization_loss = losses['Loss/localization_loss']
                    losses_dict["localization"].append(localization_loss)
                    classification_loss = losses['Loss/classification_loss']
                    losses_dict["classification"].append(classification_loss)

                    # Calculate distillation loss

                    distill_loss = distillation_loss_fn(
                        tf.nn.softmax(teacher_prediction_dict['class_predictions_with_background'] / self.temperature),
                        tf.nn.softmax(student_prediction_dict['class_predictions_with_background'] / self.temperature)
                    )

                    losses_dict["distill"].append(distill_loss)

                    total_loss = self.alpha * distill_loss + self.beta * classification_loss + self.gamma * localization_loss
                    losses_dict["total"].append(total_loss)

                gradients = tape.gradient(total_loss, student_model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, student_model.trainable_variables))

            losses_dict["distill_avg"] = np.mean(losses_dict["distill"])
            losses_dict["localization_avg"] = np.mean(losses_dict["localization"])
            losses_dict["classification_avg"] = np.mean(losses_dict["classification"])
            losses_dict["total_avg"] = np.mean(losses_dict["total"])
            all_epoch_losses.append([optimizer.learning_rate.numpy(), losses_dict["distill_avg"], losses_dict["localization_avg"], losses_dict["classification_avg"], losses_dict["total_avg"]])

            end_time = time.time()
            self.print_message(" - Time: {:.2f}s - | - Learning Rate {:.4f} - | - Distillation loss: {:.4f} - Classification loss: {:.4f} - Localization loss: {:.4f} - | - Total loss: {:.4f} \n".format(end_time - start_time, optimizer.learning_rate.numpy(), losses_dict["distill_avg"], losses_dict["classification_avg"], losses_dict["localization_avg"], losses_dict["total_avg"]))

            
            # Save checkpoint every save_every_n_epochs epochs
            if self.save_ckpt:
                if checkpoint_path and (epoch + 1) % save_every_n_epochs == 0:
                    ckpt_save_path = ckpt_manager.save()
                    self.print_message(f"Checkpoint saved at {ckpt_save_path}\n\n")

                
        end_distill = time.time()
        total_distill_time = end_distill - start_distill
        self.print_message(f"Distillation finished in {total_distill_time:.2f}s || {total_distill_time / 60:.2f}min || {total_distill_time / 3600:.2f}h\n")

        # Final save of the student model
        if self.save_ckpt:
            if checkpoint_path:
                ckpt_save_path = ckpt_manager.save()
                self.print_message(f"Final checkpoint saved at {ckpt_save_path}\n")


        return student_model, all_epoch_losses

    def get_student_model(self, model_name, model_config_path, distilled_model_path, distilled_model_name = None):
        configs = config_util.get_configs_from_pipeline_file(model_config_path + model_name + "/pipeline.config")
        model_config = configs['model']
        model = model_builder.build(model_config=model_config, is_training=True)
        if distilled_model_name == None:
            i=0
            while(True):
                distilled_model_name = model_name + f"_distilled_{i}"
                model_path = distilled_model_path + distilled_model_name
                other_files_exist = False
                if not os.path.exists(model_path):
                    os.makedirs(model_path, exist_ok=True)
                    shutil.copyfile(model_config_path + model_name + "/pipeline.config", model_path + "/pipeline.config")
                    break
                else:
                    for file in os.listdir(model_path):
                        if file != "pipeline.config":
                            other_files_exist = True
                            i+=1
                            break  
                if other_files_exist == False:
                    break     
                    

        self.localization_loss = configs['model'].ssd.loss.localization_loss
        self.localization_loss_name = str(self.localization_loss).split(" ")[0]

        self.classification_loss = configs['model'].ssd.loss.classification_loss
        self.classification_loss_name = str(self.classification_loss).split(" ")[0]
        
        self.evaluation_metrics_name = configs['eval_config'].metrics_set[0]

        self.print_message(f"\nEvaluation metrics {self.evaluation_metrics_name}\nClassification Loss {str(self.classification_loss)}\nLocalization Loss {str(self.localization_loss)}\n")

        return model, distilled_model_name

    def find_last_checkpoint(self, model_path):
        ckpt_files = os.listdir(model_path)
        ckpt_numbers = []
        for ckpt in ckpt_files:
            if ckpt.startswith("ckpt-"):
                ckpt_numbers.append(int(ckpt.split(".")[0].split("-")[1]))
        last_ckpt_number = str(max(ckpt_numbers))
        last_ckpt = "ckpt-" + last_ckpt_number
        return last_ckpt

    def load_model(self, model_name, path_to_model_dir):
        path_to_model_dir += model_name
        tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

        # Enable GPU dynamic memory allocation
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        self.print_message("\n[Loading TF2 Checkpoint]\n")
        self.print_message(f'Loading model - {model_name}... \n')
        start_time = time.time()

        # Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(path_to_model_dir + "/pipeline.config")
        model_config = configs['model']
        
        detection_model = model_builder.build(model_config=model_config, is_training=False)

        last_ckpt = self.find_last_checkpoint(path_to_model_dir)

        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(path_to_model_dir + "/" + last_ckpt).expect_partial()

        end_time = time.time()
        elapsed_time = end_time - start_time
        self.print_message(f'The model has been loaded ({last_ckpt}) - {elapsed_time:.2f}s\n\n\n')

        return detection_model

    def parse_function(self, example_proto):
        feature_description = {
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/source_id': tf.io.FixedLenFeature([], tf.string),
            'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
            'image/object/class/label': tf.io.VarLenFeature(tf.int64),
        }
        example = tf.io.parse_single_example(example_proto, feature_description)
        image = tf.io.decode_jpeg(example['image/encoded'], channels=3)
        image = tf.image.resize(image, [640, 640])
        xmin = tf.sparse.to_dense(example['image/object/bbox/xmin'])
        xmax = tf.sparse.to_dense(example['image/object/bbox/xmax'])
        ymin = tf.sparse.to_dense(example['image/object/bbox/ymin'])
        ymax = tf.sparse.to_dense(example['image/object/bbox/ymax'])
        classes = tf.sparse.to_dense(example['image/object/class/label'])

        boxes = tf.stack([ymin, xmin, ymax, xmax], axis=1)
        return image, (boxes, classes)

    def distillation_init(self, epoch, alpha_n = 0.5, beta_n = 0.5, gamma_n = 0.5, temperature_n = 10):
        
        self.print_message(f"Chosen dataset: {self.dataset_name}\n alpha = {alpha_n} || beta = {beta_n} || gamma = {gamma_n} || temperature = {temperature_n}\n")

        model, all_losses = self.distill(
            epoch_num=epoch,
            dataset=self.dataset,
            teacher_model=self.teacher_model,
            teacher_name=self.teacher_model_name,
            student_model=self.student_model,
            student_name=self.student_model_name,
            optimizer=tf.keras.optimizers.Adam(),
            distillation_loss_fn=tf.keras.losses.KLDivergence(),
            save_every_n_epochs=1,
            checkpoint_path= self.distilled_model_path + self.distilled_model_name,
            alpha = alpha_n,
            beta = beta_n,
            gamma = gamma_n,
            temperature = temperature_n 
        )

        losses_names = ['Learning Rate', 'Distillation Loss', 'Localization Loss', 'Classification Loss', 'Total Loss']
        pd.DataFrame(all_losses, columns=losses_names).to_csv(self.distilled_model_path + self.distilled_model_name + "/losses.csv", index=False)
        # Zapisanie modelu
        if self.save_ckpt:
            ckpt = tf.compat.v2.train.Checkpoint(model=model)
            ckpt.save(self.distilled_model_path + self.distilled_model_name + '/ckpt')
            self.print_message(f"Model {self.student_model_name} saved in {self.distilled_model_path + self.distilled_model_name + '/ckpt'}\n")

        return all_losses