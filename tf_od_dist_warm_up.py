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
from sklearn.preprocessing import RobustScaler

class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super().__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")

        cos_annealed_lr = tf.cos(
            self.pi
            * (tf.cast(step, tf.float32) - self.warmup_steps)
            / float(self.total_steps - self.warmup_steps)
        )
        learning_rate = 0.5 * self.learning_rate_base * (1 + cos_annealed_lr)

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )

class distiller():
    def __init__(self, teacher_model_name, student_model_name, tf_records_path = "TensorFlow/workspace/training_demo/annotations/train.record", save_log = True, save_ckpt = True):
        self.start_time = time.time()
        self.datetime = datetime.now()
        self.dynamic_memory_allocation()

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
        self.delta = 0
        self.temperature = 10

        self.localization_loss_name = None
        self.classification_loss_name = None
        self.distillation_loss_name = None

        self.localization_loss = None
        self.classification_loss = None
        self.evaluation_metrics_name = None

        self.dataset = tf.compat.v1.data.TFRecordDataset(self.tf_records_path)
        self.dataset = self.dataset.map(self.parse_function).batch(1)
        self.dataset_name = tf_records_path.split("/")[-1]
        self.dataset_length = self.image_count()

        self.teacher_model = self.load_model(self.teacher_model_name, self.models_path)
        self.student_model, self.distilled_model_name = self.get_student_model(self.student_model_name, self.models_path, self.distilled_model_path, distilled_model_name = None) # Change distilled_model_name to continue training model from a specific checkpoint

        self.checkpoint_path = self.distilled_model_path + self.distilled_model_name

    def dynamic_memory_allocation(self):
        # Enable GPU dynamic memory allocation
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    def image_count(self):
        image_count = 0
        for image, (boxes, classes) in self.dataset:
            image_count += 1
        return image_count

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
    @tf.function
    def detect_fn_teacher(self, image):
        image, shapes = self.teacher_model.preprocess(image)
        prediction_dict = self.teacher_model.predict(image, shapes)

        return prediction_dict, shapes

    #@tf.function
    def detect_fn_student(self, image):
        image, shapes = self.student_model.preprocess(image)
        prediction_dict = self.student_model.predict(image, shapes)

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

    def attention_transfer(self, attention_loss):
        if "ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8" in self.student_model_name and "ssd_resnet152_v1_fpn_640x640_coco17_tpu-8" in self.teacher_model_name:
            loss = 0
            for i in range(92, 98):
                loss += attention_loss(self.teacher_model.variables[i], self.student_model.variables[i])
            for i in range(12):
                if i == 0 or i == 2 or i == 4:
                    loss += attention_loss(self.teacher_model.variables[563 + i], tf.tile(self.student_model.variables[179 + i], [1,1,2,1]))
                else:
                    loss += attention_loss(self.teacher_model.variables[563 + i], self.student_model.variables[179 + i])
            for i in range(4):
                loss += attention_loss(self.teacher_model.variables[655 + i], self.student_model.variables[271 + i])
            for i in range(4):
                loss += attention_loss(self.teacher_model.variables[969 + i], self.student_model.variables[329 + i])

            return loss
        else:
            self.print_message("Attention transfer not implemented for this models!\n")
            return 0
        
    def normalize_losses(self, distill_loss, classification_loss, localization_loss, attention_loss):
        losses = np.vstack([distill_loss, classification_loss, localization_loss, attention_loss])
        scaler = RobustScaler()
        normalized_losses = scaler.fit_transform(losses.reshape(-1, 1))
        distill_loss = tf.convert_to_tensor(normalized_losses[0])
        classification_loss = tf.convert_to_tensor(normalized_losses[1])
        localization_loss = tf.convert_to_tensor(normalized_losses[2])
        attention_loss = tf.convert_to_tensor(normalized_losses[3])

        return distill_loss, classification_loss, localization_loss, attention_loss

    def ckpt_manager(self, optimizer):
        ckpt = tf.compat.v2.train.Checkpoint(model=self.student_model, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, self.checkpoint_path, max_to_keep=5)
        # Restore the latest checkpoint if it exists
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            self.print_message(f"Restored from {ckpt_manager.latest_checkpoint}\n")
            x = 1
            while (True):
                if ckpt_manager.latest_checkpoint[-(x + 1)] == "-":
                    start_step = int(ckpt_manager.latest_checkpoint[-x:])
                    break
                x += 1
        else:
            self.print_message("Starting training from scratch.\n")
            start_step = 0

        return ckpt_manager, start_step
    
    def distill(self, epoch_num, optimizer, distillation_loss_fn, attention_loss_fn, save_every_n_epochs=10):
        self.print_message(f"\nDistillation started [{datetime.now().day}.{datetime.now().month}.{datetime.now().year}r. | {datetime.now().hour}:{datetime.now().minute}:{datetime.now().second}]...\n")
        self.print_message(f"Teacher model: {self.teacher_model_name}  || student model: {self.student_model_name}\n")
        self.print_message(f"Distilled model: {self.distilled_model_name}\n")
        start_distill = time.time()

        start_step = 0
        self.distillation_loss_name = distillation_loss_fn.name
        # Initialize checkpoint manager
        if self.save_ckpt:
            ckpt_manager, start_step = self.ckpt_manager(optimizer)

        all_epoch_losses = []
        for epoch in range(start_step, epoch_num):
            start_time = time.time()

            is_training = True
            self.student_model._is_training = is_training  # pylint: disable=protected-access
            tf.keras.backend.set_learning_phase(is_training)

            losses_dict = {
                "distill": [],
                "localization": [],
                "classification": [],
                "attention": [],
                "total": [],
                "distill_avg": 0,
                "localization_avg": 0,
                "classification_avg": 0,
                "attention_avg": 0,
                "total_avg": 0
            }
            i = 0
            current_time = datetime.now()
            for image, (boxes, classes) in tqdm(self.dataset, total=self.dataset_length, desc=f"{datetime.now().hour}:{datetime.now().minute}:{datetime.now().second} | Epoch {epoch + 1}/{epoch_num}"):
                i+=1
                teacher_prediction_dict, _ = self.detect_fn_teacher(image)
                with tf.GradientTape() as tape:
                    student_prediction_dict, shapes = self.detect_fn_student(image)
                    reshaped_boxes, reshaped_classes = self.reshape(boxes, classes)
                    
                    self.student_model.provide_groundtruth(groundtruth_boxes_list=[reshaped_boxes],
                                                        groundtruth_classes_list=[tf.cast(reshaped_classes, tf.float32)])

                    losses = self.student_model.loss(student_prediction_dict, shapes)

                    # Calculate distillation loss
                    distill_loss = distillation_loss_fn(
                        tf.nn.softmax(teacher_prediction_dict['class_predictions_with_background'] / self.temperature),
                        tf.nn.softmax(student_prediction_dict['class_predictions_with_background'] / self.temperature)
                    )

                    if self.delta != 0:
                        attention_loss = self.attention_transfer(attention_loss_fn)
                        distill_loss, losses['Loss/classification_loss'], losses['loss/localization_loss'], attention_loss = self.normalize_losses(distill_loss, losses['Loss/classification_loss'], losses['Loss/localization_loss'], attention_loss)
                    else:
                        attention_loss = 0

                    total_loss = self.alpha * distill_loss + self.beta * losses['Loss/classification_loss'] + self.gamma * losses['Loss/localization_loss'] + self.delta * attention_loss


                    losses_dict["localization"].append(losses['Loss/localization_loss'])
                    losses_dict["classification"].append(losses['Loss/classification_loss'])
                    losses_dict["attention"].append(attention_loss)
                    losses_dict["total"].append(total_loss)
                    losses_dict["distill"].append(distill_loss)

                gradients = tape.gradient(total_loss, self.student_model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.student_model.trainable_variables))

            losses_dict["distill_avg"] = np.mean(losses_dict["distill"])
            losses_dict["localization_avg"] = np.mean(losses_dict["localization"])
            losses_dict["classification_avg"] = np.mean(losses_dict["classification"])
            losses_dict["attention_avg"] = np.mean(losses_dict["attention"])
            losses_dict["total_avg"] = np.mean(losses_dict["total"])
            all_epoch_losses.append([optimizer.learning_rate.numpy(), losses_dict["distill_avg"], losses_dict["localization_avg"], losses_dict["classification_avg"], losses_dict["attention_avg"], losses_dict["total_avg"]])

            end_time = time.time()
            self.print_message(" - Time: {:.2f}s - | - Learning Rate {:.4f} - | - AVERAGE ( - Distillation loss: {:.4f} - Classification loss: {:.4f} - Localization loss: {:.4f} - Attention loss: {:.4f} - | - Total loss: {:.4f} ) \n".format(end_time - start_time, optimizer.learning_rate.numpy(), losses_dict["distill_avg"], losses_dict["classification_avg"], losses_dict["localization_avg"], losses_dict["attention_avg"], losses_dict["total_avg"]))
            self.print_message(" - | - Distillation loss: {:} - Classification loss: {:} - Localization loss: {:} - Attention loss: {:} - | - Total loss: {:} \n".format(losses_dict["distill"][-1], losses_dict["classification"][-1], losses_dict["localization_avg"][-1], losses_dict["attention"][-1], losses_dict["total"][-1]))

            
            # Save checkpoint every save_every_n_epochs epochs
            if self.save_ckpt and (epoch + 1) % save_every_n_epochs == 0:
                ckpt_save_path = ckpt_manager.save()
                self.print_message(f"Checkpoint saved at {ckpt_save_path}\n\n")

                
        end_distill = time.time()
        total_distill_time = end_distill - start_distill
        self.print_message(f"Distillation finished in {total_distill_time:.2f}s || {total_distill_time / 60:.2f}min || {total_distill_time / 3600:.2f}h\n")

        # Final save of the student model
        if self.save_ckpt:
            ckpt_save_path = ckpt_manager.save()
            self.print_message(f"Final checkpoint saved at {ckpt_save_path}\n")

        return all_epoch_losses


    def get_student_model(self, model_name, model_config_path, distilled_model_path, distilled_model_name=None):
        configs = config_util.get_configs_from_pipeline_file(model_config_path + model_name + "/pipeline.config")
        model_config = configs['model']
        model = model_builder.build(model_config=model_config, is_training=True)
        if distilled_model_name == None:
            i = 0
            while (True):
                distilled_model_name = model_name + f"_distilled_{i}"
                model_path = distilled_model_path + distilled_model_name
                other_files_exist = False
                if not os.path.exists(model_path):
                    os.makedirs(model_path, exist_ok=True)
                    shutil.copyfile(model_config_path + model_name + "/pipeline.config",
                                    model_path + "/pipeline.config")
                    break
                else:
                    for file in os.listdir(model_path):
                        if file != "pipeline.config":
                            other_files_exist = True
                            i += 1
                            break
                if other_files_exist == False:
                    break

        self.localization_loss = configs['model'].ssd.loss.localization_loss
        self.localization_loss_name = str(self.localization_loss).split(" ")[0]

        self.classification_loss = configs['model'].ssd.loss.classification_loss
        self.classification_loss_name = str(self.classification_loss).split(" ")[0]

        self.evaluation_metrics_name = configs['eval_config'].metrics_set[0]

        self.print_message(
            f"\nEvaluation metrics {self.evaluation_metrics_name}\nClassification Loss {str(self.classification_loss)}\nLocalization Loss {str(self.localization_loss)}\n")

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

    #@tf.function
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

    def distillation_init(self, epoch, alpha_n = 1, beta_n = 1, gamma_n = 1, delta_n = 0, temperature_n = 10, learning_rate_n = 0.001):

        self.alpha = alpha_n
        self.beta = beta_n
        self.gamma = gamma_n
        self.delta = delta_n
        self.temperature = temperature_n

        self.print_message(f"Chosen dataset: {self.dataset_name}\n alpha = {alpha_n} || beta = {beta_n} || gamma = {gamma_n} || temperature = {temperature_n}\n")

        all_losses = self.distill(
            epoch_num=epoch,
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_n),
            distillation_loss_fn=tf.keras.losses.KLDivergence(),
            attention_loss_fn = tf.keras.losses.MeanSquaredError(),
            save_every_n_epochs=1,
        )

        losses_names = ['Learning Rate', 'Distillation Loss', 'Localization Loss', 'Classification Loss', 'Attention Loss', 'Total Loss']
        pd.DataFrame(all_losses, columns=losses_names).to_csv(self.distilled_model_path + self.distilled_model_name + "/losses.csv", index=False)
        # Zapisanie modelu
        if self.save_ckpt:
            ckpt = tf.compat.v2.train.Checkpoint(model=self.student_model)
            ckpt.save(self.distilled_model_path + self.distilled_model_name + '/ckpt')
            self.print_message(f"Model {self.student_model_name} saved in {self.distilled_model_path + self.distilled_model_name + '/ckpt'}\n")

        return all_losses