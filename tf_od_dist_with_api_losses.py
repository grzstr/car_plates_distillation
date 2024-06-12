import tensorflow as tf
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import dataset_util
import numpy as np
import cv2
import time
import os
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tqdm import tqdm



#@tf.function
def detect_fn(image, detection_model):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, shapes, image

def distill(epoch_num, dataset, teacher_model, teacher_name, student_model, student_name, optimizer, classification_loss_fn, localization_loss_fn, distillation_loss_fn, save_every_n_epochs=10, checkpoint_path=None):
    # Enable GPU dynamic memory allocation
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print("\nDistillation started...")
    print(f"Teacher model: {teacher_name}  || student model: {student_name}")
    start_distill = time.time()

    temperature = 10
    alpha = 0.5

    image_count = 0
    for image, (boxes, classes) in dataset:
        image_count += 1

    # Initialize checkpoint manager
    if checkpoint_path:
        ckpt = tf.compat.v2.train.Checkpoint(model=student_model, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
        # Restore the latest checkpoint if it exists
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print(f"Restored from {ckpt_manager.latest_checkpoint}")
            x = 1
            while(True):
                if ckpt_manager.latest_checkpoint[-(x+1)] == "-":
                    restored_epoch = int(ckpt_manager.latest_checkpoint[-x:])
                    break
                x += 1
            restore = True
        else:
            print("Starting training from scratch.")
            restore = False

    all_epoch_losses = []
    for epoch in range(epoch_num):
        start_time = time.time()
        
        if restore == True:
            epoch = restored_epoch
            restore = False

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
        for image, (boxes, classes) in tqdm(dataset, total=image_count, desc=f"Epoch {epoch + 1}/{epoch_num}"):
            teacher_detections, teacher_prediction_dict, _, _ = detect_fn(image, teacher_model)
            i+=1
            with tf.GradientTape() as tape:
                student_detections, student_prediction_dict, shapes, pre_image = detect_fn(image, student_model)

                boxes = tf.squeeze(boxes, axis=0)  # Usunięcie zbędnego wymiaru, jeśli jest dodany

                #class_batch_size, num_classes_per_image = tf.shape(classes)[0].numpy(), tf.shape(classes)[1].numpy()
                #if i == 7:
                #   boxes = tf.tile(boxes, [3, 1])
                #shapes = tf.tile(shapes, [1, 2])
                '''
                if boxes.shape != (1, 4):
                    print(f"Nieoczekiwany kształt `boxes`: {boxes.shape}")
                    if boxes.shape == (2, 4):
                        # Przekształcenie z (2, 4) na (1, 2)
                        boxes = tf.reshape(boxes[0, :2], (1, 2))
                        boxes = tf.tile(boxes, [1, 2])

                    if boxes.shape == (1, 2):
                        # Przekształcenie z (1, 2) na (3, 2) - dodanie brakujących wartości
                        boxes = tf.concat((boxes, tf.zeros((2, 2))), axis=0)   
                '''
                
                if boxes.shape != (1, 4):
                    #print(f"Nieoczekiwany kształt `boxes`: {boxes.shape}")
                    boxes = tf.reshape(boxes[0, :2], (1, 2))
                    boxes = tf.tile(boxes, [1, 2])

                    classes = tf.constant([1,2], dtype=tf.int32)
                    classes = tf.reshape(classes, [-1])
                    classes = tf.reshape(classes[0], (1, 1))
                    



                student_model.provide_groundtruth(groundtruth_boxes_list=[boxes], groundtruth_classes_list=[tf.cast(classes, tf.float32)])
                
                losses = student_model.loss(student_prediction_dict, shapes)

                localization_loss = losses['Loss/localization_loss']
                losses_dict["localization"].append(localization_loss)
                classification_loss = losses['Loss/classification_loss']
                losses_dict["classification"].append(classification_loss)

                # Calculate distillation loss

                distill_loss = distillation_loss_fn(
                    tf.nn.softmax(teacher_prediction_dict['class_predictions_with_background'] / temperature),
                    tf.nn.softmax(student_prediction_dict['class_predictions_with_background'] / temperature)
                )

                losses_dict["distill"].append(distill_loss)

                total_loss = alpha * distill_loss + (1 - alpha) * (classification_loss + localization_loss)
                losses_dict["total"].append(total_loss)

            gradients = tape.gradient(total_loss, student_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, student_model.trainable_variables))

        losses_dict["distill_avg"] = np.mean(losses_dict["distill"])
        losses_dict["localization_avg"] = np.mean(losses_dict["localization"])
        losses_dict["classification_avg"] = np.mean(losses_dict["classification"])
        losses_dict["total_avg"] = np.mean(losses_dict["total"])
        all_epoch_losses.append(losses_dict["total_avg"])

        end_time = time.time()
        print(" - Time: {:.2f}s - | - Distillation loss: {:.4f} - Classification loss: {:.4f} - Localization loss: {:.4f} - | - Total loss: {:.4f} ".format(end_time - start_time, losses_dict["distill_avg"], losses_dict["classification_avg"], losses_dict["localization_avg"], losses_dict["total_avg"]))

        # Save checkpoint every save_every_n_epochs epochs
        if checkpoint_path and (epoch + 1) % save_every_n_epochs == 0:
            ckpt_save_path = ckpt_manager.save()
            print(f"Checkpoint saved at {ckpt_save_path}\n")

    end_distill = time.time()
    total_distill_time = end_distill - start_distill
    print(f"Distillation finished in {total_distill_time:.2f}s || {total_distill_time / 60:.2f}min || {total_distill_time / 3600:.2f}h")

    # Final save of the student model
    if checkpoint_path:
        ckpt_save_path = ckpt_manager.save()
        print(f"Final checkpoint saved at {ckpt_save_path}")

    return student_model, all_epoch_losses

def get_student_model(model_name, model_config_path, distilled_model_path):
    configs = config_util.get_configs_from_pipeline_file(model_config_path + model_name + "/pipeline.config")
    model_config = configs['model']
    model = model_builder.build(model_config=model_config, is_training=True)
    distilled_model_name = model_name + f"_distilled_16"
    """
    i=0
    while(True):
        distilled_model_name = model_name + f"_distilled_{i}"
        model_path = distilled_model_path + distilled_model_name
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
            shutil.copyfile(model_config_path + model_name + "/pipeline.config", model_path + "/pipeline.config")
            break
        else:
            i+=1
    """
    return model, distilled_model_name

def load_model(model_name, path_to_model_dir):
    ckpt_dict = {"my_ssd_resnet50_v1_fpn":'/ckpt-31',
                    "my_ssd_resnet101_v1_fpn_640x640_coco17_tpu-8":"/ckpt-28",
                    "my_ssd_resnet101_v1_fpn_640x640_coco17_tpu-8_3":"/ckpt-26",
                    "my_ssd_resnet152_v1_fpn_640x640_coco17_tpu-8":"/ckpt-26",
                    "my_ssd_resnet152_v1_fpn_640x640_coco17_tpu-8_2":"/ckpt-26",
                    "my_ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8":"/ckpt-26",
                    "ssd_mobilenet_v1_fpn_640x640_distilled_2":"/ckpt-1",
                    "ssd_mobilenet_v1_fpn_640x640_distilled_3":"/ckpt-1",
                    "my_efficientdet_d1_coco17_tpu-32":"/ckpt-301",
                    "ssd_mobilenet_v1_fpn_640x640_distilled_4":"/ckpt-1",
                    "my_ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8": "/ckpt-51"}


    path_to_model_dir += model_name
    tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

    # Enable GPU dynamic memory allocation
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    print("\n[Loading TF2 Checkpoint]\n")
    print(f'Loading model - {model_name}... \n')
    start_time = time.time()

    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(path_to_model_dir + "/pipeline.config")
    model_config = configs['model']
    
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(path_to_model_dir + ckpt_dict[model_name]).expect_partial()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'The model has been loaded - {elapsed_time:.2f}s\n\n')

    return detection_model

def parse_function(example_proto):
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


teacher_model_name = "my_ssd_resnet50_v1_fpn"
student_model_name = "my_ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8"
label_filename = "label_map.pbtxt"

train_tfrecords_path = "TensorFlow/workspace/training_demo/annotations/train.record"
distilled_model_path = "TensorFlow/workspace/training_demo/distil_models/"
models_path = "TensorFlow/workspace/training_demo/models/"
path_to_labels = f"TensorFlow/workspace/training_demo/annotations/{label_filename}"
category_index = label_map_util.create_category_index_from_labelmap(path_to_labels, use_display_name=True)

teacher_model = load_model(teacher_model_name, models_path)
student_model, distilled_model_name = get_student_model(student_model_name, models_path, distilled_model_path)

dataset = tf.compat.v1.data.TFRecordDataset(train_tfrecords_path)
dataset = dataset.map(parse_function).batch(1)

model, all_losses = distill(
    epoch_num=50,
    dataset=dataset,
    teacher_model=teacher_model,
    teacher_name=teacher_model_name,
    student_model=student_model,
    student_name=student_model_name,
    optimizer=tf.keras.optimizers.Adam(),
    classification_loss_fn=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    localization_loss_fn=tf.keras.losses.MeanSquaredError(),
    distillation_loss_fn=tf.keras.losses.KLDivergence(),
    save_every_n_epochs=1,
    checkpoint_path= distilled_model_path + distilled_model_name 
)
# Zapisanie modelu
ckpt = tf.compat.v2.train.Checkpoint(model=model)
ckpt.save(distilled_model_path + distilled_model_name + '/Final/ckpt')
print(f"Model {student_model_name} saved in {distilled_model_path + distilled_model_name + '/Final/ckpt'}")