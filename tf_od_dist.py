import tensorflow as tf
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import dataset_util
import numpy as np
import cv2
import time
import os
from tqdm import tqdm

#@tf.function
def detect_fn(image, detection_model):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict

#@tf.function
def reshape_classses(classes, student_prediction_dict):
    # Check shapes after reshaping
    #print(f"Classes shape: {classes.shape}")
    #print(f"Student predictions shape: {student_prediction_dict['class_predictions_with_background'].shape}")
    
    class_batch_size, num_classes_per_image = tf.shape(classes)[0].numpy(), tf.shape(classes)[1].numpy()
    
    # Ensure classes and predictions are properly reshaped
    num_classes = student_prediction_dict['class_predictions_with_background'].shape[-1]
    flattened_classes = tf.reshape(classes, [-1])
    one_hot_classes = tf.one_hot(flattened_classes, depth=num_classes)

    # Repeat one-hot classes to match the number of predictions
    num_predictions = student_prediction_dict['class_predictions_with_background'].shape[1]
    reshaped_classes = tf.tile(one_hot_classes, [num_predictions, 1])
    reshaped_classes = tf.reshape(reshaped_classes, [-1, num_classes])

    student_preds = tf.reshape(student_prediction_dict['class_predictions_with_background'], [-1, num_classes])
    student_preds = tf.tile(student_preds, [num_classes_per_image, class_batch_size])
    # Check shapes after reshaping
    #print(f"Reshaped classes: {reshaped_classes.shape}")
    #print(f"Reshaped student predictions: {student_preds.shape}")

    return reshaped_classes, student_preds

#@tf.function
def compute_iou(box1, box2):
    # Ensure boxes have correct shapes
    box1 = tf.reshape(box1, [-1, 4])
    box2 = tf.reshape(box2, [-1, 4])

    x1 = tf.maximum(box1[:, tf.newaxis, 0], box2[:, 0])
    y1 = tf.maximum(box1[:, tf.newaxis, 1], box2[:, 1])
    x2 = tf.minimum(box1[:, tf.newaxis, 2], box2[:, 2])
    y2 = tf.minimum(box1[:, tf.newaxis, 3], box2[:, 3])

    intersection = tf.maximum(0.0, x2 - x1) * tf.maximum(0.0, y2 - y1)
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    iou = intersection / (box1_area[:, tf.newaxis] + box2_area - intersection)
    return iou


#@tf.function
def match_predictions_to_ground_truth(pred_boxes, true_boxes, iou_threshold=0.5):
    # Compute IoU between all predicted boxes and true boxes
    iou_matrix = compute_iou(pred_boxes, true_boxes)
    
    # Perform matching using the IoU matrix
    matches = tf.argmax(iou_matrix, axis=1)
    matched_iou = tf.reduce_max(iou_matrix, axis=1)

    # Filter matches based on IoU threshold
    valid_matches = matched_iou > iou_threshold
    matched_indices = tf.where(valid_matches)[:, 0]  # Ensure we get 1D tensor of indices

    matched_pred_boxes = tf.gather(pred_boxes, matched_indices)
    matched_true_boxes = tf.gather(true_boxes, tf.gather(matches, matched_indices))
    
    return matched_pred_boxes, matched_true_boxes

def distill(epoch_num, dataset, teacher_model, teacher_name, student_model, student_name, optimizer, classification_loss_fn, localization_loss_fn,  distillation_loss_fn):
    # Enable GPU dynamic memory allocation
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print("\nDistillation started...")
    print(f"Teacher model: {teacher_name}  || student model: {student_name}")
    start_disitll = time.time()

    temperature=10
    alpha=0.5

    image_count = 0
    for image in dataset:
        image_count +=1

    for epoch in range(epoch_num):
        start_time = time.time()
        for image, (boxes, classes) in tqdm(dataset, total=image_count, desc=f"Epoch {epoch + 1}/{epoch_num}"):
            teacher_detections, teachers_prediction_dict = detect_fn(image, teacher_model)
            with tf.GradientTape() as tape:
                student_detections, student_prediction_dict = detect_fn(image, student_model)

                reshaped_classes, student_preds = reshape_classses(classes, student_prediction_dict)

                classification_loss = classification_loss_fn(reshaped_classes, student_preds)

                matched_pred_boxes, matched_true_boxes = match_predictions_to_ground_truth(
                    student_prediction_dict['box_encodings'], student_prediction_dict['class_predictions_with_background'],
                    boxes
                )

                localization_loss = localization_loss_fn(matched_true_boxes, matched_pred_boxes)

                # Calculate distillation loss
                distill_loss = distillation_loss_fn(
                    tf.nn.softmax(teacher_detections['raw_detection_scores'] / temperature),
                    tf.nn.softmax(student_detections['raw_detection_scores'] / temperature)
                )

                total_loss = alpha * distill_loss + (1 - alpha) * (classification_loss + localization_loss)

            gradients = tape.gradient(total_loss, student_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, student_model.trainable_variables))
        end_time = time.time()
        print(" - Time: {:.2f}s - Loss: {:.4f}".format(end_time - start_time, total_loss))
    end_distill = time.time()
    total_distill_time = end_distill - start_disitll
    print(f"Distillation finished in {total_distill_time:.2f}s || {total_distill_time/60:.2f}min || {total_distill_time/3600:.2f}h")
    return student_model, total_loss

def get_student_model(model_name):
    student_model_path = f"TensorFlow/workspace/training_demo/distil_models/{model_name}"
    configs = config_util.get_configs_from_pipeline_file(student_model_path + "/pipeline.config")
    model_config = configs['model']
    model = model_builder.build(model_config=model_config, is_training=True)
    return model

def load_model(model_name):
    path_to_model_dir = "TensorFlow/workspace/training_demo/models/"
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
student_model_name = "ssd_mobilenet_v1_fpn_640x640_distilled"
label_filename = "label_map.pbtxt"

train_tfrecords_path = "TensorFlow/workspace/training_demo/annotations/train.record"
distilled_model_path = "TensorFlow/workspace/training_demo/distil_models/"
path_to_labels = f"TensorFlow/workspace/training_demo/annotations/{label_filename}"
category_index = label_map_util.create_category_index_from_labelmap(path_to_labels, use_display_name=True)

teacher_model = load_model(teacher_model_name)
student_model = get_student_model(student_model_name)

dataset = tf.compat.v1.data.TFRecordDataset(train_tfrecords_path)
dataset = dataset.map(parse_function).batch(1)

model, disitillation_loss_f = distill(epoch_num = 10,
                                      dataset = dataset,
                                      teacher_model = teacher_model,
                                      teacher_name = teacher_model_name,
                                      student_model = student_model,
                                      student_name = student_model_name,
                                      optimizer =  tf.keras.optimizers.Adam(),
                                      classification_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                                      localization_loss_fn = tf.keras.losses.MeanSquaredError(),
                                      distillation_loss_fn = tf.keras.losses.KLDivergence())


# Zapisanie modelu
ckpt = tf.compat.v2.train.Checkpoint(model=model)
ckpt.save(distilled_model_path + student_model_name + '_5/ckpt')