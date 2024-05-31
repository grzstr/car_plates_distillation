import tensorflow as tf
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from platesdetection import ModelLoader
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

    return detections

def detect_object(image, detection_model, model_name):
    path_to_images_dir = "TensorFlow/workspace/training_demo/images/train"
    image_path = path_to_images_dir + '/' + image
    image_np = np.array(cv2.imread(image_path))
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

    if model_name[-9:] == "_exported":
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image_np)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = detection_model(input_tensor)      
    else:
        detections = detect_fn(input_tensor, detection_model)
        
    return detections

def extract_number(filename):
    return int(filename.split("Cars")[1].split(".")[0])

def find_images_names(path_to_images_dir):
    images_names = []
    
    for image_name in os.listdir(path_to_images_dir):
        if image_name.endswith(('.bmp', '.jpg', '.png', '.jpeg')):
            images_names.append(image_name)
    sorted_names = sorted(images_names, key=extract_number)
    return sorted_names

def distill(epoch_num, dataset, teacher_model, teacher_name, student_model, student_name, optimizer, loss_fn,  distillation_loss_fn, alpha=0.1):
    # Enable GPU dynamic memory allocation
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print("\nDistillation started...")
    print(f"Teacher model: {teacher_name}  || student model: {student_name}")
    start_disitll = time.time()
    images = find_images_names("TensorFlow/workspace/training_demo/images/train")


    for epoch in range(epoch_num):
        start_time = time.time()

        #for data, image in zip(dataset, images):
        progress_bar = tqdm(images, desc=f"Epoch {epoch + 1}/{epoch_num}")
        for image in progress_bar:
            #imagetf, label = data
            teacher_detections = detect_object(image, teacher_model, teacher_name)
  
            with tf.GradientTape() as tape:
                student_detections = detect_object(image, student_model, student_name)

                distillation_loss = distillation_loss_fn(teacher_detections['raw_detection_scores'], student_detections['raw_detection_scores'])

            gradients = tape.gradient(distillation_loss, student_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, student_model.trainable_variables))
        end_time = time.time()
        print(" - Time: {:.2f}s - Loss: {:.4f}".format(end_time - start_time, distillation_loss))
    end_distill = time.time()
    total_distill_time = end_distill - start_disitll
    print(f"Distillation finished in {total_distill_time:.2f}s || {total_distill_time/60:.2f}min || {total_distill_time/3600:.2f}h")
    return student_model, distillation_loss

def get_student_model(model_name):
    student_model_path = f"TensorFlow/workspace/training_demo/distil_models/{model_name}"
    configs = config_util.get_configs_from_pipeline_file(student_model_path + "/pipeline.config")
    model_config = configs['model']
    model = model_builder.build(model_config=model_config, is_training=True)
    return model

# Funkcja do parsowania TFRecord
def parse_tfrecord_fn(example):
    feature_description = {
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/source_id': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    return tf.io.parse_single_example(example, feature_description)

# Funkcja do przetwarzania danych
def preprocess(example):
    image = tf.io.decode_jpeg(example['image/encoded'], channels=3)
    image = tf.image.resize(image, [300, 300]) / 255.0  # Normalizacja obrazów
    labels = tf.sparse.to_dense(example['image/object/class/label'])
    boxes = tf.stack([
        tf.sparse.to_dense(example['image/object/bbox/ymin']),
        tf.sparse.to_dense(example['image/object/bbox/xmin']),
        tf.sparse.to_dense(example['image/object/bbox/ymax']),
        tf.sparse.to_dense(example['image/object/bbox/xmax'])
    ], axis=1)
    return image, {'boxes': boxes, 'labels': labels}

# Funkcja do wczytywania datasetu z plików TFRecord
def load_tfrecord_dataset(tfrecord_file, batch_size=32, buffer_size=1024):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
    parsed_dataset = raw_dataset.map(parse_tfrecord_fn)
    dataset = parsed_dataset.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

batch_size = 4

teacher_model_name = "my_ssd_resnet50_v1_fpn"
student_model_name = "ssd_mobilenet_v1_fpn_640x640_distilled"
label_filename = "label_map.pbtxt"

train_tfrecords_path = "TensorFlow/workspace/training_demo/annotations/train.record"
distilled_model_path = "TensorFlow/workspace/training_demo/distil_models/"
path_to_labels = f"TensorFlow/workspace/training_demo/annotations/{label_filename}"
category_index = label_map_util.create_category_index_from_labelmap(path_to_labels, use_display_name=True)

loader = ModelLoader()
teacher_model = loader.load_model(teacher_model_name)
student_model = get_student_model(student_model_name)

optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
distillation_loss = tf.keras.losses.KLDivergence()
dataset = load_tfrecord_dataset(train_tfrecords_path, batch_size=batch_size)
epoch = 10

model, disitillation_loss_f = distill(epoch, dataset, teacher_model, teacher_model_name, student_model, student_model_name, optimizer, loss_fn, distillation_loss)


# Zapisanie modelu
ckpt = tf.compat.v2.train.Checkpoint(model=model)
ckpt.save(os.path.join(model, distilled_model_path + student_model_name + 'ckpt'))

#tf.saved_model.save(model, distilled_model_path + student_model_name + "_1000") 
