import tensorflow as tf
import tensorflow_models as tfm
import keras
import time
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import matplotlib.pyplot as plt
import os
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import numpy as np

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#model_name = "my_ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8"
#model_name = "my_ssd_resnet50_v1_fpn"
model_name = "my_ssd_resnet50_v1_fpn_exported"
#model_name = "my_ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8"
#model_name = "my_ssd_resnet152_v1_fpn_640x640_coco17_tpu-8"
#model_name = "my_ssd_resnet101_v1_fpn_640x640_coco17_tpu-8"

images_path = f"TensorFlow/workspace/training_demo/images"
model_path = f"TensorFlow/workspace/training_demo/models/{model_name}"
exported_model_path = f"TensorFlow/workspace/training_demo/exported-models/"

ckpt_dict = {"my_ssd_resnet50_v1_fpn":'/ckpt-31',
                    "my_ssd_resnet101_v1_fpn_640x640_coco17_tpu-8":"/ckpt-28",
                    "my_ssd_resnet152_v1_fpn_640x640_coco17_tpu-8":"/ckpt-26",
                    "my_ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8":"/ckpt-26",
                    "my_ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8": "/ckpt-51"}


AUTO = tf.data.AUTOTUNE  # Used to dynamically adjust parallelism.
BATCH_SIZE = 64

# Comes from Table 4 and "Training setup" section.
TEMPERATURE = 10  # Used to soften the logits before they go to softmax.
INIT_LR = 0.003  # Initial learning rate that will be decayed over the training period.
WEIGHT_DECAY = 0.001  # Used for regularization.
CLIP_THRESHOLD = 1.0  # Used for clipping the gradients by L2-norm.

# We will first resize the training images to a bigger size and then we will take
# random crops of a lower size.
BIGGER = 160
RESIZE = 128

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    images_path + "/train",
    labels="inferred",
    label_mode="int",  
    validation_split=0.2,  
    subset="training",
    seed=1337,  
    image_size=(640, 640),  
    batch_size=4 
)

validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
    images_path + "/train",
    labels="inferred",
    label_mode="int",  
    validation_split=0.2,  
    subset="validation",
    seed=1337, 
    image_size=(640, 640),  
    batch_size=4 
)


test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    images_path + "/test",
    labels="inferred",
    label_mode="int",  
    seed=1337,  
    image_size=(640, 640),  
    batch_size=4  
)
print(f"Number of training examples: {train_ds.cardinality()}.")
print(
    f"Number of validation examples: {validation_ds.cardinality()}."
)
print(f"Number of test examples: {test_ds.cardinality()}.")


teacher_model = keras.models.load_model(exported_model_path + f"{model_name}_keras")

def get_mobilenet():
    mobilenet = keras.applications.MobileNet(
        input_shape=(640, 640, 3),
        include_top=True,
        weights=None,
        classes=100  # Adjust this according to your needs
    )
    return mobilenet

student_model = get_mobilenet()

class Distiller(tf.keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.loss_tracker = keras.metrics.Mean(name="distillation_loss")

    @property
    def metrics(self):
        metrics = super().metrics
        metrics.append(self.loss_tracker)
        return metrics

    def compile(self, optimizer, metrics, distillation_loss_fn, temperature):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.distillation_loss_fn = distillation_loss_fn
        self.temperature = temperature

    def preprocess_for_teacher(self, x):
        x_resized = tf.image.resize(x, (640, 640))
        x_cast = tf.cast(x_resized, dtype=tf.uint8)
        return x_cast

    @tf.function
    def call(self, inputs, training=False):
        student_predictions = self.student(inputs, training=training)
        return student_predictions

    #@tf.function
    def train_step(self, data):
        x, y = g
        x_teacher = self.preprocess_for_teacher(x)

        # Debugging: wydrukuj kształty
        tf.print("x_teacher shape:", tf.shape(x_teacher))

        # Forward pass nauczyciela
        teacher_predictions = self.teacher(x_teacher)
        teacher_class_predictions = teacher_predictions['detection_classes']  # Wyodrębnij odpowiednie predykcje

        with tf.GradientTape() as tape:
            # Forward pass ucznia
            student_predictions = self.student(x, training=True)

            # Obliczanie straty
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_class_predictions / self.temperature, axis=-1),
                tf.nn.softmax(student_predictions / self.temperature, axis=-1),
            )

        # Obliczanie gradientów
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(distillation_loss, trainable_vars)

        # Aktualizacja wag
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Raportowanie postępów
        self.loss_tracker.update_state(distillation_loss)
        return {"distillation_loss": self.loss_tracker.result()}

    #@tf.function
    def test_step(self, data):
        x, y = data
        x_teacher = self.preprocess_for_teacher(x)

        # Forward passes
        teacher_predictions = self.teacher(tf.expand_dims(x_teacher, axis=0))
        student_predictions = self.student(x, training=False)

        # Obliczanie straty
        distillation_loss = self.distillation_loss_fn(
            tf.nn.softmax(teacher_predictions / self.temperature, axis=-1),
            tf.nn.softmax(student_predictions / self.temperature, axis=-1),
        )

        # Raportowanie postępów
        self.loss_tracker.update_state(distillation_loss)
        self.compiled_metrics.update_state(y, student_predictions)
        results = {m.name: m.result() for m in self.metrics}
        return results


class WarmUpCosine(keras.optimizers.schedules.LearningRateSchedule):
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

ARTIFICIAL_EPOCHS = 1000
ARTIFICIAL_BATCH_SIZE = 512
DATASET_NUM_TRAIN_EXAMPLES = 1020
TOTAL_STEPS = int(
    DATASET_NUM_TRAIN_EXAMPLES / ARTIFICIAL_BATCH_SIZE * ARTIFICIAL_EPOCHS
)
scheduled_lrs = WarmUpCosine(
    learning_rate_base=INIT_LR,
    total_steps=TOTAL_STEPS,
    warmup_learning_rate=0.0,
    warmup_steps=1500,
)

lrs = [scheduled_lrs(step) for step in range(TOTAL_STEPS)]
plt.plot(lrs)
plt.xlabel("Step", fontsize=14)
plt.ylabel("LR", fontsize=14)
plt.show()

optimizer = tfa.optimizers.AdamW(
    weight_decay=WEIGHT_DECAY, learning_rate=scheduled_lrs, clipnorm=CLIP_THRESHOLD
)

student_model = get_mobilenet()

# Kompilacja i trening modelu
distiller = Distiller(student=student_model, teacher=teacher_model)
distiller.compile(
    optimizer,
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    distillation_loss_fn=keras.losses.KLDivergence(),
    temperature=TEMPERATURE,
)

history = distiller.fit(
    train_ds,
    steps_per_epoch=int(np.ceil(DATASET_NUM_TRAIN_EXAMPLES / BATCH_SIZE)),
    validation_data=validation_ds,
    epochs=30,  # This should be at least 1000.
)

student = distiller.student
student_model.compile(metrics=["accuracy"])
_, top1_accuracy = student.evaluate(test_ds)
print(f"Top-1 accuracy on the test set: {round(top1_accuracy * 100, 2)}%")
