import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class Distiller(tf.keras.Model):
    def __init__(self, teacher, student):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, alpha=0.1, temperature=3):
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        teacher_images, student_images, labels = data

        # Forward pass through the teacher model
        teacher_predictions = self.teacher(teacher_images)

        with tf.GradientTape() as tape:
            # Forward pass through the student model
            student_predictions = self.student(student_images, training=True)

            # Compute the student loss
            student_loss = self.student_loss_fn(labels, student_predictions)

            # Compute the distillation loss
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1)
            )

            # Compute the total loss
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        self.compiled_metrics.update_state(labels, student_predictions)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        student_images, labels = data

        # Forward pass through the student model
        student_predictions = self.student(student_images, training=False)

        # Compute the student loss
        student_loss = self.student_loss_fn(labels, student_predictions)

        # Update metrics
        self.compiled_metrics.update_state(labels, student_predictions)

        return {m.name: m.result() for m in self.metrics}

# Define image sizes
teacher_image_size = (224, 224)
student_image_size = (128, 128)

images_path = "TensorFlow/workspace/training_demo/images"
model_name = "my_ssd_resnet50_v1_fpn_exported"
exported_model_path = f"TensorFlow/workspace/training_demo/exported-models/"

# Create ImageDataGenerators for resizing images
teacher_datagen = ImageDataGenerator(rescale=1./255)
student_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess images
teacher_generator = teacher_datagen.flow_from_directory(
    images_path + "/train",
    target_size=teacher_image_size,
    batch_size=32,
    class_mode='categorical'
)

student_generator = student_datagen.flow_from_directory(
    images_path + "/train",
    target_size=student_image_size,
    batch_size=32,
    class_mode='categorical'
)

# Create a function to wrap generators for distillation
def combined_generator(teacher_gen, student_gen):
    while True:
        teacher_images, labels = next(teacher_gen)
        student_images, _ = next(student_gen)
        yield (teacher_images, student_images, labels)

def get_mobilenet():
    mobilenet = keras.applications.MobileNet(
        input_shape=(640, 640, 3),
        include_top=True,
        weights=None,
        classes=100  # Adjust this according to your needs
    )
    return mobilenet

combined_gen = combined_generator(teacher_generator, student_generator)

# Assuming teacher and student models are already defined
teacher_model = keras.models.load_model(exported_model_path + f"{model_name}_keras")
student_model = get_mobilenet()

# Create a Distiller instance
distiller = Distiller(teacher=teacher_model, student=student_model)

# Compile the distiller
distiller.compile(
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[tf.keras.metrics.CategoricalAccuracy()],
    student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
    distillation_loss_fn=tf.keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=3
)

# Train the student model using the distiller
distiller.fit(combined_gen, steps_per_epoch=len(teacher_generator), epochs=10)
