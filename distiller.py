import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class Distiller(tf.keras.Model):
    def __init__(self, teacher, student):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student
        self.loss_tracker = keras.metrics.Mean(name="distillation_loss")

    @property
    def metrics(self):
        metrics = super().metrics
        metrics.append(self.loss_tracker)
        return metrics

    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, alpha=0.1, temperature=3):
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        teacher_images, student_images, labels = data

        print(f"Kształt obrazów nauczyciela: {teacher_images.shape}, dtype: {teacher_images.dtype}")
        print(f"Kształt obrazów ucznia: {student_images.shape}, dtype: {student_images.dtype}")

        teacher_predictions = self.teacher(teacher_images)

        print(f"Struktura predykcji nauczyciela: {type(teacher_predictions)}")
        print(f"Klucze w predykcjach nauczyciela: {teacher_predictions.keys()}")
        print(f"Kształt 'raw_detection_scores': {teacher_predictions['raw_detection_scores'].shape}")
        print(f"Dtype 'raw_detection_scores': {teacher_predictions['raw_detection_scores'].dtype}")

        teacher_logits = teacher_predictions['raw_detection_scores']

        print(f"Przed reshape: kształt teacher_logits: {teacher_logits.shape}")

        num_elements_teacher_logits = tf.size(teacher_logits)
        print(f"Liczba elementów w teacher_logits: {num_elements_teacher_logits}")

        batch_size = tf.shape(student_images)[0]
        num_classes = tf.shape(teacher_logits)[-1] // self.temperature  # Assuming the division is appropriate here

        # Calculate the new shape
        new_shape = tf.stack([batch_size, -1, 2])
        print(f"Nowy kształt: {new_shape}")
        
        # Ensure the new shape has the same number of elements
        num_elements_new_shape = tf.reduce_prod(new_shape)
        print(f"Liczba elementów po reshape: {num_elements_new_shape}")

        # Verify if number of elements match
        tf.debugging.assert_equal(num_elements_teacher_logits, num_elements_new_shape, message="Niezgodność liczby elementów")

        teacher_logits = tf.reshape(teacher_logits, new_shape)
        teacher_logits = tf.reduce_mean(teacher_logits, axis=1)
        teacher_logits = tf.reshape(teacher_logits, [batch_size, -1])  # Adjusted reshape

        print(f"Kształt logitów nauczyciela po przycięciu: {teacher_logits.shape}")

        with tf.GradientTape() as tape:
            student_predictions = self.student(student_images)
            print(f"Kształt predykcji ucznia: {student_predictions.shape}")

            student_loss = self.student_loss_fn(labels, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_logits / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1)
            )

            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_tracker.update_state(distillation_loss)
        return {"distillation_loss": self.loss_tracker.result()}

    def test_step(self, data):
        student_images, labels = data

        # Przejście przez model ucznia
        student_predictions = self.student(student_images, training=False)

        # Obliczanie straty ucznia
        student_loss = self.student_loss_fn(labels, student_predictions)

        # Aktualizacja metryk
        self.compiled_metrics.update_state(labels, student_predictions)
        results = {m.name: m.result() for m in self.metrics}
        return results

# Funkcja do łączenia generatorów dla destylacji
def combined_generator(teacher_gen, student_gen):
    while True:
        teacher_images, labels = next(teacher_gen)
        student_images, _ = next(student_gen)
        
        # Cast teacher_images to the expected dtype if needed
        teacher_images = tf.cast(teacher_images * 255.0, tf.uint8)
        
        yield (teacher_images, student_images, labels)

def get_mobilenet():
    mobilenet = keras.applications.MobileNet(
        input_shape=(640, 640, 3),
        include_top=True,
        weights=None,
        classes=100  # Adjust this according to your needs
    )
    return mobilenet

# Definiowanie rozmiarów obrazów
teacher_image_size = (640, 640)
student_image_size = (640, 640)

images_path = "TensorFlow/workspace/training_demo/images"
model_name = "my_ssd_resnet50_v1_fpn_exported"
exported_model_path = "TensorFlow/workspace/training_demo/exported-models/"
distilled_model_path = "TensorFlow/workspace/training_demo/distil_models/"

# Tworzenie ImageDataGenerator dla zmiany rozmiaru obrazów
teacher_datagen = ImageDataGenerator(rescale=1./255)
student_datagen = ImageDataGenerator(rescale=1./255)

# Ładowanie i wstępne przetwarzanie obrazów
teacher_generator = teacher_datagen.flow_from_directory(
    images_path + "/train",
    target_size=teacher_image_size,
    batch_size=2,
    class_mode='categorical'
)

student_generator = student_datagen.flow_from_directory(
    images_path + "/train",
    target_size=student_image_size,
    batch_size=2,
    class_mode='categorical'
)

combined_gen = combined_generator(teacher_generator, student_generator)

# Załadowanie modeli nauczyciela i ucznia
teacher_model = keras.models.load_model(exported_model_path + f"{model_name}_keras")
student_model = get_mobilenet()

# Stworzenie instancji Distillera
distiller = Distiller(teacher=teacher_model, student=student_model)

# Kompilacja distillera
distiller.compile(
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[tf.keras.metrics.CategoricalAccuracy()],
    student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
    distillation_loss_fn=tf.keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=3
)

# Trenowanie modelu ucznia przy użyciu distillera
distiller.fit(combined_gen, steps_per_epoch=len(teacher_generator), epochs=10)

'''
student = distiller.student
student_model.compile(metrics=["accuracy"])
_, top1_accuracy = student.evaluate(combined_gen)
print(f"Top-1 accuracy on the test set: {round(top1_accuracy * 100, 2)}%")
'''

# Zapisywanie wydestylowanego modelu ucznia w formacie SavedModel
student_model.save(distilled_model_path + 'MobileNet_distilled2', save_format='tf')
