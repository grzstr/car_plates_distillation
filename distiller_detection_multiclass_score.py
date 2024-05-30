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

        # Debugowanie
        print(f"Kształt obrazów nauczyciela: {teacher_images.shape}, dtype: {teacher_images.dtype}")
        print(f"Kształt obrazów ucznia: {student_images.shape}, dtype: {student_images.dtype}")

        # Przejście przez model nauczyciela
        teacher_predictions = self.teacher(teacher_images)
        print(f"Predykcje nauczyciela: {teacher_predictions}")

        # Zakładamy, że 'detection_multiclass_scores' są logitami klas
        teacher_logits = teacher_predictions['detection_multiclass_scores']
        print(f"Kształt logitów nauczyciela przed przekształceniem: {teacher_logits.shape}")

        # Przekształcenie logitów nauczyciela, aby pasowały do kształtu logitów ucznia
        # W tym przypadku usuwamy wymiar 0, który jest zbędny i transponujemy tensor
        teacher_logits = tf.squeeze(teacher_logits, axis=0)
        teacher_logits = tf.transpose(teacher_logits, perm=[1, 0])
        
        # Debugowanie kształtu logitów nauczyciela po przekształceniu
        print(f"Kształt logitów nauczyciela po przekształceniu: {teacher_logits.shape}")

        with tf.GradientTape() as tape:
            # Przejście przez model ucznia
            student_predictions = self.student(student_images, training=True)
            
            # Debugowanie kształtów
            print(f"Kształt logitów nauczyciela: {teacher_logits.shape}")
            print(f"Kształt predykcji ucznia: {student_predictions.shape}")

            # Obliczanie straty ucznia
            student_loss = self.student_loss_fn(labels, student_predictions)

            # Obliczanie straty destylacji
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_logits / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1)
            )

            # Obliczanie całkowitej straty
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Obliczanie gradientów
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Aktualizacja wag
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Report progress
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
