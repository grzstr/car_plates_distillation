import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2

exported_model_path = f"TensorFlow/workspace/training_demo/exported-models/"
model_name = "my_ssd_resnet50_v1_fpn_exported"
teacher_model = keras.models.load_model(exported_model_path + f"{model_name}_keras")

def load_and_preprocess_image(image_path, target_size):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalizacja, jeśli jest wymagana
    return img_array

image_path = "TensorFlow/workspace/training_demo/images/test/Cars411.png"
#target_size = (224, 224)  # Zmodyfikuj zgodnie z wymaganiami modelu
target_size = (640, 640)
img_array = load_and_preprocess_image(image_path, target_size)

predictions = teacher_model.predict(img_array)

def process_predictions(predictions, threshold=0.5):
    boxes, scores, classes = predictions
    results = []
    for box, score, class_id in zip(boxes, scores, classes):
        if score > threshold:
            results.append({
                "box": box,
                "score": score,
                "class_id": class_id
            })
    return results

def draw_boxes(image_path, results):
    img = cv2.imread(image_path)
    for result in results:
        box = result['box']
        score = result['score']
        class_id = result['class_id']
        
        # Konwersja z normalizowanych współrzędnych na piksele
        (height, width) = img.shape[:2]
        (startX, startY, endX, endY) = (box[1] * width, box[0] * height, box[3] * width, box[2] * height)
        
        # Rysowanie pudełka
        cv2.rectangle(img, (int(startX), int(startY)), (int(endX), int(endY)), (255, 0, 0), 2)
        label = f"Class: {class_id}, Score: {score:.2f}"
        cv2.putText(img, label, (int(startX), int(startY) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Wyświetlenie obrazu
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

results = process_predictions(predictions)
draw_boxes(image_path, results)
