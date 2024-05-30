import tensorflow as tf
import keras


# Załaduj zapisany model
saved_model_dir = f"TensorFlow/workspace/training_demo/exported-models/my_ssd_resnet50_v1_fpn_exported/saved_model"
model = tf.saved_model.load(saved_model_dir)

# Wyciągnij wagi z modelu
for layer in model.signatures['serving_default'].variables:
    print(f'Layer: {layer.name}')
    #print(f'Weights: {layer.numpy()}')

