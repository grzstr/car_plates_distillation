# Knowledge Distillation using TensorFlow Object Detection API

This repository contains the implementation of a project focused on knowledge distillation in object detection using TensorFlow Object Detection API. The project leverages Python 3.9 and TensorFlow 2.15 on an Ubuntu system, with training conducted on combined datasets sourced from Kaggle.

## Table of Contents
1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Model Training](#model-training)
4. [Datasets](#dataset)
5. [Knowledge Distillation](#knowledge-distillation)
6. [License](#license)
7. [Acknowledgements](#acknowledgements)

## Introduction

The objective of this project is to distill knowledge from a large, complex object detection model (teacher) into a smaller, more efficient model (student) while maintaining high accuracy. The process aims to transfer knowledge learned by the teacher model to the student model, which can then be used for efficient inference on edge devices or in low-resource environments. Pre-trained models from the TensorFlow Model Garden have been utilized and fine-tuned to detect license plates on a Kaggle dataset. Subsequently, EasyOCR library is employed to recognize characters on the license plate.


## Requirements
- TensorFlow 2.15
- [TensorFlow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md)
- Python 3.9
- EasyOCR
- Ubuntu (or compatible operating system)

## Model training

### Dataset
The models were trained using two combined datasets sourced from Kaggle. The datasets contain labeled images with annotations for various objects. 
- [Car License Plate Detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection)
- [Automatic Number Plate Recognition](https://www.kaggle.com/datasets/aslanahmedov/number-plate-detection)

### Teacher model

The teacher model is a complex and well-performing model trained on the combined datasets. It serves as the knowledge source for the student model.

### Student Model
The student model is a lightweight model aimed at achieving faster inference times with reduced computational resources. The training process includes:

1. **Knowledge Transfer:** Using soft labels and logits from the teacher model.
2. **Optimization:** Fine-tuning hyperparameters and architecture to balance performance and efficiency.

## Knowledge Distillation

Knowledge distillation is achieved by minimizing the Kullback-Leibler divergence between the softmax output of the teacher and student models. This technique helps the student model learn the nuances of object detection from the teacher model, enhancing its accuracy beyond what would be possible with direct training on the datasets alone.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements
This project utilizes pre-trained models from the TensorFlow Model Garden and leverages the EasyOCR library for character recognition.


