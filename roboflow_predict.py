from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
from autodistill_yolov8 import YOLOv8
from autodistill_roboflow_universe import RoboflowUniverseModel
from autodistill.utils import plot
import supervision as sv
import cv2



def predict(img_path, base_model):
    image = cv2.imread(img_path)
    detections = base_model.predict(img_path)
    # annotate image with detections
    box_annotator = sv.BoxAnnotator()

    labels = [
        f"{base_model.ontology.classes()[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _, _ in detections
    ]

    annotated_frame = box_annotator.annotate(
        scene=image.copy(), detections=detections, labels=labels
    )

    sv.plot_image(annotated_frame, (16, 16))


# define an ontology to map class names to our GroundingDINO prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
model_configs = [
    ("carplates-bi5bk", 1)
]

base_model = RoboflowUniverseModel(
    ontology=CaptionOntology(
        {
            "licence": "licence",
        }
),
    api_key="JowxUFgbZ4P1EuEjdMi1",
    model_configs=model_configs,
)

# label all images in a folder called `context_images`
base_model.label(
  input_folder="./images",
  output_folder="./dataset"
)

img_path = "./TensorFlow/workspace/training_demo/images/test/"

predict(img_path + "Cars425.png", base_model)
predict(img_path + "Cars430.png", base_model)
predict(img_path + "Cars414.png", base_model)
predict(img_path + "Cars415.png", base_model)