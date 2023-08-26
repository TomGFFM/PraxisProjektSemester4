import yaml

from pkg_resources import resource_filename
from ultralytics import YOLO

best_model = YOLO(resource_filename("minimetrobot", "models/best.pt"))

with open(resource_filename("minimetrobot", "models/labels.yaml"), 'r') as label_file:
    lables_list = yaml.safe_load(label_file)

class_labels = {index: value for index, value in enumerate(lables_list)}
