"""
This script imports the best YOLO model and corresponding class labels for object detection.

The `best_model` variable holds the YOLO model loaded from the 'best.pt' file within the 'minimetrobot' package.
The `class_labels` dictionary maps numerical indices to their respective class labels, which are read from 'labels.yaml'.

Dependencies:
    - yaml: For parsing the 'labels.yaml' file
    - YOLO from ultralytics: For the object detection model
    - pkg_resources: For locating the model and labels files within the package
"""

import yaml

from pkg_resources import resource_filename
from ultralytics import YOLO

best_model = YOLO(resource_filename("minimetrobot", "models/best.pt"))

with open(resource_filename("minimetrobot", "models/labels.yaml"), 'r') as label_file:
    lables_list = yaml.safe_load(label_file)

class_labels = {index: value for index, value in enumerate(lables_list)}
