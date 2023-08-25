from pkg_resources import resource_filename
from ultralytics import YOLO

best_model = YOLO(resource_filename("MiniMetroBot", "models/best.pt"))
