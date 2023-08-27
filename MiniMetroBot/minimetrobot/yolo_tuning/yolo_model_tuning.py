"""
This script is designed for object detection model training in the game "Mini Metro" using the YOLO algorithm.
The script iterates through different combinations of hyperparameters like batch size, optimizer, and learning rate schedule.
The performance metrics and the model checkpoints are managed and saved using Comet.ml.

Parameters:
    optimizer_pick (list): List of optimizer names to be used in the training.
    cos_lr_pick (list): List of booleans to specify whether cosine learning rate schedule should be used.
    batch_size (list): List of integers specifying the batch sizes to be used in the model training.

Comet.ml is initialized for each experiment to track metrics, hyperparameters, and model checkpoints.
The experiment's name is constructed using the hyperparameters and the current timestamp.

The YOLO model is trained using the following:
    - Pre-trained model from the path 'minimetrobot/yolo_tuning/model_source/yolov8n.pt'
    - Training data from the YAML file located at 'minimetrobot/yolo_tuning/images_annotated/data.yaml'
    - Number of epochs is set to 130, with a patience of 30 epochs for early stopping.
    - The trained model is saved in 'minimetrobot/yolo_tuning/model_tuned'.

After training, the Comet.ml experiment is ended, and metrics are saved for future analysis.

Notes:
    mAP - mean Average Precision: The average precision calculated across all existing classes.
    box_loss - Loss for the bounding boxes; the difference between the predicted and actual boxes.
    cls_loss - Classification loss; the difference between the predicted and actual class labels.

"""


import comet_ml

from comet_ml import Experiment
from datetime import datetime
from ultralytics import YOLO

optimizer_pick = ['Adam', 'Adamax', 'AdamW', 'NAdam']
cos_lr_pick = [False, True]
batch_size = [8, 16, 32, 64]

for bs in batch_size:
    for c in cos_lr_pick:
        for o in optimizer_pick:
            print(bs, o, c)
            # init and load comet_ml
            comet_ml.init()
            experiment = Experiment(
                api_key="tM7z8qhQjX8rm9axPTcOhjpma",
                project_name = "mini-metro-object-detection",
                workspace="tomgffm")

            # create numeric tag from date
            n_tag = datetime.now().strftime('%Y%m%d_%H%M')

            # set experiment name
            experiment.set_name(name=f"MiniMetro_opt_{o}_coslr_{c}_bs_{bs}_{n_tag}")

            # Load a pre-trained model
            model = YOLO('minimetrobot/yolo_tuning/model_source/yolov8n.pt')

            # Train the model on apple mps (metal performance shader) device
            model.train(data='minimetrobot/yolo_tuning/images_annotated/data.yaml',
                        epochs=130,
                        patience=30,
                        batch=bs,
                        cache=True,
                        imgsz=1024,
                        save_period=-1,
                        device='mps',
                        workers=16,
                        project='minimetrobot/yolo_tuning/model_tuned',
                        optimizer=o,
                        cos_lr=c,
                        lr0=0.001,
                        lrf=0.05,
                        verbose=True,
                        name=f"round_optimized_with_{o}_{c}_bs_{bs}_{n_tag}"
                        )

            experiment.end()













