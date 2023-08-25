import comet_ml

from comet_ml import Experiment
from datetime import datetime
from ultralytics import YOLO
from PIL import Image


# optimizer_pick = ['Adam', 'Adamax', 'AdamW', 'NAdam']
# cos_lr_pick = [False, True]
# batch_size = [8, 16, 32, 64]

optimizer_pick = ['Adam', 'Adamax']
cos_lr_pick = [True]
batch_size = [6]

for bs in batch_size:
    for c in cos_lr_pick:
        for o in optimizer_pick:
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
            model = YOLO('MiniMetroBot/yolo_tuning/model_source/yolov8n.pt')

            # Train the model on apple mps (metal performance shader) device
            model.train(data='MiniMetroBot/yolo_tuning/images_annotated/data.yaml',
                        epochs=140,
                        patience=40,
                        batch=bs,
                        cache=True,
                        imgsz=1856,
                        save_period=-1,
                        device='mps',
                        workers=24,
                        project='MiniMetroBot/yolo_tuning/model_tuned',
                        optimizer=o,
                        cos_lr=c,
                        lr0=0.001,
                        lrf=0.05,
                        verbose=True,
                        name=f"round_optimized_with_{o}_{c}_bs_{bs}_{n_tag}"
                        )

            experiment.end()

# metrics = model.val()  # evaluate model performance on the validation set
# path = model.export(format="onnx")

# load best model for smoke test
# best_model = YOLO('model_tuned/train3/weights/best.pt')
# best_model = YOLO('model_tuned/round_optimized_with_Adam/weights/best.pt')
#
#
# # predict on test_image
# im1 = Image.open("MiniMetroBot/yolo_tuning/images_raw/GameMenuItems.jpg")
# results1 = best_model.predict(source=im1, save=True)  # save plotted images
#
# im2 = Image.open("MiniMetroBot/yolo_tuning/images_raw/Bildschirmfoto 2023-07-15 um 12.24.53.png")
# results2 = best_model.predict(source=im2, save=True)  # save plotted images
#
# im3 = Image.open("MiniMetroBot/yolo_tuning/images_raw/Bildschirmfoto 2023-07-15 um 12.25.38.png")
# results3 = best_model.predict(source=im3, save=True)  # save plotted images

"""
mAP - mean Average Precision: Durchschnittlichte Precision über alle vorhandenen Klassen
box_loss - Loss der bounding boxes; Differenz der vorhergesagten und der tatsächlichen Boxen
cls_loss - Classification loss; Vorhergesagte Klasse vs. tatsächlicher Klasse
"""

















