# MiniMetroBot

## Table of Contents

1. [Description](#description)
2. [Project Scope and Targets](#project-scope-and-targets)
    - [In Scope Targets](#in-scope-targets)
    - [Out of Scope Targets](#out-of-scope-targets)
3. [Project Structure](#project-structure)
4. [Dependencies](#dependencies)
5. [Installation of the Final Bot](#installation-of-the-final-bot)
    - [Option 1: Using Git and Poetry](#option-1-using-git-and-poetry)
    - [Option 2: Using Wheel File](#option-2-using-wheel-file)
6. [Usage](#usage)
    - [Option 1: Using the Poetry Entry Point](#option-1-using-the-poetry-entry-point)
    - [Option 2: Running the Python Script Manually](#option-2-running-the-python-script-manually)
7. [YOLOv8 Model Training information](#yolov8-model-training-information)
    - [Model Information](#model-information)
    - [Training Metrics](#training-metrics)
    - [Model Training](#model-training)
8. [Contributing](#contributing)
9. [License](#license)
10. [Show your support](#show-your-support)

## Description

MiniMetroBot is an object detection bot for the game MiniMetro. It uses YOLOv8 for object detection and OpenCV for image processing.  
The recent status of the Bot is that it is capable to detect the main game object of the game Minimetro and it executes before the detection starts 
a basic automated mouse click and move.

## Project Scope and Targets
Main scope of the project is to detect game object which includes the position of the relevant objects and to add the capability to automatically perform user inputs. 
Out of scope are further steps to further train another model which actually can play the game.

### In Scope Targets
-[x] Game object detection
-[x] Basic user input
-[x] Runs on a resolution which is available on a Macbook Pro M1 Max 2021

### Out of Scope Targets (to be handled in a subsequent project)
-[ ] Fully automated gaming performed by the final bot


## Project Structure

```
MiniMetroBot/                       # Root directory              
│
├── minimetrobot/                   # Source code
│   │── core/                       # Core modules containing the main bot functionality
│   │   ├── __init__.py
│   │   └── ScreenDetector.py       # Class for detecting objects and to mock input actions (docstring available)
│   │── evidences/                  # Folder contains an evidence which shows that the bot works as discussed
│   │   └── mock_activity_start.mov # Screenvideo activity mock -> startup Minimetro game
│   │   ├── detection_evidence.mov  # Screenvideo from bot usage -> object detection while gaming
│   │── models/                     # Model files
│   │   ├── __init__.py
│   │   ├── best.pt                 # Best model from training exercise
│   │   └── labels.yaml             # List of object labels which can be detected by the model
│   │── yolo_tuning/                # Main folder which holds objects for model training (not part of whl installation)
│   │   ├── images_annotated/       # Annotated training data for YOLOv8 training
│   │   ├── images_pre_selected/    # Updated version of the raw data images
│   │   ├── images_raw/             # Raw images
│   │   ├── model_source/           
│   │   │   └── yolov8n.pt          # Base YOLOv8 model which is tuned based on the training data
│   │   ├── model_tuned
│   │   └── yolo_model_tuning.py    # Script for model tuning (docstring available)
│   ├── __init__.py
│   └── __main__.py                 # Main program used for active object detection in live game (docstring available)
│  
├── docs/                           # Documentation
│   ├── README.md                   # Project description as markdown
│   └── README.pdf                  # Markdown converted to pdf
│
├── LICENSE                         # License information
└── pyproject.toml                  # Poetry configuration file
```

## Dependencies

The project has the following dependencies:

- `python`: 3.10.9
- `comet-ml`: 3.33.6
- `ipython`: 8.14.0
- `numpy`: 1.24.2
- `opencv-contrib-python`: 4.7.0.72
- `opencv-python`: 4.7.0.72
- `pillow`: 10.0.0
- `pyyaml`: 6.0.1
- `screeninfo`: 0.8.1
- `torch`: 2.0.0
- `torchvision`: 0.15.1
- `ultralytics`: 8.0.162
- `pyautogui`: 0.9.54

To install the system and its dependencies follow the installation instructions below.

## Installation of the Final Bot

### Option 1: Using Git and Poetry

1. **Clone the repository:**
    ```bash
    git clone https://github.com/TomGFFM/PraxisProjektSemester4_30313.git
    ```

2. **Navigate to the project directory:**
    ```bash
    cd PraxisProjektSemester4_30313/MiniMetroBot
    ```

3. **Install dependencies using [Poetry](https://python-poetry.org/):**
    ```bash
    poetry install
    ```

### Option 2: Using Wheel File

1. **Download the latest wheel file, `minimetrobot-0.1.6-py3-none-any.whl`, from the repository.**

2. **Install the package using `pip`:**
    ```bash
    pip install path/to/minimetrobot-0.1.6-py3-none-any.whl
    ```

## Usage

### Option 1: Using the Poetry Entry Point 
**(notice different naming convention here. Just: "metrobot")**  
**If poetry is installed you can use the following command to run the MiniMetroBot from the environment where the package is installed:**

```bash
poetry run metrobot
```

### Option 2: Running the Python Script Manually
**Activate the virtual environment and execute the following command:**

```bash
python -m minimetrobot
```

Either of these options will start the MiniMetroBot.


## YOLOv8 Model Training information

### Model Information
The project folder "yolo_tuning" contains a subfolder "model_tuned". Here are a number of model folders available which contain trained YOLOv8 object detection models. 
The final model which was chosen and integrated into the recent version of the Bot is in the folder: "round_optimized_with_Adamax_True_bs_16_20230826_1342"

### Training Metrics
The relevant training metrics can be reviewed in the public CometML report for this project. 
Click here to visit the report page: [MiniMetroBot CometML Metrics Report.](https://www.comet.com/tomgffm/mini-metro-object-detection/reports/minimetro-bot-detection-model-tuning-report)

### Model Training
The YOLOv8 model training can be performed via the script yolo_model_tuning.py. The underlying base model is available in the folder "model_source". 
This contains the basic YOLOv8 which is used as a foundation for the detection model training. Review contained docstring for further usage and information.

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/TomGFFM/PraxisProjektSemester4_30313/issues).

## License

This project is [BSD](LICENSE) licensed.

## Show your support

Give a ⭐️ if this project helped you!
