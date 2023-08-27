"""
This script serves as the main entry point for the Mini Metro object detection program.
It imports the best pre-trained model and class labels for object detection from the 'minimetrobot' package.
The script utilizes the ScreenDetector class to perform object detection on a live screen.

Functions:
    main(): Initializes and starts the object detection process.

To execute this script, run it directly as a standalone Python program.

Example:
    python <script_name>.py

"""
from minimetrobot.models import best_model, class_labels
from minimetrobot.core import ScreenDetector

def main():
    """Main function to start the object detection on live screen."""
    # Initialize the screen detector with the path to the YOLO model
    detector = ScreenDetector(best_model, class_labels)
    detector.mock_interaction()
    detector.initialize_display(screen_number=1)
    detector.detect_on_screen()


if __name__ == "__main__":
    main()
