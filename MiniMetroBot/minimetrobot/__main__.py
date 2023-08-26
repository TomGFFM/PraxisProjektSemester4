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
