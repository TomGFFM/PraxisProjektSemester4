from MiniMetroBot.models import best_model
from MiniMetroBot.core import ScreenDetector

def main():
    """Main function to start the object detection on live screen."""
    # Initialize the screen detector with the path to the YOLO model
    detector = ScreenDetector(best_model)
    detector.initialize_display(screen_number=1)
    detector.detect_on_screen()


if __name__ == "__main__":
    main()
