import cv2
import numpy as np
from screeninfo import get_monitors
from PIL import ImageGrab


class ScreenDetector:
    """
       A class to detect and annotate objects on the computer screen in real-time using the YOLO model.

       Attributes:
       -----------
       model : object
           The trained YOLO model used for object detection.

       Methods:
       --------
       initialize():
           Sets up the necessary configurations and prepares the YOLO model for detection.

       detect_objects():
           Captures the computer screen, detects objects using the YOLO model,
           and overlays the detections on the screen.

       Notes:
       ------
       Ensure that the necessary dependencies like OpenCV, YOLO, etc., are installed and configured.

       Examples:
       ---------
       >>> detector = ScreenDetector(model=yolo_model_object)
       >>> detector.initialize()
       >>> detector.detect_objects()
       """
    def __init__(self, model):
        """Initialize the ScreenDetector with the YOLO model.

        Args:
            model (YOLO): The actual YOLO model object.
        """
        self.model = model
        self.window_name = 'Detections'

    def initialize_display(self, screen_number=1):
        """
        Initializes the display window and moves it to the specified screen number.

        Args:
            screen_number (int): The screen number to which the window should be moved.
        """
        # Create an initial black image just for the purpose of initialization
        initial_display = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imshow(self.window_name, initial_display)

        # Move the window to the specified screen
        monitors = get_monitors()
        if screen_number < len(monitors):
            monitor = monitors[screen_number]
            cv2.moveWindow(self.window_name, monitor.x, monitor.y)

    def detect_on_screen(self):
        """Detect objects on the live screen and display them."""
        while True:
            # Capture a screenshot of the entire screen
            screenshot = ImageGrab.grab()
            image = np.array(screenshot)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Run the YOLO model on the screenshot to get detections
            results = self.model(image)
            for result in results:
                for box in result.boxes.data:
                    x_min, y_min, x_max, y_max, confidence, class_id = map(int, box.tolist())
                    # Draw a rectangle around the detected object
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 4)
                    label = f"Class: {class_id}, Conf: {confidence:.2f}"
                    # Display the class and confidence of the detected object
                    cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

            # Update the window with the latest screenshot and detections
            cv2.imshow(self.window_name, image)

            # Press "Q" to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
