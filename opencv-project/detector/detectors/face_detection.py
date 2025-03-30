import cv2 as cv
from typing import Tuple, List

class FaceDetector:
    """
    A class for detecting faces in images using Haar Cascade classifier.

    Attributes:
        face_cascade (cv.CascadeClassifier): Pre-trained Haar Cascade classifier for face detection.
        scale_factor (float): Parameter specifying how much the image size is reduced at each image scale.
        min_neighbors (int): Parameter specifying how many neighbors each candidate rectangle should have.
        min_size (Tuple[int, int]): Minimum possible object size. Objects smaller than this are ignored.
    """
    def __init__(self, scale_factor: float = 1.1,
                 min_neighbors: int = 5,
                 min_size: Tuple[int, int] = (30, 30)) -> None:
        """
        Initialize the FaceDetector with Haar Cascade parameters.

        Args:
            scale_factor: Scale factor for multi-scale detection (default: 1.1).
            min_neighbors: Minimum neighbors for detection quality (default: 5).
            min_size: Minimum face size in pixels (default: (30, 30)).

        Raises:
            ValueError: If the Haar Cascade classifier fails to load.
        """
        self.face_cascade = cv.CascadeClassifier(
            cv.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        self.scale_factor: float = scale_factor
        self.min_neighbors: int = min_neighbors
        self.min_size: Tuple[int, int] = min_size

        if self.face_cascade.empty():
            raise ValueError("Failed to load Haar Cascade classifier XML file")

    def detect_faces(self, gray_image: cv.typing.MatLike) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a grayscale image using Haar Cascade detection.

        Args:
            gray_image: Input grayscale image (numpy array of type uint8).

        Returns:
            List of rectangles where faces were detected.
            Each rectangle is represented as a tuple (x, y, width, height).
        """
        return self.face_cascade.detectMultiScale(
            image=gray_image,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )

    @staticmethod
    def draw_rectangles(frame: cv.typing.MatLike,
                        faces: List[Tuple[int, int, int, int]], 
                        color: Tuple[int, int, int] = (255, 0, 0), thickness: int = 2) -> None:
        """
        Draw rectangles around detected faces on the input frame.

        Args:
            frame: Input image (BGR format) where rectangles will be drawn.
            faces: List of face rectangles (x, y, width, height).
            color: BGR color tuple for rectangles (default: red).
            thickness: Thickness of rectangle lines (default: 2).

        Note:
            This operation modifies the input frame in-place.
        """
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), color, thickness)