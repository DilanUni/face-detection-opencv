import cv2 as cv
from typing import Optional

class VideoHandler:
    """
    A class for handling video input from files or cameras.
    
    Provides functionality to:
    - Capture frames from video sources
    - Resize frames
    - Convert frames to grayscale
    - Manage video capture resources
    
    Attributes:
        cap (cv.VideoCapture): OpenCV video capture object
    """
    def __init__(self, source: str | int = 0):
        self.cap = cv.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")

    def get_frame(self) -> Optional[cv.typing.MatLike]:
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self) -> None:
        self.cap.release()

    @staticmethod
    def resize_frame(frame: cv.typing.MatLike,
                     width: int = 640.0,
                     height: int = 360.0,
                     downscale_factor: float = 1.0
                     ) -> cv.typing.MatLike:
        """
        Resize a video frame while maintaining aspect ratio.
        
        Args:
            frame: Input frame to resize
            width: Base width for scaling
            height: Base height for scaling
            downscale_factor: Multiplier for resizing (1.0 = original)
            
        Returns:
            Resized frame as numpy array
            
        Example:
            To halve the frame size: resize_frame(frame, downscale_factor=0.5)
        """
        return cv.resize(frame, (int(width * downscale_factor), int(height * downscale_factor)))

    '''
    @staticmethod
    def to_gray_scale(frame: cv.typing.MatLike) -> cv.typing.MatLike:
        return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    '''