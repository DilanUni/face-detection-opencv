import time
import cv2 as cv
from typing import Tuple

FPS_POSITION: Tuple[int, int] = (10, 30)
TEXT_COLOR: Tuple[int, int, int] = (0, 255, 0)

class FPSCounter:
    """
    A utility class for calculating and displaying frames per second (FPS) metrics.
    
    This class provides functionality to:
    - Calculate real-time FPS values
    - Display FPS counter on video frames
    - Display frame numbers on video frames
    
    Attributes:
        prev_time (float): Timestamp of previous frame processing
        fps_text (str): Formatted FPS string for display
    """
    def __init__(self) -> None:
        self.prev_time: float = 0
        self.fps_text: str = "FPS: 0"

    def update(self) -> str:
        """
        Update and calculate the current FPS value.
        
        Returns:
            str: Formatted FPS string (e.g., "FPS: 30")
            
        Note:
            This should be called once per frame for accurate measurement.
        """
        curr_time: float = time.time()
        fps_val: float = 1 / (curr_time - self.prev_time) if (curr_time - self.prev_time) != 0 else 0
        self.prev_time: float = curr_time
        self.fps_text = f"FPS: {int(fps_val)}"
        return self.fps_text

    @staticmethod
    def display_fps(frame: cv.typing.MatLike,
                    text: str,
                    color: Tuple[int, int, int] = TEXT_COLOR,
                    position: Tuple[int, int] = FPS_POSITION,
                    font_scale: float = 1.0,
                    thickness: int = 2,
                    margin: int = 10,) -> None:
        """
        Display FPS counter on a video frame.
        
        Args:
            frame: Input frame (BGR format) where text will be drawn
            text: FPS text to display
            color: Text color in BGR format (default: green)
            position: Base position coordinates (x,y)
            font_scale: Font size multiplier
            thickness: Text stroke thickness
            margin: Padding from frame edges
            
        Note:
            The text will be automatically positioned in the top-left corner
            with proper margin calculation.
        """      
        (text_width, text_height), _ = cv.getTextSize(
        text=text,
        fontFace=cv.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        thickness=thickness)
        
        x = margin 
        y = margin + text_height
        
        cv.putText(
        img=frame,
        text=text,
        org=(x, y),
        fontFace=cv.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        color=color,
        thickness=thickness) 
    
    @staticmethod
    def display_frame_number(frame: cv.typing.MatLike,
                             position: Tuple[int, int] = FPS_POSITION,
                             color: Tuple[int, int, int] = TEXT_COLOR,
                             frame_number: int = 0,
                             margin: int = 10
                             ) -> None:
        """
        Display frame number on a video frame.
        
        Args:
            frame: Input frame (BGR format) where text will be drawn
            frame_number: Current frame number to display
            color: Text color in BGR format (default: green)
            font_scale: Font size multiplier
            thickness: Text stroke thickness
            margin: Padding from frame edges
            
        Note:
            The text will be automatically positioned in the bottom-right corner
            with proper margin calculation.
        """
        height, width = frame.shape[:2]
        
        text: str = f'Frame {frame_number}'
        
        (text_width, text_height), _ = cv.getTextSize(text=text, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1.0, thickness=2)
        
        x = width - text_width - margin
        
        y = height - margin
        
        cv.putText(img=frame,
                   text=text,
                   org=(x, y),
                   fontFace= cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1.0,
                   color=color,
                   thickness=2)