import time
import os
from typing import List, Tuple
import cv2 as cv
from detectors.face_detection import FaceDetector
from utils.fps_counter import FPSCounter
from utils.video_utils import VideoHandler

def main():
    frame_number: int = 0
    last_capture_time: float = 0
    CAPTURE_DELAY: float = 2.0  # 2 seconds between captures

    # Configure output photo directory
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    photos_dir = os.path.join(parent_dir, "photos")
    
    # Create photos directory if it doesn't exist
    os.makedirs(photos_dir, exist_ok=True)

    try:
        '''
        # Load video
        video_path = 'opencv-project/videos/video.mp4'
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"File {video_path} not found")'''
        
        # Initialize video capture (0 for webcam)
        video_handler = VideoHandler(0) # Change to video path for file processing
        face_detector = FaceDetector()
        fps_counter = FPSCounter()

        while True:
            frame = video_handler.get_frame()
            if frame is None:
                break
            
            # Downscale frame for faster processing     
            down_scale_frame = VideoHandler.resize_frame(frame=frame, downscale_factor=1.0) # 1.0 for normal frame scale
            # gray_frame = cv.cvtColor(down_scale_frame, cv.COLOR_BGR2GRAY)

            # Detect faces and draw visualization rectangles
            faces = face_detector.detect_faces(down_scale_frame)
            face_detector.draw_rectangles(down_scale_frame, faces)

            # Capture cooldown period
            current_time = time.time()
            if len(faces) > 0 and (current_time - last_capture_time) >= CAPTURE_DELAY:
                timestamp = int(current_time)
                photo_path = os.path.join(photos_dir, f"face_{timestamp}.jpg")
                
                # Save original frame (without detection rectangles)
                cv.imwrite(photo_path, frame)
                print(f"[INFO] Saved face photo: {photo_path}")
                last_capture_time = current_time

            # Video metrics
            fps_text = fps_counter.update()
            FPSCounter.display_fps(down_scale_frame, fps_text)
            FPSCounter.display_frame_number(frame=down_scale_frame, frame_number=frame_number)

            cv.imshow('Face Detection', down_scale_frame)
            frame_number += 1

            if cv.waitKey(1) == 27: # Exit on ESC key
                break

    except Exception as e:
        print(f"[ERROR] Runtime exception: {e}")
    finally:
        if video_handler is not None:
            video_handler.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()