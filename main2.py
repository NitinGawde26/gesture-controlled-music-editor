import cv2
import numpy as np
import time
import logging
import os
from hand_tracker import process_frame
from audio_controller import control_audio, start_audio, stop_audio, get_audio_params, reset_audio, toggle_max_lofi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='main.log',
    filemode='w'
)
logger = logging.getLogger(__name__)

def main():
    """Main application entry point."""
    logger.info("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam")
        print("Error: Could not open webcam.")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.9
    FONT_THICKNESS = 2
    TEXT_COLOR = (0, 255, 0)
    TITLE_COLOR = (255, 255, 255)
    BG_COLOR = (0, 0, 0)
    PROGRESS_BG = (50, 50, 50)
    PROGRESS_FILL = (0, 255, 100)
    LINE_THICKNESS = 2
    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 720
    
    window_name = "Gesture Music Editor"
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, 0, 0)
    
    logger.info("Starting audio system...")
    print("Starting audio playback...")
    start_audio()
    
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                logger.error("Failed to read frame from webcam")
                print("Error: Failed to read frame from webcam")
                break
                
            frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
            frame = cv2.flip(frame, 1)
            
            frame, hands_data = process_frame(frame)
            
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            # Title
            cv2.putText(frame, "Gesture Music Editor", (10, 40), 
                        FONT, 1.2, TITLE_COLOR, 3)
            
            # FPS Counter
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(frame, fps_text, (WINDOW_WIDTH - 150, 30), 
                        FONT, 0.7, TEXT_COLOR, FONT_THICKNESS)
            
            # Controls hint
            cv2.putText(frame, "Press 'S' for lofi toggle | 'R' to reset | ESC to exit", 
                        (10, WINDOW_HEIGHT - 20), FONT, 0.5, (150, 150, 150), 1)
            
            if len(hands_data) == 2:
                if hands_data[0]["side"] == "left" and hands_data[1]["side"] == "right":
                    left_data, right_data = hands_data[0], hands_data[1]
                elif hands_data[0]["side"] == "right" and hands_data[1]["side"] == "left":
                    left_data, right_data = hands_data[1], hands_data[0]
                else:
                    hands_data.sort(key=lambda h: h["wrist"][0])
                    left_data, right_data = hands_data[0], hands_data[1]
                
                left_dist = left_data["thumb_index_dist"]
                right_dist = right_data["thumb_index_dist"]
                thumb_dist = left_data["inter_thumb_dist"]
                
                logger.info(f"Passing to control_audio - left_dist: {left_dist:.1f}, right_dist: {right_dist:.1f}, thumb_dist: {thumb_dist:.1f}")
                
                control_audio(left_dist, right_dist, thumb_dist)
                
                volume, pan, vocal_mix, lofi_intensity = get_audio_params()
                
                # Draw hand connections
                for hand in hands_data:
                    thumb_pos = (hand["thumb_x"], hand["thumb_y"])
                    index_pos = (hand["index_x"], hand["index_y"])
                    line_color = (255, 255, 255)
                    cv2.line(frame, thumb_pos, index_pos, line_color, LINE_THICKNESS)
                
                left_thumb = left_data["thumb_pos"]
                right_thumb = right_data["thumb_pos"]
                cv2.line(frame, left_thumb, right_thumb, (255, 255, 255), LINE_THICKNESS)
                
                # Feature display with progress bars
                features = [
                    {"name": "VOLUME", "value": volume, "color": (0, 100, 255)},
                    {"name": "VOCALS", "value": vocal_mix, "color": (255, 100, 0)},
                    {"name": "LOFI", "value": lofi_intensity, "color": (255, 0, 150)}
                ]
                
                start_y = 100
                bar_width = 300
                bar_height = 25
                
                for i, feature in enumerate(features):
                    y_pos = start_y + (i * 80)
                    
                    # Feature name
                    cv2.putText(frame, feature["name"], (50, y_pos), 
                                FONT, 0.8, feature["color"], 2)
                    
                    # Percentage text
                    percentage = int(feature["value"] * 100)
                    perc_text = f"{percentage}%"
                    cv2.putText(frame, perc_text, (50, y_pos + 35), 
                                FONT, 0.7, TEXT_COLOR, 2)
                    
                    # Progress bar background
                    cv2.rectangle(frame, (200, y_pos - 15), 
                                (200 + bar_width, y_pos - 15 + bar_height), 
                                PROGRESS_BG, -1)
                    
                    # Progress bar fill
                    fill_width = int(bar_width * feature["value"])
                    if fill_width > 0:
                        cv2.rectangle(frame, (200, y_pos - 15), 
                                    (200 + fill_width, y_pos - 15 + bar_height), 
                                    feature["color"], -1)
                    
                    # Progress bar border
                    cv2.rectangle(frame, (200, y_pos - 15), 
                                (200 + bar_width, y_pos - 15 + bar_height), 
                                (255, 255, 255), 2)
                
                # Pan indicator
                pan_text = f"PAN: {'←' if pan < -0.1 else '→' if pan > 0.1 else '●'} {abs(pan)*100:.0f}%"
                cv2.putText(frame, pan_text, (50, start_y + 280), 
                            FONT, 0.7, (255, 255, 0), 2)
                
            else:
                # Show message when hands not detected
                cv2.putText(frame, "Show both hands to control audio", 
                            (WINDOW_WIDTH//2 - 250, WINDOW_HEIGHT//2),
                            FONT, 1.0, (0, 100, 255), 2)
            
            cv2.imshow(window_name, frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                logger.info("ESC pressed, exiting")
                print("ESC pressed, exiting...")
                break
            elif key == ord('r') or key == ord('R'):  # Reset
                logger.info("Reset audio requested")
                print("Resetting audio...")
                reset_audio()
            elif key == ord('s') or key == ord('S'):  # Toggle lofi
                logger.info("Toggle max lofi requested")
                print("Toggling max lofi...")
                toggle_max_lofi()
    
    except Exception as e:
        logger.error(f"Error in main loop: {str(e)}")
        print(f"Error: {str(e)}")
        
    finally:
        logger.info("Cleaning up resources")
        print("Cleaning up...")
        stop_audio()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()