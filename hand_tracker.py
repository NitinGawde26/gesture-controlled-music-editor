import cv2
import mediapipe as mp
import math
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='hand_tracker.log',
    filemode='w'
)
logger = logging.getLogger(__name__)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2, 
    min_detection_confidence=0.6,  # Slightly lower for performance
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def validate_finger_distances(thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip):
    """
    Validate finger detection to prevent middle finger being mistaken for index finger.
    Returns corrected index finger position or None if detection is unreliable.
    """
    try:
        if thumb_tip is None or index_tip is None:
            return index_tip
        
        # Calculate distances between adjacent fingers
        thumb_index_dist = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))
        
        if middle_tip is not None:
            thumb_middle_dist = np.linalg.norm(np.array(thumb_tip) - np.array(middle_tip))
            index_middle_dist = np.linalg.norm(np.array(index_tip) - np.array(middle_tip))
            
            # Check if index finger is actually middle finger
            if (thumb_middle_dist < thumb_index_dist * 0.8 and 
                index_middle_dist > thumb_index_dist * 1.2):
                # Middle finger is closer to thumb than detected index finger
                # This suggests wrong finger detection
                print("WARNING: Possible middle finger detected as index finger")
                return middle_tip  # Use middle finger as corrected index
        
        # Additional validation using finger ordering
        if ring_tip is not None and middle_tip is not None:
            # Fingers should be in order: thumb, index, middle, ring
            thumb_x = thumb_tip[0]
            index_x = index_tip[0]
            middle_x = middle_tip[0]
            ring_x = ring_tip[0]
            
            # Check if fingers are in reasonable order (assumes right hand)
            if not (thumb_x < index_x < middle_x < ring_x or 
                    thumb_x > index_x > middle_x > ring_x):
                print("WARNING: Finger order seems incorrect")
                # Could implement more sophisticated correction here
        
        return index_tip
        
    except Exception as e:
        print(f"Error in finger validation: {e}")
        return index_tip
    

    
def get_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def process_frame(frame):
    """
    Process a video frame to detect and track hands.
    
    Args:
        frame: Video frame from webcam
        
    Returns:
        Tuple of (processed frame, hand data)
    """
    try:
        h, w, _ = frame.shape
        
        # Resize for faster processing
        small_frame = cv2.resize(frame, (320, 240))
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(rgb_frame)
        
        # List to store hand tracking data
        hand_data = []

        # Check if hands are detected
        if results.multi_hand_landmarks:
            hand_info = []
            
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw landmarks on original frame
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Get wrist position
                wrist = hand_landmarks.landmark[0]
                wrist_pos = (int(wrist.x * w), int(wrist.y * h))
                
                # Get finger landmarks
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]
                
                # Convert to pixel coordinates
                thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                index_pos = (int(index_tip.x * w), int(index_tip.y * h))
                
                # Calculate thumb-index distance
                thumb_index_dist = get_distance(thumb_pos, index_pos)
                
                # Identify hand side
                hand_side = "left" if wrist_pos[0] < w/2 else "right"
                if results.multi_handedness and idx < len(results.multi_handedness):
                    detected_handedness = results.multi_handedness[idx].classification[0].label
                    confidence = results.multi_handedness[idx].classification[0].score
                    logger.debug(f"Hand {idx}: MediaPipe detected {detected_handedness} with confidence {confidence:.2f}")
                    hand_side = detected_handedness.lower()
                
                hand_info.append({
                    "side": hand_side,
                    "landmarks": hand_landmarks,
                    "thumb_pos": thumb_pos,
                    "index_pos": index_pos,
                    "wrist_pos": wrist_pos,
                    "thumb_index_dist": thumb_index_dist
                })
            
            # Sort hands by X position
            hand_info.sort(key=lambda h: h["wrist_pos"][0])
            
            # Process hands
            for idx, hand in enumerate(hand_info):
                hand_data_entry = {
                    "side": hand["side"],
                    "wrist": hand["wrist_pos"],
                    "thumb_x": hand["thumb_pos"][0],
                    "thumb_y": hand["thumb_pos"][1],
                    "index_x": hand["index_pos"][0],
                    "index_y": hand["index_pos"][1],
                    "thumb_pos": hand["thumb_pos"],
                    "thumb_index_dist": hand["thumb_index_dist"]
                }
                hand_data.append(hand_data_entry)
            
            # Calculate inter-hand measurements
            if len(hand_data) == 2:
                inter_thumb_dist = get_distance(hand_data[0]["thumb_pos"], hand_data[1]["thumb_pos"])
                hand_data[0]["inter_thumb_dist"] = inter_thumb_dist
                hand_data[1]["inter_thumb_dist"] = inter_thumb_dist
                
                inter_wrist_dist = get_distance(hand_data[0]["wrist"], hand_data[1]["wrist"])
                hand_data[0]["inter_wrist_dist"] = inter_wrist_dist
                hand_data[1]["inter_wrist_dist"] = inter_wrist_dist
                
                logger.debug(f"Inter-thumb distance: {inter_thumb_dist:.2f}px")
                logger.debug(f"Inter-wrist distance: {inter_wrist_dist:.2f}px")
                logger.debug(f"Left thumb-index distance: {hand_data[0]['thumb_index_dist'] if hand_data[0]['side'] == 'left' else hand_data[1]['thumb_index_dist']:.2f}px")
                
        return frame, hand_data
        
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return frame, []

def __del__():
    hands.close()