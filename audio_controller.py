import numpy as np
import sounddevice as sd
from pydub import AudioSegment
import threading
import os
import time
import logging
import warnings
import gc
import queue
from pedalboard import Pedalboard, Reverb, LowpassFilter, PitchShift, Chorus

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Ensure SoX is in PATH
AudioSegment.converter = r"C:\Program Files (x86)\sox-14.4.2\sox.exe"

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='audio_controller.log',
    filemode='w'
)
logger = logging.getLogger(__name__)

# Set FFmpeg path
AudioSegment.converter = r"C:\Users\Administrator\Downloads\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"

# Load vocal and instrumental audio files
VOCAL_PATH = r"C:\Users\Administrator\Desktop\python_project\Gesture-Controlled Music Editor\green-day-wake-me-up-when-september-ends-official-audio-128-ytshorts.savetube.me_vocals.wav"
INSTRUMENTAL_PATH = r"C:\Users\Administrator\Desktop\python_project\Gesture-Controlled Music Editor\green-day-wake-me-up-when-september-ends-official-audio-128-ytshorts.savetube.me_instrumental.wav"

# Validate file paths
if not os.path.exists(VOCAL_PATH):
    raise FileNotFoundError(f"Vocal audio file not found at: {VOCAL_PATH}")
if not os.path.exists(INSTRUMENTAL_PATH):
    raise FileNotFoundError(f"Instrumental audio file not found at: {INSTRUMENTAL_PATH}")

try:
    # Load vocal audio
    vocal_audio = AudioSegment.from_file(VOCAL_PATH)
    sample_rate = vocal_audio.frame_rate
    vocal_data = np.array(vocal_audio.get_array_of_samples())
    if vocal_audio.channels == 2:
        vocal_data = vocal_data.reshape((-1, 2))
    else:
        vocal_data = vocal_data.reshape((-1, 1))
    vocal_data = vocal_data.astype(np.float32) / 32768.0
    logger.info(f"Loaded vocal audio: {VOCAL_PATH}, length: {len(vocal_data)} samples, rate: {sample_rate}")

    # Load instrumental audio
    instrumental_audio = AudioSegment.from_file(INSTRUMENTAL_PATH)
    if instrumental_audio.frame_rate != sample_rate:
        instrumental_audio = instrumental_audio.set_frame_rate(sample_rate)
    instrumental_data = np.array(instrumental_audio.get_array_of_samples())
    if instrumental_audio.channels == 2:
        instrumental_data = instrumental_data.reshape((-1, 2))
    else:
        instrumental_data = instrumental_data.reshape((-1, 1))
    instrumental_data = instrumental_data.astype(np.float32) / 32768.0
    logger.info(f"Loaded instrumental audio: {INSTRUMENTAL_PATH}, length: {len(instrumental_data)} samples, rate: {sample_rate}")

    # Ensure both tracks have the same length
    min_length = min(len(vocal_data), len(instrumental_data))
    vocal_data = vocal_data[:min_length]
    instrumental_data = instrumental_data[:min_length]
    audio_data = vocal_data * 0.5 + instrumental_data * 0.5  # Initial mix

    logger.info(f"Aligned audio tracks to length: {min_length} samples")
except Exception as e:
    logger.error(f"Error loading audio files: {e}", exc_info=True)
    raise

# Buffer parameters
BUFFER_SIZE = 4096 * 4
CROSSFADE_SAMPLES = 512
buffer_queue = queue.Queue(maxsize=2)

# Shared parameters
params = {
    "volume": 1.0,
    "pan": 0.0,
    "vocal_mix": 0.5,  # Controls vocal/instrumental balance (0.0 = full instrumental, 1.0 = full vocal)
    "lofi_intensity": 0.0,
    "stop": False,
    "update_buffer": False,
    "buffer_ready": True,
    "crossfade_active": False,
    "crossfade_progress": 0,
}

# Audio stream state
current_position = 0
stream_lock = threading.Lock()
buffer_lock = threading.Lock()
current_audio_data = audio_data.copy()
next_audio_data = None
last_vocal_mix = 0.5
last_lofi_intensity = 0.0
last_update_time = time.time()
last_gc_time = time.time()
gc_interval = 30.0
silence_buffer = np.zeros((int(sample_rate * 0.1), audio_data.shape[1]), dtype=np.float32)
SMOOTHING_FACTOR = 0.7
volume_smoothed = 1.0
pan_smoothed = 0.0
vocal_mix_smoothed = 0.5
previous_vocal_mix = 0.5
previous_lofi_intensity = 0.0
vocal_mix_history = []
lofi_history = []
HISTORY_LENGTH = 8  # Increased for better smoothing

def normalize_distance(dist, min_val=20, max_val=300):
    """Normalize a distance value to a 0-1 range with smooth, linear control."""
    try:
        if dist is None or np.isnan(dist) or dist < 0:
            logger.warning(f"Invalid distance: {dist}, returning 0.0")
            print(f"Invalid distance value: {dist}")
            return 0.0
        
        # Ensure distance is within valid range
        dist = max(min_val, min(dist, max_val))
        
        # Simple linear normalization for smooth control
        normalized = (dist - min_val) / (max_val - min_val)
        
        # Apply subtle smoothing curve for better feel (much gentler than before)
        # This gives slight ease-in/ease-out but maintains linearity
        enhanced = normalized * normalized * (3.0 - 2.0 * normalized)  # Smooth S-curve
        
        result = max(0.0, min(1.0, enhanced))
        
        print(f"Distance: {dist:.1f} -> Normalized: {normalized:.3f} -> Enhanced: {result:.3f}")
        return result
    except Exception as e:
        logger.error(f"Error normalizing distance: {e}", exc_info=True)
        print(f"ERROR normalizing distance: {e}")
        return 0.0




def update_audio_buffer_worker():
    """Ultra-fast buffer worker with instant response and no lag."""
    global current_audio_data, next_audio_data, current_position, last_vocal_mix, last_lofi_intensity, last_gc_time
    
    # INSTANT processing variables - no smoothing delay
    processing_vocal_mix = 0.5
    processing_lofi_intensity = 0.0
    INSTANT_RESPONSE = 0.85  # Much more aggressive for instant response
    
    try:
        with buffer_lock:
            current_audio_data = audio_data.copy()
    except Exception as e:
        logger.error(f"Buffer setup error: {e}", exc_info=True)
    
    while not params["stop"]:
        try:
            current_time = time.time()
            if current_time - last_gc_time > gc_interval:
                gc.collect()
                last_gc_time = current_time
            
            update_needed = False
            target_vocal_mix = 0
            target_lofi_intensity = 0
            
            # Check if update is needed
            with stream_lock:
                if params["update_buffer"] and params["buffer_ready"]:
                    update_needed = True
                    target_vocal_mix = params["vocal_mix"]
                    target_lofi_intensity = params["lofi_intensity"]
                    params["update_buffer"] = False
                    params["buffer_ready"] = False
            
            if update_needed:
                try:
                    # INSTANT response - minimal smoothing for immediate effect
                    change_threshold = 0.05  # Lower threshold for instant response
                    
                    vocal_change = abs(target_vocal_mix - processing_vocal_mix)
                    lofi_change = abs(target_lofi_intensity - processing_lofi_intensity)
                    
                    # INSTANT updates for any significant change
                    if vocal_change > change_threshold or lofi_change > change_threshold:
                        # Direct assignment for instant response
                        processing_vocal_mix = target_vocal_mix * INSTANT_RESPONSE + processing_vocal_mix * (1.0 - INSTANT_RESPONSE)
                        processing_lofi_intensity = target_lofi_intensity * INSTANT_RESPONSE + processing_lofi_intensity * (1.0 - INSTANT_RESPONSE)
                    else:
                        # Minimal smoothing for micro-adjustments
                        processing_vocal_mix = target_vocal_mix * 0.7 + processing_vocal_mix * 0.3
                        processing_lofi_intensity = target_lofi_intensity * 0.7 + processing_lofi_intensity * 0.3
                    
                    # Generate new audio with preserved timing
                    new_audio_data = apply_vocal_mix_and_lofi(processing_vocal_mix, processing_lofi_intensity)
                    
                    # CRITICAL: Instant buffer swap with position preservation
                    with buffer_lock:
                        preserved_position = current_position
                        old_length = len(current_audio_data)
                        new_length = len(new_audio_data)
                        
                        if old_length > 0 and new_length > 0:
                            position_ratio = preserved_position / old_length
                            adjusted_position = int(position_ratio * new_length)
                            adjusted_position = max(0, min(adjusted_position, new_length - 1))
                        else:
                            adjusted_position = 0
                        
                        # Instant atomic swap
                        current_audio_data = new_audio_data
                        current_position = adjusted_position
                    
                    with stream_lock:
                        last_vocal_mix = processing_vocal_mix
                        last_lofi_intensity = processing_lofi_intensity
                    
                    # Display percentage-based feedback
                    vocal_percent = int(processing_vocal_mix * 100)
                    lofi_percent = int(processing_lofi_intensity * 100)
                    print(f"INSTANT UPDATE: Vocal={vocal_percent}%, Lofi={lofi_percent}%, pos={adjusted_position}")
                    
                except Exception as e:
                    logger.error(f"Buffer processing error: {e}", exc_info=True)
                    print(f"BUFFER ERROR: {e}")
                
                # Reset buffer ready flag
                with stream_lock:
                    params["buffer_ready"] = True
                    
        except Exception as e:
            logger.error(f"Buffer worker error: {e}", exc_info=True)
            with stream_lock:
                params["buffer_ready"] = True
        
        # Ultra-fast update cycle for zero lag
        time.sleep(0.005)  # Reduced to 5ms for instant response



def control_audio(left_dist, right_dist, hand_dist):
    """Lag-free audio control with instant response and percentage-based feedback."""
    global last_update_time, volume_smoothed, pan_smoothed, vocal_mix_smoothed, last_vocal_mix, last_lofi_intensity
    
    try:
        print(f"\nDistance values - Left: {left_dist}, Right: {right_dist}, Hand: {hand_dist}")
        
        # Normalize distances with instant response
        left_norm = normalize_distance(left_dist, min_val=15, max_val=280)
        right_norm = normalize_distance(right_dist, min_val=20, max_val=300)
        hand_norm = normalize_distance(hand_dist, min_val=25, max_val=320)
        inverted_hand_norm = 1.0 - hand_norm
        
        current_time = time.time()
        
        # INSTANT vocal mixing with percentage-based control
        FULL_VOCAL_THRESHOLD = 35
        FULL_INSTRUMENTAL_THRESHOLD = 110
        INSTANT_ZONE = 5  # Reduced hysteresis for instant response
        
        # Instant transition zones
        if left_dist <= FULL_VOCAL_THRESHOLD - INSTANT_ZONE:
            target_vocal_mix = 1.0  # 100% vocal
        elif left_dist >= FULL_INSTRUMENTAL_THRESHOLD + INSTANT_ZONE:
            target_vocal_mix = 0.0  # 100% instrumental
        else:
            # Linear interpolation for smooth percentage control
            effective_min = FULL_VOCAL_THRESHOLD - INSTANT_ZONE
            effective_max = FULL_INSTRUMENTAL_THRESHOLD + INSTANT_ZONE
            normalized_dist = (left_dist - effective_min) / (effective_max - effective_min)
            target_vocal_mix = 1.0 - normalized_dist
        
        # INSTANT vocal response - no smoothing lag
        INSTANT_VOCAL_RESPONSE = 0.9  # Very aggressive for instant response
        vocal_mix_smoothed = vocal_mix_smoothed * (1 - INSTANT_VOCAL_RESPONSE) + target_vocal_mix * INSTANT_VOCAL_RESPONSE
        
        # INSTANT lofi response with percentage control
        lofi_target = inverted_hand_norm
        
        # Direct assignment for instant lofi response
        INSTANT_LOFI_RESPONSE = 0.85  # Aggressive for instant hand distance response
        if not hasattr(control_audio, 'lofi_smoothed'):
            control_audio.lofi_smoothed = 0.0
        
        control_audio.lofi_smoothed = control_audio.lofi_smoothed * (1 - INSTANT_LOFI_RESPONSE) + lofi_target * INSTANT_LOFI_RESPONSE
        
        with stream_lock:
            # INSTANT volume and pan with percentage feedback
            INSTANT_VOLUME_PAN = 0.8
            target_volume = max(0.1, min(1.0, right_norm ** 0.7))
            volume_smoothed = volume_smoothed * (1 - INSTANT_VOLUME_PAN) + target_volume * INSTANT_VOLUME_PAN
            params["volume"] = volume_smoothed
            
            target_pan = max(-1.0, min(1.0, -1 + 2 * left_norm))
            pan_smoothed = pan_smoothed * (1 - INSTANT_VOLUME_PAN) + target_pan * INSTANT_VOLUME_PAN
            params["pan"] = pan_smoothed
            
            new_lofi_intensity = control_audio.lofi_smoothed
            new_vocal_mix = vocal_mix_smoothed
            
            # Display percentage-based feedback
            volume_percent = int(params['volume'] * 100)
            pan_percent = int((params['pan'] + 1) * 50)  # Convert -1 to 1 range to 0-100%
            vocal_percent = int(new_vocal_mix * 100)
            lofi_percent = int(new_lofi_intensity * 100)
            
            print(f"INSTANT Control - Volume: {volume_percent}%, Pan: {pan_percent}%, "
                  f"Vocal: {vocal_percent}%, Lofi: {lofi_percent}%")
            
            # INSTANT UPDATES with minimal delay
            if current_time - last_update_time >= 0.008:  # Reduced to 8ms for instant response
                last_update_time = current_time
                
                # Very low thresholds for instant response
                vocal_change = abs(new_vocal_mix - last_vocal_mix)
                lofi_change = abs(new_lofi_intensity - last_lofi_intensity)
                
                instant_vocal_threshold = 0.01  # Very low for instant response
                instant_lofi_threshold = 0.01   # Very low for instant response
                
                if (vocal_change > instant_vocal_threshold or lofi_change > instant_lofi_threshold) and params["buffer_ready"]:
                    params["vocal_mix"] = new_vocal_mix
                    params["lofi_intensity"] = new_lofi_intensity
                    params["update_buffer"] = True
                    
                    print(f">>> INSTANT UPDATE - NO LAG <<<")
                    print(f"    Vocal: {vocal_percent}% (Δ: {vocal_change:.4f})")
                    print(f"    Lofi: {lofi_percent}% (Δ: {lofi_change:.4f})")
                    
                    last_vocal_mix = new_vocal_mix
                    last_lofi_intensity = new_lofi_intensity
                    
    except Exception as e:
        logger.error(f"Control audio error: {e}", exc_info=True)
        print(f"CONTROL ERROR: {e}")



def crossfade_buffers(buffer1, buffer2, position, crossfade_progress, fade_length):
    try:
        buf1_length = len(buffer1)
        buf2_length = len(buffer2)
        frames_needed = min(fade_length, buf1_length - position, buf2_length - position)
        if frames_needed <= 0:
            return np.zeros((0, buffer1.shape[1]), dtype=np.float32)
        if position < buf1_length:
            chunk1 = buffer1[position:position+frames_needed].copy()
        else:
            chunk1 = np.zeros((frames_needed, buffer1.shape[1]), dtype=np.float32)
        if position < buf2_length:
            chunk2 = buffer2[position:position+frames_needed].copy()
        else:
            chunk2 = np.zeros((frames_needed, buffer1.shape[1]), dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, frames_needed).reshape(-1, 1)
        fade_in = np.linspace(0.0, 1.0, frames_needed).reshape(-1, 1)
        result = chunk1 * fade_out + chunk2 * fade_in
        return result
    except Exception as e:
        logger.error(f"Crossfade error: {e}", exc_info=True)
        return np.zeros((0, buffer1.shape[1]), dtype=np.float32)

def audio_callback(outdata, frames, time_info, status):
    """Ultra-optimized audio callback with seamless buffer handling."""
    global current_position, current_audio_data
    
    try:
        if status and status.output_underflow:
            logger.warning("Audio underflow detected")
        
        # Minimal locking for parameters
        with stream_lock:
            volume = params["volume"]
            pan = params["pan"]
        
        # Quick buffer access with position preservation
        with buffer_lock:
            local_audio_data = current_audio_data
            buffer_length = len(local_audio_data)
            pos = current_position
        
        if buffer_length == 0:
            outdata.fill(0)
            return
        
        # Seamless audio processing without position jumps
        end_position = pos + frames
        
        if end_position >= buffer_length:
            # Handle wraparound smoothly
            remaining = buffer_length - pos
            
            if remaining > 0:
                # First part - from current position to end
                chunk = local_audio_data[pos:buffer_length].copy()
                chunk = chunk * volume
                
                if chunk.shape[1] == 2:
                    left_gain = min(1.0, 1.0 - max(0, pan))
                    right_gain = min(1.0, 1.0 + min(0, pan))
                    chunk = chunk * np.array([left_gain, right_gain])
                
                outdata[:remaining] = chunk
            
            # Second part - seamless loop from beginning
            remaining_frames = frames - remaining
            if remaining_frames > 0 and buffer_length > 0:
                loop_end = min(remaining_frames, buffer_length)
                
                if loop_end > 0:
                    loop_chunk = local_audio_data[0:loop_end].copy()
                    loop_chunk = loop_chunk * volume
                    
                    if loop_chunk.shape[1] == 2:
                        left_gain = min(1.0, 1.0 - max(0, pan))
                        right_gain = min(1.0, 1.0 + min(0, pan))
                        loop_chunk = loop_chunk * np.array([left_gain, right_gain])
                    
                    outdata[remaining:remaining+loop_end] = loop_chunk
                
                # Fill any remaining frames with silence
                if remaining_frames > buffer_length:
                    outdata[remaining+loop_end:] = 0
            
            # Update position atomically
            new_position = remaining_frames % buffer_length if buffer_length > 0 else 0
            current_position = new_position
            
        else:
            # Normal case - no wraparound needed
            chunk = local_audio_data[pos:end_position].copy()
            chunk = chunk * volume
            
            if chunk.shape[1] == 2:
                left_gain = min(1.0, 1.0 - max(0, pan))
                right_gain = min(1.0, 1.0 + min(0, pan))
                chunk = chunk * np.array([left_gain, right_gain])
            
            outdata[:len(chunk)] = chunk
            
            # Fill remaining with silence if needed
            if len(chunk) < frames:
                outdata[len(chunk):] = 0
            
            current_position = end_position
                
    except Exception as e:
        logger.error(f"Audio callback error: {e}", exc_info=True)
        outdata.fill(0)
        # Reset position on error to prevent stuck states
        current_position = 0


def audio_loop():
    global current_audio_data
    try:
        if len(current_audio_data) == 0:
            current_audio_data = audio_data.copy()
        stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=audio_data.shape[1],
            callback=audio_callback,
            blocksize=BUFFER_SIZE,
            device=None,
            latency='high'
        )
        stream.start()
        logger.info(f"Audio stream started: {sample_rate} Hz, blocksize: {BUFFER_SIZE}")
        while not params["stop"]:
            sd.sleep(100)
        stream.stop()
        stream.close()
        logger.info("Audio stream stopped")
    except Exception as e:
        logger.error(f"Audio stream error: {e}", exc_info=True)

def start_audio():
    logger.info("Starting audio system...")
    stream_thread = threading.Thread(target=audio_loop, daemon=True)
    stream_thread.start()
    buffer_thread = threading.Thread(target=update_audio_buffer_worker, daemon=True)
    buffer_thread.start()
    logger.info("Audio system started")

def stop_audio():
    logger.info("Stopping audio system...")
    params["stop"] = True

def get_audio_params():
    with stream_lock:
        return params["volume"], params["pan"], params["vocal_mix"], params["lofi_intensity"]

def reset_audio():
    global current_position, current_audio_data, next_audio_data, volume_smoothed, pan_smoothed, vocal_mix_smoothed, last_vocal_mix, last_lofi_intensity
    try:
        with buffer_lock:
            current_audio_data = audio_data.copy()
            next_audio_data = None
        with stream_lock:
            current_position = 0
            params["volume"] = 1.0
            params["pan"] = 0.0
            params["vocal_mix"] = 0.5
            params["lofi_intensity"] = 0.0
            params["update_buffer"] = False
            params["buffer_ready"] = True
            params["crossfade_active"] = False
            params["crossfade_progress"] = 0
            volume_smoothed = 1.0
            pan_smoothed = 0.0
            vocal_mix_smoothed = 0.5
            last_vocal_mix = 0.5
            last_lofi_intensity = 0.0
        logger.info("Audio reset to initial state")
        gc.collect()
        return True
    except Exception as e:
        logger.error(f"Error resetting audio: {e}", exc_info=True)
        return False

def toggle_max_lofi():
    """Instant toggle between maximum lofi and normal with percentage display."""
    global current_audio_data, next_audio_data, last_lofi_intensity
    
    with stream_lock:
        current_intensity = params["lofi_intensity"]
        
        if current_intensity < 0.5:
            params["lofi_intensity"] = 1.0
            print(f"\n🎵 INSTANT MAXIMUM LOFI: 100% 🎵")
            print(f"   └── Extreme slowed+reverb activated instantly!")
        else:
            params["lofi_intensity"] = 0.0
            print(f"\n🎵 INSTANT NORMAL: 0% Lofi 🎵")
        
        # Force instant update
        params["update_buffer"] = True
        last_lofi_intensity = -1.0
    
    try:
        lofi_percent = int(params['lofi_intensity'] * 100)
        print(f"INSTANT lofi toggle to: {lofi_percent}%")
        
        with stream_lock:
            vocal_mix = params["vocal_mix"]
            lofi = params["lofi_intensity"]
        
        new_audio = apply_vocal_mix_and_lofi(vocal_mix, lofi)
        
        with buffer_lock:
            next_audio_data = new_audio
            with stream_lock:
                params["crossfade_active"] = True
                params["crossfade_progress"] = 0
        
        print(f"✅ INSTANT lofi effect: {lofi_percent}%. Zero lag applied.")
        return lofi
    except Exception as e:
        print(f"❌ Error in toggle_max_lofi: {e}")
        logger.error(f"Error in toggle_max_lofi: {e}", exc_info=True)
        return params["lofi_intensity"]
    

def _handle_normal_audio(outdata, pos, frames, audio_buffer, buffer_length, volume, pan, offset):
    """Handle normal audio processing without crossfade"""
    global current_position
    
    try:
        end_position = pos + frames
        
        if end_position >= buffer_length:
            # Handle wraparound
            remaining = buffer_length - pos
            if remaining > 0:
                chunk = audio_buffer[pos:buffer_length].copy()
                chunk = chunk * volume
                if chunk.shape[1] == 2:
                    left_gain = min(1.0, 1.0 - max(0, pan))
                    right_gain = min(1.0, 1.0 + min(0, pan))
                    chunk[:, 0] *= left_gain
                    chunk[:, 1] *= right_gain
                outdata[offset:offset+remaining] = chunk
            
            # Loop remaining frames
            remaining_frames = frames - remaining
            if remaining_frames > 0:
                loop_end = min(remaining_frames, buffer_length)
                if loop_end > 0:
                    loop_chunk = audio_buffer[0:loop_end].copy()
                    loop_chunk = loop_chunk * volume
                    if loop_chunk.shape[1] == 2:
                        left_gain = min(1.0, 1.0 - max(0, pan))
                        right_gain = min(1.0, 1.0 + min(0, pan))
                        loop_chunk[:, 0] *= left_gain
                        loop_chunk[:, 1] *= right_gain
                    outdata[offset+remaining:offset+remaining+loop_end] = loop_chunk
            
            with stream_lock:
                current_position = remaining_frames % buffer_length
        else:
            # Normal case - no wraparound
            chunk = audio_buffer[pos:end_position].copy()
            chunk = chunk * volume
            if chunk.shape[1] == 2:
                left_gain = min(1.0, 1.0 - max(0, pan))
                right_gain = min(1.0, 1.0 + min(0, pan))
                chunk[:, 0] *= left_gain
                chunk[:, 1] *= right_gain
            outdata[offset:offset+len(chunk)] = chunk
            
            with stream_lock:
                current_position = end_position
                
    except Exception as e:
        logger.error(f"Error in _handle_normal_audio: {e}", exc_info=True)
        print(f"HANDLE_NORMAL_AUDIO ERROR: {e}")
        # Fill with silence on error
        outdata[offset:] = 0

import logging

logger = logging.getLogger(__name__)

# 4. APPLY VOCAL MIX AND LOFI - Enhanced debugging and error handling
def apply_vocal_mix_and_lofi(vocal_mix, lofi_intensity):
    """Instant vocal mixing and lofi effects with percentage-based control and zero lag."""
    global previous_vocal_mix, previous_lofi_intensity
    
    try:
        # Input validation and clamping
        vocal_mix = np.clip(vocal_mix, 0.0, 1.0)
        lofi_intensity = np.clip(lofi_intensity, 0.0, 1.0)
        
        # INSTANT response - minimal smoothing for zero lag
        INSTANT_SMOOTHING = 0.9  # Very aggressive for instant response
        
        # Direct assignment for instant vocal transitions
        vocal_mix = previous_vocal_mix * (1 - INSTANT_SMOOTHING) + vocal_mix * INSTANT_SMOOTHING
        
        # INSTANT lofi response for hand distance
        lofi_change = abs(lofi_intensity - previous_lofi_intensity)
        if lofi_change > 0.05:  # Any significant change gets instant response
            lofi_intensity = previous_lofi_intensity * 0.1 + lofi_intensity * 0.9  # 90% instant
        else:
            lofi_intensity = previous_lofi_intensity * 0.2 + lofi_intensity * 0.8  # 80% instant
        
        previous_vocal_mix = vocal_mix
        previous_lofi_intensity = lofi_intensity
        
        # Display percentage feedback
        vocal_percent = int(vocal_mix * 100)
        lofi_percent = int(lofi_intensity * 100)
        print(f"INSTANT Processing: Vocal={vocal_percent}%, Lofi={lofi_percent}%")
        
        # Efficient audio mixing with exact length preservation
        min_length = min(len(vocal_data), len(instrumental_data))
        vocal_part = vocal_data[:min_length].astype(np.float32)
        instrumental_part = instrumental_data[:min_length].astype(np.float32)
        
        # INSTANT crossfade mixing - no smoothing curves for immediate response
        fade_vocal = vocal_mix
        fade_instrumental = 1 - vocal_mix
        
        # Direct mixing for instant response
        audio_array = vocal_part * fade_vocal + instrumental_part * fade_instrumental
        
        # INSTANT lofi effects application
        if lofi_intensity > 0.01:  # Very low threshold for instant onset
            print(f"INSTANT Lofi Application: {lofi_percent}%")
            
            # INSTANT time stretching
            if lofi_intensity > 0.05:  # Lower threshold for instant stretch
                stretch_factor = 1.0 + (lofi_intensity * 0.3)  # Immediate stretch response
                
                if stretch_factor > 1.02:  # Very low threshold for instant effect
                    original_length = len(audio_array)
                    target_length = int(original_length * stretch_factor)
                    
                    # Fast interpolation for instant response
                    old_indices = np.linspace(0, original_length - 1, original_length)
                    new_indices = np.linspace(0, original_length - 1, target_length)
                    
                    if len(audio_array.shape) == 2:  # Stereo
                        stretched = np.zeros((target_length, 2), dtype=np.float32)
                        stretched[:, 0] = np.interp(new_indices, old_indices, audio_array[:, 0])
                        stretched[:, 1] = np.interp(new_indices, old_indices, audio_array[:, 1])
                        audio_array = stretched
                    else:  # Mono
                        audio_array = np.interp(new_indices, old_indices, audio_array).astype(np.float32)
            
            # INSTANT filtering
            try:
                from scipy import signal
                
                # Immediate cutoff calculation
                base_cutoff = 18000
                min_cutoff = 5000
                cutoff_reduction = lofi_intensity ** 0.3  # More immediate curve
                target_cutoff = base_cutoff - (cutoff_reduction * (base_cutoff - min_cutoff))
                
                nyquist = sample_rate / 2
                normalized_cutoff = min(0.98, target_cutoff / nyquist)
                
                # Instant filter application
                b, a = signal.butter(2, normalized_cutoff, btype='low')
                
                if len(audio_array.shape) == 2:
                    audio_array[:, 0] = signal.lfilter(b, a, audio_array[:, 0])
                    audio_array[:, 1] = signal.lfilter(b, a, audio_array[:, 1])
                else:
                    audio_array = signal.lfilter(b, a, audio_array)
                
            except ImportError:
                pass
            except Exception as e:
                print(f"Filter error (non-critical): {e}")
            
            # Instant noise addition
            noise_level = lofi_intensity * 0.0008
            if noise_level > 0.00005:  # Very low threshold
                noise_shape = audio_array.shape
                noise = np.random.normal(0, noise_level, noise_shape).astype(np.float32)
                noise_blend = min(1.0, lofi_intensity * 4.0)  # Instant blending
                audio_array = audio_array * (1 - noise_blend * 0.1) + noise * noise_blend
        
        # Light normalization to prevent clipping
        max_amplitude = np.max(np.abs(audio_array))
        if max_amplitude > 0.88:
            audio_array = audio_array * (0.82 / max_amplitude)
        
        return audio_array.astype(np.float32)
        
    except Exception as e:
        print(f"ERROR in processing: {e}")
        logger.error(f"Error in apply_vocal_mix_and_lofi: {e}", exc_info=True)
        
        # Safe fallback
        try:
            fallback_length = min(len(vocal_data), len(instrumental_data))
            fallback = (vocal_data[:fallback_length] * 0.5 + instrumental_data[:fallback_length] * 0.5).astype(np.float32)
            return fallback
        except:
            return np.zeros((44100, 2), dtype=np.float32)
