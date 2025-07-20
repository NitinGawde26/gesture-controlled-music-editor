# 🎶 Gesture-Controlled Music Editor

A real-time interactive audio processor that allows users to separate vocals and instrumentals, apply LoFi audio effects, and control the entire music editing pipeline using hand gestures. This project leverages **Demucs**, **Pedalboard**, and **MediaPipe** to deliver a unique, hands-free music editing experience.

---

## ✨ Features

- 🎙️ **Vocal and Instrument Separation**  
  Utilizes the Demucs deep learning model to cleanly split audio into vocals and instrumental tracks.

- 🎛️ **LoFi Audio Effects**  
  Apply real-time effects like:
  - Time-stretching  
  - Reverb and pitch shift  
  - Low-pass filtering  
  - White noise overlays

- ✋ **Gesture-Based Control**  
  Control playback, effect switching, and mixing using hand gestures powered by MediaPipe and OpenCV.

- 🔊 **Real-Time Volume Mixing**  
  Blend vocals and instrumentals with smooth real-time volume control.

---

## 📁 Project Structure

```
gesture-controlled-music-editor/
├── main.py                 # Application entry point
├── hand_tracker.py         # Gesture tracking using MediaPipe
├── audio_controller.py     # Audio processing, separation, and effect control
├── requirements.txt        # List of dependencies
└── README.md               # Project documentation
```

---

## 🛠️ Technologies Used

| Library          | Purpose                                      |
|------------------|----------------------------------------------|
| `demucs`         | Source separation (vocals & instruments)     |
| `pydub`          | Audio segment manipulation                   |
| `sounddevice`    | Real-time audio playback                     |
| `pedalboard`     | LoFi effects like reverb, pitch shift, etc.  |
| `mediapipe`      | Real-time hand tracking                      |
| `opencv-python`  | Webcam input for gesture control             |
| `numpy`          | Audio math and buffer operations             |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/gesture-controlled-music-editor.git
cd gesture-controlled-music-editor
```

### 2. Set up a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```


---


## 👨‍💻 Author

**Nitin Gawde**  
📫 nitingawde2605@gmail.com  

---

## 🙌 Contributions

Pull requests, issues, and feedback are welcome.  
If you have ideas for more gesture mappings or effects, feel free to contribute!
