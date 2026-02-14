# Gesture Based Music Controller

A real time gesture controlled music player built using computer vision. Control playback, change tracks and adjust volume using only hand gestures captured through your webcam. The project was made for **Gradient Tech Recruitment 2026** under the topic **Computer Vision - ASL Translator** using MediaPipe for hand tracking, OpenCV for video processing, and Pygame for audio playback into a fully functional gesture driven music system.

---

## Demo Video

👉 https://www.youtube.com/watch?v=xv0RFg9DShM

---

## Features

- Single pinch → Play / Pause
- Double pinch → Next Track
- Triple pinch → Previous Track
- Thumb + Middle finger pinch → Real-time volume control
- Supports `.mp3`, `.wav`, `.ogg`
- Live hand landmark visualization
- Automatic playlist loading from `./music`
- Real-time webcam tracking

---

## How It Works

The application uses MediaPipe’s Hand Landmarker to detect hand landmarks from webcam frames in real time. Load your music files in the `./music` folder and run `gesture.py`. There are a few sample files from the album Siyaah already included in the music folder.

Two pinch gestures are recognized:

### Thumb + Index Finger
Used for discrete commands based on pinch count:


| Single Pinch | Play / Pause |

| Double Pinch | Next Track |

| Triple Pinch | Previous Track |

Pinches are tracked within a sliding time window to distinguish gesture sequences.

---

### Thumb + Middle Finger
Used for continuous volume control:

1. Hold the pinch
2. Move your hand vertically (in the y axis)
3. Volume adjusts proportionally to hand movement
4. Volume defaults at 70% to start
5. While holding the pinch, move your hand up to increase and down to decrease volume

---

## Requirements

### Python Dependencies
Install using:

```bash
pip install opencv-python pygame mediapipe
```
### Standard Library Modules Used

```bash
os
math
time
collections
urllib
```