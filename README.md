# Object Detection with Webcam and Text-to-Speech (TTS) for Edge Devices

An edge computing implementation of real-time object detection with text-to-speech capabilities, optimized for embedded systems and resource-constrained environments.

## Edge Computing & Embedded AI Features

- Optimized for CPU-only execution on edge devices

- Memory-efficient frame processing with 2.5s detection intervals

- Selective object detection (max 2 objects with >50% confidence)

- Resource management with automated cache clearing

- Thread-based concurrent processing for camera feed and detection

- Frame size optimization (640x480) for embedded systems

- Minimal memory footprint with selective imports
## Prerequisites
- Python 3.7 or higher
- Git (optional)
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-repository-name.git
cd your-repository-name
```

2. Set up virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # For macOS/Linux
venv\Scripts\activate     # For Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download pre-trained models:
- Object Detection Model: Save in `./local_model/`
- Text-to-Speech Model: Save in `./mms-tts-eng/model/`

These models are required for system functionality.

## Project Structure
```
/your-repository
│
├── app.py                 # Main application file
├── requirements.txt       # Project dependencies
├── ./local_model/        # Object detection model files
└── ./mms-tts-eng/model/  # TTS model files
```

## Usage

1. Start application:
```bash
python app.py
```

2. Use Gradio interface to:
- Select camera
- Start/stop detection
- View live feed and results
- Hear detected objects via TTS

## Code Overview

### app.py
Contains core functionality:
- Live camera feed capture using OpenCV
- CPU-based object detection using HuggingFace model
- Text-to-speech conversion using VITS model
- Gradio interface implementation

### Core Functions
- `live_camera_feed(camera_index)`: Captures webcam feed
- `detect_objects()`: Processes frames for detection
- `text_to_speech(text)`: Converts detections to speech
- `play_audio(output)`: Outputs audio using sounddevice
- `start_detection(camera_option)`: Initiates detection
- `stop_detection()`: Terminates detection

### Gradio Interface Features
- Camera selection dropdown
- Start/Stop buttons
- Live feed display
- Detection results output

## Requirements
```txt
torch
transformers
gradio
opencv-python
sounddevice
numpy
```

Install via:
```bash
pip install -r requirements.txt
```

## Notes
- System operates entirely on CPU
- Real-time performance depends on CPU capabilities
- Models files are available at the github repo itself just clone it
## License
MIT License
