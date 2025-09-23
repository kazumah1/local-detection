# Local Real-Time Object Detection
A real-time object detection and labeling pipeline that combines a YOLO detector with a local VLM via Ollama.

## Features
- Face/person censoring
- Webcam and screen capture streaming
- Compatibility with any Ollama VLM that supports images
- YOLO-based detection
- Fully local setup (optionally use remote models)

## How it Works
- **Detection:** YOLO infers every --frame-delay frames and filters by --confidence
- **Storing:** Object is given a UID stored in self.prev_box_ids, and bounding boxes are stored in self.prev_boxes
- **Tracking heuristic:** Centers + IoU matching update prev_boxes and stable UIDs
- **Labeling:** Async queue crops frame per stored box -> base64 -> POST to Ollama --server with --vlm to generate text label
- **Censoring:** Pixelates class person when --censor is True, skips labeling and boundary box drawing

## Next Steps
- SAM integration for better object segmentation

## Prerequisites
1. Python 3.10+ (project currently uses a venv in `venv/`)
2. Ollama installed with a VLM pulled (e.g., `llava:7b`)
3. Model weights available locally (default: `yolo11n.pt` in repo root)
4. Screen and camera permissions if using those sources

## Getting Started
1. Create and activate a virtual environment
```
python -m venv venv
source venv/bin/activate
```
2. Install dependencies
```
pip install -r requirements.txt
```
3. Start your Ollama server (default API at `http://localhost:11434`)
```
ollama serve
```
4. Run the app
```
python main.py [args]
```

### Arguments
- `--vlm` (str): VLM model name. Default: `llava:7b`.
- `--detection-model` (str): Path to YOLO `.pt` weights. Default: `yolo11n.pt`.
- `--server` (str): VLM server URL. Default: `http://localhost:11434`.
- `--confidence` (float): Confidence threshold for object detection filtering. Default: `0.3`.
- `--frame-delay` (int): Frame interval between detection passes. Default: `5`.
- `--censor` (bool): Enable face/person pixelation. Default: `False`.
- `--source` (int): Video source: `0` = webcam, `1` = screen capture. Default: `0`.
- `-h`: Show help.

### Usage Examples
Run with defaults (webcam, local Ollama, YOLO `yolo11n.pt`):
```
python main.py
```

Enable censoring and use screen capture as the source:
```
python main.py --censor True --source 1
```

Use a different VLM and model weights path:
```
python main.py --vlm llava:13b --detection-model face/last.pt
```

### Notes
- Ensure your VLM supports image inputs via Ollama’s `/api/generate` with `images` payloads.
- The README previously suggested auto-downloading the `.pt` file; currently the code expects the file to be present locally.