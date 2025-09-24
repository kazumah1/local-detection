from sympy.logic import false
from ultralytics import YOLO, FastSAM, SAM
import cv2
import math
import cvzone
import random
from PIL import Image
from transformers import LlavaNextProcessor
import base64
import requests
from io import BytesIO
import numpy as np
import threading
from queue import Queue, Empty
import time
import pyautogui
import argparse
from collections import deque

# model = YOLO('yolo11n.pt')
# model = YOLO('face/last.pt')


# class_names = model.names
'''['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 
'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 
'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']'''
# colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(class_names))]

class RealTimeLabeler:
    def __init__(self, vlm="llava:7b", subtractor="MOG2", llm_server="http://localhost:11434", motion_threshold=0.3, frame_delay=5):
        self.vlm = vlm
        self.server = llm_server
        self.current_frame = None
        self.frame_delay = frame_delay
        self.motion_threshold = motion_threshold
        self.motion_area_threshold = 5000
        self.ssim_threshold = .9
        if subtractor == "KNN":
            self.subtractor = cv2.createBackgroundSubtractorKNN()
        else:
            self.subtractor = cv2.createBackgroundSubtractorMOG2()
        self.frame = 0

        self.label = "thinking..."

        self.frame_buffer = deque()
        self.buffer_size = 10

        self.analysis_queue = Queue()
        self.analysis_in_progress = False
        self.analysis_running = True
        self.analysis_thread = threading.Thread(target=self._analysis_worker, daemon=True)
        self.analysis_thread.start()
    
    def _analysis_worker(self):
        while self.analysis_running:
            try:
                request = self.analysis_queue.get(timeout=1.0)
                frame = request['frame']

                img_base64 = self.process_img(frame)
                label = self._get_label_from_llm(img_base64)

                self.label = label

                self.analysis_in_progress = False
            except Empty:
                continue
            except Exception as e:
                print(f"Error analyzing frame {frame}: {type(e).__name__} {e}")
                self.analysis_in_progress = False

    
    def _get_current_frame(self):
        return getattr(self, 'current_frame', None)
    
    def _get_label_from_llm(self, img_base64):
        """Common method to get label from LLM"""
        prompt = "Describe the main subject in detail in 3-10 words"

        payload = {
            "model": self.vlm,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_ctx": 2048
            },
            "images": [img_base64]
        }

        response = requests.post(
            f"{self.server}/api/generate",
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            return result.get('response', 'unknown object').strip()
        else:
            print(f"Ollama API error: {response.status_code}")
            return "unknown object"
    
    def process_img(self, img):
        """Process image using coordinates directly"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        crop_PIL = Image.fromarray(img_rgb)
        buffer = BytesIO()
        crop_PIL.save(buffer, format="JPEG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return img_base64
    
    def detect_motion(self, img):
        # returns True if there is motion, False if there isn't
        subtractor_motion = False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.frame_buffer.append(gray)
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.popleft()
        if len(self.frame_buffer) < 2:
            return True

        curr_mask = self.subtractor.apply(img)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        curr_mask = cv2.morphologyEx(curr_mask, cv2.MORPH_OPEN, kernel=kernel)
        curr_mask = cv2.morphologyEx(curr_mask, cv2.MORPH_CLOSE, kernel=kernel)

        contours, _ = cv2.findContours(curr_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_motion_area = sum(cv2.contourArea(contour) for contour in contours)
        if total_motion_area > self.motion_area_threshold:
            subtractor_motion = True

        buffer_motion = False
        if len(self.frame_buffer) >= 2:
            frame_diff = cv2.absdiff(self.frame_buffer[-1], self.frame_buffer[-2])
            _, diff_thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            diff_contours, _ = cv2.findContours(diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            diff_area = sum(cv2.contourArea(contour) for contour in diff_contours)
            if diff_area > (self.motion_area_threshold / 2):
                buffer_motion = True
        
        structural_motion = False
        if len(self.frame_buffer) >= 3:
            ssim_score = cv2.matchTemplate(self.frame_buffer[0], self.frame_buffer[-1],cv2.TM_CCOEFF_NORMED)[0][0]
            structural_motion = ssim_score < self.ssim_threshold
        
        return subtractor_motion or buffer_motion or structural_motion
        

    def run(self, source=0):
        cap = cv2.VideoCapture(0)
        cap.set(3, 320)
        cap.set(4, 320)

        while True:
            if source == 1:
                ss = pyautogui.screenshot()
                frame = np.array(ss)
                img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                success, img = cap.read()
            has_motion = self.detect_motion(img)
            print(has_motion)
            if has_motion and (self.frame % self.frame_delay == 0) and (not self.analysis_in_progress):
                frame = img.copy()
                self.analysis_queue.put({
                    'frame': frame
                })

                self.analysis_in_progress = True
                print("Motion detected")
                
            cvzone.putTextRect(img, self.label, (50, 50), scale=2, thickness=2, colorR=(0, 0, 0), colorB=(0, 0, 0))

            cv2.imshow("image", img)
            self.frame += 1
            cv2.waitKey(1)

def main():
    parser = argparse.ArgumentParser(description="Real Time Object Detector + Identifier")
    parser.add_argument('--source', default=0, type=int, choices=[0, 1], help="video source [0:webcam, 1:screen capture]")
    parser.add_argument('--vlm', default='llava:7b', type=str, help='VLM model name')
    parser.add_argument('--subtractor', default='MOG2', type=str, help='object detection model file')
    parser.add_argument('--server', default='http://localhost:11434', type=str, help='VLM model server URL')
    parser.add_argument('--motion', default=0.3, type=float, help="confidence threshold for object detection")
    parser.add_argument('--frame-delay', default=1, type=int, help='object detection intervals')

    args = parser.parse_args()
    labeler = RealTimeLabeler(
        vlm=args.vlm, 
        subtractor=args.subtractor, 
        llm_server=args.server, 
        motion_threshold=args.motion,
        frame_delay=args.frame_delay
        )
    labeler.run(args.source)
main()