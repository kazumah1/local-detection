from ultralytics import YOLO
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
from queue import Queue
import time
import pyautogui
import argparse

model = YOLO('yolo11n.pt')
# model = YOLO('face/last.pt')


class_names = model.names
'''['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 
'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 
'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']'''
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(class_names))]

class RealTimeDetector:
    def __init__(self, vlm="llava:7b", detection_model="yolo11n.pt", llm_server="http://localhost:11434", confidence_level=0.3, frame_delay=5):
        self.vlm = vlm
        self.detection_model = YOLO(detection_model)
        self.class_names = self.detection_model.names
        self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        self.colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(self.class_names))]
        self.server = llm_server
        self.confidence_level = confidence_level
        self.current_frame = None
        self.frame_delay = frame_delay
        self.frame = 0

        # need a way to store previous bounding boxes for object tracking
        self.prev_box_ids = set()
        self.prev_boxes = {}
        self.curr_box_ids = []
        self.tmp_coords = {}
        # need a way to store labels per bounding box
        self.labels = {}

        # async labeling
        self.running = True
        self.labeling_queue = Queue()
        self.labeling_in_progress = set()
        self.labeling_lock = threading.Lock()

        self.labeling_thread = threading.Thread(target=self._labeling_worker, daemon=True)
        self.labeling_thread.start()

        

        # Store class and confidence for each tracked object
        self.object_metadata = {}  # {uid: {'class': b_class, 'conf': conf}}

        self.uid = 0
        
    
    def detect_objects(self, img, censor):
        self.current_frame = img.copy()
        results = self.detection_model(img, stream=True)
        b_class = 0
        conf = 0
        for r in results:
            boxes = r.boxes
            for b in boxes:
                x1, y1, x2, y2 = b.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil(b.conf[0]*100)/100
                b_class = int(b.cls[0])

                if b_class == 0 and censor:
                    y1 = max(0, y1 - 10)    
                    y2 = min(img.shape[0], y2 + 10)
                    x1 = max(0, x1 - 10)
                    x2 = min(img.shape[1], x2 + 10)
                    roi = img[y1:y2, x1:x2]
                    h, w = roi.shape[:2]

                    temp = cv2.resize(roi, (w//25, h//25), interpolation=cv2.INTER_LINEAR)
                    pixelated_roi = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    # Replace the region
                    img[y1:y2, x1:x2] = pixelated_roi
                elif conf > 0.5:
                    uid = self._generate_uid()
                    self.curr_box_ids.append(uid)
                    self.tmp_coords[uid] = [x1, y1, x2, y2]
                    # Store metadata for this detection
                    self.object_metadata[uid] = {'class': b_class, 'conf': conf}
        self._vectorize_ious(self.curr_box_ids, self.prev_box_ids)
        print("prev boxes", self.prev_boxes)
        for uid in self.prev_boxes.keys():
            x1, y1, x2, y2 = self.prev_boxes[uid]
            
            # Get metadata for this object (use defaults if not found)
            metadata = self.object_metadata.get(uid, {'class': 0, 'conf': 0.5})
            obj_class = metadata['class']
            obj_conf = metadata['conf']
            
            # Generate label if needed
            if uid not in self.labels:
                with self.labeling_lock:
                    if uid not in self.labeling_in_progress:
                        self.labeling_queue.put({
                            'uid': uid,
                            'coords': (x1, y1, x2, y2),
                            'timestamp': time.time()
                        })
                        self.labeling_in_progress.add(uid)
                        # label = self.label_box_from_coords(img, x1, y1, x2, y2)
                        # self.labels[uid] = label
                label = "[labeling...]"
            else:
                label = self.labels[uid]

            # Draw with object-specific class and confidence
            cv2.rectangle(img, (x1, y1), (x2, y2), self.colors[obj_class], 2)
            cvzone.putTextRect(img, f'{label} {obj_conf}', (max(0, x1), max(35, y1-20)), scale=2, thickness=2, colorR=self.colors[obj_class], colorB=self.colors[obj_class])
        self.tmp_coords = {}
        self.curr_box_ids = []
    
    def _labeling_worker(self):
        while self.running:
            try:
                request = self.labeling_queue.get(timeout=1.0)
                uid = request['uid']
                x1, y1, x2, y2 = request['coords']

                with self.labeling_lock:
                    if uid not in self.prev_boxes:
                        self.labeling_in_progress.discard(uid)
                        continue
                try:
                    current_frame = self._get_current_frame()
                    label = self.label_box_from_coords(current_frame, x1, y1, x2, y2)

                    with self.labeling_lock:
                        if uid in self.prev_boxes:
                            self.labels[uid] = label
                        self.labeling_in_progress.discard(uid)
                except Exception as e:
                    print(f"Labeling error for {uid}: {e}")
                    with self.labeling_lock:
                        self.labeling_in_progress.discard(uid)
                        if uid in self.object_metadata:
                            obj_class = self.object_metadata[uid]['class']
                            self.labels[uid] = self.class_names[obj_class]
            except:
                continue
    
    def _get_current_frame(self):
        return getattr(self, 'current_frame', None)

    def label_box_from_coords(self, img, x1, y1, x2, y2):
        """Generate label from coordinates instead of box object"""
        img_base64 = self.crop_and_process_coords(img, x1, y1, x2, y2)
        return self._get_label_from_llm(img_base64)
    
    def _get_label_from_llm(self, img_base64):
        """Common method to get label from LLM"""
        prompt = "Describe this object in 3-5 words"

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
    
    def crop_and_process_coords(self, img, x1, y1, x2, y2):
        """Crop and process image using coordinates directly"""
        crop = img[y1:y2, x1:x2]

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        crop_PIL = Image.fromarray(crop_rgb)
        buffer = BytesIO()
        crop_PIL.save(buffer, format="JPEG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return img_base64
    
    def _vectorize_ious(self, curr_box_ids, prev_box_ids):
        curr_boxes = np.array(list(self.tmp_coords.values()))
        prev_boxes = np.array(list(self.prev_boxes.values()))
        curr_box_ids = np.array(list(self.tmp_coords.keys()))
        prev_box_ids = np.array(list(self.prev_boxes.keys()))

        if len(curr_boxes) == 0 :
            return []
        
        curr_centers = np.column_stack([
            (curr_boxes[:, 0] + curr_boxes[:, 2]) / 2,
            (curr_boxes[:, 1] + curr_boxes[:, 3]) / 2
        ])

        if len(prev_boxes) == 0:
            self.prev_boxes = self.tmp_coords.copy()
            return []

        prev_centers = np.column_stack([
            (prev_boxes[:, 0] + prev_boxes[:, 2]) / 2,
            (prev_boxes[:, 1] + prev_boxes[:, 3]) / 2
        ])

        # TODO: understand
        if prev_centers is not None:
            curr_centers_expanded = curr_centers[:, np.newaxis, :]
            prev_centers_expanded = prev_centers[np.newaxis, :, :]

            distances = np.sqrt(np.sum((curr_centers_expanded - prev_centers_expanded)**2, axis=2))

            distance_threshold = 100
            close = distances < distance_threshold

            curr_indices, prev_indices = np.where(close)

            match_ids = set()
            matched_curr_indices = set()
            new_prev_boxes = {}

            for curr_idx, prev_idx in zip(curr_indices, prev_indices):
                curr_box_id = curr_box_ids[curr_idx]
                prev_box_id = prev_box_ids[prev_idx]

                curr_box = self.tmp_coords[curr_box_id]
                prev_box = self.prev_boxes[prev_box_id]

                iou = self._calculate_iou(curr_box, prev_box)

                if iou > 0.3:
                    match_ids.add(prev_box_id)
                    new_prev_boxes[prev_box_id] = curr_box
                    matched_curr_indices.add(curr_box_id)
                else:
                    match_ids.add(curr_box_id)
                    new_prev_boxes[curr_box_id] = curr_box
                    matched_curr_indices.add(curr_box_id)

            for curr_box_id in curr_box_ids:
                if curr_box_id not in matched_curr_indices:
                    new_prev_boxes[curr_box_id] = self.tmp_coords[curr_box_id]
                    
            self.prev_boxes = new_prev_boxes
            self.prev_box_ids = match_ids
        else:
            self.prev_box_ids = set(self.curr_box_ids)
            self.prev_boxes = self.tmp_coords.copy()

    
    def _calculate_iou(self, curr_box, prev_box):
        x1 = max(curr_box[0], prev_box[0])
        y1 = max(curr_box[1], prev_box[1])
        x2 = min(curr_box[2], prev_box[2])
        y2 = min(curr_box[3], prev_box[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (curr_box[2] - curr_box[0]) * (curr_box[3] - curr_box[1])
        area2 = (prev_box[2] - prev_box[0]) * (prev_box[3] - prev_box[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0
    
    def _generate_uid(self):
        self.uid += 1
        return "obj_" + str(self.uid)

    def run(self, censor=False, source=0):
        cap = cv2.VideoCapture(0)
        cap.set(3, 320)
        cap.set(4, 320)
        screen_heigh, screen_width = pyautogui.size()

        while True:
            if self.frame % self.frame_delay == 0:
                if source == 1:
                    ss = pyautogui.screenshot()
                    frame = np.array(ss)
                    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    success, img = cap.read()

                # object detection + boundary box drawing
                self.detect_objects(img, censor)

                cv2.imshow("image", img)
            self.frame += 1
            cv2.waitKey(1)

def main():
    parser = argparse.ArgumentParser(description="Real Time Object Detector + Identifier")
    parser.add_argument('--censor', default=False, type=bool, help="boolean for censoring faces/people")
    parser.add_argument('--source', default=0, type=int, choices=[0, 1], help="video source [0:webcam, 1:screen capture]")
    parser.add_argument('--vlm', default='llava:7b', type=str, help='VLM model name')
    parser.add_argument('--detection-model', default='yolo11n.pt', type=str, help='object detection model file')
    parser.add_argument('--server', default='http://localhost:11434', type=str, help='VLM model server URL')
    parser.add_argument('--confidence', default=0.3, type=float, help="confidence threshold for object detection")
    parser.add_argument('--frame-delay', default=5, type=int, help='object detection intervals')

    args = parser.parse_args()
    detector = RealTimeDetector(
        vlm=args.vlm, 
        detection_model=args.detection_model, 
        llm_server=args.server, 
        confidence_level=args.confidence, 
        frame_delay=args.frame_delay
        )
    detector.run(args.censor, args.source)
main()