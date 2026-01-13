import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Union

# --- Configuration Parameters ---
VIDEO_PATH: str = "20250722-2218-56.1207212.mp4" 
YOLO_MODEL: str = 'yolov8s.pt' 
CONFIDENCE_THRESHOLD: float = 0.5 
IMGSZ: Tuple[int, int] = (640, 384) 

MONITORED_CLASSES: Dict[str, str] = {
    'person': 'People',
    'bicycle': 'Bikes/Moto',
    'motorcycle': 'Bikes/Moto',
    'car': 'Vehicles',
    'bus': 'Vehicles',
    'truck': 'Vehicles',
}

# --- Heuristic Conversion Parameters ---
HEURISTIC_MAX_OBJ_HEIGHT_PX: int = 250 
HEURISTIC_MIN_OBJ_HEIGHT_PX: int = 20  

HEURISTIC_MIN_DIST_M: float = 5.0    
HEURISTIC_MAX_DIST_M: float = 100.0  

FPS_ASSUMED: int = 30 
HEURISTIC_MIN_PIXEL_MOVE_FOR_SPEED: float = 0.2 
HEURISTIC_SPEED_FACTOR: float = 0.2 

HEURISTIC_MIN_PIXEL_Y_CHANGE_FOR_APPROACH: int = 2 

# --- Safety Thresholds ---
THRESHOLD_POTENTIAL_DANGER_PROX_M: float = 25.0 
MIN_SPEED_FOR_TTC_CALC_KMPH: float = 3.0 

TTC_WARNING_THRESHOLD_S: float = 3.0 
TTC_IMMINENT_THRESHOLD_S: float = 0.8 

THRESHOLD_NO_RISK_FAR_M: float = 80.0 

# --- Helper Functions ---
def estimate_real_world_distance_meters(bbox: Tuple[int, int, int, int], frame_height: int) -> float:
    obj_height_px: int = bbox[3] - bbox[1] 
    clipped_obj_height_px: int = np.clip(obj_height_px, HEURISTIC_MIN_OBJ_HEIGHT_PX, HEURISTIC_MAX_OBJ_HEIGHT_PX)

    normalized_height_inverse: float = (HEURISTIC_MAX_OBJ_HEIGHT_PX - clipped_obj_height_px) / \
                                      (HEURISTIC_MAX_OBJ_HEIGHT_PX - HEURISTIC_MIN_OBJ_HEIGHT_PX)
    normalized_height_inverse = np.clip(normalized_height_inverse, 0.0, 1.0) 
    
    distance_m: float = HEURISTIC_MIN_DIST_M + (HEURISTIC_MAX_DIST_M - HEURISTIC_MIN_DIST_M) * normalized_height_inverse
    
    return float(max(1.0, distance_m)) 

def estimate_real_world_speed_kmph(history: defaultdict, object_id: int, current_distance_m: float) -> float:
    if object_id not in history or len(history[object_id]) < 2:
        return 0.0 

    pos1: Tuple[int, int] = history[object_id][-2]['center'] 
    pos2: Tuple[int, int] = history[object_id][-1]['center'] 
    
    pixel_distance_moved: float = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
    
    if pixel_distance_moved < HEURISTIC_MIN_PIXEL_MOVE_FOR_SPEED:
        return 0.0

    METERS_PER_PIXEL_AT_MIN_DIST_HEURISTIC = 0.01 
    dynamic_meters_per_pixel: float = METERS_PER_PIXEL_AT_MIN_DIST_HEURISTIC * (current_distance_m / HEURISTIC_MIN_DIST_M)
    
    speed_mps: float = pixel_distance_moved * dynamic_meters_per_pixel * FPS_ASSUMED 
    speed_kmph: float = speed_mps * 3.6 
    
    return float(max(0.0, speed_kmph)) 

def calculate_ttc_seconds(distance_meters: float, speed_kmph: float) -> float:
    if speed_kmph < MIN_SPEED_FOR_TTC_CALC_KMPH: 
        return float('inf') 
    
    speed_mps: float = speed_kmph * 1000 / 3600 
    
    if speed_mps < 0.1: 
        return float('inf')

    distance_meters = max(0.1, distance_meters) 
    
    ttc_seconds: float = distance_meters / speed_mps
    
    return float(min(100.0, max(0.1, ttc_seconds))) 

# --- Main Application Logic ---
def run_safety_system() -> None:
    fps: float = 0.0 

    try:
        model: YOLO = YOLO(YOLO_MODEL)
        print(f"YOLOv8 model '{YOLO_MODEL}' loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load YOLOv8 model '{YOLO_MODEL}'. Details: {e}")
        print("Please ensure the model file is in the correct path or downloaded.")
        return

    cap: cv2.VideoCapture = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"ERROR: Could not open video source '{VIDEO_PATH}'. Please check the path and filename.")
        return 

    print(f"Opened video source: {VIDEO_PATH}")

    object_tracking_history: defaultdict[int, List[Dict[str, Any]]] = defaultdict(list)
    
    frame_idx: int = 0
    start_time: float = time.time()

    print("AI Safety System running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("INFO: End of video stream or error reading frame. Exiting.")
            break

        frame_idx += 1
        H, W = frame.shape[:2]

        current_time: float = time.time()
        if (current_time - start_time) > 0:
            fps = frame_idx / (current_time - start_time)
        else:
            fps = 0.0 

        results: List[Any] = model.track(frame, persist=True, conf=CONFIDENCE_THRESHOLD, 
                                        imgsz=IMGSZ, 
                                        classes=[idx for idx, name in model.names.items() if name in MONITORED_CLASSES], 
                                        tracker="bytetrack.yaml") 

        detected_objects_data: List[Dict[str, Any]] = [] 
        
        vehicle_count: int = 0
        people_count: int = 0
        bikes_moto_count: int = 0
        prox_warn_count: int = 0
        ttc_warn_count: int = 0
        ttc_danger_count: int = 0
        
        current_frame_overall_risk: str = "NO RISK" 
        current_frame_risk_color: Tuple[int, int, int] = (0, 255, 0) 

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()  
            track_ids = results[0].boxes.id.int().cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.int().cpu().numpy()

            for box, track_id, score, class_id in zip(boxes, track_ids, scores, class_ids):
                x, y, w, h = box
                x1, y1, x2, y2 = int(x - w/2), int(y - h/2), int(x + w/2), int(x + h/2)
                
                label: str = model.names[class_id]
                
                if label not in MONITORED_CLASSES:
                    continue 

                object_tracking_history[track_id].append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (int(x), int(y)), 
                    'frame_idx': frame_idx
                })
                if len(object_tracking_history[track_id]) > FPS_ASSUMED * 2: 
                    object_tracking_history[track_id].pop(0) 

                real_world_distance_m: float = estimate_real_world_distance_meters((x1, y1, x2, y2), H)
                real_world_speed_kmph: float = estimate_real_world_speed_kmph(object_tracking_history, track_id, real_world_distance_m) 
                ttc_seconds: float = calculate_ttc_seconds(real_world_distance_m, real_world_speed_kmph)

                is_prox_potential_danger: bool = real_world_distance_m <= THRESHOLD_POTENTIAL_DANGER_PROX_M
                
                is_approaching: bool = False
                if real_world_speed_kmph >= MIN_SPEED_FOR_TTC_CALC_KMPH and len(object_tracking_history[track_id]) >= 2:
                    prev_y2: int = object_tracking_history[track_id][-2]['bbox'][3]
                    curr_y2: int = object_tracking_history[track_id][-1]['bbox'][3]
                    if (curr_y2 - prev_y2) > HEURISTIC_MIN_PIXEL_Y_CHANGE_FOR_APPROACH: 
                        is_approaching = True
                
                is_ttc_danger: bool = is_approaching and ttc_seconds <= TTC_IMMINENT_THRESHOLD_S and ttc_seconds > 0
                is_ttc_warning: bool = is_approaching and ttc_seconds <= TTC_WARNING_THRESHOLD_S and ttc_seconds > 0 and not is_ttc_danger 

                if MONITORED_CLASSES.get(label) == 'Vehicles':
                    vehicle_count += 1
                elif MONITORED_CLASSES.get(label) == 'People':
                    people_count += 1
                elif MONITORED_CLASSES.get(label) == 'Bikes/Moto':
                    bikes_moto_count += 1
                
                if is_ttc_danger:
                    ttc_danger_count += 1
                elif is_ttc_warning:
                    ttc_warn_count += 1
                elif is_prox_potential_danger: 
                    prox_warn_count += 1

                obj_color: Tuple[int, int, int] = (0, 255, 0) 
                
                if is_ttc_danger:
                    obj_color = (0, 0, 255) 
                    current_frame_risk_color = (0, 0, 255)
                    current_frame_overall_risk = "!! IMMINENT DANGER !!"
                elif is_ttc_warning:
                    obj_color = (0, 100, 255) 
                    if current_frame_overall_risk not in ["!! IMMINENT DANGER !!"]: 
                        current_frame_overall_risk = "!! DANGER ALERT !!"
                        current_frame_risk_color = (0, 100, 255)
                elif is_prox_potential_danger: 
                    obj_color = (0, 165, 255) 
                    if current_frame_overall_risk not in ["!! IMMINENT DANGER !!", "!! DANGER ALERT !!"]:
                        current_frame_overall_risk = "CAUTION: POTENTIAL RISK"
                        current_frame_risk_color = (0, 165, 255)
                
                if real_world_distance_m > THRESHOLD_NO_RISK_FAR_M and current_frame_overall_risk == "NO RISK":
                    obj_color = (100, 255, 100) 
                
                detected_objects_data.append({
                    'id': track_id,
                    'class': label,
                    'bbox': (x1, y1, x2, y2),
                    'color': obj_color,
                    'speed_kmph': real_world_speed_kmph,
                    'ttc_s': ttc_seconds,
                    'prox_m': real_world_distance_m,
                    'is_prox_potential_danger': is_prox_potential_danger,
                    'is_ttc_warning': is_ttc_warning,
                    'is_ttc_danger': is_ttc_danger
                })

        for obj_data in detected_objects_data:
            x1, y1, x2, y2 = obj_data['bbox']
            obj_color = obj_data['color']

            cv2.rectangle(frame, (x1, y1), (x2, y2), obj_color, 2)
            
            label_line1: str = f"{obj_data['class']} ID:{obj_data['id']}"
            label_line2: str = f"SPD: {obj_data['speed_kmph']:.1f}km/h"
            
            cv2.putText(frame, label_line1, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, obj_color, 2)
            cv2.putText(frame, label_line2, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, obj_color, 2)

            if obj_data['is_prox_potential_danger'] and not obj_data['is_ttc_danger'] and not obj_data['is_ttc_warning']:
                 cv2.putText(frame, f"PROX: {obj_data['prox_m']:.1f}m", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, obj_color, 2)
            
            if obj_data['is_ttc_warning'] or obj_data['is_ttc_danger']:
                 ttc_display_val: str = f"{obj_data['ttc_s']:.1f}s" if obj_data['ttc_s'] != float('inf') else "INF"
                 cv2.putText(frame, f"TTC: {ttc_display_val}", (x1, y2 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, obj_color, 2)

        panel_x: int = 10
        panel_y: int = 10
        line_spacing: int = 20
        font_style = cv2.FONT_HERSHEY_SIMPLEX
        font_scale: float = 0.5
        font_thickness: int = 1
        text_color: Tuple[int, int, int] = (255, 255, 255) 

        panel_lines: List[str] = [
            f"Frame: {frame_idx}",
            f"FPS: {fps:.1f}",
            f"Vehicles: {vehicle_count}",
            f"People: {people_count}",
            f"Bikes/Moto: {bikes_moto_count}",
            f"Prox Warn: {prox_warn_count}", 
            f"TTC Warn: {ttc_warn_count}",   
            f"TTC Danger: {ttc_danger_count}" 
        ]

        max_text_width: int = 0
        for line in panel_lines:
            (text_width, text_height_val) = cv2.getTextSize(line, font_style, font_scale, font_thickness)[0]
            if text_width > max_text_width:
                max_text_width = text_width
        
        panel_width: int = max_text_width + 20 
        panel_height: int = len(panel_lines) * line_spacing + 20 

        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
        alpha: float = 0.6 
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        for i, line in enumerate(panel_lines):
            cv2.putText(frame, line, (panel_x + 10, panel_y + 20 + i * line_spacing), 
                        font_style, font_scale, text_color, font_thickness, cv2.LINE_AA) 

        (text_width, text_height_val) = cv2.getTextSize(current_frame_overall_risk, font_style, 0.8, 2)[0]
        text_x: int = (W - text_width) // 2
        cv2.putText(frame, current_frame_overall_risk, (text_x, 30), font_style, 0.8, current_frame_risk_color, 2)
        
        cv2.imshow("AI Safety System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("AI Safety System stopped.")

if __name__ == "__main__":
    run_safety_system()