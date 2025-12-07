import cv2
from ultralytics import YOLO
import tkinter as tk
from collections import defaultdict
import numpy as np

# --- BEÁLLÍTÁSOK ---
model = YOLO("yolov8s.pt")
video_path = r'jesz.mp4' 

# Csak ezeket a járműveket figyeljük
VEHICLE_CLASSES = [2, 3, 5, 7] 

# Magyarosítás szótár (COCO ID -> Magyar név)
HU_NAMES = {
    2: "Auto",
    3: "Motor",
    5: "Busz",
    7: "Teher" # Teherautó/Kamion
}

# --- VÁLTOZÓK ---
track_history = defaultdict(lambda: [])
counted_ids = set()
count_down = 0
count_up = 0

# --- KÉPERNYŐ MÉRETEK ---
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()

cap = cv2.VideoCapture(video_path)
cv2.namedWindow("Forgalom Statisztika", cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Átméretezés
    h, w = frame.shape[:2]
    max_w, max_h = int(screen_width * 0.8), int(screen_height * 0.8)
    scale = min(max_w / w, max_h / h, 1)
    if scale < 1:
        frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        h, w = frame.shape[:2]

    # Vonal pozíciója
    line_y = int(h * 0.6)
    cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)

    # Detektálás
    results = model.track(frame, persist=True, verbose=False, classes=VEHICLE_CLASSES)[0]

    if results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        track_ids = results.boxes.id.int().cpu().numpy()
        clss = results.boxes.cls.int().cpu().numpy()

        for box, track_id, cls in zip(boxes, track_ids, clss):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # --- MOZGÁS LOGIKA ---
            track = track_history[track_id]
            track.append((cx, cy))
            if len(track) > 30: track.pop(0)

            # Irány meghatározása
            direction_text = ""
            color_dir = (0, 255, 0) # Alap zöld

            if len(track) >= 2: # Mesterséges intelligencia által generált
                diff = track[-1][1] - track[-2][1]
                if diff > 2:   # Y nő -> Lefele
                    direction_text = "LE"
                    color_dir = (0, 0, 255) # Piros
                elif diff < -2: # Y csökken -> Felfele
                    direction_text = "FEL"
                    color_dir = (255, 0, 0) # Kék

            # --- SZÁMLÁLÁS ---
            if line_y - 15 < cy < line_y + 15:
                if track_id not in counted_ids:
                    if direction_text == "LE":
                        count_down += 1
                        counted_ids.add(track_id)
                        cv2.line(frame, (0, line_y), (w, line_y), (0, 0, 255), 4)
                    elif direction_text == "FEL":
                        count_up += 1
                        counted_ids.add(track_id)
                        cv2.line(frame, (0, line_y), (w, line_y), (255, 0, 0), 4)

            # --- KIRAJZOLÁS ---
            # Keret
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_dir, 2)
            
            # Címke összeállítása: Típus + ID + Irány
            # Pl: "Auto ID:42 LE"
            veh_type = HU_NAMES.get(cls, "Jarmu")
            label = f"{veh_type} [{track_id}] {direction_text}"
            
            # Szöveg háttere (hogy olvasható legyen)
            (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + w_text, y1), color_dir, -1)
            
            # Szöveg kiírása
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Pötty a közepére
            cv2.circle(frame, (cx, cy), 4, color_dir, -1)

    # --- ÖSSZESÍTŐ TÁBLÁZAT (Dashboard) ---
    # Háttér (kicsit nagyobb, hogy elférjen a 3 sor)
    cv2.rectangle(frame, (10, 10), (280, 115), (0, 0, 0), -1)
    
    # 1. sor: Lefele (Piros)
    cv2.putText(frame, f"Lefele:   {count_down}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # 2. sor: Felfele (Kék)
    cv2.putText(frame, f"Felfele:  {count_up}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    # 3. sor: Összesen (Fehér)
    total_count = count_up + count_down
    cv2.putText(frame, f"Osszesen: {total_count}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Forgalom Statisztika", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
