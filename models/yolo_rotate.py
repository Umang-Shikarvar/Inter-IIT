from ultralytics import YOLO
import os
import cv2
import numpy as np

class YOLOOriented:
    def __init__(self):
        print("[INFO] Loading YOLO OBB ...")
        model_path = os.path.join("models", "yolov8n-obb.pt")

        # Will auto-download if missing
        self.model = YOLO(model_path)

    def detect(self, img_path):
        res = self.model(img_path)[0]
        dets = []

        obb = res.obb
        if obb is None:
            return dets

        obb_np = obb.cpu().numpy()

        for r in obb_np:
            dets.append({
                "poly": r.xyxyxyxy.tolist()[0],
                "class": self.model.names[int(r.cls)],
                "conf": float(r.conf)
            })
        return dets

    def visualize(self, img_path, dets, save_path):
        img = cv2.imread(img_path)
        if img is None:
            return None

        for d in dets:
            if "poly" in d:
                pts = d["poly"]

                # If nested [[x,y]...] convert to flat list
                if isinstance(pts[0], list):
                    pts = [v for xy in pts for v in xy]

                # Now convert to pairs
                pts = [(int(pts[i]), int(pts[i+1])) for i in range(0, len(pts), 2)]
                pts_np = np.array(pts, dtype=np.int32)

                cv2.polylines(img, [pts_np], True, (0, 0, 255), 2)

                if "class" in d:
                    cx, cy = pts[0]
                    cv2.putText(img, d["class"], (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 255), 2)

        cv2.imwrite(save_path, img)
        return save_path
