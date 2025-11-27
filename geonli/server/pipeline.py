import logging
from datetime import datetime
from models.qwen_vlm import QwenVLM
from models.yolo_rotate import YOLOOriented
from utils.helpers import match_class

logging.basicConfig(
    filename="geonli.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

class GeoNLIPipeline:
    def __init__(self):
        logging.info("Initializing models...")
        self.caption_model = QwenVLM()
        self.vqa_model = QwenVLM()  # Same model for QA
        self.det_model = YOLOOriented()
        logging.info("Models loaded successfully.")

    def run_caption(self, img):
        logging.info(f"Caption Request | img={img}")
        try:
            out = self.caption_model.caption(img)
            logging.info(f"Caption Result: {out}")
            return {"caption": out}
        except Exception as e:
            logging.error(f"Caption Error: {e}")
            return {"error": str(e)}

    def run_vqa(self, img, question):
        logging.info(f"VQA Request | img={img} | q={question}")
        try:
            out = self.vqa_model.vqa(img, question)
            logging.info(f"VQA Result: {out}")
            return {"vqa": out}
        except Exception as e:
            logging.error(f"VQA Error: {e}")
            return {"error": str(e)}

    def run_grounding(self, img_path, grounding):
        dets = self.det_model.detect(img_path)
        
        cls = match_class(grounding)
        if cls:
            dets = [d for d in dets if cls in d["class"]]

        annotated = self.det_model.visualize(img_path, dets, "annotated.jpg")

        return {
            "count": len(dets),
            "objects": dets,
            "image": annotated
        }