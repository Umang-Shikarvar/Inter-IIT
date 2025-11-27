from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import torch
from .device import get_best_device

class QwenVLM:
    def __init__(self):
        print("[INFO] Loading Moondream2 (trust_remote_code=True) ...")
        self.device = get_best_device()
        self.model_name = "vikhyatk/moondream2"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device!="cpu" else torch.float32,
            device_map=self.device
        )

    def caption(self, img_path: str):
        image = Image.open(img_path).convert("RGB")
        return self.model.caption(image)

    def vqa(self, img_path: str, question: str):
        image = Image.open(img_path).convert("RGB")
        return self.model.answer_question(image, question)