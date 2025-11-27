from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
import tempfile, shutil, os
from .pipeline import GeoNLIPipeline

app = FastAPI()
p = GeoNLIPipeline()

@app.post("/geonli/task")
async def eval(
    image: UploadFile = File(...),
    task: str = Form(...),         # "caption", "vqa", "ground"
    question: str = Form(""),      # only for vqa
    grounding: str = Form("")      # only for grounding
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        shutil.copyfileobj(image.file, tmp)
        img_path = tmp.name

    if task == "caption":
        result = p.run_caption(img_path)
    elif task == "vqa":
        result = p.run_vqa(img_path, question)
    elif task == "ground":
        res = p.run_grounding(img_path, grounding)
        return FileResponse(res["image"], media_type="image/jpeg")
    else:
        result = {"error": "Invalid task."}

    os.remove(img_path)
    return JSONResponse(result)