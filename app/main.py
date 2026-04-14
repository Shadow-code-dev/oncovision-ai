from fastapi import FastAPI, UploadFile, File
from app.inference import predict_image

app = FastAPI()

@app.get("/")
def home():
    return {"message": "OncoVision AI API Running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    pred, confidence = predict_image(image_bytes)
    class_names = ["benign", "malignant", "normal"]

    return {
        "prediction": class_names[int(pred)],
        "confidence": float(confidence)
    }