from fastapi import FastAPI, UploadFile, File
# from app.inference import predict_image

app = FastAPI()

@app.get("/")
def home():
    return {"message": "OncoVision AI API Running!"}