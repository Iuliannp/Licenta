from fastapi import FastAPI, Request, File, UploadFile, Depends
import shutil
from fastapi.responses import HTMLResponse
from Mymodel import *
import os
import numpy as np
import math
from fastapi.templating import Jinja2Templates

classes = {
    0 : 'de',
    1 : 'en',
    2 : 'fr'
}

app = FastAPI()

templates = Jinja2Templates(directory="templates")
@app.get("/upload/", response_class=HTMLResponse)
async def upload(request: Request):
   return templates.TemplateResponse("./uploadfile.html", {"request": request})

@app.post("/uploader/")
async def create_upload_file(file: UploadFile = File(...)):
    with open("destination.wav", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    prediction = makePrediction('./destination.wav')
    os.remove("./destination.wav")
    
    accuracy = prediction.max()
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = classes[predicted_class_index]
    output = "I am {} % sure that this is {}".format(math.trunc(accuracy*100), predicted_class_label)
    
    return {"Prediction": output}


@app.get("/")
async def root():
    return {"message" : "Hello, world!"}
