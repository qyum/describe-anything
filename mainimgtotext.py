import os
from PIL import Image
import torch
from huggingface_hub import snapshot_download
from dam import DescribeAnythingModel, disable_torch_init
import time

import base64
import requests
import os
import json

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)  # for exponential backoff

import re
from typing import Union
from fastapi import FastAPI, Response, status, HTTPException,Request
from fastapi.middleware.cors import CORSMiddleware
from tenacity import retry, stop_after_attempt, wait_exponential
import time
import sys
import os
from io import BytesIO
from pydantic import BaseModel
import os
import sys
import warnings
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile, Form
warnings.filterwarnings("ignore")
import io
from fastapi import APIRouter, Depends, status, Response


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# Download the model from Hugging Face to local "checkpoints" folder
#snapshot_download('nvidia/DAM-3B', local_dir="checkpoints")  #D:\eshop1\describeanything\checkpoints


router = APIRouter()
@router.get("/")
def get_request():
    """
    Dummy function to test if server is running.
    """
    return {"Hello": "imagetotext"}



# Config params
image_path = "D:/describeanything/tshirt.jpg"  # <-- change this to your image path
model_path = "checkpoints"  # Local model directory
prompt_mode = "full+focal_crop"
conv_mode = "v1"
temperature = 0.2
top_p = 0.5
output_path = "image_description.txt"

# Disable default torch init for faster load
disable_torch_init()

# Force CPU mode
device = torch.device("cpu")

# Load model on CPU only
dam = DescribeAnythingModel(
    model_path=model_path,
    prompt_mode=prompt_mode,
    conv_mode=conv_mode,
    device_map="cpu",          # force all model parts to CPU
    torch_dtype=torch.float32  # avoid float16 issues on CPU
)

#class Item(BaseModel):


@router.post("/describe")
async def describe_image(file: UploadFile = File(...)):
    try:
        # Read image from uploaded file
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        mask = Image.new("L", image.size, 255)

        query = "<image>\nDescribe the content of this image in detail."
        description = ""
        
        for token in dam.get_description(
            [image],
            [mask],
            query=query,
            streaming=False,
            temperature=temperature,
            top_p=top_p,
            num_beams=1,
            max_new_tokens=512,
        ):
            description += token

        return JSONResponse({"description": description})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)