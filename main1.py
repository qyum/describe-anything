import os
import io
import time
import warnings
from io import BytesIO
from typing import Union

from PIL import Image
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from dam import DescribeAnythingModel, disable_torch_init

from huggingface_hub import snapshot_download
from dam import DescribeAnythingModel

#snapshot_download('nvidia/DAM-3B', local_dir="checkpoints")

warnings.filterwarnings("ignore")

# ---------------- FastAPI Setup ----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def get_request():
    """Dummy endpoint to test server."""
    return {"Hello": "eshop"}

# ---------------- Config ----------------
model_path = "checkpoints"  # Local model directory
prompt_mode = "full+focal_crop"
conv_mode = "v1"
temperature = 0.2
top_p = 0.5

# Disable default torch init for faster load
disable_torch_init()

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model directly to device
dam = DescribeAnythingModel(
    model_path=model_path,
    prompt_mode=prompt_mode,
    conv_mode=conv_mode,
).to(device)

# ---------------- Helper ----------------
def move_to_device(obj, device):
    """Recursively move tensors/images to a device."""
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, list):
        return [move_to_device(o, device) for o in obj]
    return obj

# ---------------- API Endpoint ----------------
@app.post("/describe")
async def describe_image(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        mask = Image.new("L", image.size, 255)

        query = "<image>\nDescribe the content of this image in detail."
        description = ""

        # Run inference on GPU
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
            # Ensure tokens are on CPU before converting to string
            if torch.is_tensor(token):
                token = token.cpu()
            description += str(token)

        description = description.replace("\\", "")

        print("Description:", description)
        return JSONResponse({"description": description})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
