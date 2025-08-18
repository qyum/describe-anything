import os
from PIL import Image
import torch


from huggingface_hub import snapshot_download
from dam import DescribeAnythingModel

#snapshot_download('nvidia/DAM-3B', local_dir="checkpoints")


from dam import DescribeAnythingModel, disable_torch_init

print('yes')
# Config params
image_path = "D:/describe-anything/tshirt.jpg"  # <-- change this
model_path = "checkpoints"  # If available, use the image version; otherwise keep your existing model
prompt_mode = "full+focal_crop"
conv_mode = "v1"
temperature = 0.2
top_p = 0.5
output_path = "image_description.txt"

# Disable default torch init for faster load
disable_torch_init()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
dam = DescribeAnythingModel(
    model_path=model_path,
    prompt_mode=prompt_mode,
    conv_mode=conv_mode,
).to(device)

# Load the image
image = Image.open(image_path).convert("RGB")
mask = Image.new("L", image.size, 255)  # Full mask (no masked-out areas)

# Query prompt
query = "<image>\nDescribe the content of this image in detail."

description = ""
print("Generating description ...")
for token in dam.get_description(
    [image],          # List of one image
    [mask],           # List of one mask
    query=query,
    streaming=True,
    temperature=temperature,
    top_p=top_p,
    num_beams=1,
    max_new_tokens=512,
):
    print(token, end="", flush=True)
    description += token
print("\n\nDescription generation complete.")

# Save the description
with open(output_path, "w") as f:
    f.write(description)
print(f"Description saved to {output_path}")

