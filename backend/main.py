from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import io
import base64

# Initialize FastAPI app
app = FastAPI()

# Allow CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Recreate the model architecture
num_classes = 6  # Update this if your number of classes is different
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Class index to material and recyclability mapping (update as needed)
CLASS_MAP = {
    0: {"material": "cardboard", "recyclable": True},
    1: {"material": "glass", "recyclable": True},
    2: {"material": "metal", "recyclable": True},
    3: {"material": "paper", "recyclable": True},
    4: {"material": "plastic", "recyclable": True},
    5: {"material": "trash", "recyclable": False},
}

# Load the state dictionary
try:
    model.load_state_dict(torch.load("model.pt", map_location=torch.device("cpu")))
    model.eval()
except Exception as e:
    print(f"Error loading model.pt: {e}")
    raise

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((256, 192)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")  # Ensure RGB

        # Preprocess the image
        input_tensor = preprocess(pil_image).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted_class = torch.max(output, 1)

        class_idx = int(predicted_class.item())
        class_info = CLASS_MAP.get(class_idx, {"material": "unknown", "recyclable": False})
        return JSONResponse(content={
            "prediction": class_idx,
            "material": class_info["material"],
            "recyclable": class_info["recyclable"]
        })
    except Exception as e:
        import traceback
        print("Prediction error:", e)
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
