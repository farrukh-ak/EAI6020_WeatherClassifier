from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
import torchvision.models as models

# Define the model architecture
model = models.resnet18(weights=None)  # Use weights=None for custom model
model.fc = torch.nn.Linear(model.fc.in_features, 11)  # Adjust for 11 classes

# Load the trained model weights
model.load_state_dict(torch.load("weather_modelF.pth", map_location=torch.device('cpu')))
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Define class names (replace with your actual class names)
class_names = ['dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'snow']

# Create FastAPI app
app = FastAPI()

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Weather Classification API! Use the /predict endpoint to classify images."}

# Prediction endpoint (POST method)
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess the image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # Ensure image is in RGB format
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        prediction = class_names[predicted.item()]
    
    return {"prediction": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)