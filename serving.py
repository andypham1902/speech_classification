from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import librosa
import numpy as np
from model import Model
from io import BytesIO
from fastapi import Form
app = FastAPI()

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model("tf_efficientnetv2_s.in21k_ft_in1k", n_classes=2)
from safetensors.torch import load_file

model.load_state_dict(load_file("v2s-test/checkpoint-580/model.safetensors"))
model.to(device)
model.eval()

# Test curl command
"""
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "audiobytes=@/path/to/your/audio/file.wav" \
     -F "text=Is this a question?"
"""

@app.post("/predict")
def predict(audiobytes: UploadFile = File(...), text: str = Form(...)):
    # Process the audio
    audio_content = audiobytes.file.read()
    audio, sr = librosa.load(BytesIO(audio_content), sr=48000)
    
    # Preprocess the audio
    if len(audio) < sr:
        audio = np.pad(audio, (sr - len(audio), 0))
    audio = audio[-sr:]
    mels = librosa.feature.melspectrogram(y=audio, sr=sr, fmax=sr//2, n_mels=256)
    mels = np.expand_dims(mels, axis=0)
    
    # Convert to tensor, add batch dimension, and move to GPU
    input_tensor = torch.tensor(mels).float().unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        _, logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    # Augment the text based on the prediction
    if predicted_class == 1:  # Question
        augmented_text = text + '？'
    else:  # Statement
        augmented_text = text + '。'
    
    # Return JSON response with augmented text
    return JSONResponse(
        content={
            "augmented_text": augmented_text,
        },
        status_code=200
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
