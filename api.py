from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
import torch
import numpy as np
import librosa
from model import Model
import safetensors.torch
import shutil
from df import enhance, init_df

# Constants - copied from demo.py
CHECKPOINT_PATH = "outputs/checkpoint-84740/model.safetensors"
EMOTION_LABELS = {
    0: 'Angry',
    1: 'Calm',
    2: 'Disgust',
    3: 'Fear',
    4: 'Happy',
    5: 'Neutral',
    6: 'Sad',
    7: 'Surprise',
}
EMOTION_LABELS_LIST = [EMOTION_LABELS[i] for i in range(8)]

# Load model - reuse from demo.py
def load_model():
    model = Model(
        model_name="tf_efficientnetv2_s.in21k_ft_in1k",
        n_classes=8,
    )
    model.load_state_dict(safetensors.torch.load_file(CHECKPOINT_PATH))
    model.eval()
    return model.cuda() if torch.cuda.is_available() else model

# Process audio - reuse from demo.py
def process_audio(audio_path):
    sr = 48000
    length = 4
    n_mels = 128
    data_max = 7.6293945e-06
    data_min = -80.0
    data, sr = librosa.load(audio_path, sr=sr)

    # Apply voice enhancement for visualization
    print("Enhancing audio...")
    data = enhance(denoise_model, df_state, torch.from_numpy(np.expand_dims(data, 0)))
    data = data.squeeze().numpy()

    if len(data) < length * sr:
        data = np.pad(data, (length * sr - len(data), 0))
    data = data[-(length * sr):]
    S = librosa.feature.melspectrogram(y=data, sr=sr, power=2.0)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_dB = (S_dB - data_min) / (data_max - data_min)
    S_dB = np.expand_dims(np.expand_dims(S_dB, axis=0), axis=0)
    return torch.tensor(S_dB).float()

# Initialize model at startup
model = load_model()
denoise_model, df_state, _ = init_df()

# Create FastAPI app
app = FastAPI(
    title="Emotion Classification API",
    description="API for classifying emotions in audio",
    version="1.0.0"
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict emotion from audio file
    
    Returns a dictionary with emotion labels and their corresponding probabilities
    """
    # Validate file
    if not file.filename.endswith(('.wav', '.mp3', '.ogg')):
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
    try:
        # Copy uploaded file to temp file
        with temp_file as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process audio
        processed_audio = process_audio(temp_file.name)
        
        # Make prediction
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processed_audio = processed_audio.to(device)
        
        with torch.no_grad():
            outputs = model(images=processed_audio, return_dict=True)
            probs = torch.softmax(outputs["logits"], dim=-1)
            probs = probs[0].cpu().numpy()
        
        # Create results dictionary
        results = {label: float(prob) for label, prob in zip(EMOTION_LABELS_LIST, probs)}
        # Get top 3 emotions by probability
        top_3 = sorted(results.items(), key=lambda x: x[1], reverse=True)[:3]
        results = {emotion: round(prob, 2) for emotion, prob in top_3}
        
        # Apply specific business logic for emotion prediction
        top_emotion, top_prob = top_3[0]

        # If top emotion is sad/happy with probability <= 0.64 or <= 0.85, change to neutral
        if (top_emotion == "Sad" and top_prob <= 0.64) or (top_emotion == "Happy" and top_prob <= 0.85):
            results = {"Neutral": round(results.get("Neutral", 0.0), 2)}
        else:
            # Just keep the top emotion
            results = {top_emotion: round(top_prob, 2)}


        return JSONResponse(content=results)
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

@app.get("/")
async def root():
    return {"message": "Welcome to the Emotion Classification API. Use /predict endpoint to analyze audio."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
