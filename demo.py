import os
import gradio as gr
import torch
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from dataset import EmotionDataset, collate_fn
from model import Model
import safetensors.torch
import librosa
import librosa.display
from scipy.signal import butter, filtfilt

# Constants
CHECKPOINT_PATH = "v2s_emo_augment_with_noise_2/checkpoint-1548/model.safetensors"
TEST_DATA_PATH = "emotions.csv"
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
EMOTION_LABELS = [EMOTION_LABELS[i] for i in range(8)]
# Load model
def load_model():
    model = Model(
        model_name="tf_efficientnetv2_s.in21k_ft_in1k",
        n_classes=8,
    )
    model.load_state_dict(safetensors.torch.load_file(CHECKPOINT_PATH))
    model.eval()
    return model.cuda()

# Load test dataset
def load_test_dataset():
    df = pd.read_csv(TEST_DATA_PATH)
    test_dataset = EmotionDataset(
        df[df.fold == 0],  # Adjust fold as needed
        mode="val",
    )
    return test_dataset

# Initialize model and dataset
model = load_model()
test_dataset = load_test_dataset()

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Create a butterworth bandpass filter"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, fs, lowcut=300, highcut=3400, order=5):
    """Apply bandpass filter to focus on voice frequencies"""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

def spectral_gating(data, sr, n_std_thresh=1.5):
    """Reduce noise using spectral gating"""
    # Compute spectrogram
    spec = librosa.stft(data)
    mag = np.abs(spec)
    phase = np.angle(spec)
    
    # Estimate noise profile
    mean = np.mean(mag, axis=1, keepdims=True)
    std = np.std(mag, axis=1, keepdims=True)
    thresh = mean + n_std_thresh * std
    
    # Apply soft thresholding
    mask = (mag > thresh).astype(float)
    smoothed_mask = librosa.decompose.nn_filter(mask, aggregate=np.median, metric='cosine')
    mag = mag * smoothed_mask
    
    # Reconstruct signal
    filtered_spec = mag * np.exp(1j * phase)
    return librosa.istft(filtered_spec)

def enhance_audio(data, sr):
    """Apply voice enhancement pipeline"""
    # Apply bandpass filter for voice frequency range
    filtered_data = apply_bandpass_filter(data, sr)
    
    # Apply spectral gating for noise reduction
    enhanced_data = spectral_gating(filtered_data, sr)
    
    # Normalize audio
    enhanced_data = librosa.util.normalize(enhanced_data)
    
    return enhanced_data

def process_audio(audio_path):
    sr = 48000
    length = 4
    n_mels = 128
    data, sr = librosa.load(audio_path, sr=sr)
    
    # Apply voice enhancement
    # data = enhance_audio(data, sr)
    
    # Continue with existing processing
    if len(data) < length * sr:
        data = np.pad(data, (length * sr - len(data), 0))
    data = data[-(length * sr):]
    mels = librosa.feature.melspectrogram(y=data, sr=sr, fmax=sr//2, n_mels=n_mels, hop_length=512, n_fft=2048)
    mels = np.expand_dims(mels, axis=0)
    return torch.tensor([mels]).float()

def create_spectrogram(audio_path):
    data, sr = librosa.load(audio_path)
    
    # Apply voice enhancement for visualization
    # data = enhance_audio(data, sr)
    
    plt.figure(figsize=(10, 4))
    spectrogram = librosa.feature.melspectrogram(y=data, sr=sr)
    librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    
    fig = plt.gcf()
    plt.close()
    return fig

def predict(audio=None, use_random=False):
    if use_random:
        idx = random.randint(0, len(test_dataset)-1)
        sample = test_dataset[idx]
        batch = collate_fn([sample])
        batch = {k: v.cuda() for k, v in batch.items()}
        # Get the actual audio path for playback
        audio = test_dataset.paths[idx]
    else:
        # Save audio to temporary file
        temp_path = "_tmp.wav"
        if audio is not None:
            if isinstance(audio, dict):  # New Gradio audio format
                audio_data = audio['array']
                sr = audio['sr']
                # Convert to mono if stereo
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)
                librosa.output.write_wav(temp_path, audio_data, sr)
                audio = temp_path
            elif os.path.exists(audio):
                os.system(f"cp {audio} {temp_path}")
                audio = temp_path
        else:
            return None, None, None
        processed_audio = process_audio(audio)
        batch = {
            "images": processed_audio.cuda(),
        }
    
    # Make prediction
    with torch.no_grad():
        outputs = model(images=batch["images"], return_dict=True)
        probs = torch.softmax(outputs["logits"], dim=-1)
        probs = probs[0].cpu().numpy()
    
    # Create results dictionary
    results = {label: float(prob) for label, prob in zip(EMOTION_LABELS, probs)}
    
    # Generate spectrogram
    spectrogram = create_spectrogram(audio)
    
    return results, audio, spectrogram

# Gradio interface
def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Emotion Classification Demo")
        
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(sources="microphone", type="filepath")
                file_input = gr.File(label="Upload Audio File", type="filepath")  # New upload feature
                random_btn = gr.Button("Use Random Sample")
            
            with gr.Column():
                audio_output = gr.Audio(label="Current Audio")
                spectrogram_output = gr.Plot(label="Spectrogram")
                output = gr.Label(label="Predictions")
        
        audio_input.change(
            fn=lambda x: predict(x, False),
            inputs=[audio_input],
            outputs=[output, audio_output, spectrogram_output]
        )
        
        file_input.change(
            fn=lambda x: predict(x.name, False) if x else (None, None, None),
            inputs=[file_input],
            outputs=[output, audio_output, spectrogram_output]
        )
        
        random_btn.click(
            fn=lambda: predict(None, True),
            inputs=[],
            outputs=[output, audio_output, spectrogram_output]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)
