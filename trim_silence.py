import pandas as pd
import webrtcvad
import librosa
import os
import collections
import contextlib
import sys
import wave
import webrtcvad
from pydub import AudioSegment
import io
from time import time
from tqdm import tqdm

def read_wave(path):
    """Reads a .wav file.

    Takes the path, and returns (PCM audio data, sample rate).
    Resamples the audio if the sample rate is not in (8000, 16000, 32000, 48000).
    """
    valid_sample_rates = (8000, 16000, 32000, 48000)

    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        
        pcm_data = wf.readframes(wf.getnframes())

        if sample_rate not in valid_sample_rates or sample_width != 2:
            audio = AudioSegment(
                data=pcm_data,
                sample_width=sample_width,
                frame_rate=sample_rate,
                channels=num_channels
            )
            new_sample_rate = 16000  # Example: resample to 16000 Hz
            audio = audio.set_frame_rate(new_sample_rate)
            audio = audio.set_sample_width(2)
            
            buffer = io.BytesIO()
            audio.export(buffer, format="wav")
            buffer.seek(0)
            
            with contextlib.closing(wave.open(buffer, 'rb')) as resampled_wf:
                pcm_data = resampled_wf.readframes(resampled_wf.getnframes())
                sample_rate = resampled_wf.getframerate()
        
        return pcm_data, sample_rate
    

class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data."""
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    """Filters out non-speech segments."""
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False

    voiced_frames = []
    timestamps = []

    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                timestamps.append((ring_buffer[0][0].timestamp, ring_buffer[-1][0].timestamp))
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            if not is_speech:
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    triggered = False
                    timestamps.append((voiced_frames[0].timestamp, voiced_frames[-1].timestamp))
                    ring_buffer.clear()
                    voiced_frames = []

    if voiced_frames:
        timestamps.append((voiced_frames[0].timestamp, voiced_frames[-1].timestamp))

    return timestamps

def gather_segments(intervals):
    if not intervals:
        return []
    
    # Sort intervals by the start time
    intervals.sort(key=lambda x: x[0])
    
    merged_intervals = [intervals[0]]
    
    for current in intervals[1:]:
        last = merged_intervals[-1]
        
        if current[0] <= last[1]:  # Check if there is an overlap
            merged_intervals[-1] = (last[0], max(last[1], current[1]))  # Merge intervals
        else:
            merged_intervals.append(current)
    
    # Round the intervals to 2 decimal places
    merged_intervals = [(round(interval[0], 2), round(interval[1], 2)) for interval in merged_intervals]
    
    return merged_intervals


def apply_vad(vad, audio_path):
    audio, sample_rate = read_wave(audio_path)
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 30, 300, vad, frames)
    segment = gather_segments(segments)
    
    return segment

if __name__ == "__main__":
    df = pd.read_csv("data/train_question.csv")
    audio_list = df.audio.tolist()
    vad = webrtcvad.Vad(3)
    root = "/data/lipsync/question"
    sr = 48000

    for audio_path in tqdm(audio_list):
        os.makedirs(os.path.join(root, os.path.dirname(audio_path.replace("Question-Statement", "Question-Statement_clean"))), exist_ok=True)
        path = os.path.join(root, audio_path)
        new_path = audio_path.replace("Question-Statement", "Question-Statement_clean")

        audio = AudioSegment.from_file(path)
        audio.export(os.path.join(root, new_path), format="wav", codec="pcm_s16le", parameters=["-ac", "1", "-ar", "48000"])

        segments = apply_vad(vad, os.path.join(root, new_path))
        audio = AudioSegment.from_file(os.path.join(root, new_path))
        voice_only = AudioSegment.empty()
        for i, seg in enumerate(segments):
            start, end = seg
            start = int(start * 1000)
            end = int(end * 1000)
            voice_segment = audio[start:end]
            voice_only += voice_segment
            
            voice_only.export(os.path.join(root, new_path), format="wav")
            # librosa.output.write_wav(os.path.join(root, audio_path), voice_only, sr)