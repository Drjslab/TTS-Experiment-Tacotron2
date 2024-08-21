import torch
import torchaudio
import torchaudio.transforms as transforms
import numpy as np
from model import TacotronModel  # Replace with your actual model import
import os
from Tacotron2 import Tacotron2

save_path="output"

# Function to convert text to character indices
def text_to_sequence(text):
    return torch.tensor([ord(c) for c in text], dtype=torch.long).unsqueeze(0)

# Function to convert mel-spectrogram to waveform
def mel_to_waveform(mel_spectrogram, sample_rate=22050):
    # You can use the Griffin-Lim algorithm for waveform reconstruction
    griffin_lim = transforms.GriffinLim(n_fft=1024, hop_length=256, win_length=1024)
    waveform = griffin_lim(mel_spectrogram)
    return waveform

# Load the trained Tacotron model
model = Tacotron2()
final_model_path = f"{save_path}/jp_tacotron.pt"
model.load_state_dict(torch.load(final_model_path))  # Load your trained model's weights
model.eval()

# Inference on the input text
text = "Hello World"
text_sequence = text_to_sequence(text)

with torch.no_grad():
    # Pass the text sequence to the model to get the mel-spectrogram
    mel_spectrogram = model(text_sequence)  # Replace this with your model's inference method

    # Convert the mel-spectrogram to waveform
    waveform = mel_to_waveform(mel_spectrogram.squeeze(0))

# Save the waveform as a .wav file
output_path = "output.wav"
torchaudio.save(output_path, waveform.unsqueeze(0), 22050)

print(f"Audio saved at {output_path}")
