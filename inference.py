import torch
import torchaudio
import torchaudio.transforms as transforms
import numpy as np
import os
from Tacotron2 import Tacotron2

save_path="output"

# Function to convert text to character indices
def text_to_sequence(text):
    return torch.tensor([ord(c) for c in text], dtype=torch.long).unsqueeze(0)



def mel_to_waveform(mel_spectrogram, sample_rate=22050):
    # Create an inverse Mel filter bank
    n_fft = 1024
    mel_to_stft = transforms.InverseMelScale(
        n_stft=n_fft // 2 + 1, 
        n_mels=80,  # Same as your mel-spectrogram
        sample_rate=sample_rate
    )
    
    # Project the mel-spectrogram back to STFT
    stft = mel_to_stft(mel_spectrogram)
    
    # Apply the Griffin-Lim algorithm
    griffin_lim = transforms.GriffinLim(n_fft=n_fft, hop_length=512, win_length=1024)
    waveform = griffin_lim(stft)
    
    return waveform

# Load the trained Tacotron model
model = Tacotron2()
final_model_path = f"{save_path}/jp_tacotron.pt"
model.load_state_dict(torch.load(final_model_path))  # Load your trained model's weights
model.eval()

# Inference on the input text
text = "Hello World, this world is so big, What is my name. I love India and continetal."
text_sequence = text_to_sequence(text)

print("text_sequence", text_sequence.shape)

with torch.no_grad():
    # Pass the text sequence to the model to get the mel-spectrogram
    mel_input= ""
    output = model(text_sequence, mel_input)  # Replace this with your model's inference method

    print("output",output)

    waveform = mel_to_waveform(output)
    print("p", waveform.shape)

# Save the waveform as a .wav file
output_path = "output.wav"
torchaudio.save(output_path, waveform, 22050)

print(f"Audio saved at {output_path}")
print("Plaing sound")
os.system(f"aplay {output_path}")
print("Done.")