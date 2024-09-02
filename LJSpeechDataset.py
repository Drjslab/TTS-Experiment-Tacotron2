import torchaudio
import os
import torch
import torchaudio.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

class LJSpeechDataset(Dataset):
    def __init__(self, data_path, sample_rate=22050):
        self.data_path = data_path
        self.sample_rate = sample_rate
        self.metadata = self.get_metadata()

    def get_metadata(self):
        with open(os.path.join(self.data_path, "metadata.csv"), "r", encoding="utf-8") as f:
            metadata = [line.strip().split("|") for line in f]
        return metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Load audio and text data
        wave_path = os.path.join(self.data_path, "wavs", self.metadata[idx][0] + ".wav")
        text = self.metadata[idx][2]

        # Load waveform

        # wave_path = "https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/Samples/AFsp/M1F1-Alaw-AFsp.wav"
        waveform, sample_rate = torchaudio.load(wave_path)
        

        if sample_rate != self.sample_rate:
            resample = transforms.Resample(sample_rate, self.sample_rate)
            waveform = resample(waveform)

        # Convert waveform to mel spectrogram
        

        print(waveform.shape)

        spectrogram = transforms.Spectrogram(
            n_fft=1024,
            win_length=1024,
            hop_length=256,

        )
        
        mel_spectrogram = transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            n_mels=100
        )

        mel = mel_spectrogram(waveform)
        print("mel", mel.shape)
        mel = mel.squeeze(0)
        print("mel", mel.shape)
        print("*"*10)

        specto = spectrogram(waveform).squeeze(0)


        fig,((ax1,ax2),(ax3,_)) = plt.subplots(2,2,figsize=(10,4))

        ax1.imshow(mel)
        ax1.set_title('Mel Specto')
        ax2.imshow(specto)
        ax2.set_title('Spectogram 2')
        ax3.plot(waveform.t().numpy())
        plt.tight_layout()
        plt.show()

        # Normalize the mel spectrogram
        # mel = (mel - mel.min()) / (mel.max() - mel.min())

        # Convert to numpy for plotting
       

        # Transpose mel to match the shape [time, n_mels]
        mel = mel.transpose(0, 1)
        print(mel.shape)


        # Convert text to a list of character indices
        text_indices = torch.tensor([ord(char) for char in text], dtype=torch.long)

        return mel, text_indices,  self.metadata[idx][0]

    @staticmethod
    def collate_fn(batch):
        # Separate the batch into mel spectrograms and texts
        mels, texts, name = zip(*batch)
        
        # Pad mel spectrograms to the length of the longest one in the batch
        mel_padded = pad_sequence(mels, batch_first=True, padding_value=0.0)
        
        # Pad text sequences to the length of the longest one in the batch
        text_padded = pad_sequence(texts, batch_first=True, padding_value=0)
        
        return mel_padded, text_padded, name


def mel_to_waveform(mel_spectrogram, sample_rate=22050):
    # Create an inverse Mel filter bank
    n_fft = 1024
    mel_to_stft = transforms.InverseMelScale(
        n_stft=n_fft // 2 + 1, 
        n_mels=490,  # Same as your mel-spectrogram
        sample_rate=sample_rate
    )
    
    # Project the mel-spectrogram back to STFT
    stft = mel_to_stft(mel_spectrogram)
    
    # Apply the Griffin-Lim algorithm
    griffin_lim = transforms.GriffinLim(n_fft=n_fft, hop_length=256, win_length=1024)
    waveform = griffin_lim(stft)
    
    return waveform


if __name__ == "__main__":
    data_path = "datasets"
    dataset = LJSpeechDataset(data_path)
    # dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=LJSpeechDataset.collate_fn)
    # print(type(dataloader))

    mel, text, name = dataset[5]
    # Plot the MelSpectrogram
    print(name)
    # mel_np = mel.detach().numpy()
    # # plt.figure(figsize=(10, 4))
    # # plt.imshow(mel_np, aspect='auto', origin='lower')
    # # plt.colorbar(format='%+2.0f dB')
    # # plt.title('Mel Spectrogram')
    # # plt.xlabel('Time')
    # # plt.ylabel('Mel Frequency')
    # # plt.show()

    # x = mel_to_waveform(mel,sample_rate=22050)
    #     # Save the waveform as a .wav file
    # output_path = "output_testt.wav"
    # torchaudio.save(output_path, x.unsqueeze(0), 22050)


