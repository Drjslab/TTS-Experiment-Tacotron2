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

        waveform, sample_rate = torchaudio.load(wave_path)

        print("\n") 

        if sample_rate != self.sample_rate:
            resample = transforms.Resample(sample_rate, self.sample_rate)
            waveform = resample(waveform)
        

        # Convert waveform to mel spectrogram

        spectrogram = transforms.Spectrogram(
            n_fft=1024,
            win_length=1024,
            hop_length=512,

        )
        
        mel_spectrogram = transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            win_length=1024,
            hop_length=512,
            n_mels=80
        )

        mel = mel_spectrogram(waveform)
        mel = mel.squeeze(0)

        specto = spectrogram(waveform)
        specto = specto.squeeze(0)


        '''
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(10,4))
        log_mel_Mel_spec = torch.log(mel + 1e-9)
        log_mel_spec = torch.log(specto + 1e-9)
        ax1.imshow(log_mel_Mel_spec)
        ax1.set_title('Mel Specto')
        ax2.imshow(log_mel_spec)
        ax2.set_title('Spectogram 2')
        ax3.plot(waveform.t().numpy())
        plt.tight_layout()
        plt.show()
        '''
        
        # Transpose mel to match the shape [time, n_mels]
        tensor = mel.transpose(0, 1)

        # Define the target size
        target_size = 512

        # Padding
        if tensor.size(0) < target_size:
            pad_size = target_size - tensor.size(0)
            padded_tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_size), mode='constant', value=0.0)
        else:
            # Truncating
            padded_tensor = tensor[:target_size, :]


        # Convert text to a list of character indices
        text_indices = torch.tensor([ord(char) for char in text], dtype=torch.long)
        print(padded_tensor.shape)
        return padded_tensor, text_indices,  self.metadata[idx][0]
    
    @staticmethod
    def collate_fn(batch):
        # Separate the batch into mel spectrograms and texts
        max_length = 512
        mels, texts, name = zip(*batch)
        # mymels = torch.

        # Pad mel spectrograms to the length of the longest one in the batch
        mel_padded = pad_sequence(mels, batch_first=True, padding_value=0.0)
        # mel_padded = [pad_or_truncate(mel, max_length) for mel in mels]

        print(mel_padded)
        
        # Pad text sequences to the length of the longest one in the batch
        text_padded = pad_sequence(texts, batch_first=True, padding_value=0)        
        return mel_padded, text_padded, name


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


if __name__ == "__main__":
    data_path = "datasets"
    dataset = LJSpeechDataset(data_path)
    mel, text, name = dataset[1]
    print("Final")
    print(mel.shape)
