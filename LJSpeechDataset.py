import torchaudio
import os
import torch
import torchaudio.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

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
        waveform, sample_rate = torchaudio.load(wave_path)
        if sample_rate != self.sample_rate:
            resample = transforms.Resample(sample_rate, self.sample_rate)
            waveform = resample(waveform)

        # Convert waveform to mel spectrogram
        mel_spectrogram = transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            n_mels=80
        )
        mel = mel_spectrogram(waveform).squeeze(0)

        # Normalize the mel spectrogram
        mel = (mel - mel.min()) / (mel.max() - mel.min())

        # Transpose mel to match the shape [time, n_mels]
        mel = mel.transpose(0, 1)

        # Convert text to a list of character indices
        text_indices = torch.tensor([ord(char) for char in text], dtype=torch.long)

        return mel, text_indices

    @staticmethod
    def collate_fn(batch):
        # Separate the batch into mel spectrograms and texts
        mels, texts = zip(*batch)
        
        # Pad mel spectrograms to the length of the longest one in the batch
        mel_padded = pad_sequence(mels, batch_first=True, padding_value=0.0)
        
        # Pad text sequences to the length of the longest one in the batch
        text_padded = pad_sequence(texts, batch_first=True, padding_value=0)
        
        return mel_padded, text_padded

if __name__ == "__main__":
    data_path = "datasets"
    dataset = LJSpeechDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=LJSpeechDataset.collate_fn)

    for mel, text in dataloader:
        print(mel.shape)  # [batch_size, max_time, n_mels]
        print(text.shape)  # [batch_size, max_text_length]
