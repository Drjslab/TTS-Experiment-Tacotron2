import torchaudio
import datasets
from torch.utils.data import Dataset, DataLoader
import os
import torchaudio.transforms as transform
import torch


class LJSpeechDataset(Dataset):
    def __init__(self,data_path,sample_rate=22050):
        self.data_path = data_path
        self.sample_rate = sample_rate
        self.metadata = self.get_metadata()
    
    def get_metadata(self):
        with open(os.path.join(self.data_path,"metadata.csv"), "r", encoding="utf-8") as f:
            metadata = [line.strip().split("|") for line in f]
        return metadata
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # audio-> wav -> mel -> tensor 
        wave_path = os.path.join(self.data_path,"wavs",self.metadata[idx][0]+".wav")
        text = self.metadata[idx][2]


        # A
        # load waveform,
        waveform, sample_rate = torchaudio.load(wave_path)
        if sample_rate != self.sample_rate:
            resample = transforms.Resample(sample_rate, self.sample_rate)
            waveform = resample(waveform)
        
        # get MALFrq
        mel_spectrogram = transform.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=80
        )

        mel = mel_spectrogram(waveform).squeeze(0)
        mel =mel_spectrogram(mel-mel.min()) / (mel.max() - mel.min())
        
    
        # Transpose mel to match the shape [n_mels, time] -> [time, n_mels]
        mel = mel.transpose(0, 1)        
        return torch.tensor(mel), text


if __name__ == "__main__":
    data_path = "datasets"
    dataset = LJSpeechDataset(data_path)
    print(dataset)



    
         