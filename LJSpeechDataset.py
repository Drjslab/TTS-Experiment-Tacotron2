import torchaudio
import datasets
from torch.utils.data import Dataset, DataLoader
import os
import torchaudio.transforms as transform



class LJSpeechDataset(Dataset):
    def __init__(self,data_path,sample_reate=22050):
        self.data_path = data_path
        self.sample_reate = sample_reate
        self.metadata = self.get_metadata()
    
    def get_metadata(self):
        with open(os.path.join(self.data_path,"metadata.csv"), "r", encoding="utf-8") as f:
            metadata = [line.strip().split("|") for line in f]
        return metadata
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # audio-> wav -> mel -> tensor 
        wave_path = os.path.join(self.data_path,"wav",self.metadata[idx][0]+".wav")
        text = self.metadata[idx][2]


        # A
        # load waveform,
        waveform, sample_rate = torchaudio.load(wav_path)
        if sample_rate != self.sample_rate:
            resample = transforms.Resample(sample_rate, self.sample_rate)
            waveform = resample(waveform)
        
        # get MALFrq
        mel_spectrogram = transform.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=80
        )
        
        mel =mel_spectrogram(mel-mel.min()) / (mel.max() - mel.min())

        return torch.tensor(mel), text


if __name__ == "__main__":
    data_path = "datasets"
    dataset = LJSpeechDataset(data_path)
    print(dataset)



    
         