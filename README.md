# TTS-Experiment-Tacotron2

Downlad LJDATASETS from [1]
Steps 1:
Create DAtaset load

1. Create 'datasets' folder extact ljDATaset[1] in it will show 1 folder (wavs) and 2 fils metsdata.csv and readme




# Citations 
[1] https://www.kaggle.com/datasets/mathurinache/the-lj-speech-dataset?resource=download



# Chage Logs
15 Aug 
Size mis match not working...

17 Aug 
-> Fixed padding issue at 
''' 
if output.size(1) < mel.size(1):
    output = nn.functional.pad(output, (0, 0, 0, mel.size(1) - output.size(1)))
'''
Added following methods

'''
@staticmethod
    def collate_fn(batch):
        # Separate the batch into mel spectrograms and texts
        mels, texts = zip(*batch)
        
        # Pad mel spectrograms to the length of the longest one in the batch
        mel_padded = pad_sequence(mels, batch_first=True, padding_value=0.0)
        
        # Pad text sequences to the length of the longest one in the batch
        text_padded = pad_sequence(texts, batch_first=True, padding_value=0)
        
        return mel_padded, text_padded
'''

CUDA Fail 
Suggested solutions -> 

-> Disabled globaly CUDA
or single Network only 

'''
self.encoder = nn.LSTM(n_hidden, n_hidden, n_layers, batch_first=True, bidirectional=True, use_cudnn=False)
'''