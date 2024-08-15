import torch.nn as nn



class Tacotron2(nn.Module):
    def __init__(self,n_mels=80,n_hidden=256,n_layers=2,output_size=80):
        super(Tacotron2, self).__init__()
        self.embedding = nn.Embedding(256,n_hidden)
        self.encoder = nn.LSTM(n_hidden, n_hidden, n_layers,batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(n_hidden*2, n_hidden, n_layers, batch_first=True)
        self.fc = nn.Linear(n_hidden, output_size)
    
    def forward(self, text, mel_input):
        embedded  = self.embedding(text)
        encoder_output,_ = self.encoder(embedded)
        decoder_output, _ = self.decoder(encoder_output)
        output = self.fc(decoder_output)
        return output

    