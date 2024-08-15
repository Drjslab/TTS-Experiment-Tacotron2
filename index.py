from Tacotron2 import Tacotron2
from LJSpeechDataset import LJSpeechDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import os
import torch

def train(model, dataloader, optimizer,criterion, epochs=10,save_path="output"):
    model.train()
    # Create a directory for saving checkpoints
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for epoch in range(epochs):
        epoch_loss =0
        for mel, text in tqdm(dataloader):
            text = [ord(char) for char in text[0]]
            text = torch.tensor(text).unsqueeze(0)

            print(mel.size())

            mel_input = mel[:,:-1,:]
            mel_target = mel[:,1:,:]

            print("mel_target", mel_target.size())
            print("mel_input", mel_input.size())
            print("***************")
            output = model(text,mel_input)
            
            output = output[:, :mel_target.shape[1], :] 

            loss = criterion(output, mel_target)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        torch.save({
            'epoch':epoch+1,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'loss':epoch_loss /len(dataloader) },
            os.path.join(save_path, f"model_epoch_{epoch +1}.pth")
        )

        print(f"Epoch {epoch+1}/{epochs}, Loss :{epoch_loss/len(dataloader):.4f}")

    print("Training Complete.")
    torch.save({
            'epoch': epochs,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'loss':epoch_loss/len(dataloader),
        }, os.path.join(save_path, f"Final_model.pth")
    )


if __name__ == "__main__":
    data_path = "datasets"
    dataset = LJSpeechDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = Tacotron2()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    train(model,dataloader,optimizer,criterion, epochs=10, save_path="output")






