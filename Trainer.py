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

    # Training loop
    for epoch in range(10):  # Adjust the number of epochs
        for mel, text in dataloader:
            # Move data to device (if using GPU)
            mel, text = mel.to("cuda"), text.to("cuda")
            model.to("cuda")

            # Forward pass
            optimizer.zero_grad()
            output = model(text, mel)

            # Pad output to match target length
            if output.size(1) < mel.size(1):
                output = nn.functional.pad(output, (0, 0, 0, mel.size(1) - output.size(1)))


            # Compute loss and backpropagate
            loss = criterion(output, mel)
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}], Loss: {loss.item()}")

    print("Training Complete.")
    torch.save({
            'epoch': epochs,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'loss':epoch_loss/len(dataloader),
        }, os.path.join(save_path, f"Final_model.pth")
    )


if __name__ == "__main__":
    # Initialize dataset and model
    data_path = "datasets"
    dataset = LJSpeechDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=LJSpeechDataset.collate_fn)
    
    model = Tacotron2()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    train(model,dataloader,optimizer,criterion, epochs=10, save_path="output")






