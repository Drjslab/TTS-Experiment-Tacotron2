from Tacotron2 import Tacotron2
from LJSpeechDataset import LJSpeechDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import os
import torch
import logging


# Configure logging
logging.basicConfig(
    filename='app.log',           # Log file name
    filemode='a',                 # Append mode (use 'w' for overwrite mode)
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO            # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
)

def train(model, dataloader, optimizer,criterion, epochs=10,save_path="output"):
    model.train()
    # Create a directory for saving checkpoints
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Training loop
    for epoch in range(10):  # Adjust the number of epochs
        for mel, text, name in dataloader:
            try:
                # Move data to device (if using GPU)
                mel, text = mel.to("cpu"), text.to("cpu")
                model.to("cpu")

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
                infoTxt = f"Epoch [{epoch+1}], Loss: {loss.item()} for {name}"
                logging.info(infoTxt)
                print(infoTxt)
            except Exception as e:
                errorTxt = f"Error : {e} for {name}"
                print(errorTxt)
                logging.error(infoTxt)

    print("Training Complete.")
    final_model_path = f"{save_path}/jp_tacotron.pt"
    torch.save(model.state_dict(), final_model_path)
    # torch.save({
    #         'epoch': epochs,
    #         'model_state_dict':model.state_dict(),
    #         'optimizer_state_dict':optimizer.state_dict(),
    #         'loss':epoch_loss/len(dataloader),
    #     }, os.path.join(save_path, f"Final_model.pth")
    # )


if __name__ == "__main__":
    # Initialize dataset and model
    data_path = "datasets"
    dataset = LJSpeechDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=LJSpeechDataset.collate_fn)
    
    model = Tacotron2()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    train(model,dataloader,optimizer,criterion, epochs=10, save_path="output")






