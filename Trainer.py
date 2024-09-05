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
    filemode='w',                 # Append mode (use 'w' for overwrite mode)
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO            # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
)

def train(model, dataloader, optimizer, criterion, epochs=50, save_path="output"):
    model.train()
    # Create a directory for saving checkpoints
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        for mel, text, name in dataloader:
            try:

                print("AT Trainging ", mel.shape, text.shape)
                # Move data to device (if using GPU)
                # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                device ="cpu"
                print(f"Using {device}")
                mel, text = mel.to(device), text.to(device)
                model.to(device)

                # Forward pass
                optimizer.zero_grad()
                output = model(text, mel)

                print("output", output.shape)

                # Pad output to match target length
                if output.size(1) < mel.size(1):
                    output = nn.functional.pad(output, (0, 0, 0, mel.size(1) - output.size(1)))

                # Compute loss
                loss = criterion(output, mel)
                loss.backward()
                optimizer.step()

                # Calculate Mean Absolute Error as a measure of accuracy
                mae = torch.mean(torch.abs(output - mel)).item()
                epoch_accuracy += mae

                # Update epoch lossD
                epoch_loss += loss.item()

                infoTxt = f"Epoch [{epoch + 1}], Loss: {loss.item()}, MAE: {mae} for {name}"
                logging.info(infoTxt)
                print(infoTxt)

                # input("Press enter to continue.")

            except Exception as e:
                errorTxt = f"Error: {e} for {name}"
                print(errorTxt)
                logging.error(errorTxt)

        # Calculate average loss and accuracy for the epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        avg_epoch_accuracy = epoch_accuracy / len(dataloader)

        # Log and print epoch statistics
        epoch_info = f"Epoch [{epoch + 1}/{epochs}], Avg Loss: {avg_epoch_loss:.4f}, Avg MAE: {avg_epoch_accuracy:.4f}"
        logging.info(epoch_info)
        print(epoch_info)

    print("Training Complete.")
    final_model_path = f"{save_path}/jp_tacotron.pt"
    torch.save(model.state_dict(), final_model_path)

if __name__ == "__main__":
    # Initialize dataset and model
    data_path = "datasets"
    dataset = LJSpeechDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=LJSpeechDataset.collate_fn)
    
    model = Tacotron2()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train(model, dataloader, optimizer, criterion, epochs=150, save_path="output")
