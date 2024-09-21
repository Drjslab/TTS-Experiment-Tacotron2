import re
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Function to extract epoch and loss from log line
def extract_epoch_loss(log_line):
    pattern = r"Epoch \[(\d+)/\d+\], Avg Loss: ([\d\.]+)"
    match = re.search(pattern, log_line)
    if match:
        epoch = int(match.group(1))
        avg_loss = float(match.group(2))
        return epoch, avg_loss
    return None

# Reading the log file and creating dataset
log_file = 'app.log'
epochs = []
losses = []

with open(log_file, 'r') as file:
    for line in file:
        result = extract_epoch_loss(line)
        if result:
            epoch, avg_loss = result
            epochs.append(epoch)
            losses.append(avg_loss)

# Creating DataFrame from extracted data
df = pd.DataFrame({
    'Epoch': epochs,
    'Avg Loss': losses
})

# Preparing data for linear regression
X = np.array(epochs).reshape(-1, 1)  # Epoch values as features (reshaped to 2D array)
y = np.array(losses)  # Avg Loss values as target

# Create and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict the next 10 points
next_epochs = np.array(range(epochs[-1] + 1, epochs[-1] + 11)).reshape(-1, 1)
predicted_losses = model.predict(next_epochs)

# Append predictions to the dataframe
predicted_df = pd.DataFrame({
    'Epoch': next_epochs.flatten(),
    'Avg Loss': predicted_losses
})

# Concatenate the original and predicted dataframes
combined_df = pd.concat([df, predicted_df])

# Print the combined dataframe
print("Combined Data with Predictions:")
print(combined_df)

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['Avg Loss'], marker='o', linestyle='-', color='b', label='Original Data')
plt.plot(predicted_df['Epoch'], predicted_df['Avg Loss'], marker='x', linestyle='--', color='r', label='Predicted Data')
plt.title('Epoch vs Avg Loss (with Predictions)')
plt.xlabel('Epoch')
plt.ylabel('Avg Loss')
plt.grid(True)
plt.legend()
plt.show()
