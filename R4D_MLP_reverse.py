import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# Define the list of CSV filenames
csv_filenames = ['stim_new.csv', 'Y1_new.csv', 'Y2_new.csv', 'Y3_new.csv']

# Read each CSV file into a pandas DataFrame
dataframes = [pd.read_csv(filename, header=None) for filename in csv_filenames]

# Declare the data
roi1 = dataframes[1]
roi2 = dataframes[2]
roi3 = dataframes[3]
stimuli = dataframes[0]

# Concatenate the neuron data into a single DataFrame
neuron_data = pd.concat([roi1, roi2, roi3], axis=1)

# Split the data into training and test sets
train_stimuli, test_stimuli, train_neurons, test_neurons = train_test_split(stimuli, neuron_data, test_size=0.1)

# Convert dataframes to PyTorch tensors
train_stimulus_tensor = torch.tensor(train_stimuli.values).float()
train_neuron_tensor = torch.tensor(train_neurons.values).float()
test_stimulus_tensor = torch.tensor(test_stimuli.values).float()
test_neuron_tensor = torch.tensor(test_neurons.values).float()

# Define a custom dataset
class StimulusDataset(Dataset):
    def __init__(self, stimulus_tensor, neuron_tensor):
        self.stimulus_tensor = stimulus_tensor
        self.neuron_tensor = neuron_tensor

    def __len__(self):
        return len(self.stimulus_tensor)

    def __getitem__(self, idx):
        return self.stimulus_tensor[idx], self.neuron_tensor[idx]

# Create datasets and data loaders for training and test sets
train_dataset = StimulusDataset(train_stimulus_tensor, train_neuron_tensor)
train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = StimulusDataset(test_stimulus_tensor, test_neuron_tensor)
test_data_loader = DataLoader(test_dataset, batch_size=32)

# Define a simple neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(38,128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model and optimizer
model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=2)

# Train the model using mean squared error loss
num_epochs = 200
for epoch in range(num_epochs):
    # Training loop with progress bar
    with tqdm(total=len(train_data_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
        for i, (stimulus, neurons) in enumerate(train_data_loader):
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(stimulus)
            loss = nn.L1Loss()(outputs, neurons)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update progress bar
            pbar.update(1)


    # Calculate R-squared value for test set
    test_predictions = []
    test_actuals = []
    for i, (stimulus, neurons) in enumerate(test_data_loader):
        outputs = model(stimulus)
        test_predictions.append(outputs.detach().numpy())
        test_actuals.append(neurons.detach().numpy())
    test_predictions = np.concatenate(test_predictions)
    test_actuals = np.concatenate(test_actuals)
    r2 = r2_score(test_actuals, test_predictions)

    print(f'Epoch [{epoch+1}/{num_epochs}], Test R-squared: {r2:.4f}')
# Plot test actual and test predictions
plt.figure(figsize=(50, 6))
plt.plot(test_actuals[:, 0], label='Test Actual')
plt.plot(test_predictions[:, 0], label='Test Predictions')
plt.xlabel('Time Frame')
plt.ylabel('Neuron Activity')
plt.legend()
plt.show()
