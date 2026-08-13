#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[2]:


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers,
                            dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Reshape input for LSTM: (batch, seq_len, input_size)
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # Take last timestep's output
        out = self.fc(out)
        return out


# In[22]:


def load_data(csv_path, input_size, output_size, test_split_ratio):
    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index(['datetime']).resample('h').mean()
    df = df.loc['2008-10-01 00:00:00':'2024-09-30 00:00:00']
    df = df.reset_index()
    df['Discharge, cubic feet per second'] = df['Discharge, cubic feet per second'].interpolate(method='linear')
    data = df['Discharge, cubic feet per second'].values
    datetime_values = pd.to_datetime(df['datetime']).values

    scaler = StandardScaler()
    data = scaler.fit_transform(data.reshape(-1, 1)).flatten()

    X, y, datetime_y = [], [], []
    for i in range(len(data) - input_size - output_size):
        X.append(data[i:i + input_size])
        y.append(data[i + input_size:i + input_size + output_size])
        datetime_y.append(datetime_values[i + input_size:i + input_size + output_size])

    X = np.array(X)
    y = np.array(y)
    datetime_y = np.array(datetime_y)

    test_size = int(test_split_ratio * len(X))
    train_size = len(X) - test_size
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    datetime_test = datetime_y[train_size:]

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    return (X_train, y_train), (X_test, y_test), scaler, datetime_test


# In[14]:


def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}')


# In[15]:


def mc_dropout_prediction(model, test_loader, num_mc_samples):
    model.train()  # Enable dropout during inference
    all_predictions = []

    with torch.no_grad():
        for _ in range(num_mc_samples):
            mc_predictions = []
            for batch_X, _ in test_loader:
                batch_X = batch_X.to(device)
                prediction = model(batch_X)
                mc_predictions.append(prediction)
            all_predictions.append(torch.cat(mc_predictions, dim=0))

    stacked_predictions = torch.stack(all_predictions)
    mean_prediction = stacked_predictions.mean(dim=0)
    std_prediction = stacked_predictions.std(dim=0)

    return mean_prediction, std_prediction


# In[19]:


def plot_predictions_with_uncertainty(test_targets, mean_prediction, std_prediction):
    if mean_prediction.ndim > 1:
        mean_prediction = mean_prediction[:, 0]
        std_prediction = std_prediction[:, 0]
        test_targets = test_targets[:, 0]

    test_targets = test_targets.cpu().numpy()
    mean_prediction = mean_prediction.cpu().numpy()
    std_prediction = std_prediction.cpu().numpy()

    lower_bound = mean_prediction - 1.96 * std_prediction
    upper_bound = mean_prediction + 1.96 * std_prediction

    plt.figure(figsize=(12, 6))
    plt.plot(test_targets, label='True Values', color='b')
    plt.plot(mean_prediction, label='Mean Prediction', color='r')
    plt.fill_between(range(len(mean_prediction)),
                     lower_bound,
                     upper_bound,
                     color='r', alpha=0.3, label='95% Uncertainty Interval')

    plt.title('Predictions with 95% Uncertainty')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


# In[16]:


def export_results_to_csv(test_targets, mean_prediction, lower_bound, upper_bound, datetime_test, scaler, output_file='predictions.csv'):
    test_targets = scaler.inverse_transform(test_targets.cpu().reshape(-1, 1)).flatten()
    mean_prediction = scaler.inverse_transform(mean_prediction.cpu().reshape(-1, 1)).flatten()
    lower_bound = scaler.inverse_transform(lower_bound.cpu().reshape(-1, 1)).flatten()
    upper_bound = scaler.inverse_transform(upper_bound.cpu().reshape(-1, 1)).flatten()
    datetime_test = datetime_test.flatten()

    results = pd.DataFrame({
        'Datetime': datetime_test,
        'Observation': test_targets,
        'Mean Prediction': mean_prediction,
        'Lower Bound (95%)': lower_bound,
        'Upper Bound (95%)': upper_bound
    })

    results.to_csv(output_file, index=False)
    print(f"Results exported to {output_file}")


# In[21]:


if __name__ == "__main__":
    csv_path = '/home/mostafs/Data/Discharge/Atlanta_02337410.csv'
    input_size = 24
    output_size = 1
    hidden_size = 128
    num_layers = 2
    dropout_rate = 0.1
    num_mc_samples = 20000  
    num_epochs = 50
    batch_size = 32
    test_split_ratio = 0.25

    (X_train, y_train), (X_test, y_test), scaler, datetime_test = load_data(csv_path, input_size, output_size, test_split_ratio)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_loader, nn.MSELoss(), optimizer, num_epochs)

    mean_prediction, std_prediction = mc_dropout_prediction(model, test_loader, num_mc_samples)

    lower_bound = mean_prediction - 1.96 * std_prediction
    upper_bound = mean_prediction + 1.96 * std_prediction

    # Plot results
    #plot_predictions_with_uncertainty(y_test, mean_prediction, std_prediction)

    export_results_to_csv(y_test, mean_prediction, lower_bound, upper_bound, datetime_test, scaler, output_file='predictions.csv')


# In[ ]:




