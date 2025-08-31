import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Set seaborn style for refined aesthetics
sns.set(style="whitegrid", palette="deep", font_scale=1.2)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Catchment area (km²) for Leaf River
CATCHMENT_AREA = 1944  # Update if using a different catchment

# Load and merge datasets (update paths to your files)
gridmet_file = r"D:\Rainfall_runoff_Aayush\leaf_river_basin_climate_timeseries.csv"  # Contains pr, tmmn, tmmx, etr
gleam_file = r"D:\Rainfall_runoff_Aayush\leaf_river_basin_GLEAM_timeseries.csv"  # Contains E, Ep, SMs, SMrz
runoff_file = r"D:\Rainfall_runoff_Aayush\discharge_USGS_leaf.csv"  # Contains Runoff (mm/day)

# Load data
gridmet_df = pd.read_csv(gridmet_file, parse_dates=['Date'], index_col='Date')
gleam_df = pd.read_csv(gleam_file, parse_dates=['Date'], index_col='Date')
runoff_df = pd.read_csv(runoff_file, parse_dates=['Date'], index_col='Date')

# Convert runoff from cumecs to mm/day
runoff_df['Runoff_mm'] = (runoff_df['Runoff'] * 86_400_000) / (CATCHMENT_AREA * 1_000_000)

# Merge datasets
df = gridmet_df[['pr', 'tmmn', 'tmmx']].join(gleam_df[['E', 'SMrz']], how='inner')
df = df.join(runoff_df[['Runoff_mm']], how='inner')

# Preprocess data
def preprocess_data(df, features, target, seq_length=10):
    # Compute mean temperature
    df['Temperature'] = (df['tmmn'] + df['tmmx']) / 2
    
    # Convert SMrz (m³/m³) to mm (assuming root-zone depth of 1000 mm)
    df['SoilMoisture'] = df['SMrz'] * 1000
    
    # Add lagged precipitation and soil moisture features
    df['Precip_Lag1'] = df['pr'].shift(1)
    df['Precip_Lag2'] = df['pr'].shift(2)
    df['SMrz_Lag1'] = df['SoilMoisture'].shift(1)
    
    # Drop unnecessary columns and NaNs
    df = df.drop(columns=['tmmn', 'tmmx', 'SMrz']).dropna()
    
    # Update features list
    all_features = features + ['Precip_Lag1', 'Precip_Lag2', 'SMrz_Lag1']
    
    # Normalize features and target
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    df[all_features] = scaler_x.fit_transform(df[all_features])
    df[target] = scaler_y.fit_transform(df[[target]])
    
    # Create sequences for LSTM/PIML
    def create_sequences(data, target_col, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data.iloc[i:i+seq_length][all_features].values)
            y.append(data.iloc[i+seq_length][target_col])
        return np.array(X), np.array(y)
    
    X, y = create_sequences(df, target, seq_length)
    
    # Create flattened data for RF
    X_rf = X.reshape(X.shape[0], -1)
    
    # Split into train and test sets (80% train, 20% test)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    X_train_rf, X_test_rf = X_rf[:train_size], X_rf[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, X_train_rf, X_test_rf, y_train, y_test, scaler_x, scaler_y, df, train_size, seq_length

# Define features and target
features = ['pr', 'Temperature', 'E', 'SoilMoisture']
target = 'Runoff_mm'

# Preprocess data
X_train, X_test, X_train_rf, X_test_rf, y_train, y_test, scaler_x, scaler_y, df_processed, train_size, seq_length = preprocess_data(df, features, target, seq_length=10)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        out = self.relu(out[:, -1, :])
        out = self.fc(out)
        return out

# Define PIML Model
class PIMLModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(PIMLModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        out = self.relu(out[:, -1, :])
        out = self.fc(out)
        return out

# Physics-informed loss with storage change
def physics_loss(pred, inputs, scaler_x, scaler_y):
    pred = scaler_y.inverse_transform(pred.cpu().detach().numpy())
    inputs = scaler_x.inverse_transform(inputs[:, -1, :].cpu().detach().numpy())
    rainfall = inputs[:, 0]  # Precipitation (pr)
    evaporation = inputs[:, 2]  # Actual Evaporation (E)
    soil_moisture_t = inputs[:, 3]  # SoilMoisture
    soil_moisture_t1 = inputs[:, 6]  # SMrz_Lag1
    delta_sm = soil_moisture_t - soil_moisture_t1  # Storage change
    # Water balance: P - R - E ≈ ΔSM
    physics_violation = torch.tensor((rainfall - pred - evaporation - delta_sm) ** 2)
    return torch.mean(physics_violation)

# Training function
def train_model(model, X_train, y_train, X_test, y_test, epochs, batch_size, physics_weight=0.0):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            optimizer.zero_grad()
            outputs = model(X_batch)
            data_loss = criterion(outputs, y_batch)
            if physics_weight > 0:
                phys_loss = physics_loss(outputs, X_batch, scaler_x, scaler_y)
                loss = data_loss + physics_weight * phys_loss
            else:
                loss = data_loss
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test)
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
    return train_losses, val_losses

# Train LSTM model
lstm_model = LSTMModel(input_size=len(features) + 3, hidden_size=50, num_layers=1, dropout=0.2)
train_losses_lstm, val_losses_lstm = train_model(lstm_model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, epochs=50, batch_size=16)

# Train PIML model
piml_model = PIMLModel(input_size=len(features) + 3, hidden_size=50, num_layers=1, dropout=0.2)
train_losses_piml, val_losses_piml = train_model(piml_model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, epochs=50, batch_size=16, physics_weight=0.2)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_rf, y_train.ravel())

# Predict and evaluate
lstm_model.eval()
with torch.no_grad():
    y_pred_lstm = lstm_model(X_test_tensor).numpy()
    y_pred_piml = piml_model(X_test_tensor).numpy()
y_pred_rf = rf_model.predict(X_test_rf)
y_pred_lstm_inv = scaler_y.inverse_transform(y_pred_lstm)
y_pred_piml_inv = scaler_y.inverse_transform(y_pred_piml)
y_pred_rf_inv = scaler_y.inverse_transform(y_pred_rf.reshape(-1, 1))
y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# Calculate metrics
rmse_lstm = np.sqrt(mean_squared_error(y_test_inv, y_pred_lstm_inv))
r2_lstm = r2_score(y_test_inv, y_pred_lstm_inv)
rmse_rf = np.sqrt(mean_squared_error(y_test_inv, y_pred_rf_inv))
r2_rf = r2_score(y_test_inv, y_pred_rf_inv)
rmse_piml = np.sqrt(mean_squared_error(y_test_inv, y_pred_piml_inv))
r2_piml = r2_score(y_test_inv, y_pred_piml_inv)
print(f'LSTM - RMSE: {rmse_lstm:.3f}, R²: {r2_lstm:.3f}')
print(f'Random Forest - RMSE: {rmse_rf:.3f}, R²: {r2_rf:.3f}')
print(f'PIML - RMSE: {rmse_piml:.3f}, R²: {r2_piml:.3f}')

# Plot predictions in separate subplots with refined aesthetics
fig, axes = plt.subplots(3, 1, figsize=(12, 12), dpi=100, sharex=True)
models = ["LSTM", "Random Forest", "PIML"]
predictions = [y_pred_lstm_inv, y_pred_rf_inv, y_pred_piml_inv]
rmse_scores = [rmse_lstm, rmse_rf, rmse_piml]
r2_scores = [r2_lstm, r2_rf, r2_piml]
colors = sns.color_palette("deep", 4)

for i, (model, pred, rmse, r2) in enumerate(zip(models, predictions, rmse_scores, r2_scores)):
    axes[i].plot(df_processed.index[train_size+seq_length:], pred, label=f"{model} Predicted", 
                 color=colors[i+1], linestyle='--', linewidth=2)
    axes[i].plot(df_processed.index[train_size+seq_length:], y_test_inv, label="Actual Runoff", 
                 color=colors[0], linewidth=3)
    axes[i].set_ylabel("Runoff (mm/day)", fontsize=12)
    axes[i].set_title(f"{model} Rainfall-Runoff Comparison (RMSE: {rmse:.3f}, R²: {r2:.3f})", 
                      fontsize=12, pad=10)
    axes[i].legend(loc="upper right", frameon=True, edgecolor="black")
    axes[i].grid(True, which="both", linestyle="--", alpha=0.5)

axes[-1].set_xlabel("Date", fontsize=12)
plt.suptitle("Rainfall-Runoff Prediction Comparison", fontsize=14, y=0.95)
plt.tight_layout()
plt.show()

# Plot training and validation loss with refined aesthetics
plt.figure(figsize=(10, 5), dpi=100)
plt.plot(train_losses_lstm, label="LSTM Train Loss", color=colors[0], linewidth=2)
plt.plot(val_losses_lstm, label="LSTM Val Loss", color=colors[0], linestyle="--", linewidth=2)
plt.plot(train_losses_piml, label="PIML Train Loss", color=colors[3], linewidth=2)
plt.plot(val_losses_piml, label="PIML Val Loss", color=colors[3], linestyle="--", linewidth=2)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("Training and Validation Loss", fontsize=14, pad=15)
plt.legend(loc="upper right", frameon=True, edgecolor="black")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.yscale("log")  # Log scale for better visualization
plt.tight_layout()
plt.show()

# Plot predicted vs. observed scatter plots for R²
fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=100, sharex=True, sharey=True)
for i, (model, pred, r2) in enumerate(zip(models, predictions, r2_scores)):
    axes[i].scatter(y_test_inv, pred, color=colors[i+1], alpha=0.6, edgecolors="black", s=50)
    axes[i].plot([y_test_inv.min(), y_test_inv.max()], [y_test_inv.min(), y_test_inv.max()], 
                 color="black", linestyle="--", linewidth=1)
    axes[i].set_xlabel("Observed Runoff (mm/day)", fontsize=12)
    axes[i].set_ylabel("Predicted Runoff (mm/day)", fontsize=12)
    axes[i].set_title(f"{model} (R² = {r2:.3f})", fontsize=12, pad=10)
    axes[i].grid(True, which="both", linestyle="--", alpha=0.5)
plt.suptitle("Predicted vs. Observed Runoff", fontsize=14, y=1.05)
plt.tight_layout()
plt.show()