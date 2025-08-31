import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import seaborn as sns
import warnings
import os

# --- Configuration & Setup ---
warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)
sns.set(style="whitegrid", palette="deep", font_scale=1.2)
plt.rcParams['font.family'] = 'Arial'

# --- Physical and Model Hyperparameters ---
CATCHMENT_AREA = 1944
ROOT_ZONE_DEPTH = 1000
SEQ_LENGTH = 15
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.005
PHYSICS_WEIGHT_WB = 0.5 # Weight for the Water Balance loss. TUNE THIS!

# --- File Paths (Update to your local paths) ---
gridmet_file = r"D:\Rainfall_runoff_Aayush\leaf_river_basin_climate_timeseries.csv"
gleam_file = r"D:\Rainfall_runoff_Aayush\leaf_river_basin_GLEAM_timeseries.csv"
runoff_file = r"D:\Rainfall_runoff_Aayush\discharge_USGS_leaf.csv"

# --- Data Loading and Preprocessing ---
def load_and_preprocess_data(seq_length):
    gridmet_df = pd.read_csv(gridmet_file, parse_dates=['Date'], index_col='Date')
    gleam_df = pd.read_csv(gleam_file, parse_dates=['Date'], index_col='Date')
    runoff_df = pd.read_csv(runoff_file, parse_dates=['Date'], index_col='Date')

    runoff_df['Runoff_mm'] = (runoff_df['Runoff'] * 86_400_000) / (CATCHMENT_AREA * 1_000_000)
    df = gridmet_df[['pr', 'tmmn', 'tmmx']].join(gleam_df[['E', 'SMrz']], how='inner')
    df = df.join(runoff_df[['Runoff_mm']], how='inner').rename(columns={'pr': 'Precipitation'})
    df['Temperature'] = (df['tmmn'] + df['tmmx']) / 2
    df = df.drop(columns=['tmmn', 'tmmx'])

    df['SoilMoisture'] = df['SMrz'] * ROOT_ZONE_DEPTH
    df['AntecedentPrecip7'] = df['Precipitation'].rolling(window=7, min_periods=1).mean()
    df['SoilMoisture_Lag1'] = df['SoilMoisture'].shift(1)
    df = df.drop(columns=['SMrz']).dropna()

    features = ['Precipitation', 'Temperature', 'E', 'SoilMoisture']
    target = 'Runoff_mm'
    all_features = features + ['AntecedentPrecip7', 'SoilMoisture_Lag1']

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    df[all_features] = scaler_x.fit_transform(df[all_features])
    df[[target]] = scaler_y.fit_transform(df[[target]])

    X, y = [], []
    for i in range(len(df) - seq_length):
        X.append(df.iloc[i:i+seq_length][all_features].values)
        y.append(df.iloc[i+seq_length][target])
    X, y = np.array(X), np.array(y)
    X_rf = X.reshape(X.shape[0], -1)

    train_size = int(0.8 * len(X))
    test_dates = df.index[train_size+seq_length:]

    X_train, X_test = X[:train_size], X[train_size:]
    X_train_rf, X_test_rf = X_rf[:train_size], X_rf[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return (X_train, X_test, X_train_rf, X_test_rf, y_train, y_test,
            scaler_x, scaler_y, all_features, test_dates)

# --- Model Definitions ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# --- Physics Loss Function ---
def physics_loss_water_balance(pred_norm, inputs_norm, scaler_x, scaler_y, feature_names):
    """ Water balance loss: P - E - R - dS = 0 """
    pred_real = torch.tensor(scaler_y.inverse_transform(pred_norm.detach().cpu().numpy()), device=pred_norm.device)
    inputs_t_real_np = scaler_x.inverse_transform(inputs_norm[:, -1, :].cpu().numpy())

    p_idx = feature_names.index('Precipitation')
    e_idx = feature_names.index('E')
    sm_idx = feature_names.index('SoilMoisture')
    sm_lag_idx = feature_names.index('SoilMoisture_Lag1')
    
    precipitation = torch.tensor(inputs_t_real_np[:, p_idx], device=pred_norm.device)
    evaporation = torch.tensor(inputs_t_real_np[:, e_idx], device=pred_norm.device)
    runoff = pred_real.squeeze()
    
    sm_t = torch.tensor(inputs_t_real_np[:, sm_idx], device=pred_norm.device)
    sm_t_minus_1 = torch.tensor(inputs_t_real_np[:, sm_lag_idx], device=pred_norm.device)
    delta_sm = sm_t - sm_t_minus_1
    
    residual = precipitation - evaporation - runoff - delta_sm
    return torch.mean(residual**2)

# --- Training and Evaluation Functions ---
def train_model(model, X_train, y_train, X_test, y_test, all_features, is_piml=False):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0
    patience_epochs = 15

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = 0

        for i in range(0, len(X_train), BATCH_SIZE):
            X_batch = X_train[i:i+BATCH_SIZE]
            y_batch = y_train[i:i+BATCH_SIZE]
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            data_loss = criterion(outputs, y_batch)
            
            loss = data_loss
            if is_piml:
                phys_loss = physics_loss_water_balance(outputs, X_batch, scaler_x, scaler_y, all_features)
                loss += PHYSICS_WEIGHT_WB * phys_loss
            
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test).item()

        train_losses.append(epoch_train_loss / len(X_train))
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        print(f'Epoch {epoch+1}/{EPOCHS}, Val Loss: {val_loss:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'best_model_{"piml" if is_piml else "lstm"}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience_epochs:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(f'best_model_{"piml" if is_piml else "lstm"}.pth'))
    return model, train_losses, val_losses


def calculate_kge(obs, sim): return 1 - np.sqrt((np.corrcoef(obs, sim)[0, 1] - 1)**2 + (np.std(sim)/np.std(obs) - 1)**2 + (np.mean(sim)/np.mean(obs) - 1)**2)
def calculate_nse(obs, sim): return 1 - (np.sum((sim - obs)**2) / np.sum((obs - np.mean(obs))**2))

# --- Main Execution ---
# 1. Load Data
(X_train, X_test, X_train_rf, X_test_rf, y_train, y_test,
 scaler_x, scaler_y, all_features, test_dates) = load_and_preprocess_data(SEQ_LENGTH)

# Convert to Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 2. Train Models
# Standard LSTM
print("\n--- Training Standard LSTM ---")
lstm_model = LSTMModel(X_train.shape[2], HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
lstm_model, train_losses_lstm, val_losses_lstm = train_model(lstm_model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, all_features, is_piml=False)

# PIML
print("\n--- Training PIML (Water Balance) ---")
piml_model = LSTMModel(X_train.shape[2], HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
piml_model, train_losses_piml, val_losses_piml = train_model(piml_model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, all_features, is_piml=True)

# Random Forest
print("\n--- Training Random Forest ---")
rf_model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1, max_features=0.5)
rf_model.fit(X_train_rf, y_train.ravel())

# 3. Evaluate Models
models = {"LSTM": lstm_model, "PIML": piml_model, "RF": rf_model}
results = {}

y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Store predictions for plotting
y_pred_lstm_inv = None
y_pred_piml_inv = None
y_pred_rf_inv = None

for name, model in models.items():
    if name == "RF":
        pred_norm = model.predict(X_test_rf).reshape(-1, 1)
    else:
        model.eval()
        with torch.no_grad():
            pred_norm = model(X_test_tensor).cpu().numpy()
    
    pred_inv = scaler_y.inverse_transform(pred_norm).flatten()

    if name == "LSTM":
        y_pred_lstm_inv = pred_inv
    elif name == "PIML":
        y_pred_piml_inv = pred_inv
    elif name == "RF":
        y_pred_rf_inv = pred_inv

    results[name] = {
        "pred": pred_inv,
        "RMSE": np.sqrt(mean_squared_error(y_test_inv, pred_inv)),
        "R2": r2_score(y_test_inv, pred_inv),
        "NSE": calculate_nse(y_test_inv, pred_inv),
        "KGE": calculate_kge(y_test_inv, pred_inv)
    }

print("\n--- Model Performance ---")
for name, metrics in results.items():
    print(f"{name}: KGE={metrics['KGE']:.3f}, NSE={metrics['NSE']:.3f}, R2={metrics['R2']:.3f}, RMSE={metrics['RMSE']:.3f}")

# 4. Visualization
models = ["LSTM", "Random Forest", "PIML"]
predictions = [y_pred_lstm_inv, y_pred_rf_inv, y_pred_piml_inv]
rmse_scores = [results["LSTM"]["RMSE"], results["RF"]["RMSE"], results["PIML"]["RMSE"]]
r2_scores = [results["LSTM"]["R2"], results["RF"]["R2"], results["PIML"]["R2"]]
colors = sns.color_palette("deep", 4)

# Time Series Predictions
fig, axes = plt.subplots(3, 1, figsize=(12, 12), dpi=100, sharex=True)
for i, (model, pred, rmse, r2) in enumerate(zip(models, predictions, rmse_scores, r2_scores)):
    axes[i].plot(test_dates, pred, label=f"{model} Predicted", color=colors[i+1], linestyle='--', linewidth=2)
    axes[i].plot(test_dates, y_test_inv, label="Actual Runoff", color=colors[0], linewidth=3)
    axes[i].set_ylabel("Runoff (mm/day)", fontsize=12)
    axes[i].set_title(f"{model} Rainfall-Runoff Comparison (RMSE: {rmse:.3f}, R²: {r2:.3f})", fontsize=12, pad=10)
    axes[i].legend(loc="upper right", frameon=True, edgecolor="black")
    axes[i].grid(True, which="both", linestyle="--", alpha=0.5)

axes[-1].set_xlabel("Date", fontsize=12)
plt.suptitle("Rainfall-Runoff Prediction Comparison", fontsize=14, y=0.95)
plt.tight_layout()
plt.show()

# Training/Validation Loss
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
plt.yscale("log")
plt.tight_layout()
plt.show()

# Scatter Plot of Predicted vs Observed
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


