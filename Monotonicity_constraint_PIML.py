# =============================================================================
# OPTION 3: MONOTONICITY CONSTRAINT PIML (WITH PLOTS)
# =============================================================================

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

# --- 1. CONFIGURATION & SETUP ---
warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)
sns.set(style="whitegrid", palette="deep", font_scale=1.1)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.dpi'] = 100

# --- Physical and Model Hyperparameters (TUNE THESE) ---
CATCHMENT_AREA = 1944
ROOT_ZONE_DEPTH = 1000
SEQ_LENGTH = 15
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.005
PHYSICS_WEIGHT_WB = 0.5   # Weight for the Water Balance loss
PHYSICS_WEIGHT_MONO = 10.0 # Weight for the Monotonicity loss (often needs to be higher)

# --- File Paths (UPDATE THESE TO YOUR LOCAL PATHS) ---
gridmet_file = r"D:\Rainfall_runoff_Aayush\leaf_river_basin_climate_timeseries.csv"
gleam_file = r"D:\Rainfall_runoff_Aayush\leaf_river_basin_GLEAM_timeseries.csv"
runoff_file = r"D:\Rainfall_runoff_Aayush\discharge_USGS_leaf.csv"

# --- 2. DATA LOADING & PREPROCESSING ---
# The load_and_preprocess_data function is identical to the one in Option 2
def load_and_preprocess_data(seq_length):
    # This function is identical to the one in the previous answer
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
            scaler_x, scaler_y, all_features, test_dates, train_size)

# --- 3. MODEL & LOSS FUNCTION DEFINITIONS ---
# Models (LSTM_PIML_Mono) and physics losses are identical to previous answer
class LSTM_PIML_Mono(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTM_PIML_Mono, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def physics_loss_water_balance(pred_norm, inputs_norm, scaler_x, scaler_y, feature_names):
    pred_real = torch.tensor(scaler_y.inverse_transform(pred_norm.detach().cpu().numpy()), device=pred_norm.device)
    inputs_t_real_np = scaler_x.inverse_transform(inputs_norm[:, -1, :].cpu().numpy())
    p_idx, e_idx, sm_idx, sm_lag_idx = [feature_names.index(f) for f in ['Precipitation', 'E', 'SoilMoisture', 'SoilMoisture_Lag1']]
    p = torch.tensor(inputs_t_real_np[:, p_idx], device=pred_norm.device)
    e = torch.tensor(inputs_t_real_np[:, e_idx], device=pred_norm.device)
    runoff = pred_real.squeeze()
    sm_t = torch.tensor(inputs_t_real_np[:, sm_idx], device=pred_norm.device)
    sm_t_minus_1 = torch.tensor(inputs_t_real_np[:, sm_lag_idx], device=pred_norm.device)
    delta_sm = sm_t - sm_t_minus_1
    residual = p - e - runoff - delta_sm
    return torch.mean(residual**2)

def physics_loss_monotonicity(model, inputs_norm, feature_names):
    sm_feature_idx = feature_names.index('SoilMoisture')
    inputs_with_grad = inputs_norm.clone().detach().requires_grad_(True)
    outputs = model(inputs_with_grad)
    gradients = torch.autograd.grad(outputs=outputs, inputs=inputs_with_grad, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]
    grad_for_sm = gradients[:, -1, sm_feature_idx]
    violation = torch.relu(-grad_for_sm)
    return torch.mean(violation)

# --- 4. TRAINING & EVALUATION ---
def train_model(model, X_train, y_train, X_test, y_test, model_name="lstm"):
    """Generic training function, NOW RETURNS LOSS HISTORY."""
    # This function is identical to the one in Option 2
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    best_val_loss = float('inf')
    patience, max_patience = 0, 15
    train_losses, val_losses = [], []
    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = 0
        for i in range(0, len(X_train), BATCH_SIZE):
            X_batch, y_batch = X_train[i:i+BATCH_SIZE], y_train[i:i+BATCH_SIZE]
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        train_losses.append(epoch_train_loss / (len(X_train) / BATCH_SIZE))
        model.eval()
        with torch.no_grad(): val_loss = criterion(model(X_test), y_test)
        val_losses.append(val_loss.item())
        scheduler.step(val_loss)
        print(f'Epoch {epoch+1}/{EPOCHS}, Val Loss: {val_loss.item():.6f}')
        if val_loss < best_val_loss:
            best_val_loss, patience = val_loss.item(), 0
            torch.save(model.state_dict(), f'best_model_{model_name}.pth')
        else:
            patience += 1
            if patience >= max_patience:
                print(f"Early stopping at epoch {epoch+1}"); break
    model.load_state_dict(torch.load(f'best_model_{model_name}.pth'))
    return model, train_losses, val_losses

def train_piml_monotonicity(model, X_train, y_train, X_test, y_test, all_features):
    """Specialized training for Monotonicity PIML, NOW RETURNS LOSS HISTORY."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    best_val_loss = float('inf')
    patience, max_patience = 0, 15
    train_losses, val_losses = [], []
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = 0
        for i in range(0, len(X_train), BATCH_SIZE):
            X_batch, y_batch = X_train[i:i+BATCH_SIZE], y_train[i:i+BATCH_SIZE]
            optimizer.zero_grad()
            outputs = model(X_batch)
            data_loss = criterion(outputs, y_batch)
            wb_loss = physics_loss_water_balance(outputs, X_batch, scaler_x, scaler_y, all_features)
            mono_loss = physics_loss_monotonicity(model, X_batch, all_features)
            loss = data_loss + PHYSICS_WEIGHT_WB * wb_loss + PHYSICS_WEIGHT_MONO * mono_loss
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        train_losses.append(epoch_train_loss / (len(X_train) / BATCH_SIZE))
        model.eval()
        with torch.no_grad(): val_loss = criterion(model(X_test), y_test)
        val_losses.append(val_loss.item())
        scheduler.step(val_loss)
        print(f'Epoch {epoch+1}/{EPOCHS}, Val Loss: {val_loss.item():.6f}')
        if val_loss < best_val_loss:
            best_val_loss, patience = val_loss.item(), 0
            torch.save(model.state_dict(), 'best_model_piml_mono.pth')
        else:
            patience += 1
            if patience >= max_patience:
                print(f"Early stopping at epoch {epoch+1}"); break

    model.load_state_dict(torch.load('best_model_piml_mono.pth'))
    return model, train_losses, val_losses

def calculate_kge(obs, sim): return 1 - np.sqrt((np.corrcoef(obs, sim)[0, 1] - 1)**2 + (np.std(sim)/np.std(obs) - 1)**2 + (np.mean(sim)/np.mean(obs) - 1)**2)
def calculate_nse(obs, sim): return 1 - (np.sum((sim - obs)**2) / np.sum((obs - np.mean(obs))**2))

# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    (X_train, X_test, X_train_rf, X_test_rf, y_train, y_test,
     scaler_x, scaler_y, all_features, test_dates, train_size) = load_and_preprocess_data(SEQ_LENGTH)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    print("\n--- Training Standard LSTM ---")
    lstm_model = LSTM_PIML_Mono(X_train.shape[2], HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
    lstm_model, lstm_train_loss, lstm_val_loss = train_model(lstm_model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, "lstm")
    
    print("\n--- Training Random Forest ---")
    rf_model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1, max_features=0.5)
    rf_model.fit(X_train_rf, y_train.ravel())

    print("\n--- Training PIML (Monotonicity Constraint) ---")
    piml_mono_model = LSTM_PIML_Mono(X_train.shape[2], HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
    piml_mono_model, piml_train_loss, piml_val_loss = train_piml_monotonicity(piml_mono_model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, all_features)

    models = {"LSTM": lstm_model, "Random Forest": rf_model, "PIML-Monotonicity": piml_mono_model}
    results = {}
    y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    for name, model in models.items():
        if name == "Random Forest": pred_norm = model.predict(X_test_rf).reshape(-1, 1)
        else:
            model.eval()
            with torch.no_grad(): pred_norm = model(X_test_tensor).cpu().numpy()
        pred_inv = scaler_y.inverse_transform(pred_norm).flatten()
        results[name] = {
            "pred": pred_inv,
            "RMSE": np.sqrt(mean_squared_error(y_test_inv, pred_inv)),
            "R2": r2_score(y_test_inv, pred_inv),
            "NSE": calculate_nse(y_test_inv, pred_inv),
            "KGE": calculate_kge(y_test_inv, pred_inv)
        }

    print("\n--- Model Performance Comparison ---")
    for name, metrics in results.items():
        print(f"{name}: KGE={metrics['KGE']:.3f}, NSE={metrics['NSE']:.3f}, R2={metrics['R2']:.3f}, RMSE={metrics['RMSE']:.3f}")
    
    # --- 6. PLOTTING ---
    model_names = ["LSTM", "Random Forest", "PIML-Monotonicity"]
    predictions = [results[m]["pred"] for m in model_names]
    rmse_scores = [results[m]["RMSE"] for m in model_names]
    r2_scores = [results[m]["R2"] for m in model_names]
    colors = sns.color_palette("deep", 4)
    
    # Plot 1: Time-series predictions
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), dpi=100, sharex=True)
    for i, (model_name, pred, rmse, r2) in enumerate(zip(model_names, predictions, rmse_scores, r2_scores)):
        axes[i].plot(test_dates, pred, label=f"{model_name} Predicted", color=colors[i+1], linestyle='--', linewidth=2)
        axes[i].plot(test_dates, y_test_inv, label="Observed Runoff", color=colors[0], linewidth=2.5, alpha=0.8)
        axes[i].set_ylabel("Runoff (mm/day)")
        title = f"{model_name} | KGE: {results[model_name]['KGE']:.3f}, R²: {r2:.3f}, RMSE: {rmse:.3f}"
        axes[i].set_title(title, fontsize=14)
        axes[i].legend(loc="upper right")
        axes[i].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[-1].set_xlabel("Date")
    plt.suptitle("Model Prediction Comparison on Test Set", fontsize=16, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

    # Plot 2: Training and validation loss
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(lstm_train_loss, label="LSTM Train Loss", color=colors[1], linewidth=2)
    plt.plot(lstm_val_loss, label="LSTM Val Loss", color=colors[1], linestyle="--", linewidth=2)
    plt.plot(piml_train_loss, label="PIML-Monotonicity Train Loss", color=colors[3], linewidth=2)
    plt.plot(piml_val_loss, label="PIML-Monotonicity Val Loss", color=colors[3], linestyle="--", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.yscale("log")
    plt.tight_layout()
    plt.show()

    # Plot 3: Predicted vs. observed scatter plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=100, sharex=True, sharey=True)
    fig.suptitle("Predicted vs. Observed Runoff", fontsize=16, y=1.02)
    for i, (model_name, pred, r2) in enumerate(zip(model_names, predictions, r2_scores)):
        axes[i].scatter(y_test_inv, pred, color=colors[i+1], alpha=0.5, edgecolors='k', s=40)
        min_val, max_val = min(y_test_inv.min(), pred.min()), max(y_test_inv.max(), pred.max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
        axes[i].set_xlabel("Observed Runoff (mm/day)")
        axes[i].set_title(f"{model_name} (R² = {r2:.3f})")
        axes[i].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Predicted Runoff (mm/day)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()