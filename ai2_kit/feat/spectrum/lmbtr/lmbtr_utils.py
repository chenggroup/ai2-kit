# lmbtr_utils.py

# 1. Single Import Block
import os
import re
import pickle
import logging
import concurrent.futures
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from ase import Atoms
from ase.io import read
from dscribe.descriptors import LMBTR
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Helper class for the neural network, used by multiple functions
class NeuralNetwork(nn.Module):
    """A simple fully-connected neural network for regression."""
    def __init__(self, input_dim: int, layer_size: int = 256):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, layer_size)
        self.fc2 = nn.Linear(layer_size, layer_size)
        self.fc3 = nn.Linear(layer_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Internal helper functions (prefixed with _) to support the main utility functions
def _get_xyz_from_gout(file_content: str) -> Atoms | None:
    """Extracts ASE Atoms object from the content of a Gaussian output file."""
    lines = file_content.splitlines()
    start_index, end_index = None, None
    for i, line in enumerate(lines):
        if 'Symbolic Z-matrix' in line:
            start_index = i + 2
        elif 'ITRead' in line and start_index is not None:
            end_index = i - 1
            break
            
    if start_index is not None and end_index is not None:
        coords_forces_raw = lines[start_index:end_index]
        symbols = [line.split()[0] for line in coords_forces_raw]
        positions = [list(map(float, line.split()[1:4])) for line in coords_forces_raw]
        return Atoms(symbols=symbols, positions=positions)
    return None

def _get_isotropic_from_gout(file_content: str) -> np.ndarray:
    """Extracts isotropic chemical shielding values from the content of a Gaussian output file."""
    hff_reg = re.compile(r"^\s*(\d+)\s+(\w+)\s+Isotropic\s+=\s+([\d.-]+)", re.MULTILINE)
    hff_matches = hff_reg.findall(file_content)
    return np.array([float(match[2]) for match in hff_matches])

# 2. Main Utility Functions
def create_descriptor_dataframe(
    gout_folder_path: str,
    central_atoms: List[str],
    species: List[str] = ["Li", "O", "C", "H", "N", "S", "F"],
    periodic: bool = False,
    k2_grid_min: float = 0,
    k2_grid_max: float = 6,
    k2_grid_n: int = 20,
    k2_grid_sigma: float = 0.1,
    k2_weighting_scale: float = 0.5,
    k2_weighting_threshold: float = 1e-3,
    k3_grid_min: float = 0,
    k3_grid_max: float = 180,
    k3_grid_n: int = 20,
    k3_grid_sigma: float = 0.1,
    k3_weighting_scale: float = 0.5,
    k3_weighting_threshold: float = 1e-3,
    normalization: str = "l2"
) -> pd.DataFrame:
    """
    Reads Gaussian output files from a directory, calculates LMBTR descriptors and
    extracts chemical shielding values for specified central atoms, then returns
    a combined pandas DataFrame.

    Args:
        gout_folder_path (str): Path to the directory containing Gaussian output files.
        central_atoms (List[str]): List of central atom symbols (e.g., ['Li']).
        species (List[str]): List of all species present in the systems.
        periodic (bool): Whether the system is periodic.
        k2_grid_min (float): Minimum for the k=2 grid (distance).
        k2_grid_max (float): Maximum for the k=2 grid (distance).
        k2_grid_n (int): Number of points for the k=2 grid.
        k2_grid_sigma (float): Smearing for the k=2 grid.
        k2_weighting_scale (float): Scale for the k=2 exponential weighting.
        k2_weighting_threshold (float): Cutoff threshold for k=2 weighting.
        k3_grid_min (float): Minimum for the k=3 grid (angle).
        k3_grid_max (float): Maximum for the k=3 grid (angle).
        k3_grid_n (int): Number of points for the k=3 grid.
        k3_grid_sigma (float): Smearing for the k=3 grid.
        k3_weighting_scale (float): Scale for the k=3 exponential weighting.
        k3_weighting_threshold (float): Cutoff threshold for k=3 weighting.
        normalization (str): Normalization type for LMBTR.

    Returns:
        pd.DataFrame: A DataFrame with 'X' (LMBTR descriptors) and 'y' (isotropic values) columns.
    """
    lmbtrk2 = LMBTR(
        species=species,
        geometry={"function": "distance"},
        grid={"min": k2_grid_min, "max": k2_grid_max, "n": k2_grid_n, "sigma": k2_grid_sigma},
        weighting={"function": "exp", "scale": k2_weighting_scale, "threshold": k2_weighting_threshold},
        periodic=periodic,
        normalization=normalization,
    )
    lmbtrk3 = LMBTR(
        species=species,
        geometry={"function": "angle"},
        grid={"min": k3_grid_min, "max": k3_grid_max, "n": k3_grid_n, "sigma": k3_grid_sigma},
        weighting={"function": "exp", "scale": k3_weighting_scale, "threshold": k3_weighting_threshold},
        periodic=periodic,
        normalization=normalization,
    )

    file_list = sorted([os.path.join(gout_folder_path, f) for f in os.listdir(gout_folder_path) if os.path.isfile(os.path.join(gout_folder_path, f))])
    
    all_lmbtrs = []
    all_isotropys = []

    for file_path in tqdm(file_list, desc="Processing Gaussian Output Files"):
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            atoms = _get_xyz_from_gout(content)
            if atoms is None:
                print(f"Warning: Could not parse atoms from {file_path}. Skipping.")
                continue

            isotropys_all = _get_isotropic_from_gout(content)
            
            central_atom_indices = [i for i, symbol in enumerate(atoms.get_chemical_symbols()) if symbol in central_atoms]
            
            if not central_atom_indices:
                continue

            lmbtr_descriptor_k2 = lmbtrk2.create(atoms, centers=central_atom_indices, n_jobs=-1)
            lmbtr_descriptor_k3 = lmbtrk3.create(atoms, centers=central_atom_indices, n_jobs=-1)
            
            for i in range(len(central_atom_indices)):
                descriptor = np.concatenate((lmbtr_descriptor_k2[i], lmbtr_descriptor_k3[i]), axis=0)
                all_lmbtrs.append(descriptor)
            
            cs_symbols = np.array(atoms.get_chemical_symbols())
            filtered_isotropys = np.concatenate([isotropys_all[np.where(cs_symbols == e)[0]] for e in central_atoms])
            all_isotropys.extend(filtered_isotropys.tolist())

        except Exception as e:
            print(f"Error processing file {file_path}: {e}. Skipping.")

    return pd.DataFrame(data={'X': all_lmbtrs, 'y': all_isotropys})


def extract_and_save_csv(
    df: pd.DataFrame,
    output_path: str,
    save_to_file: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts feature (X) and target (y) arrays from a DataFrame and optionally
    saves them to a CSV file.

    Args:
        df (pd.DataFrame): Input DataFrame with 'X' and 'y' columns.
        output_path (str): Path to save the CSV file (e.g., './data/cut-data.csv').
        save_to_file (bool): If True, saves the DataFrame to a CSV. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the X and y numpy arrays.
    """
    X = np.array(df['X'].to_list())
    y = df['y'].values

    if save_to_file:
        save_df = pd.DataFrame({
            'index': list(range(len(X))),
            'X': X.tolist(),
            'y': y,
        })
        save_df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
        
    return X, y


def load_features_from_csv(
    csv_path: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads feature (X) and target (y) arrays from a specified CSV file.

    Args:
        csv_path (str): The path to the input CSV file (e.g., './data/cut-data.csv').

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the X and y numpy arrays.
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    df = pd.read_csv(csv_path)
    logging.info(f"Loaded data from {csv_path}")

    X = df['X'].apply(lambda x: list(map(float, x.strip('[]').split(',')))).tolist()
    X = np.array(X)
    logging.info(f"Processed X column, shape: {X.shape}")

    y = df['y'].values
    logging.info(f"Processed y column, shape: {y.shape}")
    
    return X, y


def train_evaluate_and_save_model(
    X: np.ndarray,
    y: np.ndarray,
    model_save_path: str,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    num_epochs: int = 1000,
    save_model: bool = False
) -> Tuple[Any, np.ndarray, np.ndarray, float]:
    """
    Trains and evaluates a neural network, and optionally saves the trained model.

    Args:
        X (np.ndarray): Feature data.
        y (np.ndarray): Target data (normalized).
        model_save_path (str): Path to save the model state (e.g., './models/lmbtr_model.pth').
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the training set to use for validation.
        random_state (int): Seed for reproducibility.
        batch_size (int): Number of samples per gradient update.
        learning_rate (float): Learning rate for the Adam optimizer.
        num_epochs (int): Total number of training epochs.
        save_model (bool): If True, saves the model's state dictionary. Defaults to False.

    Returns:
        Tuple[Any, np.ndarray, np.ndarray, float]: A tuple containing the trained model,
        the true test labels (y_test), the predicted test labels (y_pred), and the final test loss.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)

    X_train_t, X_val_t, X_test_t = torch.Tensor(X_train), torch.Tensor(X_val), torch.Tensor(X_test)
    y_train_t, y_val_t, y_test_t = torch.Tensor(y_train), torch.Tensor(y_val), torch.Tensor(y_test)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size)
    
    model = NeuralNetwork(input_dim=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
        
        if epoch % 100 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets.unsqueeze(1))
                    val_loss += loss.item() * inputs.size(0)
            val_loss /= len(val_loader.dataset)
            print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}")
            
    # Final evaluation on test set
    model.eval()
    y_pred_list = []
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            test_loss += loss.item() * inputs.size(0)
            y_pred_list.extend(outputs.cpu().numpy())
    test_loss /= len(test_loader.dataset)
    print(f"Final Test Loss: {test_loss:.4f}")

    if save_model:
        torch.save(model.state_dict(), model_save_path)
        print(f"Saved PyTorch Model State to {model_save_path}")

    return model, y_test, np.array(y_pred_list), test_loss


def plot_predictions(
    y_true_normalized: np.ndarray,
    y_pred_normalized: np.ndarray,
    mean_value: float,
    std_value: float,
    figure_save_path: str,
    enable_plotting: bool = False,
    save_figure: bool = False,
    xlim: Tuple[float, float] = (90, 95),
    ylim: Tuple[float, float] = (90, 95),
    figure_dpi: int = 300
) -> Dict[str, float]:
    """
    Calculates regression metrics, and optionally plots and saves a regression plot.

    Args:
        y_true_normalized (np.ndarray): True normalized target values.
        y_pred_normalized (np.ndarray): Predicted normalized target values.
        mean_value (float): Mean used for normalization.
        std_value (float): Standard deviation used for normalization.
        figure_save_path (str): Path to save the figure (e.g., './plots/pearson.jpg').
        enable_plotting (bool): If True, displays the plot. Defaults to False.
        save_figure (bool): If True, saves the plot to a file. Defaults to False.
        xlim (Tuple[float, float]): X-axis limits for the plot.
        ylim (Tuple[float, float]): Y-axis limits for the plot.
        figure_dpi (int): DPI for the saved figure.

    Returns:
        Dict[str, float]: A dictionary containing MAE, RMSE, and R2 Score.
    """
    # Reverse normalization
    y_true_rev = y_true_normalized * std_value + mean_value
    y_pred_rev = y_pred_normalized * std_value + mean_value
    
    # Calculate metrics
    mae = mean_absolute_error(y_true_rev, y_pred_rev)
    rmse = np.sqrt(mean_squared_error(y_true_rev, y_pred_rev))
    r2 = r2_score(y_true_rev, y_pred_rev)
    
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    if enable_plotting or save_figure:
        plt.figure(figsize=(4, 4), dpi=figure_dpi)
        plt.scatter(y_true_rev, y_pred_rev, s=5, c='#f15c5c', marker='o')
        plt.xlabel(r'$^{7}\rm{Li}\ \sigma_{DFT}$ (ppm)', fontsize=18)
        plt.ylabel(r'$^{7}\rm{Li}\ \sigma_{NN}$ (ppm)', fontsize=18)
        plt.plot([min(xlim), max(ylim)], [min(xlim), max(ylim)], c='k', ls='--')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.tight_layout()

        if save_figure:
            plt.savefig(figure_save_path, dpi=figure_dpi, bbox_inches='tight')
            print(f"Figure saved to {figure_save_path}")
        if enable_plotting:
            plt.show()

    return {'mae': mae, 'rmse': rmse, 'r2': r2}

def predict_with_saved_model(
    model_path: str,
    input_dim: int,
    pkl_file_path: str,
    mean_value: float,
    std_value: float
) -> List[Dict[str, Any]]:
    """
    Loads a trained model and a PKL file with descriptors, then runs prediction.

    Args:
        model_path (str): Path to the saved .pth model file.
        input_dim (int): The input dimension of the model.
        pkl_file_path (str): Path to the .pkl file containing descriptors.
        mean_value (float): Mean used for normalization during training.
        std_value (float): Standard deviation used for normalization.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing the
        original descriptor and the reverse-normalized prediction.
    """
    model = NeuralNetwork(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    with open(pkl_file_path, 'rb') as f:
        file_index_des = pickle.load(f)
    
    descriptors = file_index_des['descriptor']
    tensor_descriptors = torch.Tensor(descriptors)
    
    with torch.no_grad():
        pred = model(tensor_descriptors)
        
    de_pred = pred * std_value + mean_value
    
    results = []
    for i in range(len(descriptors)):
        result = {
            'descriptor': descriptors[i],
            'de_pred': de_pred[i].detach().numpy()
        }
        results.append(result)
        
    return results