import copy
import json
import time
from pathlib import Path


import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from predictors.predictor import Predictor

class CustomDS(Dataset):
    """
    Simple custom torch dataset.
    :param X: data
    :param y: labels
    """
    def __init__(self, X: torch.FloatTensor, y: torch.LongTensor):
        super().__init__()   
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], self.y[idx]


class ELUCNeuralNet(torch.nn.Module):
    """
    Custom torch neural network module.
    :param in_size: number of input features
    :param hidden_sizes: list of hidden layer sizes
    :param linear_skip: whether to concatenate input to hidden layer output
    :param dropout: dropout probability
    """
    class EncBlock(torch.nn.Module):
        """
        Encoding block for neural network.
        Simple feed forward layer with ReLU activation and optional dropout.
        """
        def __init__(self, in_size: int, out_size: int, dropout: float):
            super().__init__()
            self.model = torch.nn.Sequential(
                torch.nn.Linear(in_size, out_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=dropout)
            )
        def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
            """
            Passes input through the block.
            """
            return self.model(X)

    def __init__(self, in_size: int, hidden_sizes: list[str], linear_skip: bool, dropout: float):
        super().__init__()
        self.linear_skip = linear_skip
        hidden_sizes = [in_size] + hidden_sizes
        enc_blocks = [self.EncBlock(hidden_sizes[i], hidden_sizes[i+1], dropout) for i in range(len(hidden_sizes) - 1)]
        self.enc = torch.nn.Sequential(*enc_blocks)
        out_size = hidden_sizes[-1] + in_size if linear_skip else hidden_sizes[-1]
        self.linear = torch.nn.Linear(out_size, 1)

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        """
        Performs a forward pass of the neural net.
        :param X: input data
        :return: output of the neural net
        """
        hid = self.enc(X)
        if self.linear_skip:
            hid = torch.concatenate([hid, X], dim=1)
        out = self.linear(hid)
        return out


class NeuralNetPredictor(Predictor):
    """
    Simple feed-forward neural network predictor implemented in PyTorch.
    Has the option to use wide and deep, concatenating the input to the output of the hidden layers
    in order to take advantage of the linear relationship in the data.
    """
    def __init__(self, features=None, hidden_sizes=[4096], linear_skip=True, dropout=0, device="mps",
            epochs=3, batch_size=2048, optim_params={}, train_pct=1, step_lr_params={"step_size": 1, "gamma": 0.1}): 
        # Model setup params
        self.model = None
        self.features = features
        self.hidden_sizes = hidden_sizes
        self.linear_skip = linear_skip
        self.dropout = dropout
        self.device = device

        # Training params
        self.scaler = StandardScaler()
        self.epochs = epochs
        self.batch_size = batch_size
        self.optim_params = optim_params
        self.train_pct = train_pct
        self.step_lr_params = step_lr_params
        

    def load(self, path: str):
        """
        Loads a model from a given folder containing a config.json, model.pt, and scaler.joblib.
        :param path: path to folder containing model files.
        """
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Path {path} does not exist.")
        
        # Initialize model with config
        with open(load_path / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        self.__init__(**config)

        self.model = ELUCNeuralNet(len(self.features), self.hidden_sizes, self.linear_skip, self.dropout)
        self.model.load_state_dict(torch.load(load_path / "model.pt"))
        self.model.to(self.device)
        self.model.eval()
        self.scaler = joblib.load(load_path / "scaler.joblib")


    def save(self, path: str):
        """
        Saves model, config, and scaler into format for loading.
        Generates path to folder if it does not exist.
        :param path: path to folder to save model files.
        """
        if self.model is None:
            raise ValueError("Model not fitted yet.")
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        config = {
            "features": self.features,
            "hidden_sizes": self.hidden_sizes,
            "linear_skip": self.linear_skip,
            "dropout": self.dropout,
            "device": self.device,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "optim_params": self.optim_params,
            "train_pct": self.train_pct,
            "step_lr_params": self.step_lr_params
        }
        with open(save_path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)
        torch.save(self.model.state_dict(), save_path / "model.pt")
        joblib.dump(self.scaler, save_path / "scaler.joblib")


    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val=None, y_val=None, X_test=None, y_test=None, log_path=None, verbose=False) -> dict:
        """
        Fits neural network to given data using predefined parameters and hyperparameters.
        If no features were specified we use all the columns in X_train.
        We scale based on the training data and apply it to validation and test data.
        AdamW optimizer is used with L1 loss.
        :param X_train: training data, may be unscaled and have excess features.
        :param y_train: training labels.
        :param X_val: validation data, may be unscaled and have excess features.
        :param y_val: validation labels.
        :param X_test: test data, may be unscaled and have excess features.
        :param y_test: test labels.
        :param log_path: path to log training data to tensorboard.
        :param verbose: whether to print progress bars.
        :return: dictionary of results from training containing time taken, best epoch, best loss, and test loss if applicable.
        """
        if not self.features:
            self.features = X_train.columns.tolist()
        self.model = ELUCNeuralNet(len(self.features), self.hidden_sizes, self.linear_skip, self.dropout)
        self.model.to(self.device)
        self.model.train()

        s = time.time()

        # Set up train set
        X_train = self.scaler.fit_transform(X_train[self.features])
        y_train = y_train.values
        train_ds = CustomDS(X_train, y_train)
        sampler = torch.utils.data.RandomSampler(train_ds, num_samples=int(len(train_ds) * self.train_pct))
        train_dl = DataLoader(train_ds, self.batch_size, sampler=sampler)

        if X_val is not None and y_val is not None:
            X_val = self.scaler.transform(X_val[self.features])
            y_val = y_val.values
            val_ds = CustomDS(X_val, y_val)
            val_dl = DataLoader(val_ds, self.batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(self.model.parameters(), **self.optim_params)
        loss_fn = torch.nn.L1Loss()
        if self.step_lr_params:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **self.step_lr_params)

        if log_path:
            writer = SummaryWriter(log_path)

        # Keeping track of best performance for validation
        result_dict = {}
        best_model = None
        best_loss = np.inf
        e = 0

        step = 0
        for epoch in range(self.epochs):
            self.model.train()
            # Standard training loop
            train_iter = tqdm(train_dl) if verbose else train_dl
            for X, y in train_iter:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.model(X)
                loss = loss_fn(out.squeeze(), y.squeeze())
                if log_path:
                    writer.add_scalar("loss", loss.item(), step)
                step += 1
                loss.backward()
                optimizer.step()
            
            # LR Decay
            if self.step_lr_params:
                scheduler.step()

            # Evaluate epoch
            if X_val is not None and y_val is not None:
                total = 0
                self.model.eval()
                with torch.no_grad():
                    for X, y in tqdm(val_dl):
                        X, y = X.to(self.device), y.to(self.device)
                        out = self.model(X)
                        loss = loss_fn(out.squeeze(), y.squeeze())
                        total += loss.item() * y.shape[0]
                
                if log_path:
                    writer.add_scalar("val_loss", total / len(val_ds), step)
                
                if total < best_loss:
                    best_model = copy.deepcopy(self.model.state_dict())
                    best_loss = total
                    e = time.time()
                    result_dict["best_epoch"] = epoch
                    result_dict["best_loss"] = total / len(val_ds)
                    result_dict["time"] = e - s

                print(f"epoch {epoch} mae {total / len(val_ds)}")
        
        if best_model:
            self.model.load_state_dict(best_model)
        else:
            e = time.time()
            result_dict["time"] = e - s

        # If we provide a test dataset
        if X_test is not None and y_test is not None:
            y_pred = self.predict(X_test)
            y_true = y_test.values
            mae = np.mean(np.abs(y_pred - y_true))
            result_dict["test_loss"] = mae

        return result_dict


    def predict(self, X_test: pd.DataFrame) -> np.array:
        """
        Generates prediction from model for given test data.
        :param X_test: test data to predict on.
        :return: array of predictions.
        """
        X_test = self.scaler.transform(X_test[self.features])
        test_ds = CustomDS(X_test, np.zeros(len(X_test)))
        test_dl = DataLoader(test_ds, self.batch_size, shuffle=False)
        pred_list = []
        with torch.no_grad():
            self.model.eval()
            for X, _ in test_dl:
                X = X.to(self.device)
                pred_list.append(self.model(X))

        if len(pred_list) > 1:
            y_pred = torch.concatenate(pred_list, dim=0).cpu().numpy()
        else:
            y_pred = pred_list[0].cpu().numpy()
        return y_pred
