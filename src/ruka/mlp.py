import cv2
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import torch
import torch.nn as nn
import tqdm

from typing import Dict, Callable, Optional

import ruka.util.tensorboard as tb


DEVICE = "cuda:0"


# -------------------------------------------------------------------- Utils --


def cycle(iterable, limit=None):
    num = 0

    while True:
        for x in iterable:
            if limit is not None and num >= limit:
                return
            num += 1
            yield x


# ---------------------------------------------------------------------- MLP --
        
        
class MLP(nn.Module):
    def __init__(
            self, 
            inp_size: int, 
            hid_size: int, 
            num_layers: int,
            dropout: float,
            loss: nn.Module = nn.MSELoss(reduction='sum')
        ):
        
        super().__init__()
        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(inp_size, hid_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            inp_size = hid_size
        layers.append(nn.Linear(inp_size, 1))
        self.layers = nn.Sequential(*layers)
        self.loss = loss
    
    def forward(self, X, y_ref=None):
        """
        X:     [..., inp_size]
        y_ref: [...]
        --------------------
        return: 
            y_pred: [...]      if y_ref is None
            loss:   []         if y_ref is not None
        """
        # Compute prediction.
        y_pred = self.layers(X)  # [..., 1]
        y_pred = y_pred[..., 0]  # [...]
        if y_ref is None:
            return y_pred
        
        # Compute loss.
        return self.loss(y_pred, y_ref)


class MCLinear(nn.Module):
    def __init__(self, num_features: int, block_size: int):
        # Check args.
        assert num_features % block_size == 0
        
        # Init parent.
        super().__init__()

        # Init weights.
        self.weight = nn.Parameter(torch.empty((num_features, num_features)))
        self.bias = nn.Parameter(torch.empty(num_features))
        self.reset_parameters()

        # Init masks.
        self.mask_diag = torch.block_diag(*[
            torch.ones(block_size, block_size)
            for _ in range(num_features // block_size)
        ])
        self.mask_triu = torch.triu(torch.ones(num_features, num_features))
        self.mask_triu *= (1 - self.mask_diag)
        self.mask_diag = self.mask_diag.to(DEVICE).detach()
        self.mask_triu = self.mask_triu.to(DEVICE).detach()
        
        
    def reset_parameters(self):
        # Init weight.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # Init bias.
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, input):
        """
        input:  [..., num_features]
        ---------------------------
        return: [..., num_features]
        """
        A = input.detach() @ (self.weight * self.mask_triu)
        B = input @ (self.weight * self.mask_diag)
        C = self.bias
        return A + B + C
        

class MCMLP(nn.Module):
    LEVEL = -1
    
    def __init__(
            self, 
            inp_size: int, 
            hid_size: int,
            num_layers: int,
            block_size: int,
            dropout: float,
            loss: nn.Module = nn.MSELoss(reduction='sum')
        ):

        super().__init__()
        layers = [nn.Linear(inp_size, hid_size)]
        for i in range(1, num_layers):
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(MCLinear(hid_size, block_size))
        self.num_blocks = hid_size // block_size
        self.block_size = block_size
        self.layers = nn.Sequential(*layers)
        self.loss = loss

    
    def forward(self, X, y_ref=None, level: int = None):
        """
        X:     [..., inp_size]
        y_ref: [...]
        --------------------
        return: 
            y_pred: [...]      if y_ref is None
            loss:   []         if y_ref is not None
        """
        # Parse arguments.
        if level is None:
            level = self.LEVEL
            
        # Compute prediction.
        # - [..., hid_size]
        y_pred = self.layers(X)  
        # - [..., block_size]
        y_pred = y_pred[..., (self.block_size - 1)::self.block_size]  
        
        if y_ref is None:
            # - [...]
            return y_pred[..., level]
        
        # Compute loss.
        L = self.loss(y_pred, y_ref[..., None] + 0 * y_pred) / self.num_blocks
        return L


class FourierFeatures(nn.Module):
    def __init__(self, inp_size: int, out_size: int, sigma: float):
        super().__init__()
        assert out_size % 2 == 0
        self.weights = torch.randn(inp_size, out_size // 2) * sigma
        self.weights = self.weights.to(DEVICE)
    
    def forward(self, input):
        """
        input:  [..., inp_size]
        -----------------------
        return: [..., out_size]
        """
        proj = input @ self.weights
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


# ----------------------------------------------------------------- Training --


def train(
        model: nn.Module, 
        train_X: np.array, 
        train_y: np.array,
        num_train_examples: int,
        batch_size: int,
        lr: float, 
    ):
    """
    train_X: [train_n, inp_size], np.array
    train_y: [train_n],           np.array
    """
    # Create optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create train dataset.
    train_X = torch.Tensor(train_X).to(DEVICE)
    train_y = torch.Tensor(train_y).to(DEVICE)
    dataset = torch.utils.data.TensorDataset(train_X, train_y)

    # Create train dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, 
        shuffle=True
    )
    assert num_train_examples % batch_size == 0
    num_train_batches = num_train_examples // batch_size
    train = cycle(dataloader, limit=num_train_batches)

    # Train.
    model.train()
    pbar = tqdm.tqdm(train, total=num_train_batches, file=sys.stdout)
    for batch_no, (X, y) in enumerate(pbar):
        tb.step(batch_no)

        # - Compute loss.
        loss = model(X, y_ref=y) / batch_size
        
        # - Update.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # - Report.
        loss = loss.item()
        tb.scalar('train_loss', loss)
        if batch_no % 100 == 0 or batch_no + 1 == num_train_batches:
            pbar.set_description(f'Train loss: {loss:.3f}')

    # Back to eval mode.
    model.eval()
    tb.flush(wait=True)


# ---------------------------------------------------------------- Functions --


class Function:
    def sample(self, n: int) -> np.array:
        """
        return: (X, y)
            X: [n, inp_size]
            y: [n]
        """
        raise NotImplementedError()

    def visualize(
            self, 
            models: Dict[str, nn.Module], 
            train_X: Optional[np.array] = None, 
            train_y: Optional[np.array] = None, 
            **kwargs
        ):
        """
        train_X: [..., inp_size], np.array
        train_X: [...],           np.array
        model(X, **kwargs)
            X:      [..., inp_size], torch.Tensor
            return: [...],           torch.Tensor
        """
        raise NotImplementedError()

    def get_inp_size(self) -> int:
        raise NotImplementedError()


class Fn1DUniform(Function):
    def __init__(
            self, 
            start: float, 
            end: float, 
            fn: Callable, 
            noise_sigma: float = 0
        ):
        """
        fn(X)
            X:      [...], np.array
            return: [...], np.array
        """
        self.start = start
        self.end = end
        self.fn = fn
        self.noise_sigma = noise_sigma

    def sample(self, n: int) -> np.array:
        """
        return: (X, y)
            X: [n, 1], np.array
            y: [n],    np.array
        """
        X = np.random.rand(n, 1)
        y = self.fn(X) + np.random.normal(size=(n, 1)) * self.noise_sigma
        return X, y[:, 0]

    def visualize(
            self, 
            models: Dict[str, nn.Module], 
            train_X: Optional[np.array] = None, 
            train_y: Optional[np.array] = None, 
            **kwargs
        ):
        """
        train_X: [..., 1], np.array
        train_y: [...],    np.array

        model(X)
            X:      [..., 1], torch.Tensor
            return: [...],    torch.Tensor
        """
        # Reference.
        X = np.arange(self.start, self.end, 0.001)  # [n], np.array
        y_ref = self.fn(X)                          # [n], np.array
        
        # Prediction.
        model_predictions = {}
        for key, model in models.items():
            # - [n], torch.Tensor
            y_pred = model(torch.Tensor(X[:, None]).to(DEVICE))  
            # - [n], np.array
            y_pred = y_pred.cpu().detach().numpy()
            model_predictions[key] = y_pred 

        # Plot everything.
        plt.figure(figsize=(11, 8))
        if train_X is not None and train_y is not None:
            plt.plot(train_X[:, 0], train_y, 'rx')
        plt.plot(X, y_ref, label='ref')
        for key, y_pred in model_predictions.items():
            plt.plot(X, y_pred, label=key)
        plt.legend()
        plt.show()

    def get_inp_size(self) -> int:
        return 1


class Image2DUniform:
    def __init__(self, image_path: str):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.image = image / 256
        #image = np.zeros(shape=(512, 512))
        #image[128:128+256, 128:128+256] = 1
        #self.image = image
        self.w, self.h = image.shape

    def sample(self, n: int) -> np.array:
        """
        return: (X, y)
            X: [n, 2], np.array
            y: [n],    np.array
        """
        X = np.random.rand(n, 2)
        Xs = X * np.array([self.w, self.h])
        y = self.image[Xs[:, 0].astype(np.int32), Xs[:, 1].astype(np.int32)]
        return X, y

    def visualize(
            self, 
            models: Dict[str, nn.Module], 
            train_X: Optional[np.array] = None, 
            train_y: Optional[np.array] = None, 
            **kwargs
        ):
        """
        train_X: [..., 2], np.array
        train_y: [...],    np.array

        model(X)
            X:      [..., 2], torch.Tensor
            return: [...],    torch.Tensor
        """
        # Plan.
        n_images = len(models) + 1
        cols = 2
        rows = math.ceil(n_images / 2)
        fig = plt.figure(figsize=(10, 5 * rows))

        # Reference.
        ax = fig.add_subplot(rows, cols, 1)
        ax.imshow(self.image, cmap='gray')
        if train_X is not None:
            ax.plot(train_X[..., 0], train_X[..., 1], 'rx')
        ax.axis('off')
        ax.title.set_text('ref')
        
        # Coordinates for prediction.
        # - cx, cy: [w, h], np.array, int32
        cy, cx = np.meshgrid(np.arange(self.w), np.arange(self.h))
        # - [w, h, 2], np.array, int32
        X = np.concatenate([cx[..., None], cy[..., None]], axis=-1)
        X = X / np.array([self.w, self.h])

        # Prediction.
        model_predictions = {}
        for key, model in models.items():
            # - [w, h], torch.Tensor
            y_pred = model(torch.Tensor(X).to(DEVICE))  
            # - [w, h], np.array
            y_pred = y_pred.cpu().detach().numpy()
            model_predictions[key] = y_pred 

        # Plot everything.
        for i, (key, y_pred) in enumerate(model_predictions.items()):
            ax = fig.add_subplot(rows, cols, i + 2)
            ax.imshow(y_pred, cmap='gray')
            if train_X is not None:
                ax.plot(train_X[..., 0], train_X[..., 1], 'rx')
            ax.axis('off')
            ax.title.set_text(key)

        plt.show()

    def get_inp_size(self) -> int:
        return 2


# refactor loss
# sin activ
# separate fourier embeddings properly
# residual

# -----------------------------------------------------------------------------


def plot_images(images):
    n_images = len(images)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure()
    
