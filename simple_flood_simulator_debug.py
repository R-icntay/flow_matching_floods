#!/usr/bin/env python
# coding: utf-8

# ## In this notebook, we create a simple month conditioned flood diffusion model

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

import rasterio
from rasterio.enums import Resampling
from natsort import natsorted
from tqdm import tqdm
import os
from pathlib import Path
import re
from datetime import datetime
from scipy.ndimage import distance_transform_edt

def imshow_normalized(tensor_img, mean = (0.5, ), std = (0.5, )):
    """
    Function that displays a normalized tensor image using matplotlib.
    
    """
    assert tensor_img.dim() == 3, "Input tensor must be 3-dimensional (C, H, W)"
    img = tensor_img.clone().cpu().numpy()
    assert len(mean) == img.shape[0] and len(std) == img.shape[0], "Mean and std must match number of channels"
    for c in range(img.shape[0]):
        img[c] = img[c] * std[c] + mean[c]  # Unnormalize

    # Transpose to H, W, C
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    plt.axis('off')


# In[2]:


class FloodDataset(Dataset):
    def __init__(self, root_dir, target_size = (1024, 1024), transform = None):
        self.root_dir = Path(root_dir)
        self.target_size = target_size
        self.transform = transform

        image_paths = natsorted(list(self.root_dir.rglob('*.tif')))
        self.tile_to_int_map = self.tile_to_int(image_paths)
        
        if not image_paths:
            raise ValueError(f"No .tif files found in {root_dir}")
        
        self.path_month = []

        for p in image_paths:
            try:
                month = self._infer_month_from_path(p)
                self.path_month.append((p, month))
            except ValueError as e:
                print(f"Skipping {p.name}: {e}")

        # self.months = [self._infer_month_from_path(p) for p in self.image_paths]
        
    def __len__(self):
        return len(self.path_month)
    
    def __getitem__(self, idx):
        # Obtain image path and corresponding month
        img_path, month = self.path_month[idx]
        # month = int(self._infer_month_from_path(img_path))

        # Resample image to target size
        arr = self._resample(img_path)
      
        # Arr is currently uint8 values, 0, 1, 2
        mask = np.where(arr == 2, 1, 0).astype(np.uint8)
        # mask = mask[None, :, :]  # 1, H, W
        mask = torch.from_numpy(mask).to(torch.float32).unsqueeze(0)  # 1, H, W
        arr = None

        if self.transform:
            mask = self.transform(mask)

        meta = {
            "path": str(img_path),
            "month": month
        }

        return mask, int(month - 1), meta

    
    def _resample(self, p: Path) -> np.ndarray:
        """
        Use rasterio to resample the first band to target size.
        """
        with rasterio.open(p) as src:
            arr = src.read(
                1,
                out_shape = self.target_size,
                resampling = Resampling.nearest # Since these are binary flood maps
            )

        return arr.astype(np.uint8)
    
    @staticmethod
    def tile_to_int(image_paths):
        tile_list = []
        pattern = re.compile(r'([A-Z]\d+[A-Z]\d+)')
        for p in image_paths:
            match = pattern.search(p.stem)
            if match:
                tile_list.append(match.group(1))
            else:
                tile_list.append(None)

        # Get unique tiles and map to integers
        unique_tiles = natsorted(set(filter(None, tile_list)))

        tile_to_int_map = {tile: idx for idx, tile in enumerate(unique_tiles)}
        return tile_to_int_map

    @staticmethod
    def _infer_month_from_path(path):
        filename = path.stem

        match = re.match(r'^(\d{4}-\d{2})', filename)
        if match:
            year_month = match.group(1)
            year_month = datetime.strptime(year_month, "%Y-%m")
            return year_month.month
        else:
            raise ValueError(f"Filename {filename} does not match expected pattern 'YYYY-MM...'.")
            # print(f"Pattern not found in filename: {filename}. Defaulting to None.")
            # return None


# In[3]:


class FloodDataset(Dataset):
    def __init__(self, root_dir, target_size = (256, 256), to_sdf = False, sdf_threshold = 10, transform = None):
        self.root_dir = Path(root_dir)
        self.target_size = target_size
        self.to_sdf = to_sdf
        self.sdf_threshold = sdf_threshold
        self.transform = transform

        image_paths = natsorted(list(self.root_dir.rglob('*.tif')))
        self.tile_to_int_map = self.tile_to_int(image_paths)
        
        if not image_paths:
            raise ValueError(f"No .tif files found in {root_dir}")
        
        self.path_month = []

        for p in image_paths:
            try:
                month = self._infer_month_from_path(p)
                self.path_month.append((p, month))
            except ValueError as e:
                print(f"Skipping {p.name}: {e}")

        # self.months = [self._infer_month_from_path(p) for p in self.image_paths]
        
    def __len__(self):
        return len(self.path_month)
    
    def __getitem__(self, idx):
        # Obtain image path and corresponding month
        img_path, month = self.path_month[idx]
        # month = int(self._infer_month_from_path(img_path))

        # Resample image to target size
        arr = self._resample(img_path)
      
        # Arr is currently uint8 values, 0, 1, 2
        mask = np.where(arr == 2, 1, 0).astype(np.uint8)
        # mask = mask[None, :, :]  # 1, H, W

        if self.to_sdf:
            mask = self.mask_to_sdf(mask, truncation_threshold = self.sdf_threshold)
        
        mask = torch.from_numpy(mask).to(torch.float32).unsqueeze(0)  # 1, H, W
        arr = None

        if self.transform:
            mask = self.transform(mask)

        meta = {
            "path": str(img_path),
            "month": month
        }

        return mask, int(month - 1), meta

    
    def _resample(self, p: Path) -> np.ndarray:
        """
        Use rasterio to resample the first band to target size.
        """
        with rasterio.open(p) as src:
            arr = src.read(
                1,
                out_shape = self.target_size,
                resampling = Resampling.nearest # Since these are binary flood maps
            )

        return arr.astype(np.uint8)
    
    @staticmethod
    def tile_to_int(image_paths):
        tile_list = []
        pattern = re.compile(r'([A-Z]\d+[A-Z]\d+)')
        for p in image_paths:
            match = pattern.search(p.stem)
            if match:
                tile_list.append(match.group(1))
            else:
                tile_list.append(None)

        # Get unique tiles and map to integers
        unique_tiles = natsorted(set(filter(None, tile_list)))

        tile_to_int_map = {tile: idx for idx, tile in enumerate(unique_tiles)}
        return tile_to_int_map

    @staticmethod
    def _infer_month_from_path(path):
        filename = path.stem

        match = re.match(r'^(\d{4}-\d{2})', filename)
        if match:
            year_month = match.group(1)
            year_month = datetime.strptime(year_month, "%Y-%m")
            return year_month.month
        else:
            raise ValueError(f"Filename {filename} does not match expected pattern 'YYYY-MM...'.")
            # print(f"Pattern not found in filename: {filename}. Defaulting to None.")
            # return None


    @staticmethod
    def mask_to_sdf(binary_mask, truncation_threshold):
        """
        Convert a binary mask to a signed distance function (SDF).
        
        Args:
            binary_mask (numpy.ndarray): Binary mask where 1 represents flood and 0 represents dry area.
            truncation_threshold (float): Maximum absolute distance value for truncation.

        Returns:
            numpy.ndarray: Signed distance function representation of the input mask.
        """
        # Ensure binary_mask is a numpy array
        binary_mask = np.asarray(binary_mask).astype(np.float32)

        # Compute distance from background to nearest flood pixel
        # These are positive distances for dry area
        dist_outside = distance_transform_edt(1 - binary_mask)

        # Compute distance from flood to nearest background pixel
        # (These will be the negative distances for the 'wet' area)
        dist_inside = distance_transform_edt(binary_mask)

        # Combine: sdf = dist_outside - dist_inside
        sdf = dist_outside - dist_inside

        # Truncate distances
        # This prevents the loss from being dominated by pixels far from the edge
        sdf_truncated = np.clip(sdf, -truncation_threshold, truncation_threshold)

        # Normalize to [-1, 1]
        sdf_normalized = sdf_truncated / truncation_threshold
        # print(sdf_normalized.min(), sdf_normalized.max())
        return sdf_normalized
    
    @staticmethod
    def sdf_to_mask(sdf):
        binary_mask = (sdf <= 0).astype(np.uint8)
        return binary_mask


# In[4]:


# Define data directory
data_dir = Path("flood_data/monthly_flood_maps")
BATCH_SIZE = 8 

# Define transforms
transform = v2.Compose([
    # Convert to float32 tensor: used if input is numpy array
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale = False),

    # Apply random horizontal flip
    v2.RandomHorizontalFlip(p = 0.5),
    # Apply normalization to center data around 0, this will scale values to [-1, 1]
    # v2.Normalize(mean = [0.5], std = [0.5])
])

    
# Create dataset
target_size = (256, 256)
flood_dataset = FloodDataset(data_dir, target_size = target_size, to_sdf = True, sdf_threshold = 10, transform = transform)

# Define data loader
flood_dataloader = DataLoader(
    dataset = flood_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers = 4,
)


# ## Creating Sampleable datasets

# In[5]:


from abc import ABC, abstractmethod
from typing import Optional, List, Type, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class Sampleable(ABC):
    """
    Distribution which can be sampled from
    """
    @abstractmethod
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            num_samples (int): number of samples to draw
        Returns:
            samples (torch.Tensor): samples drawn from the distribution [batch_size, ...]
            labels (Optional[torch.Tensor]): optional labels associated with samples [batch_size, label_dim]
        """
        pass


# Next we create a sampleable dataset for Gaussian, which is the initial distribution $P_{init}$ which we aim to transform into the flood distribution $P_{flood}$ using our flow matching model.

# In[6]:


class IsotropicGaussian(nn.Module, Sampleable):
    """
    Sampleable wrapper around torch.randn
    """
    def __init__(self, shape: List[int], std: float = 1.0):
        super().__init__()
        self.shape = shape
        self.std = std
        self.register_buffer("dummy", torch.zeros(1))

    def sample(self, num_samples):
        # Sample from standard normal and scale by std
        samples = torch.randn((num_samples, *self.shape)) * self.std
        return samples.to(self.dummy.device), None # [B, C, H, W], None


# Next we create create conditional probability paths where we will sample both the data and label.

# In[7]:


class ConditionalProbabilityPath(nn.Module, ABC):
    """
    Abstract base class for conditional probability paths
    p_0(.|z) = p_init
    p_1(.|z) = p_data 
    """
    def __init__(self, p_simple, p_data):
        super().__init__()
        self.p_simple = p_simple
        self.p_data = p_data

    def sample_marginal_path(self, t):
        """
        Sample from the marginal distribution at time t: p_t(x) = \int p_t(x|z)p(z) dz
        z ~ p_data(z), x ~ p_t(x|z)

        Args:
            t: [B, 1, 1, 1]
        Returns:
            x: transition state samples [B, C, H, W]
        """
        num_samples = t.shape[0]

        # Sample conditioning variable z ~ p_data(z)
        z, _ = self.sample_conditioning_variable(num_samples) # [B, C, H, W]

        # Sample from the conditional probability path x ~ p_t(x|z)
        x = self.sample_conditional_path(z, t) # [B, C, H, W]

        return x
    

    @abstractmethod
    def sample_conditioning_variable(self, num_samples):
        """
        Samples the conditioning variable z and guiding label y
        Args:
            num_samples (int): number of samples to draw
        Returns:
            -z [B, C, H, W]
            -y [B, label_dim]
        """
        pass

    @abstractmethod
    def sample_conditional_path(self, z, t):
        """
        Sample from the conditional distribution at time t: x ~ p_t(x|z)
        Args:
            z: conditioning variable [B, C, H, W]
            t: time [B, 1, 1, 1]
        Returns:
            x: transition state samples from p_t(.|z) [B, C, H, W]  
        """
        pass

    @abstractmethod
    def conditional_vector_field(self, x, z, t):
        """
        Compute the conditional vector field u_t(x|z) 
        Args:
            x: transition state [B, C, H, W]
            z: conditioning variable [B, C, H, W]
            t: time [B, 1, 1, 1]
        Returns:
            u: conditional vector field [B, C, H, W]
        """
        pass

    @abstractmethod
    def conditional_score(self, x, z, t):
        """
        Compute the conditional score \nabla_x log p_t(x|z)
        Args:
            x: transition state [B, C, H, W]
            z: conditioning variable [B, C, H, W]
            t: time [B, 1, 1, 1]
        Returns:
            s: conditional score [B, C, H, W]
        """
        pass


# In[8]:


## Creating noise schedulers
import torch
from torch.func import vmap, jacrev
from abc import ABC, abstractmethod

class Alpha(ABC):
    """
    Base class for alhpa_t function that scale the data
    """
    def __init_(self):
        super().__init__()
        # Check alpha_t(0) == 0
        assert torch.allclose(
            self(torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1)),
        )
        # Check alpha_t(1) == 1
        assert torch.allclose(
            self(torch.ones(1, 1, 1, 1), torch.ones(1, 1, 1, 1)),
        )

    @abstractmethod
    def __call__(self, t):
        """
        Evaluates alpha_t. Should satisfy: self(0.0) = 0.0, self(1.0) = 1.0.
         Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - alpha_t (num_samples, 1, 1, 1)
         
        """
        pass

    def dt(self, t):
        """
        Evaluates d/dt alpha_t

        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1, 1, 1)
        """
        t = t.unsqueeze(1)
        # Jacobian wrt a single variable is derivative
        # Define derivative of self.__call__(t_i) for single input
        dt = jacrev(self)
        # Vectorize over batch dimension
        dt = vmap(dt)(t)
        return dt.view(-1, 1, 1, 1)
    

class Beta(ABC):
    """
    Base class for beta_t function that scale the noise
    """
    def __init__(self):
        super().__init__()
        # Check beta_t(0) == 1
        assert torch.allclose(
            self(torch.zeros(1, 1, 1, 1)),
            torch.ones(1, 1, 1, 1)
        )
        # Check beta_t(1) == 0
        assert torch.allclose(
            self(torch.ones(1, 1, 1, 1)),
            torch.zeros(1, 1, 1, 1)
        )

    @abstractmethod
    def __call__(self, t):
        """
        Evaluates beta_t. Should satisfy: self(0.0) = 1.0, self(1.0) = 0.0.
         Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - beta_t (num_samples, 1, 1, 1)
         
        """
        pass

    def dt(self, t):
        """
        Evaluates d/dt beta_t

        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - d/dt beta_t (num_samples, 1, 1, 1)
        """
        t = t.unsqueeze(1)
        dt = vmap(jacrev(self))(t)
        return dt.view(-1, 1, 1, 1)
    

# Implement linear alpha and beta functions
class LinearAlpha(Alpha):
    """
    Implements alpha_t = t
    """
    def __call__(self, t):
        return t
    
    def dt(self, t):
        """"
        Evaluates d/dt alpha_t
        """
        return torch.ones_like(t)
    
class LinearBeta(Beta):
    """
    Implements beta_t = 1 - t
    """
    def __call__(self, t):
        return 1.0 - t
    
    def dt(self, t):
        """"
        Evaluates d/dt beta_t
        """
        return torch.ones_like(t) * -1.0


# In[9]:


## Instantiate Gaussian Conditional Probability Path
class GaussianConditionalProbabilityPath(ConditionalProbabilityPath):
    """
    Conditional probability path where p_t(.|z) = N(x; alpha_t * z, beta_t^2 * I)
    For linear interpolation (alpha_t = t, beta_t = 1-t), this is optimal transport.
    """
    def __init__(self, p_data, p_simple_shape, alpha, beta):
        p_simple = IsotropicGaussian(shape = p_simple_shape, std = 1.0)
        super().__init__(p_simple, p_data)
        self.alpha = alpha
        self.beta = beta

    def sample_conditioning_variable(self, num_samples):
        """
        Samples the conditioning variable z and guiding label y
        Args:
            num_samples (int): number of samples to draw
        Returns:
            -z [B, C, H, W]
            -y [B, label_dim]
        """
        return self.p_data.sample(num_samples)
    
    def sample_conditional_path(self, z, t):
        """
        Sample from the conditional probability path x ~p_t(.|z) = N(x; alpha_t * z, beta_t^2 I)

         x = alpha_t * z + beta_t * eps, eps ~ N(0, I)
        Args:
            z: conditioning variable [B, C, H, W]
            t: time [B, 1, 1, 1]
        Returns:
            x: transition state from p_t(.|z) [B, C, H, W]
        
        """
        noise = torch.randn_like(z)  # eps ~ N(0, I)
        return self.alpha(t) * z + self.beta(t) * noise
    

    def conditional_vector_field(self, x, z, t):
        """
        Computes the conditional vector field for linear optimal transport path.
        
        For alpha_t = t, beta_t = 1-t (linear interpolation):
        x_t = t*z + (1-t)*eps
        
        The conditional vector field is:
        u_t(x|z) = (z - x) / (1 - t)
        
        Args:
            x: transition state [B, C, H, W]
            z: conditioning variable [B, C, H, W]
            t: time [B, 1, 1, 1]
        Returns:
            u_t(x|z): conditional vector field [B, C, H, W]
        """
        # Use the simplified OT formula with numerical stability
        eps = 1e-5
        return (z - x) / (1.0 - t + eps)
    

    def conditional_score(self, x, z, t):
        """
        Computes the conditional score \nabla_x log p_t(x|z) = (alpha_t * z - x) / beta_t^2
        The score points towards the direction of higher data likelihood
        """
        alpha_t = self.alpha(t) # [B, 1, 1, 1]
        beta_t = self.beta(t)   # [B, 1, 1, 1]
        eps = 1e-5
        return (z * alpha_t - x) / (beta_t ** 2 + eps) # [B, C, H, W]




# In[10]:


## Update ODE, SDE and simulator classes
class ODE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt, t, **kwargs):
        """
        Computes the drift coefficient of the ODE ie vector field u_t(x|z)
        Args:
        xt: transition state at time t [B, C, H, W]
            t: time [B, 1, 1, 1]
        Returns:
            u_t(xt|z): drift coefficient [B, C, H, W]
        """
        pass

class SDE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt, t, **kwargs):
        """
        Computes the drift coefficient of the SDE u_t(xt|z)
        vector field + score
        Args:
            xt: transition state at time t [B, C, H, W]
            t: time [B, 1, 1, 1]
        Returns:
            u_t(xt|z): drift coefficient [B, C, H, W]
        """
        pass

    @abstractmethod
    def diffusion_coefficient(self, xt, t):
        """
        Computes the diffusion coefficient of the SDE: sigma * dWT
        """
        pass


class Simulator(ABC):
    @abstractmethod
    def step(self, xt, t, dt, **kwargs):
        """
        Performs a single simulation step
        x_t+h = x_t + dt * u_t(x_t|z)
        Args:
            xt: transition state at time t [B, C, H, W]
            t: time [B, 1, 1, 1]
            dt: time step (scalar)
        Returns:
            x_t+h: transition state at time t+h [B, C, H, W]
        """
        pass

    @torch.no_grad()
    def simulate(self, x, ts, **kwargs):
        """"
        Simulates the ODE/SDE from time t=0 to t=1
        Args:
            x0: initial state [B, C, H, W]
            ts: all time steps [B, nts, 1, 1, 1]

        Returns:
            xts: Final transition state at t=1 [B, C, H, W]
        """
        nts = ts.shape[1]
        for t_idx in tqdm(range(nts - 1)):
            t = ts[:, t_idx] # [B, 1, 1, 1]
            h = ts[:, t_idx + 1] - t # [B, 1, 1, 1]
            x = self.step(x, t, h, **kwargs) # [B, C, H, W]
        return x
    
    @torch.no_grad()
    def simulate_with_trajectory(self, x, ts, **kwargs):
        """
        Simulates the ODE/SDE from time t=0 to t=1 and records the full trajectory
        Args:
            x0: initial state [B, C, H, W]
            ts: all time steps [B, nts, 1, 1, 1]
        Returns:
            xts: all transition states [B, nts, C, H, W]
        """
        xs = [x.clone()]
        nts = ts.shape[1]
        for t_idx in tqdm(range(nts - 1)):
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - t
            x = self.step(x, t, h, **kwargs) # [B, C, H, W]
            xs.append(x.clone())

        return torch.stack(xs, dim = 1) # [B, nts, C, H, W]


# In[11]:


# Implement Euler and Euler-Maruyama simulators
class EulerSimulator(Simulator):
    def __init__(self, ode):
        self.ode = ode

    def step(self, xt, t, h, **kwargs):
        """
        Performs a single simulation step
        x_t+h = x_t + h * u_t(x_t|z)
        """
        x_t_plus_h = xt + h * self.ode.drift_coefficient(xt, t, **kwargs)
        return x_t_plus_h
    
class EulerMaruyamaSimulator(Simulator):
    def __init__(self, sde):
        self.sde = sde

    def step(self, xt, t, h, **kwargs):
        """
        Performs a single simulation step
        x_t+h = x_t + h * u_t(x_t|z) + diffusion_coef * (sqrt(h) * eps, eps ~ N(0, I))
        """
        noise = torch.randn_like(xt) * torch.sqrt(h) # Noise with a variance of h
        x_t_plus_h = xt + h * self.sde.drift_coefficient(xt, t, **kwargs) + self.sde.diffusion_coefficient(xt, t, **kwargs) * noise
        return x_t_plus_h
    
def record_every(num_timesteps, record_every):
    """
    Returns a list of indices to record
    Args:
        num_timesteps: total number of timesteps
        record_every: record every n timesteps
    Returns:
        indices: list of indices to record
    
    """
    if record_every == 1:
        return torch.arange(0, num_timesteps)
    
    return torch.cat(
        [
            torch.arange(0, num_timesteps, record_every),
            torch.tensor([num_timesteps - 1])
        ]
    )

# 1MB = 1024 ** 2 bytes 
MiB = 1024 ** 2

def model_size_b(model):
    """
    Returns model size in bytes. Based on https://discuss.pytorch.org/t/finding-model-size/130275/2
    """
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size

class Trainer(ABC):
    def __init__(self, model, grad_clip_norm = 1.0):
        super().__init__()
        self.model = model
        self.grad_clip_norm = grad_clip_norm
        self.losses = []

    @abstractmethod
    def get_train_loss(self, **kwargs):
        pass

    def get_optimizer(self, lr):
        return torch.optim.Adam(self.model.parameters(), lr = lr)


    def get_polynomial_decay_schedule_with_warmup(self, optimizer, num_warmup_epochs, num_training_epochs, lr_end = 1e-8, power = 1.0, last_epoch = -1):
        """
        Returns a learning rate scheduler with polynomial decay and warmup.
        """
        def lr_lambda(current_epoch):
            # Warmup phase: linearly increase learning rate
            if current_epoch < num_warmup_epochs:
                return float(current_epoch) / float(max(1, num_warmup_epochs))

            # Decay phase
            elif current_epoch > num_training_epochs:
                return lr_end
            
            else:
                # Calculate the decay factor
                total_decay_epochs = num_training_epochs - num_warmup_epochs
                last_epoch_in_decay = current_epoch - num_warmup_epochs

                decay_factor = (1 - (last_epoch_in_decay / total_decay_epochs)) ** power
                return decay_factor

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
                



    
    def train(self, num_epochs, device, lr = 1e-3, warmup_epochs = 100, **kwargs):
        # Report model size
        size_b = model_size_b(self.model)
        print(f"Model size: {size_b / MiB:.3f} MiB")

        # Move model to device and set to train mode
        self.model.to(device)
        self.model.train()
        optimizer = self.get_optimizer(lr)
        scheduler = self.get_polynomial_decay_schedule_with_warmup(optimizer, warmup_epochs, num_epochs)

        pbar = tqdm(enumerate(range(num_epochs)))
        for idx, epoch in pbar:

            # Zero out previous accumulated gradients
            optimizer.zero_grad()

            # Forward pass and compute loss
            loss = self.get_train_loss(**kwargs)
            self.losses.append(loss.item())

            # Gradient of loss wrt model parameters
            loss.backward()

            # Gradient clipping
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            # Update model parameters
            optimizer.step()

            # Update learning rate after each epoch
            scheduler.step()
            

            pbar.set_description(f"Epoch {idx}, loss: {loss.item():.3f}")

        # Final mode to eval
        self.model.eval()
        return self.losses



# In[12]:


## Instantiate Gaussian Conditional Probability Path

# Define the abstract base class for sampling conditional probability paths
class ConditionalProbabilityPath(nn.Module, ABC):
    """
    Abstract base class for conditional probability paths
    p_0(.|z) = p_init
    p_1(.|z) = p_data 
    """
    def __init__(self, p_simple, p_data):
        super().__init__()
        self.p_simple = p_simple
        self.p_data = p_data

    def sample_marginal_path(self, t):
        """
        Sample from the marginal distribution at time t: p_t(x) = \int p_t(x|z)p(z) dz
        z ~ p_data(z), x ~ p_t(x|z)

        Args:
            t: [B, 1, 1, 1]
        Returns:
            x: transition state samples [B, C, H, W]
        """
        num_samples = t.shape[0]

        # Sample conditioning variable z ~ p_data(z)
        z, _ = self.sample_conditioning_variable(num_samples) # [B, C, H, W]

        # Sample from the conditional probability path x ~ p_t(x|z)
        x = self.sample_conditional_path(z, t) # [B, C, H, W]

        return x
    

    @abstractmethod
    def sample_conditioning_variable(self, num_samples):
        """
        Samples the conditioning variable z and guiding label y
        Args:
            num_samples (int): number of samples to draw
        Returns:
            -z [B, C, H, W]
            -y [B, label_dim]
        """
        pass

    @abstractmethod
    def sample_conditional_path(self, z, t):
        """
        Sample from the conditional distribution at time t: x ~ p_t(x|z)
        Args:
            z: conditioning variable [B, C, H, W]
            t: time [B, 1, 1, 1]
        Returns:
            x: transition state samples from p_t(.|z) [B, C, H, W]  
        """
        pass

    @abstractmethod
    def conditional_vector_field(self, x, z, t):
        """
        Compute the conditional vector field u_t(x|z) 
        Args:
            x: transition state [B, C, H, W]
            z: conditioning variable [B, C, H, W]
            t: time [B, 1, 1, 1]
        Returns:
            u: conditional vector field [B, C, H, W]
        """
        pass

    @abstractmethod
    def conditional_score(self, x, z, t):
        """
        Compute the conditional score \nabla_x log p_t(x|z)
        Args:
            x: transition state [B, C, H, W]
            z: conditioning variable [B, C, H, W]
            t: time [B, 1, 1, 1]
        Returns:
            s: conditional score [B, C, H, W]
        """
        pass

class GaussianConditionalProbabilityPath(ConditionalProbabilityPath):
    """
    Conditional probability path where p_t(.|z) = N(x; alpha_t * z, beta_t^2 * I)
    For linear interpolation (alpha_t = t, beta_t = 1-t), this is optimal transport.
    """
    def __init__(self, p_data, p_simple_shape, alpha, beta):
        p_simple = IsotropicGaussian(shape = p_simple_shape, std = 1.0)
        super().__init__(p_simple, p_data)
        self.alpha = alpha
        self.beta = beta

    def sample_conditioning_variable(self, num_samples):
        """
        Samples the conditioning variable z and guiding label y
        Args:
            num_samples (int): number of samples to draw
        Returns:
            -z [B, C, H, W]
            -y [B, label_dim]
        """
        return self.p_data.sample(num_samples)
    
    def sample_conditional_path(self, z, t):
        """
        Sample from the conditional probability path x ~p_t(.|z) = N(x; alpha_t * z, beta_t^2 I)

         x = alpha_t * z + beta_t * eps, eps ~ N(0, I)
        Args:
            z: conditioning variable [B, C, H, W]
            t: time [B, 1, 1, 1]
        Returns:
            x: transition state from p_t(.|z) [B, C, H, W]
        
        """
        noise = torch.randn_like(z)  # eps ~ N(0, I)
        return self.alpha(t) * z + self.beta(t) * noise
    

    def conditional_vector_field(self, x, z, t):
        """
        Computes the conditional vector field for linear optimal transport path.
        
        For alpha_t = t, beta_t = 1-t (linear interpolation):
        x_t = t*z + (1-t)*eps
        
        The conditional vector field is simply:
        u_t(x|z) = z - eps = z - (x_t - t*z)/(1-t)
        
        But we can use the simpler form that doesn't have singularity:
        u_t(x|z) = z - x_0 where x_0 is the noise
        
        Since x_t = t*z + (1-t)*x_0, we have x_0 = (x_t - t*z)/(1-t)
        So u_t = z - (x_t - t*z)/(1-t) = (z - x_t)/(1-t)
        
        For numerical stability, we use the equivalent form:
        u_t(x|z) = (z - x) / (1 - t + 1e-8)
        
        Args:
            x: transition state [B, C, H, W]
            z: conditioning variable [B, C, H, W]
            t: time [B, 1, 1, 1]
        Returns:
            u_t(x|z): conditional vector field [B, C, H, W]
        """
        # Use the simplified OT formula with numerical stability
        # u_t(x|z) = (z - x) / (1 - t)
        eps = 1e-5
        return (z - x) / (1.0 - t + eps)
    

    def conditional_score(self, x, z, t):
        """
        Computes the conditional score \nabla_x log p_t(x|z) = (alpha_t * z - x) / beta_t^2
        The score points towards the direction of higher data likelihood
        """
        alpha_t = self.alpha(t) # [B, 1, 1, 1]
        beta_t = self.beta(t)   # [B, 1, 1, 1]
        eps = 1e-5
        return (z * alpha_t - x) / (beta_t ** 2 + eps) # [B, C, H, W]




# In[13]:


## Create sampleable wrapper for flood map dataset
class flood_map_sampler(nn.Module, Sampleable):
    """
    Sampleable wrapper for flood map dataset
    """
    def __init__(self):
        super().__init__()
        self.dataset = flood_dataset
        self.register_buffer("dummy", torch.zeros(1))

    def sample(self, num_samples):
        if num_samples > len(self.dataset):
            raise ValueError(f"Requested {num_samples} samples, but dataset only has {len(self.dataset)} samples.")
        
        # indices = [9]*num_samples #torch.randperm(len(self.dataset))[:num_samples] ** overfit 1 sample
        indices = torch.randperm(len(self.dataset))[:num_samples]
        samples = [self.dataset[i][0] for i in indices]
        
        month_labels = [self.dataset[i][1] for i in indices]
        location_labels = [re.search(r'([A-Z]\d+[A-Z]\d+)', self.dataset[i][2]["path"]).group(1) for i in indices]
        location_labels = [self.dataset.tile_to_int_map[loc] for loc in location_labels]


        samples = torch.stack(samples, dim = 0).to(self.dummy)  # B, C, H, W
        month_labels = torch.tensor(month_labels, device = self.dummy.device) # B
        location_labels = torch.tensor(location_labels, device = self.dummy.device) # B
        return samples, month_labels, location_labels

## Pinit sampler
class IsotropicGaussian(nn.Module, Sampleable):
    """
    Sampleable wrapper around torch.randn
    """
    def __init__(self, shape: List[int], std: float = 1.0):
        super().__init__()
        self.shape = shape
        self.std = std
        self.register_buffer("dummy", torch.zeros(1))

    def sample(self, num_samples):
        # Sample from standard normal and scale by std
        samples = torch.randn((num_samples, *self.shape)) * self.std
        return samples.to(self.dummy.device), None # [B, C, H, W], None


# In[27]:


import matplotlib.pyplot as plt
from torchvision.utils import make_grid
get_ipython().run_line_magic('matplotlib', 'inline')
num_rows = 1
num_cols = 1
num_timesteps = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Initialize data sampler
p_data = flood_map_sampler().to(device)

# Initialize Gaussian conditional probability path
path = GaussianConditionalProbabilityPath(
    p_data = p_data,
    p_simple_shape = (1, *target_size),
    alpha = LinearAlpha(),
    beta = LinearBeta()
).to(device)

# Draw sampled from p_data
num_samples = num_rows * num_cols
sampled_data, month_labels, location_labels = path.p_data.sample(num_samples)
z = sampled_data.view(-1, 1, *target_size)  # B, C, H, W
z.min(), z.max()


# Define time steps
ts = torch.linspace(0, 1, num_timesteps).to(device)

# Setup plots
fig, axes = plt.subplots(1, num_timesteps, figsize=(6 * num_cols * num_timesteps, 6 * num_rows))


for tidx, t in enumerate(ts):
    tt = t.view(1, 1, 1, 1).expand(num_samples, 1, 1, 1)  # B, 1, 1, 1

    # Sample from conditional probability path p_t(.|z)
    # x_t = alpha_t * z + beta_t * noise
    xt = path.sample_conditional_path(z, tt)  # B, C, H, W
    grid = make_grid(xt, nrow = num_rows, normalize = False, )
    axes[tidx].imshow(grid.permute(1, 2, 0).cpu(), cmap = 'viridis')
    axes[tidx].set_title(f"t={t.item():.2f}")
    axes[tidx].axis('off')

# plt.tight_layout()
# plt.savefig("flood_simulation1.png", dpi = 500)
# plt.show()


# In[28]:


plt.figure(figsize = (10, 10))
rows, cols = 4, 4
metadata_list = []
for i in range(1, cols * rows + 1):
    sample_idx = torch.randperm(len(flood_dataset))[0]
    # Get a sample
    sample = flood_dataset[sample_idx]
    img, label, meta = sample
    metadata_list.append(Path(meta['path']).stem.split('_flood_map')[0])
    plt.subplot(rows, cols, i)
    imshow_normalized(img)
    plt.title(f"{Path(meta['path']).stem.split('_flood_map')[0]}")

plt.tight_layout()
# plt.savefig("flood_samples_sdf.png", dpi = 1000)


# In[29]:


## Define conditional vector field as a CNN for now
class ConditionalVectorFieldCNN(nn.Module, ABC):
    """
    CNN parameterization of the learned marginal vector field u_t^theta(x|z)
    """
    @abstractmethod
    def forward(self, x, t, y):
        """
        Args:
            x: transition state [B, C, H, W]
            t: time [B, 1, 1, 1]
            y: guiding label [B, label_dim]
        Returns:
            u_t^theta(x|z): conditional vector field [B, C, H, W]
        """
        pass

class CFGVectorFieldODE(ODE):
    def __init__(self, net: ConditionalVectorFieldCNN, guidance_scale = 1.0):
        self.net = net
        self.guidance_scale = guidance_scale

    def drift_coefficient(self, x, t, y):
        """
        Predicts the vector field
        x: transition state [B, C, H, W]
        t: time [B, 1, 1, 1]
        y: guiding label [B, label_dim]
        Returns:
            u_t^theta(x|y) = wu_t^theta(x|y) + (1-w)u_t^theta(x|0)
        """
        # Guided model prediction
        guided_vector_field = self.net(x, t, y)  # [B, C, H, W]

        # Unguided model prediction
        unguided_y_month = torch.ones_like(y[0]) * 12 # Month label 12 for unguided
        unguided_y_location = torch.ones_like(y[1]) * len(p_data.dataset.tile_to_int_map)  # Max location label for unguided
        
        unguided_y = torch.stack([unguided_y_month, unguided_y_location], dim = 0) # [2, B]
        unguided_vector_field = self.net(x, t, unguided_y)  # [B, C, H, W]

        # Combine guided and unguided predictions
        combined_vector_field = (self.guidance_scale * guided_vector_field) + ((1 - self.guidance_scale) * unguided_vector_field)
        
        return combined_vector_field


# In[30]:


## Create CFG trainer
## Labels ae dropped with prop n

class CFGTrainer(Trainer):
    def __init__(self, path, model, eta, device, grad_clip_norm=1.0, **kwargs):
        assert eta >= 0 and eta <1
        super().__init__(model, grad_clip_norm, **kwargs)
        self.eta = eta
        self.path = path
        self.device = device

    def get_train_loss(self, batch_size):
        # Sample time from a uniform distribution
        # t ~ uniform(0, 1-eps) to avoid singularity at t=1
        t_max = 0.999  # Avoid t=1 where vector field becomes singular
        t = torch.rand(batch_size, 1, 1, 1, device = self.device) * t_max  # [B, 1, 1, 1]

        # Sample data point z and labels y from p_data 
        # z, y ~ p_data
        z, month_labels, location_labels = self.path.sample_conditioning_variable(batch_size) # z: [B, C, H, W], month_labels: [B], location_labels: [B]

        # Randomly drop labels with probability eta
        drop_mask = torch.rand(batch_size, device = month_labels.device) < self.eta  # [B]
        month_labels = month_labels.masked_fill(drop_mask, 12) # Where month label 12 indicates no guidance
        location_labels = location_labels.masked_fill(drop_mask, len(self.path.p_data.dataset.tile_to_int_map)) # Where location label len(...) indicates no guidance           
        y = torch.stack([month_labels, location_labels], dim = 0)  # [2, B]

        # Sample from the conditional probability path
        # x_t ~ p_t(.|z) x = alpha_t * z + beta_t * noise
        x_t = self.path.sample_conditional_path(z, t)  # [B, C, H, W]

        # Compute the conditional vector field
        # u_t(x|z) = (z - x) / (1 - t) for optimal transport path
        ut_ref = self.path.conditional_vector_field(x_t, z, t)  # [B, C, H, W]

        # Compute the model predicted vector field u_t^theta(x, y)
        ut_theta = self.model(x_t, t, y) # [B, C, H, W]

        # Compute MSE loss between reference and model predicted vector fields
        error = torch.square(ut_theta - ut_ref)  # [B, C, H, W]
        loss = torch.mean(error)  # scalar (average over all pixels and batch)
        return loss
        


# ## Unet vector field model
# Next we create a Unet model to be used as the vector field model for flow matching. The Unet will take in both the noisy flood data and the month label & location as input and output the vector field needed for flow matching.

# In[31]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResidualLayer(nn.Module):
    def __init__(self, channels, time_embed_dim, y_embed_dim):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.SiLU(),
            nn.GroupNorm(num_groups = 32,  num_channels = channels),
            nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size = 3, padding = 1)
        )
        self.block2 = nn.Sequential(
            nn.SiLU(),
            nn.GroupNorm(num_groups = 32,  num_channels = channels),
            nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size = 3, padding = 1)
        )

        # Convert time embedding to a shape that can be added to feature map
        # [B, time_embed_dim] -> [N, channels]
        self.time_adapter = nn.Sequential(
            nn.Linear(in_features = time_embed_dim, out_features = time_embed_dim),
            nn.SiLU(),
            nn.Linear(in_features = time_embed_dim, out_features = channels)
        )

        # Convert y embedding to a shape that can be added to feature map
        # [B, y_embed_dim] -> [B, channels]
        self.y_adapter = nn.Sequential(
            nn.Linear(in_features = y_embed_dim, out_features = y_embed_dim),
            nn.SiLU(),
            nn.Linear(in_features = y_embed_dim, out_features = channels)
        )

    def forward(self, x, t_embed, y_embed):
        """
        Args:
            x: feature map [B, C, H, W]
            t_embed: time embedding [B, time_embed_dim]
            y_embed: y embedding [B, y_embed_dim]
            
        """
        
        # Create residual
        res = x.clone() # [B, C, H, W]

        # Transform time embedding to match feature map shape
        t_embed = self.time_adapter(t_embed) # [B, time_embed_dim] -> [B, channels]
        t_embed = t_embed[:, :, None, None] # [B, C] -> [B, C, 1, 1]

        # Transform y embedding to match feature map shape
        y_embed = self.y_adapter(y_embed) # [B, y_embed_dim] -> [B, channels]
        y_embed = y_embed[:, :, None, None] # [B, C] -> [B, C, 1, 1]

        # Initial Conv + BN + SiLU
        x = self.block1(x)  # [B, C, H, W] -> [B, C, H, W]

        # Add time and y embeddings
        x = x + t_embed + y_embed  # [B, C, H, W] + [B, C, 1, 1] + [B, C, 1, 1] -> [B, C, H, W]

        # Second Conv + BN + SiLU
        x = self.block2(x)  # [B, C, H, W] -> [B, C, H, W]

        # Add residual
        x = x + res  # [B, C, H, W] + [B, C, H, W] -> [B, C, H, W]

        return x
    

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_residual_layers, t_embed_dim, y_embed_dim):
        super().__init__()
        # 1. The Processing Blocks
        self.res_blocks = nn.ModuleList([
            ResidualLayer(channels=in_channels, time_embed_dim=t_embed_dim, y_embed_dim=y_embed_dim) 
            for _ in range(num_residual_layers)
        ])
        
        # 2. The Downsampler (Separate)
        self.downsample_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, t_embed, y_embed):
        # Pass through residual blocks
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)
        
        # SAVE THIS X for the skip connection (High Resolution)
        skip_feature = x.clone()

        # DOWNSAMPLE for the next layer (Low Resolution)
        x_down = self.downsample_conv(x)

        return x_down, skip_feature

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_residual_layers, t_embed_dim, y_embed_dim):
        super().__init__()
        
        # 1. Upsample
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"), 
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        )
        
        # 2. Residual Blocks
        # IMPORTANT: If using concatenation (recommended), input channels would be out_channels * 2
        # If using addition (your current method), input is out_channels. 
        # Let's stick to your Addition method for minimal code changes, 
        # but typically Concatenation is better for U-Nets.
        self.res_blocks = nn.ModuleList([
            ResidualLayer(channels=out_channels, time_embed_dim=t_embed_dim, y_embed_dim=y_embed_dim) 
            for _ in range(num_residual_layers)
        ])

    def forward(self, x, skip_connection, t_embed, y_embed):
        # Upsample: [B, C_in, H, W] -> [B, C_out, H*2, W*2]
        x = self.upsample(x)

        # Add Skip Connection
        # x is [B, C_out, H*2, W*2]
        # skip_connection is [B, C_out, H*2, W*2] (Now that we fixed the encoder!)
        assert x.shape == skip_connection.shape, "Shape mismatch between upsampled x and skip connection"
        x = x + skip_connection 

        # Pass through blocks
        for block in self.res_blocks: 
            x = block(x, t_embed, y_embed)

        return x

class Midcoder(nn.Module):
    def __init__(self, channels, num_residual_layers, t_embed_dim, y_embed_dim):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResidualLayer(channels = channels, time_embed_dim = t_embed_dim, y_embed_dim = y_embed_dim) for _ in range(num_residual_layers)
        ])

    def forward(self, x, t_embed, y_embed):
        """
        Args: 
            x: feature map [B, C, H, W]
            t_embed: time embedding [B, t_embed_dim]
            y_embed: y embedding [B, y_embed_dim]
        """
        # Pass through residual blocks: [B, C, H, W] -> [B, C, H, W]
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)

        return x
    

    
class FourierEncoder(nn.Module):
    """
    Used to encode the time step t into a higher dimensional space using Fourier features.
    Based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/karras_unet.py#L183
    """
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, "Fourier feature dimension must be even"
        self.half_dim = dim // 2 # Half dimension for sin and cos
        self.weights = nn.Parameter(torch.randn(1, self.half_dim)) # [1, half_dim] random initialized weights for frequency scaling

    def forward(self, t):
        """
        Args:
            t: [B] or [B, 1, 1, 1] time step tensor
        Returns:
            t_embed: [B, t_embed_dim] Learned Fourier encoded time embedding

        """
        t = t.view(-1, 1) # [B, ...] -> [B, 1]
        # w = 2pi*f
        freqs = t * self.weights * 2 * math.pi # [B, 1] * [1, half_dim] -> [B, half_dim]
        sin_embed = torch.sin(freqs) # [B, half_dim]
        cos_embed = torch.cos(freqs) # [B, half_dim]
        t_embed = torch.cat([sin_embed, cos_embed], dim = -1) # [B, half_dim], [B, half_dim] -> [B, t_embed_dim]

        return t_embed * math.sqrt(2) # SO that the variance of each time embedding is 1. since var of sin and cos is 0.5 each


# # Simulate
# dim = 128
# half_dim = dim // 2
# weights = torch.randn(1, half_dim)
# t = torch.rand(10000, 1)  # Many random t for stats

# freqs = t * weights * 2 * math.pi
# sin_embed = torch.sin(freqs)
# cos_embed = torch.cos(freqs)
# embed = torch.cat([sin_embed, cos_embed], dim=-1)

# print("Variance without sqrt(2):", embed.var(dim=1).mean())  # ~0.5
# print("Variance with sqrt(2):", (embed * math.sqrt(2)).var(dim=1).mean())  # ~1.0


class ConditionEmbedding(nn.Module):
    """
    Embeds the guiding labels y ie for moth and location into a higher dimensional space
    
    """
    def __init__(self, num_month_embeddings, num_location_embeddings, y_embed_dim):
        super().__init__()
        self.month_embedding = nn.Embedding(num_embeddings = num_month_embeddings, embedding_dim = y_embed_dim)
        self.location_embedding = nn.Embedding(num_embeddings = num_location_embeddings, embedding_dim = y_embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(in_features = 2 * y_embed_dim, out_features = y_embed_dim),
            nn.SiLU(),
            nn.Linear(in_features = y_embed_dim, out_features = y_embed_dim)
        )
    def forward(self, y):
        """
        Args:
            y: guiding labels [2, B] where y[0] is month labels and y[1] is location labels
        Returns:
            y_embed: embedded guiding labels [B, y_embed_dim]
        """
        month_labels = y[0].long()  # [B]
        location_labels = y[1].long()  # [B]

        month_embed = self.month_embedding(month_labels)  # [B, y_embed_dim]
        location_embed = self.location_embedding(location_labels)  # [B, y_embed_dim]

        # Concatenate month and location embeddings
        y_embed = torch.cat([month_embed, location_embed], dim = -1)  # [B, 2 * y_embed_dim]

        # Project to final y embedding
        y_embed = self.proj(y_embed)  # [B, y_embed_dim]

        return y_embed
    



# In[32]:


from typing import Optional, List, Type, Tuple, Dict
## Create Unet
class FloodUNet(ConditionalVectorFieldCNN):
    def __init__(self, channels, num_residual_layers, t_embed_dim, y_embed_dim):
        super().__init__()

        # Initial conv
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channels[0], kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=channels[0]),
            nn.SiLU()
        )

        self.time_embedder = FourierEncoder(dim=t_embed_dim)
        
        # (Assuming p_data global availability for map size)
        self.y_embedder = ConditionEmbedding(
            num_month_embeddings=13,  
            num_location_embeddings=len(p_data.dataset.tile_to_int_map) + 1,
            y_embed_dim=y_embed_dim
        )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # Create Encoders
        for (curr_c, next_c) in zip(channels[:-1], channels[1:]):
            self.encoders.append(
                Encoder(curr_c, next_c, num_residual_layers, t_embed_dim, y_embed_dim)
            )

        # Create Decoders (Note: in_channels/out_channels swapped compared to encoder)
        # We iterate backwards through channels to match the encoder structure
        rev_channels = channels[::-1]
        for (curr_c, next_c) in zip(rev_channels[:-1], rev_channels[1:]):
            self.decoders.append(
                Decoder(curr_c, next_c, num_residual_layers, t_embed_dim, y_embed_dim)
            )

        self.midcoder = Midcoder(channels[-1], num_residual_layers, t_embed_dim, y_embed_dim)
        self.final_conv = nn.Conv2d(channels[0], 1, kernel_size=3, padding=1)

    def forward(self, x, t, y):
        t_embed = self.time_embedder(t)
        y_embed = self.y_embedder(y)

        x = self.initial_conv(x)

        residuals = []

        # Encoder Pass
        for encoder in self.encoders:
            # x becomes the downsampled input for next layer
            # skip becomes the residual to save
            x, skip = encoder(x, t_embed, y_embed) 
            residuals.append(skip)

        # Midcoder
        x = self.midcoder(x, t_embed, y_embed)

        # Decoder Pass
        for decoder in self.decoders:
            skip = residuals.pop()
            x = decoder(x, skip, t_embed, y_embed)

        return self.final_conv(x)


# In[33]:


## Train unet 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize probabilistic path
p_data = flood_map_sampler()
path = GaussianConditionalProbabilityPath(
    p_data = p_data,
    p_simple_shape = [1, *target_size],
    alpha = LinearAlpha(),
    beta = LinearBeta()
).to(device)

# Initialize model
floodnet = FloodUNet(
    channels = [32, 64, 128, 256, 512], #  [128, 128, 256, 512, 1024, 1024]
    num_residual_layers = 2,
    t_embed_dim = 64,
    y_embed_dim = 64
).to(device)


# Initialize trainer
trainer = CFGTrainer(
    path = path,
    model = floodnet,
    eta = 0.1,  # No label dropping for initial training
    device = device,
    grad_clip_norm = 1.0
)

# Train model
trainer.train(num_epochs = 5000, device = device, lr = 1e-4, warmup_epochs = 500, batch_size = 8)

# Save model
torch.save(floodnet.state_dict(), "floodnet_cfg_sdf.pth")

# # Load
# model = MyModel(...)               # create the model instance
# model.load_state_dict(torch.load("model_weights.pth"))
# model.eval()


# In[34]:


## Plot losses
plt.plot(trainer.losses[:])
plt.yscale('log')
plt.xlabel("Epoch")
# Save losses plot
# plt.savefig("training_losses2_cfg.png", dpi = 500)
plt.show()


# In[35]:


## Visualize the flow: noise → data trajectory
floodnet.eval()

with torch.no_grad():
    # Setup ODE and simulator
    ode = CFGVectorFieldODE(net=floodnet, guidance_scale=1.0)
    simulator = EulerSimulator(ode=ode)
    
    # Sample initial noise from p_simple
    num_samples = 1
    x0, _ = path.p_simple.sample(num_samples)
    x0 = x0.to(device)
    
    # Get ground truth data and labels
    z, month_labels, location_labels = path.sample_conditioning_variable(num_samples)
    z = z.to(device)
    month_labels = month_labels.to(device)
    location_labels = location_labels.to(device)
    y = torch.stack([month_labels, location_labels], dim=0)
    
    # Define time steps - use more steps for smoother visualization
    num_steps = 50
    ts = torch.linspace(0, 1, num_steps).view(1, -1, 1, 1).expand(num_samples, -1, 1, 1).to(device)
    
    # Simulate WITH trajectory recording
    trajectory = simulator.simulate_with_trajectory(x=x0, ts=ts, y=y)  # [B, num_steps, C, H, W]
    
    # Select frames to display (e.g., 8 frames evenly spaced)
    num_frames = 8
    frame_indices = torch.linspace(0, num_steps - 1, num_frames).long()
    
    # Plot the trajectory
    fig, axes = plt.subplots(1, num_frames + 1, figsize=(3 * (num_frames + 1), 3))
    
    for i, idx in enumerate(frame_indices):
        t_val = ts[0, idx, 0, 0].item()
        frame = trajectory[0, idx, 0].cpu().numpy()
        axes[i].imshow(frame, cmap='viridis', vmin=-1, vmax=1)
        axes[i].set_title(f't = {t_val:.2f}')
        axes[i].axis('off')
    
    # Show ground truth for comparison
    axes[-1].imshow(z[0, 0].cpu().numpy(), cmap='viridis', vmin=-1, vmax=1)
    axes[-1].set_title('Ground Truth')
    axes[-1].axis('off')
    
    plt.suptitle('Flow Matching: Noise → Data Transformation', fontsize=14)
    plt.tight_layout()
    plt.savefig("flow_trajectory.png", dpi=150, bbox_inches='tight')
    plt.show()


# In[73]:


## Create animated GIF of the flow (optional - requires imageio)
# pip install imageio if not installed

try:
    import imageio
    
    # Normalize trajectory for visualization
    traj_np = trajectory[0, :, 0].cpu().numpy()  # [num_steps, H, W]
    
    # Normalize to 0-255 for GIF
    vmin, vmax = -1, 1
    traj_normalized = np.clip((traj_np - vmin) / (vmax - vmin), 0, 1)
    traj_uint8 = (traj_normalized * 255).astype(np.uint8)
    
    # Apply colormap (viridis)
    from matplotlib import cm
    frames = []
    for i in range(traj_uint8.shape[0]):
        # Apply viridis colormap
        colored = (cm.viridis(traj_normalized[i])[:, :, :3] * 255).astype(np.uint8)
        frames.append(colored)
    
    # Add ground truth at the end (repeat a few times)
    gt_normalized = np.clip((z[0, 0].cpu().numpy() - vmin) / (vmax - vmin), 0, 1)
    gt_colored = (cm.viridis(gt_normalized)[:, :, :3] * 255).astype(np.uint8)
    for _ in range(10):
        frames.append(gt_colored)
    
    # Save as GIF
    imageio.mimsave('flow_animation.gif', frames, fps=10, loop=0)
    print("Saved animation to flow_animation.gif")
    
    # Display in notebook (if IPython available)
    from IPython.display import Image, display
    display(Image(filename='flow_animation.gif'))
    
except ImportError:
    print("Install imageio for GIF animation: pip install imageio")


# In[54]:


# Put model in eval mode
floodnet.eval()
with torch.no_grad():
    # Setup ode and simulator
    ode = CFGVectorFieldODE(net = floodnet, guidance_scale = 1.0)
    simulator = EulerSimulator(ode = ode)
    # Sample from p_simple
    num_samples = 1
    x0, _ = path.p_simple.sample(num_samples)  # [B, C, H, W]
    x0 = x0.to(device)

    # Sample conditioning variable z and labels y from p_data
    z, month_labels, location_labels = path.sample_conditioning_variable(num_samples)
    z = z.to(device)
    month_labels = month_labels.to(device)
    location_labels = location_labels.to(device)
    y = torch.stack([month_labels, location_labels], dim = 0)  # [2, B]

    # Define time steps for simulation
    num_steps = 100
    ts = torch.linspace(0, 1, num_steps).view(1, -1, 1, 1).expand(num_samples, -1, 1, 1).to(device)
    
    # Simulate forward process to t=1
    x1 =  simulator.simulate(x = x0, ts = ts, y = y)  # [B, C, H, W]
    # Binarize output
    # x1 = (x1 > 1).float()

    #Plot generated samples
    fig, axs = plt.subplots(1, 2, figsize = (12, 6))
    axs[0].imshow(z[0, 0].cpu(), cmap = 'gray',)
    axs[0].set_title("Data sample z")
    axs[0].axis('off')

    axs[1].imshow(x1[0, 0].cpu(), cmap = 'gray', )
    axs[1].set_title("Simulated x at t=1")
    axs[1].axis('off')

    plt.show()
  
    




# In[44]:


get_ipython().run_line_magic('matplotlib', 'inline')

#Plot generated samples
fig, axs = plt.subplots(1, 2, figsize = (12, 6))
axs[0].imshow(z[0, 0].cpu(), cmap = 'gray',)
axs[0].set_title("Data sample z")
axs[0].axis('off')

axs[1].imshow(x1[0, 0].cpu() < 0.15, cmap = 'gray', )
axs[1].set_title("Simulated x at t=1")
axs[1].axis('off')


# In[49]:


## Histogram of x1 values
plt.hist(x1.cpu().numpy().flatten(), bins = 2)
plt.title("Histogram of x1 values at t=1")


# ## To do
# 
# - effect of sdf
# - marginal prob path
# - How to prevent/balance Overfit
# - Effects of train/val
# - How this varies to Reimanian manifolds
