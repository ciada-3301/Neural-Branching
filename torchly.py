"""
NeuralNet - A Simple PyTorch Wrapper for Easy Neural Network Training
Author: Arkadyuti Maiti
Version: 1.0.0

A comprehensive yet simple wrapper over PyTorch for creating, training, 
and managing neural networks with minimal code.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import os
import copy
from typing import List, Union, Dict, Tuple, Optional, Any, Callable
import warnings
from collections import defaultdict
import pickle


class Model:
    """
    Main Model class for creating and training neural networks.
    
    Examples:
        >>> model = Model([2, 16, 16, 2], activation="relu")
        >>> model.train([X_train], y_train, epochs=100)
        >>> weights = model.extract_layer(1)
        >>> model.save("model.pt")
        >>> loaded_model = Model.load("model.pt")
    """
    
    def __init__(
        self,
        architecture: Union[List[int], List[Dict]],
        activation: Union[str, List[str]] = "relu",
        dropout: Optional[float] = None,
        batch_norm: bool = False,
        layer_names: Optional[List[str]] = None,
        l1_reg: float = 0.0,
        l2_reg: float = 0.0,
        optimizer: str = "adam",
        lr: float = 0.001,
        seed: Optional[int] = None
    ):
        """
        Initialize a neural network model.
        
        Args:
            architecture: List of layer sizes [input, hidden1, hidden2, ..., output]
                         or list of dicts with layer specifications
            activation: Activation function ('relu', 'tanh', 'sigmoid', 'leaky_relu')
                       or list of activations for each layer
            dropout: Dropout rate (0.0 to 1.0)
            batch_norm: Whether to use batch normalization
            layer_names: Optional names for each layer
            l1_reg: L1 regularization strength
            l2_reg: L2 regularization strength
            optimizer: Optimizer type ('adam', 'sgd', 'rmsprop')
            lr: Learning rate
            seed: Random seed for reproducibility
        """
        if seed is not None:
            self.set_seed(seed)
        
        self.architecture = architecture
        self.activation_type = activation
        self.dropout_rate = dropout
        self.use_batch_norm = batch_norm
        self.layer_names = layer_names or []
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.lr = lr
        self.seed = seed
        
        # Build the network
        self.network = self._build_network()
        
        # Initialize optimizer
        self.optimizer_type = optimizer
        self.optimizer = self._create_optimizer(optimizer, lr)
        
        # Training history
        self.history = defaultdict(list)
        
        # Device management
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)
        
        # Normalization parameters
        self.normalizer_mean = None
        self.normalizer_std = None
        self.normalizer_min = None
        self.normalizer_max = None
        
    def _build_network(self) -> nn.Module:
        """Build the neural network architecture."""
        if isinstance(self.architecture[0], dict):
            return self._build_custom_network()
        else:
            return self._build_sequential_network()
    
    def _build_sequential_network(self) -> nn.Module:
        """Build a simple sequential network from layer sizes."""
        layers = []
        architecture = self.architecture
        
        # Handle activation list
        if isinstance(self.activation_type, str):
            activations = [self.activation_type] * (len(architecture) - 1)
        else:
            activations = self.activation_type
        
        for i in range(len(architecture) - 1):
            # Linear layer
            layers.append(nn.Linear(architecture[i], architecture[i + 1]))
            
            # Batch normalization
            if self.use_batch_norm and i < len(architecture) - 2:
                layers.append(nn.BatchNorm1d(architecture[i + 1]))
            
            # Activation
            if i < len(architecture) - 2:
                layers.append(self._get_activation(activations[i]))
            
            # Dropout
            if self.dropout_rate and i < len(architecture) - 2:
                layers.append(nn.Dropout(self.dropout_rate))
        
        return nn.Sequential(*layers)
    
    def _build_custom_network(self) -> nn.Module:
        """Build network from custom layer specifications."""
        # Placeholder for custom architectures
        # Can be extended for Conv2D, MaxPool, etc.
        raise NotImplementedError("Custom architectures not yet implemented")
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'softmax': nn.Softmax(dim=1),
            'none': nn.Identity()
        }
        return activations.get(activation.lower(), nn.ReLU())
    
    def _create_optimizer(self, optimizer_type: str, lr: float) -> optim.Optimizer:
        """Create optimizer."""
        optimizers = {
            'adam': optim.Adam(self.network.parameters(), lr=lr),
            'sgd': optim.SGD(self.network.parameters(), lr=lr),
            'rmsprop': optim.RMSprop(self.network.parameters(), lr=lr),
            'adamw': optim.AdamW(self.network.parameters(), lr=lr)
        }
        return optimizers.get(optimizer_type.lower(), optim.Adam(self.network.parameters(), lr=lr))
    
    def train(
        self,
        X: List[np.ndarray],
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        loss: Union[str, Callable] = "mse",
        validation_data: Optional[Tuple] = None,
        early_stopping: bool = False,
        patience: int = 10,
        callbacks: Optional[List[str]] = None,
        lr_schedule: Optional[str] = None,
        grad_clip: Optional[float] = None,
        verbose: int = 1,
        resume_from: Optional[str] = None,
        shuffle: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the neural network.
        
        Args:
            X: List of input arrays (can be single array in list)
            y: Target array
            epochs: Number of training epochs
            batch_size: Batch size for training
            loss: Loss function ('mse', 'cross_entropy', 'bce') or custom function
            validation_data: Tuple of (X_val, y_val) for validation
            early_stopping: Whether to use early stopping
            patience: Patience for early stopping
            callbacks: List of callback names
            lr_schedule: Learning rate schedule type
            grad_clip: Gradient clipping value
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
            resume_from: Path to checkpoint to resume from
            shuffle: Whether to shuffle training data
            
        Returns:
            Dictionary containing training history
        """
        # Convert inputs to tensors
                # Convert inputs to tensors
        X_tensor = torch.FloatTensor(X[0]).to(self.device)
        
        # FIX: Check if we are doing cross_entropy to set correct dtype and shape
        if loss == "cross_entropy":
            y_tensor = torch.LongTensor(y).to(self.device)
            # Do NOT unsqueeze(1) for cross_entropy; it must stay 1D
        else:
            y_tensor = torch.FloatTensor(y).to(self.device)
            if len(y_tensor.shape) == 1:
                y_tensor = y_tensor.unsqueeze(1)

        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
        # Setup loss function
        criterion = self._get_loss_function(loss)
        
        # Setup learning rate scheduler
        scheduler = None
        if lr_schedule:
            scheduler = self._get_lr_scheduler(lr_schedule)
        
        # Early stopping setup
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.network.train()
            epoch_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.network(batch_X)
                
                # Compute loss
                batch_loss = criterion(outputs, batch_y)
                
                # Add regularization
                if self.l1_reg > 0 or self.l2_reg > 0:
                    batch_loss = batch_loss + self._compute_regularization()
                
                # Backward pass
                batch_loss.backward()
                
                # Gradient clipping
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), grad_clip)
                
                # Update weights
                self.optimizer.step()
                
                epoch_loss += batch_loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            self.history['loss'].append(avg_loss)
            
            # Validation phase
            val_loss = None
            if validation_data:
                val_loss = self._validate(validation_data, criterion)
                self.history['val_loss'].append(val_loss)
            
            # Learning rate scheduling
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss if val_loss else avg_loss)
                else:
                    scheduler.step()
            
            # Early stopping
            if early_stopping and validation_data:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = copy.deepcopy(self.network.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose > 0:
                            print(f"Early stopping at epoch {epoch + 1}")
                        if best_model_state:
                            self.network.load_state_dict(best_model_state)
                        break
            
            # Verbose output
            if verbose > 0 and (epoch + 1) % max(1, epochs // 10) == 0:
                msg = f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}"
                if val_loss:
                    msg += f" - Val Loss: {val_loss:.4f}"
                print(msg)
        
        return dict(self.history)
    
    def _validate(self, validation_data: Tuple, criterion: nn.Module) -> float:
        """Validate on validation set."""
        self.network.eval()
        X_val, y_val = validation_data
        
        X_val_tensor = torch.FloatTensor(X_val[0]).to(self.device)
        
        # FIX: Check loss type to determine correct tensor dtype and shape
        if isinstance(criterion, nn.CrossEntropyLoss):
            # Cross entropy needs LongTensor and must stay 1D
            y_val_tensor = torch.LongTensor(y_val).to(self.device)
            # Do NOT unsqueeze for cross_entropy
        else:
            # Other losses (MSE, BCE, etc.) use FloatTensor
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            if len(y_val_tensor.shape) == 1:
                y_val_tensor = y_val_tensor.unsqueeze(1)
        
        with torch.no_grad():
            outputs = self.network(X_val_tensor)
            val_loss = criterion(outputs, y_val_tensor).item()
    
        return val_loss
    
    def _get_loss_function(self, loss: Union[str, Callable]) -> nn.Module:
        """Get loss function."""
        if callable(loss):
            return loss
        
        losses = {
            'mse': nn.MSELoss(),
            'mae': nn.L1Loss(),
            'cross_entropy': nn.CrossEntropyLoss(),
            'bce': nn.BCELoss(),
            'bce_logits': nn.BCEWithLogitsLoss(),
            'huber': nn.HuberLoss()
        }
        return losses.get(loss.lower(), nn.MSELoss())
    
    def _get_lr_scheduler(self, schedule_type: str):
        """Get learning rate scheduler."""
        schedulers = {
            'step': optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1),
            'exponential': optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95),
            'cosine': optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50),
            'plateau': optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5)
        }
        return schedulers.get(schedule_type.lower())
    
    def _compute_regularization(self) -> torch.Tensor:
        """Compute L1 and L2 regularization."""
        l1_loss = 0.0
        l2_loss = 0.0
        
        for param in self.network.parameters():
            if self.l1_reg > 0:
                l1_loss += torch.sum(torch.abs(param))
            if self.l2_reg > 0:
                l2_loss += torch.sum(param ** 2)
        
        return self.l1_reg * l1_loss + self.l2_reg * l2_loss
    
    def predict(
        self,
        X: List[np.ndarray],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Make predictions on input data.
        
        Args:
            X: List of input arrays
            batch_size: Batch size for prediction
            
        Returns:
            Predictions as numpy array
        """
        self.network.eval()
        X_tensor = torch.FloatTensor(X[0]).to(self.device)
        
        with torch.no_grad():
            if batch_size:
                predictions = []
                for i in range(0, len(X_tensor), batch_size):
                    batch = X_tensor[i:i + batch_size]
                    pred = self.network(batch)
                    predictions.append(pred)
                outputs = torch.cat(predictions, dim=0)
            else:
                outputs = self.network(X_tensor)
        
        return outputs.cpu().numpy()
    
    def predict_proba(self, X: List[np.ndarray]) -> np.ndarray:
        """Get prediction probabilities (applies softmax)."""
        predictions = self.predict(X)
        return torch.softmax(torch.FloatTensor(predictions), dim=1).numpy()
    
    def predict_classes(self, X: List[np.ndarray]) -> np.ndarray:
        """Get predicted class labels."""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
    def evaluate(
        self,
        X: List[np.ndarray],
        y: np.ndarray,
        metrics: Optional[List[str]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate model on test data.
        
        Args:
            X: Input data
            y: True labels
            metrics: List of metric names
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        self.network.eval()
        X_tensor = torch.FloatTensor(X[0]).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        if len(y_tensor.shape) == 1:
            y_tensor = y_tensor.unsqueeze(1)
        
        with torch.no_grad():
            outputs = self.network(X_tensor)
            loss = nn.MSELoss()(outputs, y_tensor).item()
        
        metrics_dict = {}
        if metrics:
            predictions = outputs.cpu().numpy()
            y_true = y
            
            for metric in metrics:
                if metric == 'accuracy':
                    pred_classes = np.argmax(predictions, axis=1)
                    true_classes = np.argmax(y_true, axis=1) if len(y_true.shape) > 1 else y_true
                    metrics_dict['accuracy'] = np.mean(pred_classes == true_classes)
                elif metric == 'mse':
                    metrics_dict['mse'] = np.mean((predictions - y_true) ** 2)
                elif metric == 'mae':
                    metrics_dict['mae'] = np.mean(np.abs(predictions - y_true))
        
        return loss, metrics_dict
    
    def extract_layer(
        self,
        layer: Union[int, str],
        include_bias: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Extract weights from a specific layer.
        
        Args:
            layer: Layer index or name
            include_bias: Whether to include bias terms
            
        Returns:
            Dictionary with 'weights' and optionally 'bias'
        """
        # Get the layer
        if isinstance(layer, str):
            # Find layer by name
            target_layer = None
            for name, module in self.network.named_modules():
                if name == layer:
                    target_layer = module
                    break
        else:
            # Get layer by index
            layers = [m for m in self.network.modules() if isinstance(m, nn.Linear)]
            if layer >= len(layers):
                raise IndexError(f"Layer {layer} out of range")
            target_layer = layers[layer]
        
        if target_layer is None:
            raise ValueError(f"Layer {layer} not found")
        
        result = {
            'weights': target_layer.weight.detach().cpu().numpy()
        }
        
        if include_bias and hasattr(target_layer, 'bias') and target_layer.bias is not None:
            result['bias'] = target_layer.bias.detach().cpu().numpy()
        
        return result
    
    def extract_all_layers(self) -> Dict[int, Dict[str, np.ndarray]]:
        """Extract weights from all layers."""
        all_weights = {}
        layers = [m for m in self.network.modules() if isinstance(m, nn.Linear)]
        
        for i, layer in enumerate(layers):
            all_weights[i] = {
                'weights': layer.weight.detach().cpu().numpy(),
                'bias': layer.bias.detach().cpu().numpy() if layer.bias is not None else None
            }
        
        return all_weights
    
    def set_layer_weights(
        self,
        layer: Union[int, str],
        weights: np.ndarray,
        bias: Optional[np.ndarray] = None
    ):
        """Set weights for a specific layer."""
        # Get the layer
        if isinstance(layer, str):
            target_layer = None
            for name, module in self.network.named_modules():
                if name == layer:
                    target_layer = module
                    break
        else:
            layers = [m for m in self.network.modules() if isinstance(m, nn.Linear)]
            target_layer = layers[layer]
        
        # Set weights
        target_layer.weight.data = torch.FloatTensor(weights).to(self.device)
        if bias is not None and target_layer.bias is not None:
            target_layer.bias.data = torch.FloatTensor(bias).to(self.device)
    
    def freeze_layer(self, layer: Union[int, str]):
        """Freeze a specific layer (stop training)."""
        if isinstance(layer, str):
            target_layer = None
            for name, module in self.network.named_modules():
                if name == layer:
                    target_layer = module
                    break
        else:
            layers = [m for m in self.network.modules() if isinstance(m, nn.Linear)]
            target_layer = layers[layer]
        
        for param in target_layer.parameters():
            param.requires_grad = False
    
    def unfreeze_layer(self, layer: Union[int, str]):
        """Unfreeze a specific layer (resume training)."""
        if isinstance(layer, str):
            target_layer = None
            for name, module in self.network.named_modules():
                if name == layer:
                    target_layer = module
                    break
        else:
            layers = [m for m in self.network.modules() if isinstance(m, nn.Linear)]
            target_layer = layers[layer]
        
        for param in target_layer.parameters():
            param.requires_grad = True
    
    def freeze_all(self):
        """Freeze all layers."""
        for param in self.network.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all layers."""
        for param in self.network.parameters():
            param.requires_grad = True
    
    def freeze_layers(self, layers: List[int]):
        """Freeze specific layers by index."""
        for layer_idx in layers:
            self.freeze_layer(layer_idx)
    
    def layer_info(self, layer: Union[int, str]) -> Dict[str, Any]:
        """Get information about a specific layer."""
        if isinstance(layer, str):
            target_layer = None
            for name, module in self.network.named_modules():
                if name == layer:
                    target_layer = module
                    break
        else:
            layers = [m for m in self.network.modules() if isinstance(m, nn.Linear)]
            target_layer = layers[layer]
        
        if isinstance(target_layer, nn.Linear):
            return {
                'type': 'linear',
                'input_size': target_layer.in_features,
                'output_size': target_layer.out_features,
                'has_bias': target_layer.bias is not None
            }
        else:
            return {'type': str(type(target_layer))}
    
    def summary(self):
        """Print model summary."""
        print("=" * 70)
        print(f"{'Layer':<20} {'Type':<20} {'Output Shape':<20}")
        print("=" * 70)
        
        total_params = 0
        for i, (name, module) in enumerate(self.network.named_modules()):
            if isinstance(module, nn.Linear):
                params = module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
                total_params += params
                print(f"{name or f'Layer {i}':<20} {'Linear':<20} {str(module.out_features):<20}")
            elif isinstance(module, (nn.ReLU, nn.Tanh, nn.Sigmoid)):
                print(f"{name or f'Layer {i}':<20} {type(module).__name__:<20} {'-':<20}")
            elif isinstance(module, nn.Dropout):
                print(f"{name or f'Layer {i}':<20} {'Dropout':<20} {'-':<20}")
            elif isinstance(module, nn.BatchNorm1d):
                params = module.num_features * 2  # gamma and beta
                total_params += params
                print(f"{name or f'Layer {i}':<20} {'BatchNorm1d':<20} {str(module.num_features):<20}")
        
        print("=" * 70)
        print(f"Total parameters: {total_params:,}")
        print("=" * 70)
    
    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.network.parameters())
    
    def save(
        self,
        filepath: str,
        include_optimizer: bool = True,
        include_history: bool = True
    ):
        """
        Save complete model to file.
        
        Args:
            filepath: Path to save file
            include_optimizer: Whether to save optimizer state
            include_history: Whether to save training history
        """
        save_dict = {
            'model_state_dict': self.network.state_dict(),
            'architecture': self.architecture,
            'activation_type': self.activation_type,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'layer_names': self.layer_names,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg,
            'optimizer_type': self.optimizer_type,
            'lr': self.lr,
            'seed': self.seed
        }
        
        if include_optimizer:
            save_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        
        if include_history:
            save_dict['history'] = dict(self.history)
        
        # Save normalization parameters if they exist
        if self.normalizer_mean is not None:
            save_dict['normalizer_mean'] = self.normalizer_mean
            save_dict['normalizer_std'] = self.normalizer_std
        
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")
    
    def save_weights(self, filepath: str):
        """Save only model weights."""
        torch.save(self.network.state_dict(), filepath)
        print(f"Weights saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, resume_training: bool = False) -> 'Model':
        """
        Load complete model from file.
        
        Args:
            filepath: Path to model file
            resume_training: Whether to restore optimizer state
            
        Returns:
            Loaded Model instance
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Create model
        model = cls(
            architecture=checkpoint['architecture'],
            activation=checkpoint['activation_type'],
            dropout=checkpoint.get('dropout_rate'),
            batch_norm=checkpoint.get('use_batch_norm', False),
            layer_names=checkpoint.get('layer_names', []),
            l1_reg=checkpoint.get('l1_reg', 0.0),
            l2_reg=checkpoint.get('l2_reg', 0.0),
            optimizer=checkpoint.get('optimizer_type', 'adam'),
            lr=checkpoint.get('lr', 0.001),
            seed=checkpoint.get('seed')
        )
        
        # Load state
        model.network.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if resuming training
        if resume_training and 'optimizer_state_dict' in checkpoint:
            model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load history
        if 'history' in checkpoint:
            model.history = defaultdict(list, checkpoint['history'])
        
        # Load normalization parameters
        if 'normalizer_mean' in checkpoint:
            model.normalizer_mean = checkpoint['normalizer_mean']
            model.normalizer_std = checkpoint['normalizer_std']
        
        print(f"Model loaded from {filepath}")
        return model
    
    def load_weights(self, filepath: str):
        """Load only model weights."""
        self.network.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"Weights loaded from {filepath}")
    
    def save_architecture(self, filepath: str):
        """Save architecture as JSON."""
        arch_dict = {
            'architecture': self.architecture,
            'activation_type': self.activation_type,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'layer_names': self.layer_names
        }
        
        with open(filepath, 'w') as f:
            json.dump(arch_dict, f, indent=2)
        
        print(f"Architecture saved to {filepath}")
    
    @classmethod
    def from_architecture(cls, filepath: str) -> 'Model':
        """Load architecture from JSON and initialize with random weights."""
        with open(filepath, 'r') as f:
            arch_dict = json.load(f)
        
        return cls(
            architecture=arch_dict['architecture'],
            activation=arch_dict.get('activation_type', 'relu'),
            dropout=arch_dict.get('dropout_rate'),
            batch_norm=arch_dict.get('use_batch_norm', False),
            layer_names=arch_dict.get('layer_names', [])
        )
    
    def to_gpu(self, device: int = 0):
        """Move model to GPU."""
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{device}')
            self.network.to(self.device)
            print(f"Model moved to GPU {device}")
        else:
            print("CUDA not available, model remains on CPU")
    
    def to_cpu(self):
        """Move model to CPU."""
        self.device = torch.device('cpu')
        self.network.to(self.device)
        print("Model moved to CPU")
    
    def auto_device(self):
        """Automatically select best device."""
        if torch.cuda.is_available():
            self.to_gpu()
        else:
            self.to_cpu()
    
    def get_device(self) -> str:
        """Get current device."""
        return str(self.device)
    
    def set_optimizer(self, optimizer_type: str, lr: Optional[float] = None, **kwargs):
        """Set optimizer type and parameters."""
        self.optimizer_type = optimizer_type
        if lr is not None:
            self.lr = lr
        
        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr, **kwargs)
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(self.network.parameters(), lr=self.lr, **kwargs)
        elif optimizer_type.lower() == 'rmsprop':
            self.optimizer = optim.RMSprop(self.network.parameters(), lr=self.lr, **kwargs)
        elif optimizer_type.lower() == 'adamw':
            self.optimizer = optim.AdamW(self.network.parameters(), lr=self.lr, **kwargs)
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def set_lr(self, lr: float):
        """Set learning rate."""
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_history(self, metric: str) -> List[float]:
        """Get specific metric from history."""
        return self.history.get(metric, [])
    
    def plot_history(self):
        """Plot training history."""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot loss
            if 'loss' in self.history:
                axes[0].plot(self.history['loss'], label='Train Loss')
            if 'val_loss' in self.history:
                axes[0].plot(self.history['val_loss'], label='Val Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training History')
            axes[0].legend()
            axes[0].grid(True)
            
            # Plot other metrics
            other_metrics = [k for k in self.history.keys() if k not in ['loss', 'val_loss']]
            if other_metrics:
                for metric in other_metrics:
                    axes[1].plot(self.history[metric], label=metric)
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Metric Value')
                axes[1].set_title('Metrics')
                axes[1].legend()
                axes[1].grid(True)
            
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("matplotlib not installed. Cannot plot history.")
    
    def export_history(self, filepath: str):
        """Export training history to CSV."""
        import csv
        
        if not self.history:
            print("No history to export")
            return
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            headers = ['epoch'] + list(self.history.keys())
            writer.writerow(headers)
            
            # Data
            num_epochs = len(next(iter(self.history.values())))
            for i in range(num_epochs):
                row = [i + 1]
                for metric in self.history.keys():
                    row.append(self.history[metric][i] if i < len(self.history[metric]) else '')
                writer.writerow(row)
        
        print(f"History exported to {filepath}")
    
    def clone(self, reinitialize: bool = False, seed: Optional[int] = None) -> 'Model':
        """Clone the model."""
        if seed is not None:
            self.set_seed(seed)
        
        new_model = Model(
            architecture=self.architecture,
            activation=self.activation_type,
            dropout=self.dropout_rate,
            batch_norm=self.use_batch_norm,
            layer_names=self.layer_names,
            l1_reg=self.l1_reg,
            l2_reg=self.l2_reg,
            optimizer=self.optimizer_type,
            lr=self.lr,
            seed=seed
        )
        
        if not reinitialize:
            new_model.network.load_state_dict(copy.deepcopy(self.network.state_dict()))
        
        return new_model
    
    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def get_seed(self) -> Optional[int]:
        """Get current seed."""
        return self.seed
    
    def normalize(
        self,
        X: np.ndarray,
        method: str = "standard"
    ) -> np.ndarray:
        """
        Normalize data.
        
        Args:
            X: Input data
            method: 'standard' (z-score) or 'minmax'
            
        Returns:
            Normalized data
        """
        if method == "standard":
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            std[std == 0] = 1  # Avoid division by zero
            return (X - mean) / std
        elif method == "minmax":
            min_val = np.min(X, axis=0)
            max_val = np.max(X, axis=0)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1
            return (X - min_val) / range_val
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def fit_normalizer(self, X: np.ndarray, method: str = "standard"):
        """Fit normalizer on training data."""
        if method == "standard":
            self.normalizer_mean = np.mean(X, axis=0)
            self.normalizer_std = np.std(X, axis=0)
            self.normalizer_std[self.normalizer_std == 0] = 1
        elif method == "minmax":
            self.normalizer_min = np.min(X, axis=0)
            self.normalizer_max = np.max(X, axis=0)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted normalizer."""
        if self.normalizer_mean is not None:
            return (X - self.normalizer_mean) / self.normalizer_std
        elif self.normalizer_min is not None:
            range_val = self.normalizer_max - self.normalizer_min
            range_val[range_val == 0] = 1
            return (X - self.normalizer_min) / range_val
        else:
            raise ValueError("Normalizer not fitted. Call fit_normalizer first.")
    
    def get_activations(self, X: List[np.ndarray], layer: Optional[int] = None) -> Dict:
        """Get activations from specified layer or all layers."""
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        
        # Register hooks
        hooks = []
        linear_layers = [(name, m) for name, m in self.network.named_modules() if isinstance(m, nn.Linear)]
        
        for i, (name, module) in enumerate(linear_layers):
            if layer is None or i == layer:
                hooks.append(module.register_forward_hook(hook_fn(f"layer_{i}")))
        
        # Forward pass
        X_tensor = torch.FloatTensor(X[0]).to(self.device)
        with torch.no_grad():
            self.network(X_tensor)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Convert to numpy
        result = {k: v.cpu().numpy() for k, v in activations.items()}
        return result
    
    def create_dataloader(
        self,
        X: List[np.ndarray],
        y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True
    ) -> DataLoader:
        """Create PyTorch DataLoader."""
        X_tensor = torch.FloatTensor(X[0])
        y_tensor = torch.FloatTensor(y)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    @staticmethod
    def compare(model1: 'Model', model2: 'Model') -> Dict:
        """Compare two models."""
        diff = {
            'architecture_match': model1.architecture == model2.architecture,
            'param_count_1': model1.count_parameters(),
            'param_count_2': model2.count_parameters(),
            'device_1': model1.get_device(),
            'device_2': model2.get_device()
        }
        return diff
    
    def equals(self, other: 'Model') -> bool:
        """Check if two models are identical."""
        if self.architecture != other.architecture:
            return False
        
        # Compare state dicts
        for (k1, v1), (k2, v2) in zip(
            self.network.state_dict().items(),
            other.network.state_dict().items()
        ):
            if k1 != k2:
                return False
            if not torch.equal(v1, v2):
                return False
        
        return True
    
    @classmethod
    def quick_fit(
        cls,
        X: List[np.ndarray],
        y: np.ndarray,
        task: str = "regression",
        hidden_layers: Optional[List[int]] = None,
        epochs: int = 100
    ) -> 'Model':
        """
        Quick model training with sensible defaults.
        
        Args:
            X: Input data
            y: Target data
            task: 'regression' or 'classification'
            hidden_layers: Hidden layer sizes (auto if None)
            epochs: Training epochs
            
        Returns:
            Trained model
        """
        # Auto determine architecture
        input_size = X[0].shape[1]
        output_size = y.shape[1] if len(y.shape) > 1 else 1
        
        if hidden_layers is None:
            hidden_layers = [max(input_size * 2, 32), max(input_size, 16)]
        
        architecture = [input_size] + hidden_layers + [output_size]
        
        # Determine activation and loss
        if task == "classification":
            activation = "relu"
            loss = "cross_entropy" if output_size > 1 else "bce"
        else:
            activation = "relu"
            loss = "mse"
        
        # Create and train model
        model = cls(architecture, activation=activation)
        model.train([X[0]], y, epochs=epochs, loss=loss, verbose=1)
        
        return model


# Utility function for one-hot encoding
def one_hot_encode(y: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
    """One-hot encode labels."""
    if num_classes is None:
        num_classes = int(np.max(y)) + 1
    
    encoded = np.zeros((len(y), num_classes))
    encoded[np.arange(len(y)), y.astype(int)] = 1
    return encoded


# Example usage
if __name__ == "__main__":
    print("NeuralNet Library - Example Usage")
    print("=" * 50)
    
    # Generate sample data
    np.random.seed(42)
    X_train = np.random.randn(1000, 10)
    y_train = np.random.randn(1000, 1)
    X_test = np.random.randn(200, 10)
    y_test = np.random.randn(200, 1)
    
    # Create model
    print("\n1. Creating model...")
    model = Model([10, 64, 32, 1], activation="relu", dropout=0.2)
    model.summary()
    
    # Train model
    print("\n2. Training model...")
    model.train(
        [X_train], y_train,
        epochs=50,
        batch_size=32,
        validation_data=([X_test], y_test),
        verbose=1
    )
    
    # Extract layer weights
    print("\n3. Extracting layer weights...")
    weights = model.extract_layer(1)
    print(f"Layer 1 weights shape: {weights['weights'].shape}")
    
    # Make predictions
    print("\n4. Making predictions...")
    predictions = model.predict([X_test])
    print(f"Predictions shape: {predictions.shape}")
    
    # Evaluate
    print("\n5. Evaluating model...")
    loss, metrics = model.evaluate([X_test], y_test, metrics=['mse', 'mae'])
    print(f"Test Loss: {loss:.4f}")
    print(f"Metrics: {metrics}")
    
    # Save model
    print("\n6. Saving model...")
    model.save("example_model.pt")
    
    # Load model
    print("\n7. Loading model...")
    loaded_model = Model.load("example_model.pt")
    
    # Verify loaded model
    print("\n8. Verifying loaded model...")
    loaded_predictions = loaded_model.predict([X_test])
    print(f"Predictions match: {np.allclose(predictions, loaded_predictions)}")
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")