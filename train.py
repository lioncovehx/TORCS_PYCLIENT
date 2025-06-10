import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from model import DrivingModel
import os
import torch.nn.functional as F

class DrivingDataset(Dataset):
    def __init__(self, csv_file):
        # Load the CSV file
        self.data = pd.read_csv(csv_file)
        
        # Track sensors (19 values)
        track_cols = [f'track_{i}' for i in range(19)]
        
        # Opponent sensors (36 values)
        opponent_cols = [f'opponent_{i}' for i in range(36)]
        
        # Speed and RPM related features
        speed_cols = ['speedX', 'speedY', 'speedZ', 'rpm']
        
        # Wheel spin velocities (4 values)
        wheel_spin_cols = [f'wheelSpinVel_{i}' for i in range(4)]
        
        # Additional useful features
        extra_cols = ['angle', 'trackPos', 'is_recovering']
        
        # Target values (what we want to predict)
        target_cols = ['steer', 'accel', 'brake', 'gear']
        
        # Combine all feature columns
        feature_cols = track_cols + opponent_cols + speed_cols + wheel_spin_cols + extra_cols
        
        # Verify feature count
        assert len(feature_cols) == 66, f"Expected 66 features, got {len(feature_cols)}"
        
        # Extract features and targets
        self.features = self.data[feature_cols].values
        self.targets = self.data[target_cols].values
        
        # Calculate normalization parameters
        self.means = np.mean(self.features, axis=0)
        self.stds = np.std(self.features, axis=0) + 1e-8  # Add small epsilon to avoid division by zero
        
        # Save normalization parameters for the driver
        np.save('means.npy', self.means)
        np.save('stds.npy', self.stds)
        
        # Normalize features
        self.features = (self.features - self.means) / self.stds
        
        # Convert to torch tensors
        self.features = torch.FloatTensor(self.features)  # Shape: [num_samples, 66]
        self.targets = torch.FloatTensor(self.targets)    # Shape: [num_samples, 4]
        
        print(f"Loaded dataset with {len(self.features)} samples")
        print(f"Feature shape: {self.features.shape}")
        print(f"Target shape: {self.targets.shape}")
        print("\nFeature breakdown:")
        print(f"- Track sensors: {len(track_cols)}")
        print(f"- Opponent sensors: {len(opponent_cols)}")
        print(f"- Speed/RPM: {len(speed_cols)}")
        print(f"- Wheel spin: {len(wheel_spin_cols)}")
        print(f"- Additional features: {len(extra_cols)}")
        print(f"Total features: {len(feature_cols)}")
        
        # Print feature ranges for verification
        print("\nFeature ranges before normalization:")
        for i, col in enumerate(feature_cols):
            min_val = self.data[col].min()
            max_val = self.data[col].max()
            print(f"{col}: [{min_val:.2f}, {max_val:.2f}]")
        
        print("\nFeature ranges after normalization:")
        for i, col in enumerate(feature_cols):
            min_val = self.features[:, i].min().item()
            max_val = self.features[:, i].max().item()
            print(f"{col}: [{min_val:.2f}, {max_val:.2f}]")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]  # Returns [66], [4]

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss, _ = model.get_loss(outputs, batch_y, batch_x)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss, _ = model.get_loss(outputs, batch_y, batch_x)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print('-' * 50)

def main():
    # Hyperparameters
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    
    # Create model with correct input size (66 features)
    model = DrivingModel(input_size=66)
    
    # Load dataset
    dataset = DrivingDataset('merged_data.csv')
    
    # Print model summary
    print("\nModel Summary:")
    print(f"Input size: {model.input_size}")
    print(f"Feature breakdown:")
    print("- Track sensors: 19")
    print("- Opponent sensors: 36")
    print("- Speed/RPM: 4")
    print("- Wheel spin: 4")
    print("- Additional features: 3")
    print(f"Total features: 66")
    
    # Split into train and validation sets (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Train the model
    train_model(model, train_loader, val_loader, num_epochs, learning_rate)

if __name__ == "__main__":
    main() 