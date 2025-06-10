import torch
import torch.nn as nn
import torch.nn.functional as F

class AccelScale(nn.Module):
    def __init__(self):
        super(AccelScale, self).__init__()
    
    def forward(self, x):
        return x * 0.7 + 0.3  # Scale to range [0.3, 1.0]

class FeatureAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FeatureAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: [batch_size, input_size]
        attention_weights = self.attention(x)
        return x * attention_weights

class DrivingModel(nn.Module):
    def __init__(self, input_size=66, hidden_size=128, dropout=0.3):
        super(DrivingModel, self).__init__()
        
        # Store input size as instance variable
        self.input_size = input_size
        
        # Loss weights - focus on gear shifting and steering
        self.steer_weight = 2.0  # Increased for better steering
        self.accel_weight = 1.0
        self.brake_weight = 1.0
        self.gear_weight = 4.0  # Increased for better gear shifting
        
        # Track position processor (19 sensors)
        self.track_processor = nn.Sequential(
            nn.Linear(19, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Opponent processor (36 sensors)
        self.opponent_processor = nn.Sequential(
            nn.Linear(36, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Speed and RPM processor (4 values) - simplified for gear prediction
        self.speed_processor = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Wheel spin processor (4 values)
        self.wheel_spin_processor = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Additional features processor (3 values: angle, trackPos, is_recovering)
        self.extra_processor = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(160, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output heads
        # Gear: 8 classes (-1 to 6) with direct speed/RPM input
        self.gear_head = nn.Sequential(
            nn.Linear(hidden_size + 32, 64),  # Added speed features directly
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 8)  # 8 classes for gears (-1 to 6)
        )
        
        # Steering: continuous value with direct track and angle info
        self.steer_head = nn.Sequential(
            nn.Linear(hidden_size + 32 + 3, 64),  # Added track features and angle/trackPos directly
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Tanh()  # Bound between -1 and 1
        )
        
        # Acceleration: binary output
        self.accel_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Binary output
        )
        
        # Brake: binary output
        self.brake_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Binary output
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Process each feature group
        track_features = self.track_processor(x[:, :19])  # First 19 are track sensors
        opponent_features = self.opponent_processor(x[:, 19:55])  # Next 36 are opponent sensors
        speed_features = self.speed_processor(x[:, 55:59])  # Next 4 are speed/RPM
        wheel_spin_features = self.wheel_spin_processor(x[:, 59:63])  # Next 4 are wheel spins
        extra_features = self.extra_processor(x[:, 63:66])  # Last 3 are extra features
        
        # Combine all features
        combined_features = torch.cat([
            track_features,      # 32
            opponent_features,   # 32
            speed_features,      # 32
            wheel_spin_features, # 32
            extra_features       # 32
        ], dim=-1)  # Total: 160 features
        
        # Process through shared layers
        shared_features = self.shared(combined_features)
        
        # Get outputs for each control
        # Gear with direct speed features
        gear_logits = self.gear_head(torch.cat([shared_features, speed_features], dim=-1))
        
        # Steering with direct track features and angle/trackPos
        steer = self.steer_head(torch.cat([
            shared_features,
            track_features,  # Direct track sensor info
            x[:, 63:66]     # Direct angle and trackPos
        ], dim=-1)).squeeze(-1)
        
        # Other controls
        accel = self.accel_head(shared_features).squeeze(-1)
        brake = self.brake_head(shared_features).squeeze(-1)
        
        return steer, accel, brake, gear_logits
    
    def get_loss(self, outputs, targets, features=None):
        steer, accel, brake, gear_logits = outputs
        
        # Clamp predictions to valid ranges
        steer = torch.clamp(steer, -1.0, 1.0)
        accel = torch.clamp(accel, 0.0, 1.0)
        brake = torch.clamp(brake, 0.0, 1.0)
        
        # Convert gear targets to class indices (0-7 for gears -1 to 6)
        gear_targets = torch.round(targets[:, 3]).clamp(-1, 6).long() + 1  # Shift to 0-7 range
        
        # Calculate individual losses
        steer_loss = F.mse_loss(steer, targets[:, 0])
        accel_loss = F.binary_cross_entropy(accel, targets[:, 1])  # Changed to BCE
        brake_loss = F.binary_cross_entropy(brake, targets[:, 2])  # Changed to BCE
        
        # Enhanced gear loss with label smoothing and class weights
        class_weights = torch.ones(8, device=gear_logits.device)
        class_weights[0] = 1.5  # Weight for neutral gear (-1)
        class_weights[1] = 1.2  # Weight for first gear
        class_weights[6] = 1.2  # Weight for sixth gear
        gear_loss = F.cross_entropy(gear_logits, gear_targets, 
                                  label_smoothing=0.1,
                                  weight=class_weights)
        
        # Add penalty for incorrect gear shifts
        current_gear = targets[:, 3]
        next_gear = torch.roll(current_gear, -1)
        gear_diff = torch.abs(next_gear - current_gear)
        gear_shift_penalty = torch.mean((gear_diff > 1).float())
        
        # Add speed-based braking penalty if features are provided
        if features is not None:
            # speedX is at index 55 (after track sensors (19) + opponent sensors (36))
            speed = features[:, 55]
            speed_threshold = 0.5  # Normalized speed threshold
            brake_penalty = torch.mean(((speed > speed_threshold) & (brake < 0.5)).float())
            
            # Add acceleration-brake consistency penalty
            consistency_penalty = torch.mean(((accel > 0.5) & (brake > 0.5)).float())
        else:
            brake_penalty = torch.tensor(0.0, device=targets.device)
            consistency_penalty = torch.tensor(0.0, device=targets.device)
        
        # Combine losses with weights
        total_loss = (
            self.steer_weight * steer_loss +
            self.accel_weight * accel_loss +
            self.brake_weight * (brake_loss + brake_penalty + consistency_penalty) +
            self.gear_weight * (gear_loss + 0.5 * gear_shift_penalty)
        )
        
        # Return both total loss and individual losses for monitoring
        return total_loss, {
            'steer_loss': steer_loss.item(),
            'accel_loss': accel_loss.item(),
            'brake_loss': brake_loss.item(),
            'gear_loss': gear_loss.item(),
            'gear_shift_penalty': gear_shift_penalty.item(),
            'brake_penalty': brake_penalty.item(),
            'consistency_penalty': consistency_penalty.item()
        }
    
    def predict(self, x):
        with torch.no_grad():
            steer, accel, brake, gear_logits = self.forward(x)
            
            # Convert gear logits to actual gear values (-1 to 6)
            gear_pred = torch.argmax(gear_logits, dim=-1) - 1  # Shift back to -1 to 6 range
            
            # Clamp continuous predictions to valid ranges
            steer = torch.clamp(steer, -1.0, 1.0)
            accel = torch.clamp(accel, 0.0, 1.0)
            brake = torch.clamp(brake, 0.0, 1.0)
            
            return torch.stack([steer, accel, brake, gear_pred.float()], dim=-1)
