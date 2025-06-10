import carState
import msgParser
import carControl
import torch
import numpy as np
import os
import csv
from model import DrivingModel

class Driver(object):
    def __init__(self, stage):
        self.stage = stage
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load normalization parameters
        self.means = np.load('means.npy')
        self.stds = np.load('stds.npy') + 1e-8
        
        # Load the model with correct input size (66 features)
        self.model = DrivingModel(input_size=66)  # 19 track + 36 opponent + 4 speed + 4 wheel + 3 extra
        try:
            self.model.load_state_dict(torch.load("best_model.pth", map_location=self.device))
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Continuing with randomly initialized weights.")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Data logging
        self.data_file = "telemetry_roadtrackWheel1_AI_DRIVING.csv"
        
        # Initialize control values
        self.steer = 0.0
        self.accel = 0.2  # Start with moderate acceleration
        self.brake = 0.0
        self.gear = 1  # Start in gear 1
        
        # Set initial gear
        self.control.setGear(1)
        
        # Wheel spin thresholds
        self.wheel_spin_threshold = 100  # Threshold for detecting wheel spin
        self.prev_wheel_spins = [0, 0, 0, 0]  # Track previous wheel spins for detection

    def init(self):
        '''Return init string with rangefinder angles'''
        self.angles = [0 for x in range(19)]
        
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        
        for i in range(5, 9):
            self.angles[i] = -20 + (i-5) * 5
            self.angles[18 - i] = 20 - (i-5) * 5
            
        return self.parser.stringify({'init': self.angles})

    def extract_features(self, carstate):
        """Extract and normalize features for the model"""
        # Track sensors (19 values)
        track = carstate.getTrack()
        
        # Opponent sensors (36 values)
        opponents = carstate.getOpponents()
        
        # Speed and RPM (4 values)
        speed_x = carstate.getSpeedX()
        speed_y = carstate.getSpeedY()
        speed_z = carstate.getSpeedZ()
        rpm = carstate.getRpm()
        
        # Wheel spin velocities (4 values)
        wheel_spins = carstate.getWheelSpinVel()
        
        # Additional features (3 values)
        angle = carstate.getAngle()
        track_pos = carstate.getTrackPos()
        
        # Check if car is recovering (off track, spinning, or wheel spin)
        is_recovering = (
            abs(track_pos) > 1.0 or  # Off track
            sum(track) < 1.0 or  # No track sensors detecting track
            any(abs(spin) > self.wheel_spin_threshold for spin in wheel_spins)  # Excessive wheel spin
        )
        
        # Combine all features
        features = [
            *track,  # 19 track sensors
            *opponents,  # 36 opponent sensors
            speed_x, speed_y, speed_z, rpm,  # 4 speed/RPM values
            *wheel_spins,  # 4 wheel spin values
            angle, track_pos, 1 if is_recovering else 0  # 3 additional features
        ]
        
        return np.array(features, dtype=np.float32)

    def log_data(self):
        """Log car sensor data to CSV file"""
        try:
            file_exists = os.path.isfile(self.data_file)
            
            with open(self.data_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                
                if not file_exists:
                    writer.writerow([
                        "angle", "trackPos", "speedX", "speedY", "speedZ", "rpm", "gear", 
                        "steer", "accel", "brake",
                        *[f"wheelSpinVel_{i}" for i in range(4)],
                        *[f"track_{i}" for i in range(19)],
                        *[f"opponent_{i}" for i in range(36)]
                    ])
                
                writer.writerow([
                    self.state.getAngle(),
                    self.state.getTrackPos(),
                    self.state.getSpeedX(),
                    self.state.getSpeedY(),
                    self.state.getSpeedZ(),
                    self.state.getRpm(),
                    self.state.getGear(),
                    self.control.getSteer(),
                    self.control.getAccel(),
                    self.control.getBrake(),
                    *self.state.getWheelSpinVel(),
                    *self.state.getTrack(),
                    *self.state.getOpponents()
                ])
        except Exception as e:
            print(f"Error logging data: {e}")

    def drive(self, msg):
        """Main driving function"""
        self.state.setFromMsg(msg)
        
        # Extract and normalize features
        features = self.extract_features(self.state)
        normalized_features = (features - self.means) / self.stds
        
        # Convert to tensor and add batch dimension
        input_tensor = torch.FloatTensor(normalized_features).unsqueeze(0).to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(input_tensor)
            steer, accel, brake, gear_logits = outputs
            
            # Convert to numpy and get gear
            steer = steer.cpu().numpy()[0]
            accel = accel.cpu().numpy()[0]
            brake = brake.cpu().numpy()[0]
            
            # Get gear probabilities and convert to actual gear
            gear_probs = torch.softmax(gear_logits, dim=-1).cpu().numpy()[0]
            gear = np.argmax(gear_probs) - 1  # Convert from 0-7 to -1 to 6
            
            # Never allow gear 0 (neutral) unless explicitly needed
            if gear == 0:
                gear = 1
            
            # Debug prints
            #print(f"Model Outputs - Steer: {steer:.3f}, Accel: {accel:.3f}, Brake: {brake:.3f}, Gear: {gear}")
            #print(f"Gear probabilities: {gear_probs}")
        
        # Apply controls directly from model
        self.steer = float(steer)
        self.accel = float(accel)
        self.brake = float(brake)
        self.gear = int(gear)
        
        # Apply controls
        self.control.setSteer(self.steer)
        self.control.setAccel(self.accel)
        self.control.setBrake(self.brake)
        self.control.setGear(self.gear)
        
        # Log data
        #self.log_data()
        
        return self.control.toMsg()
