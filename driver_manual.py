import msgParser
import carState
import carControl
import csv
import os
import time
import threading
from pynput import keyboard

class Driver(object):
    def __init__(self, stage):
        self.stage = stage
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()
        
        self.data_file = "telemetry_roadtrackWheel1_normaldrive.csv"  # Change this manually for different scenarios
        
        # Manual control variables
        self.manual_steer = 0.0
        self.manual_accel = 0.0
        self.manual_brake = 0.0
        self.running = True
        
        self.steer_sensitivity = 0.25  
        self.steer_lock = 0.5  
        self.steering_damping = 0.75  

        self.prev_rpm = None
        self.max_speed = 300  

        # Initializing flags for key states
        self.key_state = {
            'up': False,
            'down': False,
            'left': False,
            'right': False,
            'z': False,
            'x': False
        }
        self.prev_shift_up = False
        self.prev_shift_down = False

        # Create a thread that runs continuously to handle keyboard input
        self.input_thread = threading.Thread(target=self.keyboard_input_loop, daemon=True)
        self.input_thread.start()
        
        self.gear_thread = threading.Thread(target=self.gear_shift_loop, daemon=True)
        self.gear_thread.start()

        # Start listening to the keyboard inputs
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener.start()

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

    def on_press(self, key):
        try:
            if hasattr(key, 'char') and key.char:
                key_char = key.char.lower()
                if key_char in self.key_state:
                    self.key_state[key_char] = True
                    #print(f"[Key Press] {key_char} key pressed")
            elif key == keyboard.Key.up:
                self.key_state['up'] = True
            elif key == keyboard.Key.down:
                self.key_state['down'] = True
            elif key == keyboard.Key.left:
                self.key_state['left'] = True
            elif key == keyboard.Key.right:
                self.key_state['right'] = True
        except Exception as e:
            print(f"Error in on_press: {e}")

    def on_release(self, key):
        try:
            if hasattr(key, 'char') and key.char:
                key_char = key.char.lower()
                if key_char in self.key_state:
                    self.key_state[key_char] = False
                    #print(f"[Key Release] {key_char} key released")
            elif key == keyboard.Key.up:
                self.key_state['up'] = False
            elif key == keyboard.Key.down:
                self.key_state['down'] = False
            elif key == keyboard.Key.left:
                self.key_state['left'] = False
            elif key == keyboard.Key.right:
                self.key_state['right'] = False
        except Exception as e:
            print(f"Error in on_release: {e}")

    def keyboard_input_loop(self):
        """Continuously handles keyboard input in a separate thread."""
        while self.running:
            # Acceleration
            if self.key_state['up']:
                self.manual_accel = min(self.manual_accel + 0.1, 1.0)
                #print(f'Accelerating: {self.manual_accel}')  # Debug print to track acceleration
            else:
                self.manual_accel = max(self.manual_accel - 0.1, 0.0)

            # Braking
            if self.key_state['down']:
                self.manual_brake = min(self.manual_brake + 0.1, 1.0)
                #print(f'Braking: {self.manual_brake}')  # Debug print to track brake
            else:
                self.manual_brake = max(self.manual_brake - 0.1, 0.0)

            # Steering
            steering_input = 0
            if self.key_state['left']:
                steering_input = self.steer_sensitivity
            elif self.key_state['right']:
                steering_input = -self.steer_sensitivity
            
            try:
                track_pos = self.state.getTrackPos() or 0
                car_angle = self.state.getAngle() or 0
                
                # Minimal track and angle compensation
                adjusted_steer = (steering_input - track_pos * 0.1 - car_angle * 0.05) / self.steer_lock
                
                # Apply soft damping
                if steering_input != 0:
                    self.manual_steer = self.manual_steer * self.steering_damping + adjusted_steer * (1 - self.steering_damping)
                else:
                    # Gradually return to center with more stability
                    self.manual_steer *= 0.8
                    if abs(self.manual_steer) < 0.05:
                        self.manual_steer = 0.0
                
                # Ensure steering stays within bounds
                self.manual_steer = max(min(self.manual_steer, 1.0), -1.0)
                
                #print(f'Steering: {self.manual_steer}')  # Debug print to track steering
                
            except Exception as e:
                #print(f"Error in steering: {e}")
                self.manual_steer = steering_input
            # Small delay to prevent high CPU usage
            time.sleep(0.05)
    def gear_shift_loop(self):
        while self.running:
            shift_up = self.key_state['z']
            shift_down = self.key_state['x']
            gear = self.state.getGear()
            speed_x = self.state.getSpeedX()
            speed_x = speed_x if speed_x is not None else 0

            #print(f"[Gear Thread] Current Gear: {gear}, SpeedX: {speed_x:.2f}, Brake: {self.manual_brake:.2f}, Z: {shift_up}, X: {shift_down}")

            if shift_up and not self.prev_shift_up and gear < 6:
                gear += 1
                #print(f"[Gear Thread] >> Gear shifted UP to {gear}")

            if shift_down and not self.prev_shift_down and gear > 1:
                gear -= 1
                #print(f"[Gear Thread] >> Gear shifted DOWN to {gear}")

            if abs(speed_x) < 1 and self.manual_brake > 0.5:
                gear = -1
                #print(f"[Gear Thread] >> Shifted to REVERSE")

            self.control.setGear(gear)

            self.prev_shift_up = shift_up
            self.prev_shift_down = shift_down

            time.sleep(0.05)

    def drive(self, msg):
        self.state.setFromMsg(msg)
        self.log_data()
        self.apply_manual_control()
        return self.control.toMsg()

    def manual_gear_shift(self):
        rpm = self.state.getRpm()
        gear = self.state.getGear()
        speed_x = self.state.getSpeedX()

        shift_up = self.key_state['z']
        shift_down = self.key_state['x']

        if shift_up and gear < 6:
           gear += 1

        if shift_down and gear > 1:
           gear -= 1

        if abs(speed_x) < 1 and self.manual_brake > 0.5:
           gear = -1  # Shift to reverse

        self.control.setGear(gear)

    def log_data(self):
        """Logs car sensor data into a CSV file."""
        try:
            file_exists = os.path.isfile(self.data_file)
            
            with open(self.data_file, mode='a', newline='') as file:
                writer = csv.writer(file)

                # Write header only once
                if not file_exists:
                    writer.writerow([
                        "angle", "trackPos", "speedX", "speedY", "speedZ", "rpm", "gear", 
                        "steer", "accel", "brake", "curLapTime", "lastLapTime", "racePos","DistfromStart",
                        *[f"wheelSpinVel_{i}" for i in range(4)],
                        *[f"track_{i}" for i in range(19)],  
                        *[f"opponent_{i}" for i in range(36)],
                        "COGfromTrack"
                    ])
                    print('HEADER WROTEEEEEEE')

                # Write sensor data using only available getters
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
                    self.state.getCurLapTime(),
                    self.state.getLastLapTime(),
                    self.state.getRacePos(),
                    self.state.getDistFromStart(),
                    *self.state.getWheelSpinVel(),
                    *self.state.getTrack(),
                    *self.state.getOpponents(),
                    self.state.getZ()
                ])
                #print('logging 5')  
        except Exception as e:
            print(f"Error logging data: {e}")

    def apply_manual_control(self):
        """Applies manual inputs from the keyboard"""
        self.control.setSteer(self.manual_steer)
        self.control.setAccel(self.manual_accel)
        self.control.setBrake(self.manual_brake)
    
    def onShutDown(self):
        self.running = False
        self.listener.stop()
        self.listener.join()
    
    def onRestart(self):
        pass
