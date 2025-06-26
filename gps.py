import os
import cv2 as cv
import numpy as np
import pandas as pd
from pyproj import Transformer
from datetime import datetime
import matplotlib.pyplot as plt

# GPS-only localization with Kalman
tf = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

class Custom_Kalman_Filter:
    def __init__(self, dt, process_noise=1.0, measurement_noise=5.0):
        
        # State vector [x, y, vx, vy]^T
        self.x = np.zeros((4,1)) # Initially assume zero position and velocity
        
        # Covariance matrix (initial uncertainty)
        self.P = np.eye(4) * 1000 # High uncertainty initially
        
        #  Measurement matrix (we only observe x and y)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])
        
        # Measurement noise covariance (R)
        self.R = np.eye(2) * (measurement_noise**2)
        
        # Identity matrix (for covariance update)
        self.I = np.eye(4)
        
        # Store noise parameter
        self.process_noise = process_noise
        
        # Initialize F and Q with initial dt
        self.update_matrices(dt)
        
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, z):
        y = z - self.H @ self.x # Innovation (residual)
        s = self.H @ self.P @ self.H.T + self.R # Innovation covariance
        k = self.P @ self.H.T @ np.linalg.inv(s) # kalman gain
        self.x = self.x + k @ y # State update
        self.P = (self.I - k @ self.H) @ self.P # Covariance update
    
    def update_matrices(self, dt):
        # State transition matrix (dynamic)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        # Process noise covariance (Q)
        q = self.process_noise ** 2
        
        dt1 = dt * dt
        dt2 = dt1 * dt
        dt3 = dt2 * dt
        
        self.Q = q * np.array([
            [dt3/4, 0, dt2/2, 0],
            [0, dt3/4, 0, dt2/2],
            [dt2/2, 0, dt1, 0],
            [0, dt2/2, 0, dt1]
        ])
        
    def get_state(self):
        return self.x[:2]
    
# Parse Data Function to Exctract the data of GPS+IMU

def parse_data(dataset):
    with open(os.path.join(dataset, 'timestamps.txt'), 'r') as f:
        time_stamp = [line.strip() for line in f.readlines()]
        
    data = []
    dataset = os.path.join(dataset, 'data')
    for i, file_name in enumerate(sorted(os.listdir(dataset))): # each time access to file in the dataset folder and then open it to read data from it
        with open(os.path.join(dataset, file_name), "r") as f:
            line = f.readline() # read each line in the file
            values = list(map(float, line.strip().split())) # convert the read data into list format
            # Extract the relevent attribute that we will need it in this project
            attributes = {
                "timestamp": time_stamp[i],
                "lat": values[0],
                "lon": values[1],
                "alt": values[2],
                "ax": values[11],
                "ay": values[12],
                "az": values[13]
            }
            data.append(attributes)
            
    df = pd.DataFrame(data)
    df.to_csv("./gps_imu_data.csv", index=False)
    return df

def convert_to_cartisian(gps_data):
    linear_positions = []
    lat0, lon0, _ = gps_data[0] # extract the first lat and lon to use it as orign
    x0, y0= tf.transform(lon0, lat0)
    
    for lat, lon, _ in gps_data:
        x, y= tf.transform(lon, lat)
        rel_x = x - x0
        rel_y = y - y0 
        linear_positions.append((rel_x, rel_y))
    
    return linear_positions

if __name__=="__main__":
    if not os.path.exists("./gps_imu_data.csv"):
        data_path = "./kitti_data"
        gi_data = parse_data(data_path)
        print(f"Parsed {len(gi_data)} frames.")
        print(gi_data.head())
    else:
        gi_data = pd.read_csv("./gps_imu_data.csv", sep=',')
        print(gi_data[['lat', 'lon', 'alt']].head())
        cartisian_coordinate = convert_to_cartisian(gi_data[['lat', 'lon', 'alt']].to_numpy())
        print(cartisian_coordinate[:5])
        
        gi_data['timestamp'] = pd.to_datetime(gi_data['timestamp'])
        timestamps = gi_data['timestamp'].tolist()
        kf = Custom_Kalman_Filter(dt=0.1, process_noise=1.0, measurement_noise=5.0)

        # Initialize position from first GPS point
        kf.x[0, 0] = cartisian_coordinate[0][0]
        kf.x[1, 0] = cartisian_coordinate[0][1]
        
        filtered_positions = []
        
        for i in range(1, len(cartisian_coordinate)):
            # compute actual dt
            dt = (timestamps[i] - timestamps[i-1]).total_seconds()
            
            # update Kalman matrices
            kf.update_matrices(dt)
            
            # predict next state
            kf.predict()
            
            # update using GPS measurement
            z = np.array(cartisian_coordinate[i]).reshape(2, 1)
            kf.update(z)
            # store filtred position
            x_filtered, y_filtered = kf.get_state().flatten()
            filtered_positions.append((x_filtered, y_filtered))
        
        print(f"filterd positons:\n {filtered_positions[:5]}")
        
        raw_x = [pt[0] for pt in cartisian_coordinate]
        raw_y = [pt[1] for pt in cartisian_coordinate]
        
        filtered_x = [pt[0] for pt in filtered_positions]
        filtered_y = [pt[1] for pt in filtered_positions]
        
        plt.figure(figsize=(10, 6))
        plt.plot(raw_x, raw_y, label="Raw GPS", linestyle='--', alpha=0.6)
        plt.plot(filtered_x, filtered_y, label="Filtered Trajectory", linewidth=2)
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title("GPS Trajectory vs Kalman Filter Output")
        plt.legend()
        plt.grid(True)
        plt.show()