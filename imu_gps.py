import os
import utm
import random
import folium
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
        
    def predict(self, acceleration):
        if acceleration is not None:
            ax, ay = acceleration
            u = np.array([[ax], [ay]])
            self.x = self.F @ self.x + self.B @ u
        else:
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
        
        self.B = np.array([
            [0.5 * dt ** 2, 0],
            [0, 0.5 * dt ** 2],
            [dt, 0],
            [0, dt]
        ])
        
    def get_state(self):
        return self.x[:2]
    
    def get_velocity(self):
        return self.x[2:4]
    
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
        
        # Extract acceleration values from IMU
        ax_values = gi_data['ax'].to_numpy()
        ay_values = gi_data['ay'].to_numpy()
        
        # simulate dropout
        gps_drop_indices = set(random.sample(range(len(cartisian_coordinate)), 50))  # drop 50 random GPS readings
        
        filtered_positions = []
        headings = []
        export_csv = []
        
        for i in range(1, len(cartisian_coordinate)):
            # compute actual dt
            dt = (timestamps[i] - timestamps[i-1]).total_seconds()
            
            # update Kalman matrices
            kf.update_matrices(dt)
            
            # predict next state using IMU
            acceleration = (ax_values[i], ay_values[i])
            kf.predict(acceleration=acceleration)
            
            # update using GPS measurement
            if i not in gps_drop_indices:
                z = np.array(cartisian_coordinate[i]).reshape(2, 1)
                kf.update(z)
                gps_used = True
            else: 
                gps_used = False
                
            # store filtred position
            x_filtered, y_filtered = kf.get_state().flatten()
            vx, vy = kf.get_velocity().flatten()
            
            theta = np.degrees(np.arctan2(vy, vx))
            
            export_csv.append({
                'timestamp': timestamps[i].strftime('%Y-%m-%d %H:%M:%S.%f'), 
                'x': x_filtered, 
                'y': y_filtered, 
                'vx':vx, 
                'vy': vy, 
                'heading_deg': theta, 
                'gps_used': gps_used
            })
            
            filtered_positions.append((x_filtered, y_filtered))
            headings.append(theta)
            
        export_csv = pd.DataFrame(export_csv)
        export_csv.to_csv("./filtered_trajectory_with_heading.csv", index=False)
        print(f"filterd positons:\n {filtered_positions[:5]}")
        
        raw_x = [pt[0] for pt in cartisian_coordinate]
        raw_y = [pt[1] for pt in cartisian_coordinate]
        
        filtered_x = [pt[0] for pt in filtered_positions]
        filtered_y = [pt[1] for pt in filtered_positions]
        arrow_u = [np.cos(theta) for theta in headings]  # x component
        arrow_v = [np.sin(theta) for theta in headings]  # y component

        plt.figure(figsize=(10, 6))
        plt.plot(raw_x, raw_y, label="Raw GPS", linestyle='--', alpha=0.6)
        plt.plot(filtered_x, filtered_y, label="Filtered Trajectory", linewidth=2)
        # Plot heading arrows every N steps to avoid clutter
        N = max(1, min(20, len(filtered_x) // 5))  # adaptive step size
        plt.quiver(
            filtered_x[::N], filtered_y[::N],
            arrow_u[::N], arrow_v[::N],
            angles='xy', scale_units='xy', scale=0.6,
            color='red', width=0.008, headwidth=4, headlength=6,
            label="Heading Direction"
        )
        # Highlight dropout zones
        for i in gps_drop_indices:
            if i < len(filtered_positions):
                plt.plot(filtered_positions[i][0], filtered_positions[i][1], 'rx', markersize=5, label="GPS Dropped" if i == min(gps_drop_indices) else "")

        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title("IMU-Aided Kalman Filter Trajectory vs. Raw GPS Path")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()
        
        # --- Real Map Visualization ---
        # 1. Set origin from the first GPS point
        origin_lat = gi_data['lat'].iloc[0]
        origin_lon = gi_data['lon'].iloc[0]
        origin_easting, origin_northing, zone_number, zone_letter = utm.from_latlon(origin_lat, origin_lon)

        # 2. Convert filtered positions back to (lat, lon)
        filtered_latlon = [
            utm.to_latlon(origin_easting + x, origin_northing + y, zone_number, zone_letter)
            for x, y in filtered_positions
        ]
        raw_latlon = list(zip(gi_data['lat'], gi_data['lon']))

        m = folium.Map(location=filtered_latlon[0], zoom_start=17)
        folium.PolyLine(filtered_latlon, color='red', weight=3, tooltip='Filtered Trajectory').add_to(m)
        folium.PolyLine(raw_latlon, color='blue', weight=1, tooltip='Raw GPS').add_to(m)
        folium.Marker(location=filtered_latlon[0], tooltip="Start", icon=folium.Icon(color='green')).add_to(m)
        folium.Marker(location=filtered_latlon[-1], tooltip="End", icon=folium.Icon(color='red')).add_to(m)
        m.save("./trajectory_map.html")

        # --- Speed Plot ---
        speeds = [np.sqrt(row['vx']**2 + row['vy']**2) for _, row in export_csv.iterrows()]
        plt.figure(figsize=(10, 4))
        plt.plot(speeds, label="Speed [m/s]", color='green')
        plt.title("Speed Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Speed [m/s]")
        plt.grid(True)
        plt.legend()
        plt.show()

        # --- Error Plot ---
        errors = [np.sqrt((fx - rx) ** 2 + (fy - ry) ** 2)
                for (fx, fy), (rx, ry) in zip(filtered_positions, cartisian_coordinate[1:])]
        plt.figure(figsize=(10, 4))
        plt.plot(errors, label="Position Error [m]", color='orange')
        plt.title("Position Error Between Raw GPS and Filtered Trajectory")
        plt.xlabel("Time Step")
        plt.ylabel("Error [m]")
        plt.grid(True)
        plt.legend()
        plt.show()