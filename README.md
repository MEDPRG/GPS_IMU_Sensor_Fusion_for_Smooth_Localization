# 🛰️ GPS+IMU Sensor Fusion using Kalman Filter

This project implements a **sensor fusion algorithm** for estimating accurate 2D trajectories using raw **GPS and IMU data**. By applying a custom-built **Kalman Filter**, the system combines noisy, low-frequency GPS data with high-rate inertial measurements (acceleration from IMU) to generate a **smooth, robust, and consistent trajectory**, even during GPS dropouts.

---

## 📚 Table of Contents

- [📌 Overview](#-overview)
- [🧠 System Architecture](#-system-architecture)
- [🔧 Requirements](#-requirements)
- [📁 Directory Structure](#-directory-structure)
- [🚀 Running the Project](#-running-the-project)
- [📤 Outputs](#-outputs)
- [📊 Visualizations](#-visualizations)
- [🧪 Kalman Filter Model](#-kalman-filter-model)
- [📈 Real-Map Integration](#-real-map-integration)
- [📌 Notes and Assumptions](#-notes-and-assumptions)
- [📜 License](#-license)
- [👤 Author](#-author)

---

## 📌 Overview

Raw GPS signals suffer from:
- Limited accuracy in urban environments.
- Inconsistent frequency.
- High noise sensitivity.

IMU sensors (accelerometers) provide:
- High-frequency motion data.
- Drift without correction.

To overcome the weaknesses of each sensor type, this project fuses **position from GPS** and **acceleration from IMU** using a **Kalman Filter**, resulting in:
- More reliable velocity estimation.
- Improved trajectory stability.
- Handling of GPS signal dropout scenarios.

---

## 🧠 System Architecture

### 1. **Data Ingestion**
- `kitti_data/` folder contains:
  - `timestamps.txt`: List of timestamps for each frame.
  - `data/`: One file per frame, containing GPS + IMU readings.

### 2. **Data Parsing**
- Extracts latitude, longitude, altitude, and IMU accelerations.
- Saves cleaned data to `gps_imu_data.csv`.

### 3. **Coordinate Transformation**
- Converts `(lat, lon)` to metric coordinates (EPSG:3857 or UTM).
- Relative origin used for local reference.

### 4. **Kalman Filter Fusion**
- Prediction using IMU (ax, ay).
- Correction using GPS (x, y).
- Handles variable time-steps.
- Simulates random GPS dropouts.

### 5. **Trajectory Export**
- Saves filtered positions, velocities, heading angles, and GPS usage to CSV.

### 6. **Visualization**
- Plots raw vs filtered trajectories, heading arrows.
- Shows position error and speed profile.
- Saves interactive map as HTML (`folium`).

---

## 🔧 Requirements

Install the following dependencies:

```bash
pip install numpy pandas matplotlib opencv-python folium utm pyproj
```

Python version: `>=3.7`

---

## 📁 Directory Structure

```bash
project_root/
├── kitti_data/                        # Input folder with raw sensor files
│   ├── data/
│   └── timestamps.txt
├── gps_imu_data.csv                   # Parsed and cleaned GPS+IMU data
├── filtered_trajectory_with_heading.csv  # Filtered trajectory results
├── trajectory_map.html                # Folium map output
├── main.py                            # Main Python script
└── README.md                          # This documentation
```

---

## 🚀 Running the Project

1. Ensure the dataset is placed under `./kitti_data/`
2. Run the script:

```bash
python main.py
```

The program will:
- Parse GPS/IMU data
- Run Kalman Filter
- Simulate GPS dropouts
- Export results
- Plot trajectories, speed, and errors
- Generate map visualization

---

## 📤 Outputs

### `filtered_trajectory_with_heading.csv`

| timestamp               | x     | y     | vx   | vy   | heading_deg | gps_used |
|------------------------|-------|-------|------|------|-------------|----------|
| 2021-08-01 12:00:00.100| 5.32  | 3.14  | 0.24 | 0.31 | 52.3        | True     |
| ...                    | ...   | ...   | ...  | ...  | ...         | ...      |

- `x, y`: Local Cartesian coordinates
- `vx, vy`: Estimated velocity components
- `heading_deg`: Angle in degrees
- `gps_used`: Boolean indicating if GPS was used in that step

### `trajectory_map.html`

- Interactive map with:
  - 🔵 Raw GPS path
  - 🔴 Filtered trajectory
  - 🟢 Start & 🔴 End markers

---

## 📊 Visualizations

The system generates the following plots:

- 📈 **Trajectory Plot**: Raw vs Filtered path + heading arrows  
- 🚀 **Speed Over Time**: Derived from estimated velocity
- ❌ **Error Over Time**: Euclidean error between raw and filtered positions
- 🗺 **Folium Map**: HTML view of GPS tracks on satellite map

---

## 🧪 Kalman Filter Model

The Kalman filter maintains a 4D state vector:

```
x = [pos_x, pos_y, vel_x, vel_y]^T
```

### Prediction Step
Uses acceleration input:

```
x' = F · x + B · u
P' = F · P · F^T + Q
```

### Update Step
When GPS is available:

```
y = z - H · x'
S = H · P' · H^T + R
K = P' · H^T · S^-1
x = x' + K · y
P = (I - K · H) · P'
```

Where:
- `F`: state transition matrix
- `B`: control input matrix
- `Q`: process noise
- `R`: measurement noise
- `H`: measurement matrix

The system adapts `F`, `Q`, and `B` for each `dt`.

---

## 📈 Real-Map Integration

- Converts filtered (x, y) back to (lat, lon) using UTM projection.
- Visualized with `folium` to show:
  - Trajectory on real map
  - Start and end markers
  - Tooltips and color coding

---

## 📌 Notes and Assumptions

- Simulated GPS dropout: 50 random time steps removed
- GPS and IMU are assumed time-synchronized
- Earth curvature is negligible at local scale (use UTM or EPSG:3857)
- Initial velocity is assumed to be zero

---

## 📜 License

This project is licensed under the MIT License.  
Feel free to use, modify, and share with attribution.

---

## 👤 Author

**Mohammed El Amine Hoceini**  
MSc in Autonomous Systems  
Eötvös Loránd University, Budapest, Hungary
