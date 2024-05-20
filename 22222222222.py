import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd

class CVFilter:
    def __init__(self):
        # Initialize filter parameters
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.pf = np.eye(6)  # Filter state covariance matrix
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.clusters = []  # Store clusters
        self.hypotheses = []  # Store hypotheses

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        # Initialize filter state
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = time
        print("Initialized filter state:")
        print("Sf:", self.Sf)
        print("pf:", self.pf)

    def predict_step(self, current_time):
        # Predict step
        dt = current_time - self.Meas_Time
        Phi = np.eye(6)
        Phi[0, 3] = dt
        Phi[1, 4] = dt
        Phi[2, 5] = dt
        Q = np.eye(6) * self.plant_noise
        self.Sf = np.dot(Phi, self.Sf)
        self.pf = np.dot(np.dot(Phi, self.pf), Phi.T) + Q
        print("Predicted filter state:")
        print("Sf:", self.Sf)
        print("pf:", self.pf)

    def update_step(self, measurements):
        # Perform JPDA update step
        print("Performing JPDA update step...")

        # Cluster determination
        self.clusters = [measurements]
        print(f"Generated Clusters: {self.clusters}")

        # Hypothesis generation
        from itertools import permutations
        num_measurements = len(measurements)
        self.hypotheses = list(permutations(range(num_measurements)))
        print(f"Generated Hypotheses: {self.hypotheses}")

        # Calculate probabilities for each hypothesis
        hypothesis_probabilities = np.zeros(len(self.hypotheses))
        for h_idx, hypothesis in enumerate(self.hypotheses):
            likelihood = 1.0
            for i, meas_idx in enumerate(hypothesis):
                Z = np.array(measurements[meas_idx][:3]).reshape(3, 1)
                Inn = Z - np.dot(self.H, self.Sf)  # Innovation
                S = np.dot(self.H, np.dot(self.pf, self.H.T)) + self.R  # Innovation covariance
                L = np.exp(-0.5 * np.dot(Inn.T, np.dot(np.linalg.inv(S), Inn))) / np.sqrt(np.linalg.det(2 * np.pi * S))
                likelihood *= L
            hypothesis_probabilities[h_idx] = likelihood

        # Normalize hypothesis probabilities
        hypothesis_probabilities /= np.sum(hypothesis_probabilities)
        print(f"Hypothesis Probabilities: {hypothesis_probabilities}")

        # Calculate association weights
        num_targets = 1
        association_weights = np.zeros((num_measurements, num_targets))
        for meas_idx in range(num_measurements):
            for h_idx, hypothesis in enumerate(self.hypotheses):
                if hypothesis[meas_idx] < num_targets:
                    association_weights[meas_idx, hypothesis[meas_idx]] += hypothesis_probabilities[h_idx]

        print(f"Association Weights: {association_weights}")

        # Update state estimate
        for t in range(num_targets):
            weight_sum = np.sum(association_weights[:, t])
            if weight_sum > 0:
                z_hat = np.zeros((3, 1))
                for meas_idx in range(num_measurements):
                    Z = np.array(measurements[meas_idx][:3]).reshape(3, 1)
                    z_hat += association_weights[meas_idx, t] * Z
                z_hat /= weight_sum

                Inn = z_hat - np.dot(self.H, self.Sf)
                S = np.dot(self.H, np.dot(self.pf, self.H.T)) + self.R
                K = np.dot(np.dot(self.pf, self.H.T), np.linalg.inv(S))
                self.Sf = self.Sf + np.dot(K, Inn)
                self.pf = np.dot(np.eye(6) - np.dot(K, self.H), self.pf)
                print(f"Updated state for target {t}: Sf = {self.Sf}, pf = {self.pf}")

# Function to convert spherical coordinates to Cartesian coordinates
def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

# Function to convert Cartesian coordinates to spherical coordinates
def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    el = math.atan(z / np.sqrt(x**2 + y**2)) * 180 / 3.14
    az = math.atan2(y, x) * 180 / 3.14
    if az < 0:
        az += 360
    return r, az, el

def cart2sph2(x: float, y: float, z: float, filtered_values_csv):
    r, az, el = [], [], []
    for i in range(len(filtered_values_csv)):
        r.append(np.sqrt(x[i]**2 + y[i]**2 + z[i]**2))
        el.append(math.atan(z[i] / np.sqrt(x[i]**2 + y[i]**2)) * 180 / 3.14)
        az.append(math.atan2(y[i], x[i]) * 180 / 3.14)
        if az[i] < 0:
            az[i] += 360
    return r, az, el

# Function to read measurements from CSV file
def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            mr = float(row[10])  # MR column
            ma = float(row[11])  # MA column
            me = float(row[12])  # ME column
            mt = float(row[13])  # MT column
            x, y, z = sph2cart(ma, me, mr)
            r, az, el = cart2sph(x, y, z)
            measurements.append((r, az, el, mt))
    return measurements

# Create an instance of the CVFilter class
kalman_filter = CVFilter()

# Define the path to your CSV file containing measurements
csv_file_path = 'data_57.csv'  # Provide the path to your CSV file

# Read measurements from CSV file
measurements = read_measurements_from_csv(csv_file_path)

# Perform filtering
time_list = []
r_list = []
az_list = []
el_list = []

for i, (r, az, el, mt) in enumerate(measurements):
    if i == 0:
        # Initialize filter state with the first measurement
        kalman_filter.initialize_filter_state(r, az, el, 0, 0, 0, mt)
    elif i == 1:
        # Initialize filter state with the second measurement and compute velocity
        prev_r, prev_az, prev_el, prev_mt = measurements[i - 1]
        dt = mt - prev_mt
        vx = (r - prev_r) / dt
        vy = (az - prev_az) / dt
        vz = (el - prev_el) / dt
        kalman_filter.initialize_filter_state(prev_r, prev_az, prev_el, vx, vy, vz, mt)
    else:
        # Predict step
        kalman_filter.predict_step(mt)
        # Update step with current measurement
        possible_measurements = measurements[:i + 1]  # Use all previous and current measurements
        kalman_filter.update_step(possible_measurements)

    # Append data for plotting
    time_list.append(mt)
    r_list.append(kalman_filter.Sf[0, 0])
    az_list.append(kalman_filter.Sf[1, 0])
    el_list.append(kalman_filter.Sf[2, 0])

# Plot results
plt.figure(figsize=(12, 6))

# Plot r
plt.subplot(3, 1, 1)
plt.plot(time_list, r_list, label='Estimated r')
plt.xlabel('Time')
plt.ylabel('Range (r)')
plt.legend()

# Plot az
plt.subplot(3, 1, 2)
plt.plot(time_list, az_list, label='Estimated az')
plt.xlabel('Time')
plt.ylabel('Azimuth (az)')
plt.legend()

# Plot el
plt.subplot(3, 1, 3)
plt.plot(time_list, el_list, label='Estimated el')
plt.xlabel('Time')
plt.ylabel('Elevation (el)')
plt.legend()

plt.tight_layout()
plt.show()

# Save filtered results to CSV
filtered_data = pd.DataFrame({
    'Time': time_list,
    'Estimated r': r_list,
    'Estimated az': az_list,
    'Estimated el': el_list
})
filtered_data.to_csv('filtered_values.csv', index=False)
