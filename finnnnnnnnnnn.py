import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from scipy.stats import chi2

class CVFilter:
    def __init__(self):
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.pf = np.eye(6)  # Filter state covariance matrix
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = time

    def predict_step(self, current_time):
        dt = current_time - self.Meas_Time
        Phi = np.eye(6)
        Phi[0, 3] = dt
        Phi[1, 4] = dt
        Phi[2, 5] = dt
        Q = np.eye(6) * self.plant_noise
        self.Sf = np.dot(Phi, self.Sf)
        self.pf = np.dot(np.dot(Phi, self.pf), Phi.T) + Q
        self.Meas_Time = current_time

    def chi_squared_gating(self, Z, gate_threshold=7.815):  # Default gate for 95% confidence with 3 DOF
        gated_measurements = []
        for measurement in Z:
            Inn = measurement - np.dot(self.H, self.Sf)
            S = np.dot(self.H, np.dot(self.pf, self.H.T)) + self.R
            distance = np.dot(Inn.T, np.dot(np.linalg.inv(S), Inn))
            if distance < gate_threshold:
                gated_measurements.append(measurement)
        return gated_measurements

    def update_step(self, measurements):
        gated_measurements = self.chi_squared_gating([np.array(m[:3]).reshape(-1, 1) for m in measurements])
        if not gated_measurements:
            return

        num_meas = len(gated_measurements)
        num_hypotheses = 2 ** num_meas
        likelihoods = np.zeros(num_hypotheses)
        hypotheses = []

        for h in range(num_hypotheses):
            hypothesis = []
            for m in range(num_meas):
                if h & (1 << m):
                    hypothesis.append(m)
            hypotheses.append(hypothesis)
            likelihood = 0
            for m in hypothesis:
                Z = gated_measurements[m]
                Inn = Z - np.dot(self.H, self.Sf)  # Innovation
                S = np.dot(self.H, np.dot(self.pf, self.H.T)) + self.R
                try:
                    L = -0.5 * np.dot(Inn.T, np.dot(np.linalg.inv(S), Inn)) - 0.5 * np.log(np.linalg.det(2 * np.pi * S))
                except np.linalg.LinAlgError:
                    L = -np.inf
                likelihood += L
            likelihood = np.array(likelihood)  # Convert to numpy array
            likelihoods[h] = likelihood.item() if likelihood.size == 1 else likelihood[0]

        max_likelihood = np.max(likelihoods)
        normalized_likelihoods = np.exp(likelihoods - max_likelihood)
        hypothesis_probs = normalized_likelihoods / np.sum(normalized_likelihoods)
        weights = np.zeros((num_meas, 1))

        for m in range(num_meas):
            weight = 0
            for h, hypothesis in enumerate(hypotheses):
                if m in hypothesis:
                    weight += hypothesis_probs[h]
            weights[m] = weight

        for m in range(num_meas):
            Z = gated_measurements[m]
            Inn = Z - np.dot(self.H, self.Sf)  # Innovation
            S = np.dot(self.H, np.dot(self.pf, self.H.T)) + self.R
            K = np.dot(np.dot(self.pf, self.H.T), np.linalg.inv(S))
            self.Sf += weights[m] * np.dot(K, Inn)
            self.pf = np.dot(np.eye(6) - np.dot(K, self.H), self.pf)

def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    el = math.atan(z / np.sqrt(x**2 + y**2)) * 180 / np.pi
    az = math.atan2(y, x) * 180 / np.pi
    if az < 0.0:
        az += 360.0
    return r, az, el

def cart2sph2(x, y, z, filtered_values_csv):
    r = []
    az = []
    el = []
    for i in range(len(filtered_values_csv)):
        r.append(np.sqrt(x[i]**2 + y[i]**2 + z[i]**2))
        el.append(math.atan(z[i] / np.sqrt(x[i]**2 + y[i]**2)) * 180 / np.pi)
        az.append(math.atan2(y[i], x[i]) * 180 / np.pi)
        if az[i] < 0.0:
            az[i] += 360.0
    return r, az, el

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
            x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
            print("Cartesian coordinates (x, y, z):", x, y, z)
            r, az, el = cart2sph(x, y, z)  # Convert Cartesian to spherical coordinates
            print("Spherical coordinates (r, az, el):", r, az, el)
            measurements.append((r, az, el, mt))
    return measurements

# Create an instance of the CVFilter class
kalman_filter = CVFilter()

# Define the path to your CSV file containing measurements
csv_file_path = 'data_57.csv'  # Provide the path to your CSV file

# Read measurements from CSV file
measurements = read_measurements_from_csv(csv_file_path)

csv_file_predicted = "data_57.csv"
df_predicted = pd.read_csv(csv_file_predicted)
filtered_values_csv = df_predicted[['FT', 'FX', 'FY', 'FZ']].values

A = cart2sph2(filtered_values_csv[:, 1], filtered_values_csv[:, 2], filtered_values_csv[:, 3], filtered_values_csv)

time_list = []
r_list = []
az_list = []
el_list = []

# Iterate through measurements
for i, (x, y, z, mt) in enumerate(measurements):
    r, az, el = cart2sph(x, y, z)
    if i == 0:
        kalman_filter.initialize_filter_state(x, y, z, 0, 0, 0, mt)
    elif i == 1:
        prev_x, prev_y, prev_z = measurements[i - 1][:3]
        dt = mt - measurements[i - 1][3]
        vx = (x - prev_x) / dt
        vy = (y - prev_y) / dt
        vz = (z - prev_z) / dt
        kalman_filter.initialize_filter_state(x, y, z, vx, vy, vz, mt)
    else:
        kalman_filter.predict_step(mt)
        possible_measurements = [measurements[idx] for idx in range(i - 2, i + 1)]
        kalman_filter.update_step(possible_measurements)
        time_list.append(mt)
        r_list.append(kalman_filter.Sf[0][0])
        az_list.append(kalman_filter.Sf[1][0])
        el_list.append(kalman_filter.Sf[2][0])

# Plot range (r) vs. time
plt.figure(figsize=(12, 6))
plt.scatter(time_list, r_list, label='Filtered Range (Code)', color='green', marker='o')
plt.scatter(filtered_values_csv[:, 0], A[0], label='Filtered Range (Track ID 57)', color='red', marker='*')
plt.xlabel('Time')
plt.ylabel('Range (r)')
plt.title('Range vs. Time')
plt.grid(True)
plt.legend()
plt.show()

# Plot azimuth (az) vs. time
plt.figure(figsize=(12, 6))
plt.scatter(time_list, az_list, label='Filtered Azimuth (Code)', color='green', marker='o')
plt.scatter(filtered_values_csv[:, 0], A[1], label='Filtered Azimuth (Track ID 57)', color='red', marker='*')
plt.xlabel('Time')
plt.ylabel('Azimuth (az)')
plt.title('Azimuth vs. Time')
plt.grid(True)
plt.legend()
plt.show()

# Plot elevation (el) vs. time
plt.figure(figsize=(12, 6))
plt.scatter(time_list, el_list, label='Filtered Elevation (Code)', color='green', marker='o')
plt.scatter(filtered_values_csv[:, 0], A[2], label='Filtered Elevation (Track ID 57)', color='red', marker='*')
plt.xlabel('Time')
plt.ylabel('Elevation (el)')
plt.title('Elevation vs. Time')
plt.grid(True)
plt.legend()
plt.show()
