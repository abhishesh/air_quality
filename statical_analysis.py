import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the path to the dataset directory
dataset_dir = "dataset"  # Update this to the path where your CSV files are stored

# Load all CSV files into a single DataFrame
all_files = [f for f in os.listdir(dataset_dir) if f.endswith(".csv")]
data_list = []

for file in all_files:
    file_path = os.path.join(dataset_dir, file)
    df = pd.read_csv(file_path, delimiter=";")
    data_list.append(df)

# Concatenate all CSVs into one DataFrame
data = pd.concat(data_list, ignore_index=True)
print(data)

# Preprocess the data (assuming 'CO2 ppm' and 'HUMIDITY %' columns)
data["CO2 ppm"] = pd.to_numeric(data["CO2 ppm"], errors="coerce")
data["HUMIDITY %"] = pd.to_numeric(data["HUMIDITY %"], errors="coerce")

# Drop rows with missing CO2 or HUMIDITY values
data = data.dropna(subset=["CO2 ppm", "HUMIDITY %"])

# Define thresholds for "party" conditions
co2_threshold = 2000  # CO2 ppm threshold for spike
humidity_threshold = 70  # HUMIDITY percentage threshold

# Create a condition for a "party"
party_condition = (data["CO2 ppm"] > co2_threshold) & (
    data["HUMIDITY %"] < humidity_threshold
)

# Detect party events
party_times = data[party_condition]

# Print out the times or records when a party might have occurred
print("Party detected at the following times or data points:")
print(party_times)

# Plotting CO2 and HUMIDITY with detected party times
plt.figure(figsize=(10, 5))
plt.plot(data["CO2 ppm"], label="CO2 ppm", color="blue")
plt.plot(data["HUMIDITY %"], label="HUMIDITY %", color="green")
plt.scatter(
    party_times.index,
    party_times["CO2 ppm"],
    color="red",
    label="Party Detected",
    s=100,
    zorder=5,
)
plt.xlabel("Time")
plt.ylabel("CO2 ppm / HUMIDITY %")
plt.legend()
plt.title("Party Detection Based on CO2 and HUMIDITY")
plt.show()
