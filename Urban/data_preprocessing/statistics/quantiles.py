import os
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt

output_dir_path = "F:\\City_Team\\COHO\\Preprocessed_Dataset"

# Output Pickle 파일 경로 설정
output_pickle_path = os.path.join(output_dir_path, "bounding_box_statistics.pkl")

with open(output_pickle_path, "rb") as file:
    bbox_data = pickle.load(file)

# Plot each list
plt.figure(figsize=(12, 8))
quantiles = {}
for idx, key in enumerate(bbox_data):
    data_list = bbox_data[key]

    # Calculate the 15% and 85% quantiles
    q25 = np.percentile(data_list, 25)
    q75 = np.percentile(data_list, 75)
    quantiles[key] = {"25%": q25, "75%": q75}

    plt.subplot(2, 2, idx + 1)

    if "min" in key:
        plt.hist(data_list, bins=1000, range=(-2000, 0), edgecolor='black', alpha=0.7)
    elif "max" in key:
        plt.hist(data_list, bins=1000, range=(0, 2000), edgecolor='black', alpha=0.7)

    plt.axvline(q25, color='red', linestyle='--', label='25% Quantile')
    plt.axvline(q75, color='blue', linestyle='--', label='75% Quantile')
    plt.title(f'Histogram of Data {key}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()

    print(f"Key: {key}, 25% Quantile: {q25}, 75% Quantile: {q75}")

plt.tight_layout()
plt.show()

output_pickle_path = os.path.join(output_dir_path, "bounding_box_quantiles.pkl")
with open(output_pickle_path, "wb") as file:
    pickle.dump(quantiles, file)

print(quantiles)