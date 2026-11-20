import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    PROJECT_ROOT / "data" / "VIBRATIONAL"
    / "building_health_monitoring_dataset.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "vibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

features = [
    "Accel_X (m/s^2)",
    "Accel_Y (m/s^2)",
    "Accel_Z (m/s^2)",
    "Strain (με)",
    "Temp (°C)"
]

# Handle missing values for diagnostic plots only
data = df[features].interpolate(
    method="linear",
    limit_direction="both"
)

data["Acceleration Magnitude"] = np.sqrt(
    data["Accel_X (m/s^2)"]**2
    + data["Accel_Y (m/s^2)"]**2
    + data["Accel_Z (m/s^2)"]**2
)

data["Condition Label"] = df["Condition Label"]

# D5.1: Feature distributions
plot_features = features + ["Acceleration Magnitude"]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for i, feature in enumerate(plot_features):
    data.boxplot(
        column=feature,
        by="Condition Label",
        ax=axes[i]
    )
    axes[i].set_title(feature)
    axes[i].set_xlabel("Condition Label")

fig.suptitle("Sensor Feature Distributions by Class")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "feature_distributions.png",
    dpi=150
)
plt.show()

# D5.2: Temperature and time correlation
data["Row Index"] = np.arange(len(data))

print("\nTEMPERATURE CORRELATION")
print(
    "Temperature vs row index:",
    data["Temp (°C)"].corr(data["Row Index"])
)

print("\nTEMPERATURE BY CLASS")
print(
    data.groupby("Condition Label")["Temp (°C)"].describe()
)

print("\nFEATURE CORRELATION WITH LABEL")
print(
    data[features + ["Condition Label"]].corr()
    ["Condition Label"]
)

# D5.3: PCA visualisation
X = StandardScaler().fit_transform(data[features])

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(9, 6))

for class_id in [0, 1, 2]:
    mask = data["Condition Label"].to_numpy() == class_id

    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=f"Class {class_id}",
        alpha=0.6
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA: Sensor Measurements by Condition Class")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig(
    OUTPUT_DIR / "pca_features.png",
    dpi=150
)
plt.show()

print("\nExplained variance ratio:", pca.explained_variance_ratio_)
print("Total explained variance:", sum(pca.explained_variance_ratio_))

print("\nD5 FEATURE DIAGNOSTICS COMPLETED")