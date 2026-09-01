from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

input_file = Path("/data/tests/junjun/nonlinear_unwrap/simulation/nonlinear_unwrap/bspline_sim_seasonal/observed_seasonal_phase_csv/observed_phase_noise_0p2rad.csv")
output_file = Path("true_height_error.png")

df = pd.read_csv(input_file)
longitude = df["longitude"].values
latitude = df["latitude"].values
delta_h_true = df["delta_h_true_m"].values

lon_min, lon_max = longitude.min(), longitude.max()
lat_min, lat_max = latitude.min(), latitude.max()

plt.figure(figsize=(7, 6))
sc = plt.scatter(longitude, latitude, c=delta_h_true, s=5, cmap="jet")
cbar = plt.colorbar(sc)
cbar.set_label("Height error / m")
plt.xlabel("Longitude / degree")
plt.ylabel("Latitude / degree")
plt.title("True Height Error")
plt.xlim(lon_min, lon_max)
plt.ylim(lat_min, lat_max)
plt.tight_layout()
plt.savefig(str(output_file), dpi=300, bbox_inches="tight")
plt.close()

print("Height error range: %.4f ~ %.4f m" % (delta_h_true.min(), delta_h_true.max()))
print("Figure saved to: %s" % output_file)