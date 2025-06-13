import seaborn as sns
import matplotlib
import scipy as sc
from sympy.calculus.util import *
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import pickle
from collections import defaultdict
from scipy.optimize import curve_fit
from collections import defaultdict, Counter


def linear_func(x, a, b):
    return a * x + b

def beta_opt(Vt, m, g, rho, Sw, B, k, Cdpro, delta=1):
    try:
        return np.clip(
            ((8 * k * m**2 * g**2)
             / (delta * np.pi * rho**2 * B**2 * Cdpro * Sw * Vt**4))**(1/3),
            0, 1
        )
    except RuntimeWarning:
        return np.inf

def Di(Vt, k, m, g, B, rho, Sw, Cdpro, delta):
    beta = beta_opt(Vt, m, g, rho, Sw, B, k, Cdpro, delta)
    return (2 * k * m**2 * g**2) / (Vt**2 * np.pi * (beta * B)**2 * rho)

def Db(Vt, rho, Sb, Cdb):
    return rho * Vt**2 * Sb * Cdb / 2

def Dpro(Vt, rho, Sw, Cdpro, m, g, B, k, delta):
    beta = beta_opt(Vt, m, g, rho, Sw, B, k, Cdpro, delta)
    epsilon = 1 - delta * (1 - beta)
    return rho * Vt**2 * epsilon * Sw * Cdpro / 2

def Vz_pennycuick(Vt, m, g, rho, Sb, Sw, B, Cdpro=0.014, Cdb=0.1, delta=1, e=1):
    k = 1.1
    return ((Di(Vt, k, m, g, B, rho, Sw, Cdpro, delta)
             + Db(Vt, rho, Sb, Cdb)
             + Dpro(Vt, rho, Sw, Cdpro, m, g, B, k, delta)) * Vt
            / (m * g))

# --- Load species data ---
with open('species_dict.pkl', 'rb') as f:
    full_objective = pickle.load(f)

#objective = full_objective

keys_to_copy = ["An","Ar","Gf_W","Gh","Vg","Cc","Ge","Fn"]
objective = {key: full_objective[key] for key in keys_to_copy}


#Grid parameters
CDBmin, CDBmax = 0.1, 0.3
CDpromin, CDpromax = 0.001, 0.097
num_points = 100
Cdpro_scaled_grid = np.linspace(0, 1, num_points)
Cdb_scaled_grid   = np.linspace(0, 1, num_points)

grid_results = {'pooled': []}


global_data = []
for bird, obj in objective.items():
    m, Sb, Sw, B = obj['parameters']['m'], obj['parameters']['Sb'], obj['parameters']['Sw'], obj['parameters']['B']
    Vh = np.array(obj['velocities']['Horizontal'], float)
    Vz = -np.array(obj['velocities']['Vertical'],   float)
    Err= np.array(obj['velocities']['Vertical_error'], float)
    for vh, vz_o, err in zip(Vh, Vz, Err):
        global_data.append((vh, vz_o, err, m, Sb, Sw, B))

for Cd_s in Cdpro_scaled_grid:
    Cdpro = CDpromin + Cd_s * (CDpromax - CDpromin)
    for Db_s in Cdb_scaled_grid:
        Cdb = CDBmin + Db_s * (CDBmax - CDBmin)
        preds, obs, errs = [], [], []
        for vh, vz_o, err, m, Sb, Sw, B in global_data:
            preds.append(Vz_pennycuick(vh, m, 9.8, 1.01, Sb, Sw, B, Cdpro, Cdb))
            obs.append(vz_o)
            errs.append(err)
        preds = np.array(preds)
        obs   = np.array(obs)
        errs  = np.array(errs)
        loss = np.sqrt(np.sum((obs - preds)**2 / errs**2) / np.sum(1 / errs**2))
        grid_results['pooled'].append({'Cdpro': Cdpro, 'Cdb': Cdb, 'loss': loss})


df = pd.DataFrame(grid_results['pooled'])
heatmap_data = df.pivot_table(values='loss', index='Cdb', columns='Cdpro')

opt_idx = df['loss'].idxmin()
best = df.loc[opt_idx]
best_Cdpro, best_Cdb, best_loss = best['Cdpro'], best['Cdb'], best['loss']
print(f"Global best C_D_pro = {best_Cdpro:.5f}")
print(f"Global best C_D_b   = {best_Cdb:.5f}")
print(f"Minimum loss        = {best_loss:.4f}")

min_loss_indices = heatmap_data.idxmin(axis=0)

valid_mask = (min_loss_indices > 0.10) & (min_loss_indices < 0.30)
min_x_values = heatmap_data.columns[valid_mask]
min_y_values = min_loss_indices[valid_mask]
# fit line
popt, _ = curve_fit(linear_func, min_x_values, min_y_values)
a, b = popt
y_min, y_max = heatmap_data.index.min(), heatmap_data.index.max()
fitted_y = np.linspace(y_min, y_max, 200)
fitted_x = (fitted_y - b) / a
cdb_counts = Counter(min_y_values)
most_cdb, freq = cdb_counts.most_common(1)[0]
print(f"Most frequent column-minimum Cdb = {most_cdb:.3f} (in {freq}/{len(min_y_values)} columns)")
#heatmap
plt.figure(figsize=(8,7))
ax = sns.heatmap(heatmap_data, cmap='cividis_r', vmin=0, vmax=1.5, cbar_kws={'label':'Loss'})
# invert y-axis to start from smallest Cdb
ax.set_ylim(ax.get_ylim()[::-1])
# Pennycuick default marker
px = np.argmin(np.abs(heatmap_data.columns - 0.014))
py = np.argmin(np.abs(heatmap_data.index   - 0.1))
ax.scatter(px+0.1, py+0.8, c='red', s=100, marker='*', label='Pennycuick default')
# overlay column minima points
norm_x_pts = (min_x_values - heatmap_data.columns.min())/(heatmap_data.columns.max()-heatmap_data.columns.min())*len(heatmap_data.columns)
norm_y_pts = (min_y_values - heatmap_data.index.min())/(heatmap_data.index.max()-heatmap_data.index.min())*len(heatmap_data.index)
#ax.scatter(norm_x_pts, norm_y_pts, s=50, c='blue', marker='o', label='Column minima')

norm_x = (fitted_x - heatmap_data.columns.min()) / (heatmap_data.columns.max() - heatmap_data.columns.min()) * len(heatmap_data.columns)
norm_y = (fitted_y - heatmap_data.index.min()) / (heatmap_data.index.max() - heatmap_data.index.min()) * len(heatmap_data.index)
ax.plot(norm_x, norm_y, '--', lw=2, color='black', label='Linear Fit')

desired_cdb = [0.10, 0.15, 0.20, 0.25, 0.30]
y_positions = [np.argmin(np.abs(heatmap_data.index - v)) for v in desired_cdb]
ax.set_yticks(y_positions)
ax.set_yticklabels([f"{v:.2f}" for v in desired_cdb])

desired_cdpro = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
x_positions = [np.argmin(np.abs(heatmap_data.columns - v)) for v in desired_cdpro]
ax.set_xticks(x_positions)
ax.set_xticklabels([f"{v:.3f}" for v in desired_cdpro], rotation=45)


ax.set_xlabel(r"$C_{D_{pro}}$")
ax.set_ylabel(r"$C_{D_{b}}$")
ax.legend()
plt.title("Heatmap of General Fit (Unacc Curve)")
plt.savefig("Heatmap_acc.svg")
plt.show(block=True)

print("\nOriginal (x, y) pairs:")
for x, y in zip(min_x_values, min_y_values):
    print(f"({x:.4f}, {y:.4f})")
print(f"lin_a = {a}, lin_b = {b}")
