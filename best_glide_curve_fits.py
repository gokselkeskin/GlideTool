import seaborn as sns
import matplotlib
import scipy as sc
from sympy.calculus.util import *

matplotlib.use("QtAgg")
matplotlib.interactive(True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import pickle
from collections import defaultdict
from scipy.optimize import curve_fit


def linear_func(x, a, b):
    return a * x + b


def beta_opt(Vt, m, g, rho, Sw, B, k, Cdpro, delta=1):
    try:
        value = np.clip(
            ((8 * k * m ** 2 * g ** 2)
             / (delta * np.pi * rho ** 2 * B ** 2 * Cdpro * Sw * Vt ** 4)
             ) ** (1 / 3),
            0, 1
        )
    except RuntimeWarning:
        value = np.inf
    return value


def Di(Vt, k, m, g, B, rho, Sw, Cdpro, delta):
    beta = beta_opt(Vt, m, g, rho, Sw, B, k, Cdpro, delta)
    return (2 * k * m ** 2 * g ** 2) / (Vt ** 2 * np.pi * (beta * B) ** 2 * rho)


def Db(Vt, rho, Sb, Cdb):
    return 0.5 * rho * Vt ** 2 * Sb * Cdb


def Dpro(Vt, rho, Sw, Cdpro, m, g, B, k, delta):
    beta = beta_opt(Vt, m, g, rho, Sw, B, k, Cdpro, delta)
    epsilon = 1 - delta * (1 - beta)
    return 0.5 * rho * Vt ** 2 * epsilon * Sw * Cdpro


def Vz_pennycuick(
        Vt, m, g, rho, Sb, Sw, B,
        Cdpro=0.014, Cdb=0.1, delta=1, e=1
):
    # Using k=1.1 as in your original code
    k = 1.1
    return (
            (
                    Di(Vt, k, m, g, B, rho, Sw, Cdpro, delta)
                    + Db(Vt, rho, Sb, Cdb)
                    + Dpro(Vt, rho, Sw, Cdpro, m, g, B, k, delta)
            )
            * Vt
            / (m * g)
    )


with open('species_dict_updated.pkl', 'rb') as f:
    full_objective = pickle.load(f)

#####select more species or load all species
#keys_to_copy = ["Cc"]
#objective = {k: full_objective[k] for k in keys_to_copy}
objective = full_objective

# Define Grid Search Ranges
CDBmin, CDBmax = 0.1, 0.3
CDpromin, CDpromax = 0.001, 0.097
num_points = 101
Cdpro_grid = np.linspace(0, 1, num_points)
Cdb_grid = np.linspace(0, 1, num_points)

# Grid Search for Each Bird

grid_results = {}

for bird, objective_dict in objective.items():
    m = objective_dict['parameters']['m']
    Sb = objective_dict['parameters']['Sb']
    Sw = objective_dict['parameters']['Sw']
    B = objective_dict['parameters']['B']

    #Observed horizontal velocities
    Vh_array = np.array(objective_dict["velocities"]['Horizontal']).astype(float)
    #Observed vertical velocities (already negative, but we store as negative for clarity)
    vz_array_objective = - np.array(objective_dict["velocities"]['Vertical']).astype(float)
    #Observed vertical speed error
    vz_array_err_objective = np.array(objective_dict["velocities"]['Vertical_error']).astype(float)

    grid_list = []
    for Cdpro_scaled in Cdpro_grid:
        for Cdb_scaled in Cdb_grid:
            #Convert scaled values -> real values
            Cdpro = CDpromin + Cdpro_scaled * (CDpromax - CDpromin)
            Cdb = CDBmin + Cdb_scaled * (CDBmax - CDBmin)

            #Predict vertical speeds at each measured horizontal velocity
            pred_vz = [
                Vz_pennycuick(
                    vh, m=m, g=9.8, rho=1.01,
                    Sb=Sb, Sw=Sw, B=B,
                    Cdpro=Cdpro, Cdb=Cdb, delta=1, e=1
                )
                for vh in Vh_array
            ]
            pred_vz = np.array(pred_vz)
            # Weighted RMSE
            loss = np.sqrt(
                np.sum((vz_array_objective - pred_vz) ** 2 / vz_array_err_objective)
                / np.sum(1 / vz_array_err_objective)
            )

            grid_list.append({
                'Cdpro_scaled': Cdpro_scaled,
                'Cdb_scaled': Cdb_scaled,
                'Cdpro': Cdpro,
                'Cdb': Cdb,
                'loss': loss
            })

    grid_results[bird] = pd.DataFrame(grid_list)

# Fit line in minimal-loss region

for bird, df in grid_results.items():
    # Pivot the data for minimal-loss analysis: shape (Cdb, Cdpro)
    heatmap_data = df.pivot_table(
        values="loss",
        index="Cdb",
        columns="Cdpro",
        aggfunc=np.mean
    )

    # For each column (Cdpro), find the row (Cdb) where loss is minimal
    min_loss_indices = heatmap_data.idxmin(axis=0)

    # Filter out-of-bounds region
    valid_mask = (min_loss_indices > 0.10) & (min_loss_indices < 0.30)
    min_x_values = heatmap_data.columns[valid_mask]  # Cdpro
    min_y_values = min_loss_indices[valid_mask]  # Cdb

    if len(min_x_values) < 2:
        print(f"[WARNING] Bird {bird} has insufficient valid points for linear fit.")
        continue

    # Linear fit y = a*x + b
    popt, _ = curve_fit(linear_func, min_x_values, min_y_values)
    a, b = popt

    y_min = heatmap_data.index.min()
    y_max = heatmap_data.index.max()

    num_new_points = 5
    # point_y = np.linspace(y_min, y_max, num_new_points)
    point_y = np.array([0.1, 0.2, 0.25, 0.3, 0.4])
    point_x = (point_y - b) / a

    # glide polar
    fig, ax = plt.subplots(figsize=(8, 6))

    objective_dict = full_objective[bird]
    Vh_array = np.array(objective_dict["velocities"]['Horizontal']).astype(float)
    # Observed vertical speeds (negative)
    vz_array_objective = - np.array(objective_dict["velocities"]['Vertical']).astype(float)
    # Observed errors
    vz_array_err_objective = np.array(objective_dict["velocities"]['Vertical_error']).astype(float)

    # Plot the observations with error bars
    ax.errorbar(
        Vh_array,
        -vz_array_objective,
        yerr=vz_array_err_objective,
        fmt='o',
        label="Measured Gliding Points"
    )

    # 5 to 20 m/s
    velocity_array = np.linspace(5, 20, 50)


    def compute_loss(cdpro_val, cdb_val):

        pred_vz = np.array([
            Vz_pennycuick(
                vh, m=objective_dict['parameters']['m'],
                g=9.8, rho=1.01,
                Sb=objective_dict['parameters']['Sb'],
                Sw=objective_dict['parameters']['Sw'],
                B=objective_dict['parameters']['B'],
                Cdpro=cdpro_val, Cdb=cdb_val,
                delta=1, e=1
            )
            for vh in Vh_array
        ])

        return np.sqrt(
            np.sum((vz_array_objective - pred_vz) ** 2 / vz_array_err_objective)
            / np.sum(1 / vz_array_err_objective)
        )


    for i in range(num_new_points):
        cdpro_i = point_x[i]
        cdb_i = point_y[i]

        # Compute polar
        Vz_fit = [
            Vz_pennycuick(
                vh, m=objective_dict['parameters']['m'],
                g=9.8, rho=1.01,
                Sb=objective_dict['parameters']['Sb'],
                Sw=objective_dict['parameters']['Sw'],
                B=objective_dict['parameters']['B'],
                Cdpro=cdpro_i, Cdb=cdb_i,
                delta=1, e=1
            )
            for vh in velocity_array
        ]
        Vz_fit_plot = -1 * np.array(Vz_fit)

        # Compute loss at measurement points
        loss_i = compute_loss(cdpro_i, cdb_i)

        ax.plot(
            velocity_array, Vz_fit_plot,
            label=(
                f"Fit {i + 1} "
                f"(Cdpro={cdpro_i:.3f}, Cdb={cdb_i:.2f}, loss={loss_i:.2f})"
            ),
            lw=1.75
        )

    ax.set_xlim(0, 20)
    ax.set_ylim(-3, 0)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_yticks([0, -1, -2, -3])

    latin_names = {
        "Fn": "Falco naumanni",
        "Fp": "Falco peregrinus",
        "Hl": "Haliaeetus leucocephalus",
        "Av": "Aquila verreauxii",
        "Ar": "Aquila rapax",
        "An": "Aquila nipalensis",
        "Gf_C": "Gyps fulvus (1)",
        "Gf_W": "Gyps fulvus (2)",
        "Gh": "Gyps himalayensis",
        "Gr": "Gyps rueppellii",
        "Vg": "Vultur gryphus",
        "Cc": "Ciconia ciconia",
        "Ge": "Geronticus eremita"
    }

    full_name = latin_names.get(bird, "Unknown Species")

    ax.set_title(r"Calculated glide polars of $\mathit{" + full_name.replace(" ", r"\ ") + "}$",
                 fontdict={'family': 'Arial', 'size': 14})

    ax.set_xlabel("Horizontal Velocity (m/s)", fontdict={'family': 'Arial', 'size': 14})
    ax.set_ylabel("Vertical Speed (m/s)", fontdict={'family': 'Arial', 'size': 14})
    ax.grid(True)
    ax.legend()
    # plt.savefig(f"Glide Polar Variations for {bird}.png", dpi=500)
    plt.savefig(f"Glide Polar Variations for {bird}_updated.svg")
    plt.show(block=True)

    for i in range(num_new_points):
        cdpro_i = point_x[i]
        cdb_i = point_y[i]
        loss_i = compute_loss(cdpro_i, cdb_i)
        print(f"  Point {i + 1}: (Cdpro={cdpro_i:.4f}, Cdb={cdb_i:.4f}), loss={loss_i:.4f}")
    print(f"Linear Fit:  Cdb = a*Cdpro + b => a={a:.4f}, b={b:.4f}")
