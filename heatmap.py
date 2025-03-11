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
import csv

def linear_func(x, a, b):
    return a * x + b


def f_linear(x,a,b):
    return (a * x) + b


def beta_opt(Vt, m, g, rho, Sw, B, k, Cdpro, delta=1):
    try:
        value = np.clip(((8 * k * m ** 2 * g ** 2)
                         / (delta * np.pi * rho ** 2 * B ** 2 * Cdpro * Sw * Vt ** 4)
                         ) ** (1 / 3),
                        0, 1, )
    except RuntimeWarning:
        value = np.inf

    return value

def beta_hat(Vt, m, g, rho, Sw, B, k, Cdpro, delta=1):
    try:
        value = ((8 * k * m ** 2 * g ** 2)
                 / (delta * np.pi * rho ** 2 * B ** 2 * Cdpro * Sw * Vt ** 4)) ** (1 / 3)
    except RuntimeWarning:
        value = np.inf

    return value

def Di(Vt, k, m, g, B, rho, Sw, Cdpro, delta):
    beta = beta_opt(Vt, m, g, rho, Sw, B, k, Cdpro, delta)
    value = ((2 * k * m ** 2 * g ** 2)
             / (Vt ** 2 * np.pi * (beta * B) ** 2 * rho)
             )
    return value


def Db(Vt, rho, Sb, Cdb):
    value = rho * Vt ** 2 * Sb * Cdb / 2
    return value


def Dpro(Vt, rho, Sw, Cdpro, m, g, B, k, delta):
    beta = beta_opt(Vt, m, g, rho, Sw, B, k, Cdpro, delta)
    epsilon = 1 - delta * (1 - beta)
    value = rho * Vt ** 2 * epsilon * Sw * Cdpro / 2
    return value


def Vz_pennycuick(Vt, m, g, rho, Sb, Sw, B, Cdpro=0.014, Cdb=0.1, delta=1, e=1):
    AR = B ** 2 / Sw
    k= 1.1
    beta = np.clip(beta_opt(Vt, m, g, rho, Sw, B, k, Cdpro, delta), 0, 1, )

    epsilon = 1 - delta * (1 - beta)

    value = ((Di(Vt, k, m, g, B, rho, Sw, Cdpro=Cdpro, delta=delta)
              + Db(Vt=Vt, rho=rho, Sb=Sb, Cdb=Cdb)
              + Dpro(Vt=Vt, rho=rho, Sw=Sw, Cdpro=Cdpro, m=m, g=g, B=B, k=k, delta=delta)) * Vt
             / (m * g))
    return value


with open('species_dict.pkl', 'rb') as f:
    full_objective = pickle.load(f)
    #objective = pickle.load(f)

####################### Choose the species or load the full data#################

keys_to_copy = ["Fn"]
objective = {key: full_objective[key] for key in keys_to_copy}

#objective = full_objective


baseline_parameters = {'Cdb': 0.1, 'Cdpro': 0.014}

CDBmin = 0.1
CDBmax = 0.3


CDpromin = 0.001
#CDpromin = 0
CDpromax = 0.097


version = "V2"

num_points = 101
Cdpro_grid = np.linspace(0, 1, num_points)
Cdb_grid = np.linspace(0, 1, num_points)

# Initialize results dictionary
grid_results = {}

# Perform grid search
for bird, objective_dict in objective.items():
    m = objective_dict['parameters']['m']
    Sb = objective_dict['parameters']['Sb']
    Sw = objective_dict['parameters']['Sw']
    B = objective_dict['parameters']['B']

    grid_results[bird] = []

    for Cdpro_scaled in Cdpro_grid:
        #plt.figure()
        figure_title= CDpromin + Cdpro_scaled * (CDpromax - CDpromin)
        for Cdb_scaled in Cdb_grid:
            # Convert scaled values to real values
            Cdpro = CDpromin + Cdpro_scaled * (CDpromax - CDpromin)
            Cdb = CDBmin + Cdb_scaled * (CDBmax - CDBmin)

            delta = 1
            e = 1

            vz_array = []
            Vh_array = np.array(objective_dict["velocities"]['Horizontal']).astype(float)
            for i in Vh_array:
                vz_array.append(
                    Vz_pennycuick(i, m=m, g=9.8, rho=1.01, Sb=Sb, Sw=Sw, B=B, Cdpro=Cdpro, Cdb=Cdb, delta=delta, e=e))

            vz_array = np.array(vz_array)
            vz_array_objective = - np.array(objective_dict["velocities"]['Vertical']).astype(float)
            vz_array_err_objective = np.array(objective_dict["velocities"]['Vertical_error']).astype(float)

            # Calculate loss
            loss = np.sqrt(np.sum((vz_array_objective - vz_array) ** 2 / vz_array_err_objective) / np.sum(
                1 / vz_array_err_objective))

            grid_results[bird].append({
                'Cdpro_scaled': Cdpro_scaled,
                'Cdb_scaled': Cdb_scaled,
                'Cdpro': Cdpro,
                'Cdb': Cdb,
                'Vz_array': vz_array,
                'loss': loss,
                'x' : Vh_array,
                'y_predict' : vz_array * -1,
                'y_original' : vz_array_objective * -1,
                'y_original_error' : vz_array_err_objective
            })


for bird, data in grid_results.items():

    df = pd.DataFrame(data)


    heatmap_data = df.pivot_table(values="loss", index="Cdb", columns="Cdpro", aggfunc=np.mean)


    plt.figure(figsize=(8, 7.2))

    sns.heatmap(heatmap_data, cmap="cividis_r",vmin=0, vmax=1.5, cbar_kws={'label': 'Loss (m/s)'})

    ax = plt.gca()


    x_ticks = ax.get_xticks()



    y_ticks = ax.get_yticks()

    ax.set_ylim(ax.get_ylim()[::-1])  # Invert y-axis order

    var_name = f"original_{bird}"


    Cdb_penny = 0.1
    Cdpro_penny = 0.014

    Penny_x_index= np.argmin(np.abs(heatmap_data.columns - Cdpro_penny))
    Penny_y_index = np.argmin(np.abs(heatmap_data.index - Cdb_penny))

    ax.scatter(Penny_x_index + 0.1, Penny_y_index + 0.8, c="red", s=100, marker="*",  linewidths=1.5,
               label="Default Pennycuick value")


    min_loss_indices = heatmap_data.idxmin(axis=0)  # Get row index of minimum value for each column

    valid_mask = (min_loss_indices > 0.10) & (min_loss_indices < 0.30)

    min_x_values = heatmap_data.columns[valid_mask]
    min_y_values = min_loss_indices[valid_mask]


    popt, _ = curve_fit(linear_func, min_x_values, min_y_values)
    a, b = popt

    y_min = heatmap_data.index.min()
    y_max = heatmap_data.index.max()

    fitted_y = np.linspace(y_min, y_max, 200)
    fitted_x = (fitted_y - b) / a

    norm_x = (
                     (fitted_x - heatmap_data.columns.min())
                     / (heatmap_data.columns.max() - heatmap_data.columns.min())
             ) * len(heatmap_data.columns)

    norm_y = (
                     (fitted_y - heatmap_data.index.min())
                     / (heatmap_data.index.max() - heatmap_data.index.min())
             ) * len(heatmap_data.index)


    ax.plot(
        norm_x,
        norm_y,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Linear Fit"
    )



    normalized_x_values = (
                                  (min_x_values - heatmap_data.columns.min()) /
                                  (heatmap_data.columns.max() - heatmap_data.columns.min())
                          ) * len(heatmap_data.columns)

    normalized_y_values = (
                                  (min_y_values - heatmap_data.index.min()) /
                                  (heatmap_data.index.max() - heatmap_data.index.min())
                          ) * len(heatmap_data.index)

    specific_x_labels = {0.001, 0.09}
    specific_y_labels = {0.10, 0.30}

    x_labels = [f"{heatmap_data.columns[int(tick)]:.3f}" if round(float(heatmap_data.columns[int(tick)]),
                                                                  3) in specific_x_labels
                else "" for tick in x_ticks if int(tick) < len(heatmap_data.columns)]
    ax.set_xticklabels(x_labels)

    y_labels = [
        f"{heatmap_data.index[int(tick)]:.2f}" if round(float(heatmap_data.index[int(tick)]), 2) in specific_y_labels
        else "" for tick in y_ticks if int(tick) < len(heatmap_data.index)]
    ax.set_yticklabels(y_labels)


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

    plt.title(r"Heatmap of Loss for $\mathit{" + full_name.replace(" ", r"\ ") + "}$", fontsize=15,fontdict={'family': 'Arial', 'size': 15})

    #plt.xlabel("CDpro")
    plt.xlabel(r"$C_{D_{\mathrm{pro}}}$", fontdict={'family': 'Arial', 'size': 14})

    #plt.ylabel("CDb")
    plt.ylabel(r"$C_{D_{\mathrm{b}}}$",fontdict={'family': 'Arial', 'size': 14})
    #plt.savefig(f"Heatmap of Loss for {bird}.png", dpi=500)
    plt.show(block=True)

    results = bird, a,b

    with open("Heatmap_line_results.csv", "a+", newline="") as result_updater:
        csv_writer = csv.writer(result_updater)
        csv_writer.writerow(results)
        result_updater.close()