import matplotlib
matplotlib.use("QtAgg")
matplotlib.interactive(True)
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy as sc

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
    k= induced_drag_factor
    beta = np.clip(beta_opt(Vt, m, g, rho, Sw, B, k, Cdpro, delta), 0, 1, )

    epsilon = 1 - delta * (1 - beta)

    value = ((Di(Vt, k, m, g, B, rho, Sw, Cdpro=Cdpro, delta=delta)
              + Db(Vt=Vt, rho=rho, Sb=Sb, Cdb=Cdb)
              + Dpro(Vt=Vt, rho=rho, Sw=Sw, Cdpro=Cdpro, m=m, g=g, B=B, k=k, delta=delta)) * Vt
             / (m * g))
    return value


with open('species_dict.pkl', 'rb') as f:
    full_objective = pickle.load(f)


keys_to_copy = ["Fn", "Gf_W", "Vg", "Ge", "Cc","Gh","Ar","An"]

objective = {key: full_objective[key] for key in keys_to_copy}


baseline_parameters = {'Cdb': 0.1, 'Cdpro': 0.014}
CDBmin = 0.1
CDBmax = 0.3

CDpromin = 0.003
CDpromax = 0.097

optimized_results = []
best_fitting_parameters_per_bird = {}

induced_drag_factor = 1.1



def wrapper_multiple(p, objective, baseline_parameters, debug=False, penalty_weight=0.01):
    loss_list = []
    for bird, objective_dict in objective.items():
        loss = wrapper(p, objective_dict=objective_dict, baseline_parameters=baseline_parameters, g=9.8, rho=1.01, debug=debug, penalty_weight=penalty_weight)
        loss_list.append(loss)
        if bird not in best_fitting_parameters_per_bird or loss < best_fitting_parameters_per_bird[bird]['loss']:
            best_fitting_parameters_per_bird[bird] = {
                'Cdb': CDBmin + p[0] * (CDBmax - CDBmin),
                'Cdpro': CDpromin + p[1] * (CDpromax - CDpromin),
                'loss': loss
            }
    overall_loss = np.mean(loss_list)
    return overall_loss


def wrapper(p, objective_dict, baseline_parameters, g=9.8, rho=1.01, debug=False, penalty_weight=0.01):
    m = objective_dict['parameters']['m']
    Sb = objective_dict['parameters']['Sb']
    Sw = objective_dict['parameters']['Sw']
    B = objective_dict['parameters']['B']

    Cdb = CDBmin + p[0] * (CDBmax - CDBmin)
    Cdpro = CDpromin + p[1] * (CDpromax - CDpromin)

    delta = 1
    e = 1

    vz_array = []
    Vh_array = np.array(objective_dict["velocities"]['Horizontal']).astype(float)
    for i in Vh_array:
        vz_array.append(Vz_pennycuick(i, m=m, g=g, rho=rho, Sb=Sb, Sw=Sw, B=B, Cdpro=Cdpro, Cdb=Cdb, delta=delta, e=e))
    vz_array_objective = - np.array(objective_dict["velocities"]['Vertical']).astype(float)
    vz_array_err_objective = np.array(objective_dict["velocities"]['Vertical_error']).astype(float)
    vz_array = np.array(vz_array)

    loss = np.sqrt(np.sum((vz_array_objective - vz_array) ** 2 / vz_array_err_objective) / np.sum(1 / vz_array_err_objective))

    penalty_weight_cdpro = penalty_weight * 3
    penalty_weight_cdb = penalty_weight

    soft_penalty_cdpro = penalty_weight_cdpro * ((Cdpro - CDpromin) ** 2 + (Cdpro - CDpromax) ** 2) / ((CDpromax - CDpromin) ** 2)
    soft_penalty_cdb = penalty_weight_cdb * ((Cdb - CDBmin) ** 2 + (Cdb - CDBmax) ** 2) / ((CDBmax - CDBmin) ** 2)

    total_loss = loss + soft_penalty_cdpro + soft_penalty_cdb

    return total_loss

def wrapper_both(p):
    return wrapper_multiple([p[0], p[1]], objective, baseline_parameters)

def optimize_cdpro_only(objective, baseline_parameters, Cdb_fixed):
    def wrapper_cdpro_only(p_cdpro):
        p = [Cdb_fixed, p_cdpro[0]]
        return wrapper_multiple(p, objective, baseline_parameters)

    initial_guess = [(CDpromax + CDpromin) / 2]
    bounds = [(0, 1)]
    result = minimize(wrapper_cdpro_only, initial_guess, bounds=bounds)
    return result.x[0]  # Optimized Cdpro

def optimize_cdb_only(objective, baseline_parameters, Cdpro_fixed):
    def wrapper_cdb_only(p_cdb):
        p = [p_cdb[0], Cdpro_fixed]  # Fix Cdpro, optimize only Cdb
        return wrapper_multiple(p, objective, baseline_parameters)

    initial_guess = [(CDBmax + CDBmin) / 2]
    bounds = [(0, 1)]
    result = minimize(wrapper_cdb_only, initial_guess, bounds=bounds)
    return result.x[0]


def full_optimization_process(objective, baseline_parameters):
    #Start with initial Cdb value
    Cdb_fixed = (CDBmax + CDBmin) / 2  # Midpoint as initial guess

    #Optimize Cdpro
    optimized_Cdpro = optimize_cdpro_only(objective, baseline_parameters, Cdb_fixed)

    #Optimize Cdb
    optimized_Cdb = optimize_cdb_only(objective, baseline_parameters, optimized_Cdpro)

    return optimized_Cdb, optimized_Cdpro


result = minimize(wrapper_multiple, args=(objective, baseline_parameters), x0=[(CDBmax + CDBmin) / 2, 0.01],
                  method="CG")
optimized_Cdb, optimized_Cdpro = result.x
# Optimized Cdb
optimized_Cdb_final =CDBmin + optimized_Cdb * (CDBmax - CDBmin)
optimized_Cdpro_final =CDpromin + optimized_Cdpro * (CDpromax - CDpromin)
#Run the full optimization process


print(f"Optimized Cdb: {CDBmin + optimized_Cdb * (CDBmax - CDBmin)}")
print(f"Optimized Cdpro: {CDpromin + optimized_Cdpro  * (CDpromax - CDpromin)}")

my_bounds = [[0, 1], [0, 1]]
x0 = np.array([0.5, 0.5])

best_parameters = {}
for bird in best_fitting_parameters_per_bird:
    best_parameters[bird] = best_fitting_parameters_per_bird[bird]

def closed_form_polar_curve(Vt, m, g, Sb, Sw, e, rho, delta, B, Cdpro, Cdb):
    AR = B ** 2 / Sw
    k = induced_drag_factor
    B_hat = beta_hat(Vt=Vt, m=m, g=g, rho=rho, Sw=Sw, B=B, k=k, Cdpro=Cdpro, delta=delta)
    if B_hat < 1:
        B0 = ((8 * k * m ** 2 * g ** 2) / (delta * np.pi * rho ** 2 * B ** 2 * Cdpro * Sw)) ** (1 / 3)
        Vz = 1 / (m * g) * (((2 * k * m ** 2 * g ** 2) / (np.pi * B0 ** 2 * B ** 2 * rho) * Vt ** (5 / 3)) + ((1 - delta) * rho * Sw * Cdpro / 2 * Vt ** 3) + (delta * rho * Sw * Cdpro / 2 * B0 * Vt ** (5 / 3)) + (rho * Sb * Cdb / 2 * Vt ** 3))
        return Vz
    else:
        Vz = 1 / (m * g) * (((2 * k * m ** 2 * g ** 2) / (np.pi * Vt ** (8 / 3) * B ** 2 * rho) * Vt ** (5 / 3)) + ((1 - delta) * rho * Sw * Cdpro / 2 * Vt ** 3) + (delta * rho * Sw * Cdpro / 2 * Vt ** (4 / 3) * Vt ** (5 / 3)) + (rho * Sb * Cdb / 2 * Vt ** 3))
        return Vz

def closed_form_derivative(Vt, m, g, Sb, Sw, e, rho, delta, B, Cdpro, Cdb):
    AR = B ** 2 / Sw
    k = induced_drag_factor
    B_hat = beta_hat(Vt=Vt, m=m, g=g, rho=rho, Sw=Sw, B=B, k=k, Cdpro=Cdpro, delta=delta)
    if B_hat < 1:
        B0 = ((8 * k * m ** 2 * g ** 2) / (delta * np.pi * rho ** 2 * B ** 2 * Cdpro * Sw)) ** (1 / 3)
        Vz_prime = 1 / (m * g) * (5 / 3 * 2 * k * m ** 2 * g ** 2 / (np.pi * B0 ** 2 * B ** 2 * rho) * Vt ** (2 / 3) + (3 * (1 - delta) * rho * Sw * Cdpro / 2 * Vt ** 2) + (5 / 3 * (delta * rho * Sw * Cdpro / 2) * B0 * Vt ** (2 / 3)) + (3 * rho * Sb * Cdb / 2 * Vt ** 2))
        return Vz_prime
    else:
        Vz_prime = 1 / (m * g) * ((-2 * k * m ** 2 * g ** 2) / (np.pi * B ** 2 * rho * Vt ** 2) + (3 * (1 - delta) * rho * Sw * Cdpro / 2 * Vt ** 2) + ((3 / 2 * delta * rho * Sw * Cdpro) * (Vt ** 2)) + (3 * rho * Sb * Cdb / 2 * Vt ** 2))
        return Vz_prime

def optimal_strategy(Vt, m, g, Sb, Sw, e, rho, delta, B, Cdpro, Cdb):
    climbrate = closed_form_polar_curve(Vt=Vt, m=m, g=g, Sb=Sb, Sw=Sw, e=e, rho=rho, delta=delta, B=B, Cdpro=Cdpro, Cdb=Cdb) - Vt * closed_form_derivative(Vt=Vt, m=m, g=g, Sb=Sb, Sw=Sw, e=e, rho=rho, delta=delta, B=B, Cdpro=Cdpro, Cdb=Cdb)
    return climbrate

def result(objective, best_parameters, g=9.8, rho=1.01):
    for bird, objective_dict in objective.items():
        m = objective_dict['parameters']['m']
        Sb = objective_dict['parameters']['Sb']
        Sw = objective_dict['parameters']['Sw']
        B = objective_dict['parameters']['B']
        Cdb = best_parameters[bird]['Cdb']
        Cdpro = best_parameters[bird]['Cdpro']
        loss = best_parameters[bird]['loss']
        delta = 1
        e = 1
        preferences = objective_dict["preferences"]


        Vh_array = np.array(objective_dict["velocities"]['Horizontal']).astype(float)
        vz_array = [Vz_pennycuick(i, m=m, g=g, rho=rho, Sb=Sb, Sw=Sw, B=B, Cdpro=Cdpro, Cdb=Cdb, delta=delta, e=e) for i in Vh_array]
        #vz_array_objective = -np.array(objective_dict["velocities"]['Vertical']).astype(float)

        range_array = np.arange(5, 30.1, 0.1)
        standart_range = np.array(
            [Vz_pennycuick(i, m=m, g=g, rho=rho, Sb=Sb, Sw=Sw, B=B, Cdpro=Cdpro, Cdb=Cdb, delta=delta, e=e) for i in
             range_array])

        glide_polar = [
            closed_form_polar_curve(i, m=m, g=g, rho=rho, Sb=Sb, Sw=Sw, B=B, Cdpro=Cdpro, Cdb=Cdb, delta=delta, e=e) for
            i in
            range_array]

        derivative = [
            closed_form_derivative(i, m=m, g=g, rho=rho, Sb=Sb, Sw=Sw, B=B, Cdpro=Cdpro, Cdb=Cdb, delta=delta, e=e) for
            i in
            range_array]

        Vclimb = [optimal_strategy(i, m=m, g=g, rho=rho, Sb=Sb, Sw=Sw, B=B, Cdpro=Cdpro, Cdb=Cdb, delta=delta, e=e) for
                  i in range_array]

        opt_array = np.arange(0.01, 30.01, 0.01)

        Vclimb2 = [optimal_strategy(i, m=m, g=g, rho=rho, Sb=Sb, Sw=Sw, B=B, Cdpro=Cdpro, Cdb=Cdb, delta=delta, e=e) for
                   i in
                   opt_array]

        polar_from_zero = [
            closed_form_polar_curve(i, m=m, g=g, rho=rho, Sb=Sb, Sw=Sw, B=B, Cdpro=Cdpro, Cdb=Cdb, delta=delta, e=e) for
            i in
            opt_array]

        vz_array_objective = np.array(objective_dict["velocities"]['Vertical']).astype(float)
        vz_array_err_objective = np.array(objective_dict["velocities"]['Vertical_error']).astype(float)
        glide_polar = np.array(glide_polar) * -1
        derivative = np.array(derivative) * -1
        Vclimb = np.array(Vclimb) * -1
        standart_range = -standart_range
        Vclimb2 = np.array(Vclimb2) * -1

        polar_from_zero = np.array(polar_from_zero) * -1

        best_glide_speed_0 = opt_array[np.argmin(np.abs(Vclimb2 - 0))]
        best_glide_speed_1 = opt_array[np.argmin(np.abs(Vclimb2 - 1))]
        best_glide_speed_2 = opt_array[np.argmin(np.abs(Vclimb2 - 2))]
        best_glide_speed_3 = opt_array[np.argmin(np.abs(Vclimb2 - 3))]

        min_sink_x = np.where(polar_from_zero == max(polar_from_zero))
        min_sink_x = int(min_sink_x[0][0])
        speed_at_min_sink = opt_array[min_sink_x]
        min_sink_speed = closed_form_polar_curve(speed_at_min_sink, m=m, g=g, rho=rho, Sb=Sb, Sw=Sw, B=B, Cdpro=Cdpro,
                                                 Cdb=Cdb, delta=delta, e=e)
        min_sink_speed = -min_sink_speed

        best_glide_y = closed_form_polar_curve(best_glide_speed_0, m=m, g=g, rho=rho, Sb=Sb, Sw=Sw, B=B, Cdpro=Cdpro,
                                               Cdb=Cdb, delta=delta, e=e)
        best_glide_y = -best_glide_y

        best_glide_ratio = abs(best_glide_speed_0 / best_glide_y)

        linear_fit, _ = sc.optimize.curve_fit(f_linear, [0, 1, 2, 3],
                                              [best_glide_speed_0, best_glide_speed_1, best_glide_speed_2,
                                               best_glide_speed_3])
        a_linear, b_linear = linear_fit
        x_linear_fit = f_linear(np.arange(0, 3.1, 0.1), a_linear, b_linear)

        fig = plt.figure()
        plt.cla()
        plt.grid()

        plt.errorbar(Vh_array, vz_array_objective, vz_array_err_objective, fmt='o',label="observed gliding points")

        plt.plot(x_linear_fit, np.arange(0, 3.1, 0.1), c="red")

        # plt.plot(Vh_array,vz_array,c="green")
        # plt.plot(range_array, standart_range, '-', c="black")
        # plt.plot(range_array, derivative, '--', c="blue")

        plt.plot(range_array, glide_polar, c="red",label="estimated glide polar")

        #plt.plot(range_array, Vclimb, '--', c="red")

        plt.title(f"Glide Polar of {bird}", fontsize=20)
        plt.scatter(speed_at_min_sink, min_sink_speed, marker="o", c="red",label="minimum sink")
        plt.scatter(best_glide_speed_0, best_glide_y, marker="^", c="red", label= "maximum glide")

        plt.xlabel("Horizontal Speed (m/s)",fontsize=15)
        plt.ylabel("Vertical Speed (m/s)",fontsize=15)

        plt.legend()
        plt.ylim((-3, 0))
        plt.xlim((0, 20.5))
        # plt.title(f'{Cdpro=:.2g}, {Cdb=:.2g}, {delta=:.2g}, {e=:.2g} ')
        fig.set_size_inches(8, 11)



        #plt.close(fig)
        plt.show(block=True)