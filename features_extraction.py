
import numpy as np
import pandas as pd

# fix random seed for reproducibility
np.random.seed(7)


def get_3d_sq_vel(x):
    square_speed = np.array(x.velX) ** 2 + np.array(x.velY) ** 2 + np.array(x.velZ) ** 2
    return square_speed.tolist()


def get_3d_vel(x):
    return (np.array(get_3d_sq_vel(x)) ** 0.5).tolist()


def total_energy_per_unit_mass(x):
    return (np.array(get_3d_sq_vel(x)) * 0.5 + np.array(x.posZ)).tolist()


def mypolyfit_y(y, deg):
    y = np.array(y)
    x = np.arange(0, y.shape[0]) / 2
    a = np.polyfit(x, y, deg)
    yy = 0
    for j in range(len(a)):
        yy += a[-j - 1] * x ** j
    return a, yy


def smooth_velocity(x, deg):
    a_x, vel_x = mypolyfit_y(x.velX, deg)
    a_y, vel_y = mypolyfit_y(x.velY, deg)
    a_z, vel_z = mypolyfit_y(x.velZ, deg)
    return (a_x, a_y, a_z), (vel_x, vel_y, vel_z)


def get_smoothed_columns(x):
    (a_x, a_y, a_z), (vel_x, vel_y, vel_z) = smooth_velocity(x, 1)
    smoothed_velocity_sq = vel_x ** 2 + vel_y ** 2 + vel_z ** 2
    smoothed_velocity = smoothed_velocity_sq ** 0.5
    _, smooth_pos_x = mypolyfit_y(x.posX, 2)
    energy = smoothed_velocity_sq * 0.5 + np.array(smooth_pos_x)

    return pd.Series([a_x[0], a_y[0], a_z[0], smoothed_velocity.tolist(), energy.tolist(),
                      vel_x, vel_y, vel_z])


def get_min_max_avg(x, columns_names):
    l = []
    for column_name in columns_names:
        data = x[column_name]
        l += [min(data), max(data), np.mean(data)]
    return pd.Series(l)


def get_polyfit_coefs(y, deg=4):
    coefs, _ = mypolyfit_y(y, deg)
    return coefs.tolist()


def get_coef_features(x, columns_names, deg):
    l = []
    for column in columns_names:
        coefs = get_polyfit_coefs(x[column], deg)
        l += coefs
    return pd.Series(l)


def create_features(df_dropped, deg):
    df_dropped2 = pd.concat([df_dropped.apply(get_3d_vel, axis=1),
                             df_dropped.apply(get_3d_sq_vel, axis=1),
                             df_dropped.apply(total_energy_per_unit_mass, axis=1)], axis=1)
    df_dropped2.columns = ["3dvel", "3dsq_vel", "e"]
    all_columns = ['posX', 'posY', 'posZ', 'velX', 'velY', 'velZ']
    df1 = df_dropped.apply(lambda x: get_coef_features(x, all_columns, deg), axis=1)
    df2 = df_dropped.apply(lambda x: get_min_max_avg(x, all_columns), axis=1)
    df3 = df_dropped2.apply(lambda x: get_coef_features(x, ["3dvel", "3dsq_vel", "e"], deg), axis=1)
    as_and_smoothed = df_dropped.apply(get_smoothed_columns, axis=1)
    as_and_smoothed.columns = ["ax", "ay", "az", "velocity", "energy", "velX", "velY", "velZ"]
    df4 = as_and_smoothed.apply(lambda x: get_min_max_avg(x, ["velocity", "energy", "velX", "velY", "velZ"]), axis=1)
    df_features = pd.concat([df1, df2, df3, df4], axis=1)

    return df_features



