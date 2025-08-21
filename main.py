import matplotlib.pyplot as plt
import numpy as np
import json

from periodogram import Periodogram
from simulate_phase import SimulatedPhase


def lab_period(param, check_times):

    mae_d = np.zeros(check_times)
    mae_h = np.zeros(check_times)

    phase = SimulatedPhase(param)
    unwrapping = Periodogram(param)

    data = {}
    # Repeated test
    for i in range(check_times):
        # 得到观测相位
        obs_phase, Bn, displace = phase.generate_sub_phase(i)

        # 估计 形变量
        d_est, h_est = unwrapping.estimation(obs_phase, Bn)

        mae_d[i] = np.mean(np.abs(d_est - displace))
        mae_h[i] = np.mean(np.abs(h_est - param["param_simulation"]["height"]))

        print("第 ", i, " 次，mae_d is ", mae_d[i])
        print("第 ", i, " 次，mae_h is ", mae_h[i])
        print("----------------------------------------------------------------")

    mae_d = np.mean(mae_d)
    mae_h = np.mean(mae_h)

    print("最后结果：")
    print("mae_d is ", mae_d)
    print("mae_h is ", mae_h)

    return data

if __name__ == "__main__":
    with open("params.json") as f:
        param = json.load(f)
    lab_period(param, 1000) 