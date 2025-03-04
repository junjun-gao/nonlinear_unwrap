import matplotlib.pyplot as plt
import numpy as np
import json

class Periodogram:
    """Periodogram method for temporal phase unwrapping"""

    def __init__(self, param) -> None:

        self.param  = param

    def estimation(self, obs_phase, terrain_phase, displac):
        """
            Linear-Periodogram method
        """
        # ! 相位相减，然后缠绕，再去进行解缠操作
        # remove height term from delta_phase
        displac_phase = obs_phase - terrain_phase
        complex_signal = np.exp(1j * displac_phase)
        displac_phase = np.angle(complex_signal)
        
        # 计算相位差
        phase_diff = np.diff(displac_phase, append=displac_phase[-1])
        
        # 计算相位跳变
        phase_jumps = (phase_diff + np.pi) % (2 * np.pi) - np.pi
        
        # 解缠相位
        unwrapped_phase = np.cumsum(phase_jumps) + displac_phase[0]

        wavelength = self.param["sentinel-1"]["wavelength"]

        d_est = (unwrapped_phase * wavelength) / (4 * np.pi)

        return d_est

class SimulatedPhase:

    def __init__(self, param_file):
        self.param_file = param_file

    def generate_sub_phase(self, check_num):
        # h 相关的参数
        np.random.seed(check_num)
        # 卫星参数导入
        wavelength = self.param_file["sentinel-1"]["wavelength"]

        sigma_Bn = self.param_file["sentinel-1"]["Bn"]
        Bn = np.random.normal(0, sigma_Bn, self.param_file["Nifg"])

        incidence_angle = self.param_file["sentinel-1"]["incidence_angle"] * np.pi / 180

        R = self.param_file["sentinel-1"]["H"] / np.cos(incidence_angle)

        terrain_height = self.param_file["param_simulation"]["height"]

        # ! 在 形变仿真值的生成是否合理？是在 -0.15 到 0.15 之间随机生成一个小数，然后保留小数点后三位
        displace = self.param_file["param_simulation"]["disp"]
        displace = np.array([x * 1e-3 for x in displace])                # 转换单位为： mm -> m

        # x = range(len(displace))
        # plt.plot(x, displace, marker='o')
        # plt.title('displace')
        # plt.xlabel('index')
        # plt.ylabel('displace')
        # plt.savefig('line_plot.png', dpi=300)

        # add_radom_noise
        self.noise = np.random.normal(0, np.pi * self.param_file["noise_level"] / 180, self.param_file["Nifg"])

        # 生成真实相位
        terrain_phase  = (4 * np.pi * Bn * terrain_height) / (wavelength * R * np.sin(incidence_angle))
        displace_phase = (4 * np.pi * displace) / (wavelength)
        phase_true = terrain_phase + displace_phase + self.noise

        # 获取缠绕相位
        complex_signal = np.exp(1j * phase_true)
        obs_phase = np.angle(complex_signal)

        return obs_phase, terrain_phase, displace

def lab_period(param, check_times):
    # Initializes the data space
    mae_d = np.zeros(check_times)

    phase = SimulatedPhase(param)
    unwrapping = Periodogram(param)

    data = {}
    # Repeated test
    for i in range(check_times):
        # 得到观测相位
        obs_phase, terrain_phase, displace  = phase.generate_sub_phase(i)
        
        # 估计 形变量
        d_est = unwrapping.estimation(obs_phase, terrain_phase, displace)
        # print("d_est is ", d_est)
        # print("dislacement is ", displac)

        mae_d[i] = np.mean(np.abs(d_est - displace))

        print("第 ", i, " 次，mae_d is ", mae_d[i])
        
    # ! 这样的结果，算不算好呢？？？
    mae_d = np.mean(mae_d)

    print("最后结果：")
    print("mae_d is ", mae_d)

    return data

if __name__ == "__main__":
    with open("params.json") as f:
        param = json.load(f)
    lab_period(param, 1000)
    
