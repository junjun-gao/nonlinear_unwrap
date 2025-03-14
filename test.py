'''
Author: junjun-gao gao.junjun@outlook.com
Date: 2025-03-14 13:48:48
LastEditors: junjun-gao gao.junjun@outlook.com
LastEditTime: 2025-03-14 13:50:10
FilePath: /periodogram/test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np

def unwrap_phase(phase_data):
    # 初始化解缠后的数据
    unwrapped_data = np.copy(phase_data)

    for i in range(1, len(phase_data)):
        # 计算相邻元素的差值
        diff = phase_data[i] - phase_data[i - 1]

        # 如果差值大于pi，则说明发生了正向跳跃，进行解缠
        if diff > np.pi:
            unwrapped_data[i:] -= 2 * np.pi

        # 如果差值小于负pi，则说明发生了反向跳跃，进行解缠
        elif diff < -np.pi:
            unwrapped_data[i:] += 2 * np.pi

    return unwrapped_data

# 测试数据
phase_data = np.array([0.1, 3.2, -3.1, -2.9, 3.0, -2.8])
unwrapped_data = unwrap_phase(phase_data)

print("原始相位数据:", phase_data)
print("解缠后的相位数据:", unwrapped_data)

