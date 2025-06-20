"""
学生模板：波动方程FTCS解
文件：wave_equation_ftcs_student.py
重要：函数名称必须与参考答案一致！
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def u_t(x, C=1, d=0.1, sigma=0.3, L=1):
    """
    计算初始速度剖面 psi(x)。
    
    参数:
        x (np.ndarray): 位置数组。
        C (float): 振幅常数。
        d (float): 指数项的偏移量。
        sigma (float): 指数项的宽度。
        L (float): 弦的长度。
    返回:
        np.ndarray: 初始速度剖面。
    """
    # 实现公式: ψ(x) = C * [x*(L-x)/L^2] * exp(-(x-d)^2/(2*sigma^2))
    return C * (x * (L - x) / L**2) * np.exp(-(x - d)**2 / (2 * sigma**2))

def solve_wave_equation_ftcs(parameters):
    """
    使用FTCS有限差分法求解一维波动方程。
    
    参数:
        parameters (dict): 包含以下参数的字典：
            - 'a': 波速 (m/s)。
            - 'L': 弦的长度 (m)。
            - 'd': 初始速度剖面的偏移量 (m)。
            - 'C': 初始速度剖面的振幅常数 (m/s)。
            - 'sigma': 初始速度剖面的宽度 (m)。
            - 'dx': 空间步长 (m)。
            - 'dt': 时间步长 (s)。
            - 'total_time': 总模拟时间 (s)。
    返回:
        tuple: 包含以下内容的元组：
            - np.ndarray: 解数组 u(x, t)。
            - np.ndarray: 空间数组 x。
            - np.ndarray: 时间数组 t。
    """
    # 从参数字典中提取参数
    a = parameters['a']          # 波速
    L = parameters['L']          # 弦长
    d = parameters['d']          # 初始速度偏移量
    C = parameters['C']          # 初始速度振幅
    sigma = parameters['sigma']  # 初始速度宽度
    dx = parameters['dx']        # 空间步长
    dt = parameters['dt']        # 时间步长
    total_time = parameters['total_time']  # 总模拟时间
    
    # 计算空间和时间网格点数
    nx = int(L / dx) + 1  # 空间点数 (包括边界点)
    nt = int(total_time / dt) + 1  # 时间步数
    
    # 创建空间和时间网格
    x = np.linspace(0, L, nx)
    t = np.linspace(0, total_time, nt)
    
    # 初始化解数组 (nx个空间点 × nt个时间步)
    u = np.zeros((nx, nt))
    
    # 计算稳定性条件参数 c
    c = (a * dt / dx) ** 2
    
    # 检查稳定性条件
    if c >= 1:
        print(f"警告: c = {c:.6f} >= 1, 解可能不稳定!")
    
    # 初始条件: t=0 时刻位移为零
    u[:, 0] = 0
    
    # 计算第一个时间步 (t=1) 的值
    # 使用公式: u(x, dt) = ψ(x) * dt
    u[1:-1, 1] = u_t(x[1:-1], C, d, sigma, L) * dt
    
    # 应用边界条件 (弦两端固定)
    u[0, :] = 0    # x=0 处
    u[-1, :] = 0   # x=L 处
    
    # FTCS 主循环
    for j in range(1, nt - 1):
        # 使用FTCS公式更新内部点
        u[1:-1, j+1] = c * (u[2:, j] + u[:-2, j]) + 2 * (1 - c) * u[1:-1, j] - u[1:-1, j-1]
        
        # 确保边界条件 (虽然边界值在循环外已设置，但再次设置确保正确)
        u[0, j+1] = 0
        u[-1, j+1] = 0
    
    return u, x, t

if __name__ == "__main__":
    # 演示和测试
    params = {
        'a': 100,
        'L': 1,
        'd': 0.1,
        'C': 1,
        'sigma': 0.3,
        'dx': 0.01,
        'dt': 5e-5,
        'total_time': 0.1
    }
    u_sol, x_sol, t_sol = solve_wave_equation_ftcs(params)

    # 创建动画
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, xlim=(0, params['L']), ylim=(-0.002, 0.002))
    line, = ax.plot([], [], 'g-', lw=2)
    ax.set_title("1D Wave Equation (FTCS)")
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Displacement (m)")
    ax.grid(True)

    def update(frame):
        """更新动画帧的函数"""
        line.set_data(x_sol, u_sol[:, frame])
        ax.set_title(f"1D Wave Equation (FTCS) - Time = {t_sol[frame]:.4f} s")
        return line,

    # 创建动画
    ani = FuncAnimation(fig, update, frames=t_sol.size, interval=50, blit=True)
    
    # ani.save('wave_animation.gif', writer='pillow', fps=20)
    
    plt.show()

