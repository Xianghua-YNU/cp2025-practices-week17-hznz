"""
学生模板：松弛迭代法解常微分方程
文件：relaxation_method_student.py
重要：函数名称必须与参考答案一致！
"""
import numpy as np
import matplotlib.pyplot as plt

def solve_ode(h, g, max_iter=10000, tol=1e-6):
    """
    实现松弛迭代法求解常微分方程 d²x/dt² = -g
    边界条件：x(0) = x(10) = 0（抛体运动问题）
    
    参数:
        h (float): 时间步长
        g (float): 重力加速度
        max_iter (int): 最大迭代次数
        tol (float): 收敛容差
    
    返回:
        tuple: (时间数组, 解数组)
    """
    # 初始化时间数组（确保包含t=10）
    t = np.arange(0, 10 + h, h)
    
    # 初始化解数组（满足边界条件）
    x = np.zeros_like(t)
    
    # 初始化变化量
    delta = float('inf')
    iteration = 0
    
    # 松弛迭代算法
    while delta > tol and iteration < max_iter:
        # 保存上一次的解
        x_old = x.copy()
        
        # 更新内部点 (索引1到倒数第二个)
        # 迭代公式: x_i = 0.5*(h²g + x_{i+1} + x_{i-1})
        x[1:-1] = 0.5 * (h*h*g + x_old[2:] + x_old[:-2])
        
        # 计算最大变化量
        delta = np.max(np.abs(x - x_old))
        
        # 更新迭代计数
        iteration += 1
    
    # 检查是否达到最大迭代次数
    if iteration >= max_iter:
        print(f"警告: 达到最大迭代次数 {max_iter} 次，当前误差 {delta:.2e} > {tol:.2e}")
    
    return t, x

if __name__ == "__main__":
    # 测试参数
    h = 10 / 100  # 时间步长
    g = 9.8       # 重力加速度
    
    # 调用求解函数
    t, x = solve_ode(h, g)
    
    # 绘制结果
    plt.plot(t, x)
    plt.xlabel('时间 (s)')
    plt.ylabel('高度 (m)')
    plt.title('抛体运动轨迹 (松弛迭代法)')
    plt.grid()
    plt.show()
