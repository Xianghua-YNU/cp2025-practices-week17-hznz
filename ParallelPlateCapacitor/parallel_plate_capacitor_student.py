"""学生模板：ParallelPlateCapacitor
文件：parallel_plate_capacitor_student.py
重要：函数名称必须与参考答案一致！
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def solve_laplace_jacobi(xgrid, ygrid, w, d, tol=1e-5):
    """
    使用Jacobi迭代法求解拉普拉斯方程
    
    参数:
        xgrid (int): x方向网格点数
        ygrid (int): y方向网格点数
        w (int): 平行板宽度
        d (int): 平行板间距
        tol (float): 收敛容差
    
    返回:
        tuple: (potential_array, iterations, convergence_history)
    """
    # 初始化电势网格
    u = np.zeros((ygrid, xgrid))
    
    # 计算平行板位置
    xL = (xgrid - w) // 2
    xR = (xgrid + w) // 2
    yB = (ygrid - d) // 2
    yT = (ygrid + d) // 2
    
    # 设置边界条件
    # 箱体边界接地
    u[0, :] = 0      # 下边界
    u[-1, :] = 0     # 上边界
    u[:, 0] = 0      # 左边界
    u[:, -1] = 0     # 右边界
    
    # 平行板电势
    u[yB, xL:xR] = -100  # 下板
    u[yT, xL:xR] = 100   # 上板
    
    # 迭代参数
    iteration = 0
    max_change = 1.0
    convergence_history = []
    
    # Jacobi迭代
    while max_change > tol:
        u_old = u.copy()
        max_change = 0.0
        
        # 遍历内部点
        for i in range(1, ygrid-1):
            for j in range(1, xgrid-1):
                # 跳过平行板区域
                if (i == yB and j >= xL and j < xR) or (i == yT and j >= xL and j < xR):
                    continue
                
                # Jacobi迭代公式
                new_value = 0.25 * (u_old[i+1, j] + u_old[i-1, j] + 
                                    u_old[i, j+1] + u_old[i, j-1])
                
                # 更新最大变化量
                change = abs(new_value - u[i, j])
                if change > max_change:
                    max_change = change
                
                u[i, j] = new_value
        
        # 重置平行板电势（迭代可能改变了它们）
        u[yB, xL:xR] = -100
        u[yT, xL:xR] = 100
        
        iteration += 1
        convergence_history.append(max_change)
    
    return u, iteration, convergence_history

def solve_laplace_sor(xgrid, ygrid, w, d, omega=1.25, Niter=1000, tol=1e-5):
    """
    实现SOR算法求解平行板电容器的电势分布
    
    参数:
        xgrid (int): x方向网格点数
        ygrid (int): y方向网格点数
        w (int): 平行板宽度
        d (int): 平行板间距
        omega (float): 松弛因子
        Niter (int): 最大迭代次数
        tol (float): 收敛容差
    返回:
        tuple: (电势分布数组, 迭代次数, 收敛历史)
    """
    # 初始化电势网格
    u = np.zeros((ygrid, xgrid))
    
    # 计算平行板位置
    xL = (xgrid - w) // 2
    xR = (xgrid + w) // 2
    yB = (ygrid - d) // 2
    yT = (ygrid + d) // 2
    
    # 设置边界条件
    # 箱体边界接地
    u[0, :] = 0      # 下边界
    u[-1, :] = 0     # 上边界
    u[:, 0] = 0      # 左边界
    u[:, -1] = 0     # 右边界
    
    # 平行板电势
    u[yB, xL:xR] = -100  # 下板
    u[yT, xL:xR] = 100   # 上板
    
    # 迭代参数
    iteration = 0
    max_change = 1.0
    convergence_history = []
    
    # SOR迭代
    for it in range(Niter):
        max_change = 0.0
        
        # 遍历内部点
        for i in range(1, ygrid-1):
            for j in range(1, xgrid-1):
                # 跳过平行板区域
                if (i == yB and j >= xL and j < xR) or (i == yT and j >= xL and j < xR):
                    continue
                
                # 计算Jacobi迭代值
                jacobi_value = 0.25 * (u[i+1, j] + u[i-1, j] + 
                                       u[i, j+1] + u[i, j-1])
                
                # 应用SOR公式
                new_value = u[i, j] + omega * (jacobi_value - u[i, j])
                
                # 更新最大变化量
                change = abs(new_value - u[i, j])
                if change > max_change:
                    max_change = change
                
                u[i, j] = new_value
        
        # 重置平行板电势
        u[yB, xL:xR] = -100
        u[yT, xL:xR] = 100
        
        iteration += 1
        convergence_history.append(max_change)
        
        # 检查收敛条件
        if max_change < tol:
            break
    
    return u, iteration, convergence_history

def plot_results(x, y, u, method_name):
    """
    绘制三维电势分布、等势线和电场线
    
    参数:
        x (array): X坐标数组
        y (array): Y坐标数组
        u (array): 电势分布数组
        method_name (str): 方法名称
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
    
    # 创建3D图形
    fig = plt.figure(figsize=(14, 6))
    
    # 第一个子图：3D电势分布
    ax1 = fig.add_subplot(121, projection='3d')
    X, Y = np.meshgrid(x, y)
    ax1.plot_wireframe(X, Y, u, rstride=2, cstride=2, linewidth=0.5, color='blue')
    
    # 添加等势线投影
    contours = ax1.contour(X, Y, u, 10, cmap='viridis', offset=np.min(u)-10)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('电势 (V)')
    ax1.set_title(f'{method_name} 三维电势分布')
    
    # 第二个子图：2D等势线和电场线
    ax2 = fig.add_subplot(122)
    
    # 绘制等势线
    contour_plot = ax2.contour(X, Y, u, 15, cmap='viridis')
    plt.clabel(contour_plot, inline=1, fontsize=8)
    
    # 计算电场（电场强度是电势梯度的负值）
    Ey, Ex = np.gradient(-u, np.diff(y)[0], np.diff(x)[0])
    
    # 绘制电场线
    ax2.streamplot(X, Y, Ex, Ey, density=2, color='red', linewidth=1, arrowsize=1)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title(f'{method_name} 等势线与电场线')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 网格参数
    xgrid, ygrid = 60, 60  # 网格尺寸
    w, d = 30, 20          # 平行板宽度和间距
    
    # Jacobi方法测试
    start_time = time.time()
    u_jacobi, iter_jacobi, conv_jacobi = solve_laplace_jacobi(xgrid, ygrid, w, d)
    jacobi_time = time.time() - start_time
    
    # SOR方法测试
    start_time = time.time()
    u_sor, iter_sor, conv_sor = solve_laplace_sor(xgrid, ygrid, w, d)
    sor_time = time.time() - start_time
    
    # 创建坐标数组
    x = np.linspace(0, 1, xgrid)
    y = np.linspace(0, 1, ygrid)
    
    # 绘制结果
    plot_results(x, y, u_jacobi, "Jacobi方法")
    plot_results(x, y, u_sor, "SOR方法")
    
    # 输出性能比较
    print(f"Jacobi方法: 迭代次数={iter_jacobi}, 计算时间={jacobi_time:.4f}秒")
    print(f"SOR方法: 迭代次数={iter_sor}, 计算时间={sor_time:.4f}秒")
    
    # 绘制收敛历史
    plt.figure()
    plt.semilogy(conv_jacobi, label='Jacobi')
    plt.semilogy(conv_sor, label='SOR')
    plt.xlabel('迭代次数')
    plt.ylabel('最大变化量 (log scale)')
    plt.title('收敛历史比较')
    plt.legend()
    plt.grid(True)
    plt.show()
