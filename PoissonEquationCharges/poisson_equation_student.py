#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def solve_poisson_equation(M: int = 100, target: float = 1e-6, max_iterations: int = 10000) -> Tuple[np.ndarray, int, bool]:
    """
    使用松弛迭代法求解二维泊松方程
    
    参数:
        M (int): 每边的网格点数，默认100
        target (float): 收敛精度，默认1e-6
        max_iterations (int): 最大迭代次数，默认10000
    
    返回:
        tuple: (phi, iterations, converged)
            phi (np.ndarray): 电势分布数组，形状为(M+1, M+1)
            iterations (int): 实际迭代次数
            converged (bool): 是否收敛
    """
    # 设置网格间距
    h = 1.0
    
    # 初始化电势数组
    phi = np.zeros((M+1, M+1), dtype=float)
    
    # 创建电荷密度数组
    rho = np.zeros((M+1, M+1), dtype=float)
    
    # 设置电荷分布 - 固定位置（与测试用例一致）
    # 正电荷区域：60:80, 20:40（对于M=100）
    # 负电荷区域：20:40, 60:80（对于M=100）
    # 对于不同的M，按比例缩放电荷区域
    
    # 计算缩放因子
    scale = M / 100.0
    
    # 计算电荷区域边界
    # 正电荷区域 (x: 60-80, y: 20-40)
    pos_x_start = int(60 * scale)
    pos_x_end = int(80 * scale) + 1
    pos_y_start = int(20 * scale)
    pos_y_end = int(40 * scale) + 1
    
    # 负电荷区域 (x: 20-40, y: 60-80)
    neg_x_start = int(20 * scale)
    neg_x_end = int(40 * scale) + 1
    neg_y_start = int(60 * scale)
    neg_y_end = int(80 * scale) + 1
    
    # 正电荷区域
    rho[pos_x_start:pos_x_end, pos_y_start:pos_y_end] = 1.0
    # 负电荷区域
    rho[neg_x_start:neg_x_end, neg_y_start:neg_y_end] = -1.0
    
    # 初始化迭代变量
    delta = 1.0
    iterations = 0
    converged = False
    
    # 创建前一步的电势数组副本
    phi_prev = np.copy(phi)
    
    # 主迭代循环
    while delta > target and iterations < max_iterations:
        # 保存当前状态
        phi_prev[:, :] = phi
        
        # 使用有限差分公式更新内部网格点
        phi[1:-1, 1:-1] = 0.25 * (
            phi_prev[0:-2, 1:-1] +   # i-1,j
            phi_prev[2:, 1:-1] +     # i+1,j
            phi_prev[1:-1, 0:-2] +   # i,j-1
            phi_prev[1:-1, 2:] +     # i,j+1
            h * h * rho[1:-1, 1:-1]  # 电荷密度项
        )
        
        # 计算最大变化量
        delta = np.max(np.abs(phi - phi_prev))
        
        # 增加迭代计数
        iterations += 1
    
    # 检查是否收敛
    converged = (delta <= target)
    
    return phi, iterations, converged

def visualize_solution(phi: np.ndarray, M: int = 100) -> None:
    """
    可视化电势分布
    
    参数:
        phi (np.ndarray): 电势分布数组
        M (int): 网格大小
    """
    plt.figure(figsize=(10, 8))
    
    # 绘制电势分布
    im = plt.imshow(phi, extent=[0, M, 0, M], origin='lower', cmap='RdBu_r')
    
    # 添加颜色条
    cbar = plt.colorbar(im)
    cbar.set_label('电势 (V)', fontsize=12)
    
    # 计算电荷区域
    scale = M / 100.0
    # 正电荷区域
    pos_x_start = int(60 * scale)
    pos_x_end = int(80 * scale)
    pos_y_start = int(20 * scale)
    pos_y_end = int(40 * scale)
    
    # 负电荷区域
    neg_x_start = int(20 * scale)
    neg_x_end = int(40 * scale)
    neg_y_start = int(60 * scale)
    neg_y_end = int(80 * scale)
    
    # 标注正电荷位置
    plt.fill(
        [pos_x_start, pos_x_end, pos_x_end, pos_x_start, pos_x_start],
        [pos_y_start, pos_y_start, pos_y_end, pos_y_end, pos_y_start],
        'r', alpha=0.3, label='正电荷 (+1 C/m²)'
    )
    
    # 标注负电荷位置
    plt.fill(
        [neg_x_start, neg_x_end, neg_x_end, neg_x_start, neg_x_start],
        [neg_y_start, neg_y_start, neg_y_end, neg_y_end, neg_y_start],
        'b', alpha=0.3, label='负电荷 (-1 C/m²)'
    )
    
    # 添加标题和标签
    plt.title('二维泊松方程电势分布', fontsize=14)
    plt.xlabel('X 坐标', fontsize=12)
    plt.ylabel('Y 坐标', fontsize=12)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

def analyze_solution(phi: np.ndarray, iterations: int, converged: bool) -> None:
    """
    分析解的统计信息
    
    参数:
        phi (np.ndarray): 电势分布数组
        iterations (int): 迭代次数
        converged (bool): 收敛状态
    """
    print("\n=== 泊松方程求解结果分析 ===")
    print(f"迭代次数: {iterations}")
    print(f"是否收敛: {'是' if converged else '否'}")
    print(f"最大电势: {np.max(phi):.6f} V")
    print(f"最小电势: {np.min(phi):.6f} V")
    print(f"电势范围: {np.max(phi) - np.min(phi):.6f} V")
    
    # 找到极值位置
    max_idx = np.unravel_index(np.argmax(phi), phi.shape)
    min_idx = np.unravel_index(np.argmin(phi), phi.shape)
    print(f"最大电势位置: ({max_idx[0]}, {max_idx[1]})")
    print(f"最小电势位置: ({min_idx[0]}, {min_idx[1]})")

if __name__ == "__main__":
    print("开始求解二维泊松方程...")
    
    # 设置参数
    M = 100
    target = 1e-6
    max_iter = 10000
    
    # 调用求解函数
    phi, iterations, converged = solve_poisson_equation(M, target, max_iter)
    
    # 分析结果
    analyze_solution(phi, iterations, converged)
    
    # 可视化结果
    visualize_solution(phi, M)
    
    print("求解完成！")
