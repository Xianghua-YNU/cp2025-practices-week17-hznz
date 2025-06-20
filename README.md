[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/AIbOye9O)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=19806740)
# 计算物理练习题: 双曲方程和椭圆方程

**课程：** 计算物理 **主题：** 偏微分方程数值解法

## 学习目标
- 掌握椭圆型偏微分方程（拉普拉斯方程、泊松方程）的数值求解方法
- 理解双曲型偏微分方程（波动方程）的有限差分格式
- 熟练运用松弛迭代法、Jacobi方法、SOR方法等数值技巧
- 培养物理建模与数值计算相结合的科学计算思维

## 项目列表

### 椭圆型方程项目
1. **[平行板电容器](./ParallelPlateCapacitor/项目说明.md)** - 使用Jacobi和SOR方法求解理想平行板电容器的拉普拉斯方程
2. **[有限厚平行板电容器](./FiniteThicknessCapacitor/项目说明.md)** - 分析有限厚度导体板电容器中的电荷分布
3. **[正负电荷泊松方程](./PoissonEquationCharges/项目说明.md)** - 使用松弛迭代法求解正负电荷构成的二维泊松方程
4. **[松弛法求解常微分方程](./RelaxationMethodODE/项目说明.md)** - 应用松弛迭代技术求解边值问题

### 双曲型方程项目
5. **[波动方程FTCS解法](./WaveEquationFTCS/项目说明.md)** - 使用前向时间中心空间差分格式求解一维波动方程

## 技术要求
- **编程语言：** Python 3.8+
- **主要依赖：** numpy, scipy, matplotlib, pandas
- **开发环境：** 支持Jupyter Notebook或Python脚本
- **代码规范：** 遵循PEP 8标准，包含详细注释和文档字符串

## 提交要求
- 完成各项目的学生模板代码实现
- 填写实验报告，包含算法分析和结果讨论
- 确保代码通过自动化测试，风格规范
- 提交前运行测试验证功能正确性

## 评分标准
- **代码实现 (70%)：** 算法正确性、代码质量、测试通过率
- **实验报告 (20%)：** 物理理解、结果分析、问题讨论
- **代码规范 (10%)：** 注释完整性、命名规范、结构清晰
