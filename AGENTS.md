# CLAUDE.md — CSST Emulator (CEmulator)

## 软件包概览

> **重要：在进行任何代码修改前，先读取软件包概览文件获取完整的项目结构和模块信息：**
> `.claude/project-overview.md`
> **每次重新概览文件后，写入改动信息**

## 核心开发规则

### 1. 计算性能：正确性优先，然后加速

- 始终先确保数值计算正确（与 CLASS/CAMB 或模拟数据对比验证）
- 正确后最大化速度，手段包括：
  - **向量化 numpy 运算**（避免 Python 循环遍历数组元素）
  - **预计算插值表**（参考 `fermi_dirac_F_interp.npz` 的做法）
  - **共享网格上的单次梯形积分**（参考 `cosmology.py` 中 `comoving_distance` 的实现）

### 2. 缓存策略：避免重复计算

- **所有仅依赖宇宙学参数的插值结果必须缓存在实例变量中**，并在 `_sync_cosmologies()` 中清除
- 已有缓存模式参考：
  - `cosmology.py:83-86` — 共动距离插值缓存，`a_min` 变化时失效
  - `Emulator.py:950-963` — 按质量定义键值的插值缓存
- **禁止重复计算同一量**（如线性 Pk、sigma(R)、增长因子），应存储复用

### 3. LaTeX 公式要求

**每个科学计算函数必须在 docstring 中写入 LaTeX 公式**：
- 行内公式使用 `` :math:`...` ``
- 显示公式使用 `.. math::`
- 函数必须写明输入输出的单位和坐标系（proper/comoving）

### 4. 单位规范

| 物理量 | 单位 |
|--------|------|
| 波数 k | **h/Mpc** |
| 距离 r, χ | **Mpc/h**（共动），或 **Mpc**（`comoving_distance` 返回的不含 h 的 Mpc） |
| 质量 M | **Msun/h** |
| 数密度 n | **(Mpc/h)⁻³** |
| 功率谱 P(k) | **(Mpc/h)³** |
| H₀ | **km/s/Mpc**；h₀ = H₀/100（无量纲） |
| 中微子质量 mnu | **eV** |

**h 因子的区分：**
- `k [h/Mpc]` — 含 h 的波数
- `M [M_sun/h]` — 含 h 的质量
- `χ(z) [Mpc]` — `comoving_distance()` 返回的是 Mpc（乘以了 c/H₀），不含 h
- docstring 中始终标注单位，如 `k : array-like, wavenumber [h/Mpc]`

### 5. Proper 与 Comoving 坐标系

- **仿真器所有量默认使用 comoving 坐标系**
- 标度因子 a 关系：`r_prop(a) = a × r_com`，其中 `a = 1/(1+z)`
- `comoving_distance(z)` 返回共动距离 [Mpc]
- cb（冷暗物质+重子）vs M（含中微子的总物质）的区分由参数 `Pcb` 控制

### 6. 代码风格

- 注释使用英文
- 遵循 numpy docstring 格式
- 使用 `np.atleast_1d()` 规范化输入
- 红移数组必须严格递增（从高 z 到低 z）
- 参数范围检查使用 `checkbound=True`

### 7. 推送后的发布确认

- 每次成功执行远程推送（`git push`）后，必须询问用户是否需要变更版本号，以及是否需要将当前开发分支合并到 `master`。
- 两项决定分别确认；未经用户明确确认，不自动修改版本号或合并到 `master`。
