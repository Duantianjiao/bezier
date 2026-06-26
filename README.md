# bezier
bezier曲线演示
# GPU 加速的贝塞尔曲线绘制（Taichi 实现）

这是一个基于 Taichi 编程语言实现的贝塞尔曲线绘制程序，通过 GPU 并行计算和显存优化，实现了 60FPS 流畅绘制，核心优化点在于减少 CPU-GPU 内存通信次数、利用 GPU 并行化像素绘制。

## 功能特点
- 支持鼠标点击添加任意数量控制点（上限 100 个）
- 实时绘制贝塞尔曲线，GPU 加速像素渲染
- 支持快捷键 `c` 清空画布
- 红色圆点显示控制点，灰色线条连接控制点
- 绿色曲线为贝塞尔曲线（De Casteljau 算法实现）

## 技术栈
- Python 3.x
- Taichi（GPU 计算框架）
- NumPy（数据处理）

## 核心优化思路
1. **显存缓冲区复用**：新增 GPU 端曲线坐标缓冲区，避免频繁内存申请
2. **批量数据传输**：CPU 端计算完所有曲线点后一次性传输到 GPU（1 次通信替代 1000+ 次）
3. **GPU 并行绘制**：像素点亮操作交给 GPU 并行执行，大幅提升渲染效率
4. **De Casteljau 算法**：纯 Python 递归实现贝塞尔曲线核心计算

## 快速开始

### 环境安装
```bash
# 安装 Taichi（支持 GPU 版本）
pip install taichi

# 安装 NumPy
pip install numpy
```

### 运行程序
```bash
python bezier.py
```

### 操作说明
- **添加控制点**：鼠标左键点击画布任意位置
- **清空画布**：按下键盘 `c` 键
- **退出程序**：关闭窗口即可

## 核心代码解析

### 1. 初始化配置
```python
import taichi as ti
import numpy as np

# 使用 GPU 后端
ti.init(arch=ti.gpu)

WIDTH = 800
HEIGHT = 800
MAX_CONTROL_POINTS = 100
NUM_SEGMENTS = 1000  # 曲线采样点数量

# 像素缓冲区（GPU 端）
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
# 曲线坐标 GPU 缓冲区（核心优化点）
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS + 1)
```

### 2. De Casteljau 算法实现
```python
def de_casteljau(points, t):
    """纯 Python 递归实现 De Casteljau 算法"""
    if len(points) == 1:
        return points[0]
    next_points = []
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]
        x = (1.0 - t) * p0[0] + t * p1[0]
        y = (1.0 - t) * p0[1] + t * p1[1]
        next_points.append([x, y])
    return de_casteljau(next_points, t)
```

### 3. GPU 并行绘制核心
```python
@ti.kernel
def draw_curve_kernel(n: ti.i32):
    # Taichi 自动将该循环在 GPU 上并行执行
    for i in range(n):
        pt = curve_points_field[i]
        x_pixel = ti.cast(pt[0] * WIDTH, ti.i32)
        y_pixel = ti.cast(pt[1] * HEIGHT, ti.i32)
        if 0 <= x_pixel < WIDTH and 0 <= y_pixel < HEIGHT:
            pixels[x_pixel, y_pixel] = ti.Vector([0.0, 1.0, 0.0])
```

### 4. 主循环逻辑
```python
def main():
    window = ti.ui.Window("Bezier Curve (60 FPS Restored)", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    control_points = []

    while window.running:
        # 事件处理（添加控制点/清空）
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB:
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()
                    control_points.append(pos)
            elif e.key == 'c':
                control_points = []

        # 清空像素缓冲区
        clear_pixels()

        # 曲线计算与绘制
        current_count = len(control_points)
        if current_count >= 2:
            # CPU 端计算所有曲线点
            curve_points_np = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
            for t_int in range(NUM_SEGMENTS + 1):
                t = t_int / NUM_SEGMENTS
                curve_points_np[t_int] = de_casteljau(control_points, t)
            
            # 批量传输到 GPU（核心优化）
            curve_points_field.from_numpy(curve_points_np)
            
            # GPU 并行绘制
            draw_curve_kernel(NUM_SEGMENTS + 1)

        # 绘制控制点和辅助线
        # ...（省略 UI 绘制代码）
        
        window.show()
```

## 性能对比
| 优化方式 | 帧率 | 内存通信次数 | 绘制耗时 |
|----------|------|--------------|----------|
| 原生 CPU 逐点绘制 | ~10 FPS | 1000+ 次/帧 | ~100ms/帧 |
| GPU 批量传输+并行绘制 | ~60 FPS | 1 次/帧 | ~16ms/帧 |


## 演示视频
<img width="480" height="285" alt="bezier曲线演示" src="https://github.com/user-attachments/assets/834a77e7-98e6-488b-b2ca-8984c849a452" />
