# Phase 1: 核心管线设计规格书

> 基于移动设备的人体建模与尺寸测量 — 一期算法验证
> 日期：2026-06-29

---

## 1. 目标

搭建从2张照片到6项人体尺寸的端到端算法管线，在本地GPU环境运行，验证精度 ≤ ±2cm（MAE）。

### 1.1 六项核心尺寸

| 尺寸 | 类型 | 定义 |
|------|------|------|
| 胸围 | 围度 | 腋窝高度水平截面凸包周长 |
| 腰围 | 围度 | 躯干最窄处水平截面凸包周长 |
| 臀围 | 围度 | 髋部最宽处水平截面凸包周长 |
| 肩宽 | 长度 | 左右肩峰关节点欧氏距离 |
| 袖长 | 长度 | 肩峰→肘→腕沿mesh表面测地线距离 |
| 裤长 | 长度 | 髋关节→膝→踝沿mesh表面测地线距离 |

---

## 2. 整体架构

```
正面照片 + 侧面照片 + 身高(cm)
         │
    ┌────▼────┐  模块1: 输入预处理
    │ 校验/缩放/裁剪
    └────┬────┘
         │ RGB图像
    ┌────▼────┐  模块2: MediaPipe Pose (CPU, ~0.05s)
    │ 33个3D关键点 × 2视角
    └────┬────┘
         │ keypoints
    ┌────▼────┐  模块3: SMPLify-X 拟合 (GPU, ~10-30s)
    │ SMPL-X mesh (10475顶点)
    └────┬────┘
         │ vertices + faces
    ┌────▼────┐  模块4: 尺寸测量提取 (CPU, ~0.1s)
    │ 6项尺寸 (cm)
    └────┬────┘
         │
    ┌────▼────┐  模块5: 输出
    │ JSON + 可视化
    └─────────┘
```

五个模块通过明确的输入/输出接口连接，每个模块可独立测试。

---

## 3. 模块详细设计

### 3.1 输入预处理 (src/input/preprocess.py)

**输入规格**：
- `front.jpg`：正面全身照，A-pose站立，紧身衣
- `side.jpg`：侧向全身照
- `height_cm`：float，用户提供的物理身高

**预处理流程**：
1. 校验：两张图均为RGB三通道、分辨率 ≥ 512px任一边、人物占画面比例 > 60%（由MediaPipe初步检测）
2. 缩放：短边统一缩放到1024px，保持宽高比
3. 裁剪：运行MediaPipe获取人体bounding box，扩展10%边距后裁剪

**输出**：
- `img_front`: numpy array (1024×N×3)
- `img_side`: numpy array (1024×N×3)
- `height_cm`: float

---

### 3.2 关键点检测 (src/keypoint/detect.py)

**依赖**：`mediapipe.solutions.pose.Pose()`

**输出格式**：每张图片输出 33×4 的numpy数组
- `[x_px, y_px, z_rel, visibility]` — x/y为像素坐标，z_rel为MediaPipe相对深度（非公制），visibility ∈ [0,1]

**33个关键点索引**（MediaPipe标准）：

| 索引 | 名称 | 索引 | 名称 |
|------|------|------|------|
| 0 | 鼻尖 | 11-12 | 左右肩 |
| 1-4 | 眼/耳 | 13-14 | 左右肘 |
| 5-6 | 左右肩 | 15-16 | 左右腕 |
| 7-8 | 左右耳下 | 23-24 | 左右髋 |
| 9-10 | 嘴 | 25-26 | 左右膝 |
| — | — | 27-32 | 左右踝/脚跟/脚尖 |

---

### 3.3 SMPL-X体型拟合 (src/fitting/) — 核心模块

**依赖**：
- `smplx` 官方Python包
- `torch >= 2.0`
- SMPL-X模型文件 `.pkl`（从smpl-x.is.tue.mpg.de注册下载）

**初始化**：
- 加载SMPL-X模型（`SMPLX('path/to/SMPLX_MALE.pkl')`或对应性别模型）
- 中性姿态（θ=0，T-pose）+ 零体型（β=0）作为优化起点
- 从关键点分布估算初始相机焦距

**已知限制**：
- MediaPipe仅提供33个身体关键点，不含手部和面部。SMPL-X手部/面部姿态参数不受关键点约束，将保持在初始化值附近，不影响身体尺寸提取。
- 性别默认使用中性模型(SMPLX_NEUTRAL)，男性/女性模型需在config中手动指定。

**三阶段优化**：

**Stage 1 — 姿态拟合**
| 参数 | 值 |
|------|------|
| 优化变量 | θ（全局旋转 + 23个关节，共55×3维） |
| 固定变量 | β = 0 |
| 损失函数 | L = w_kp * ‖π(J(θ)) − kp_obs‖₂ + w_reg * ‖θ‖₂ |
| 优化器 | L-BFGS |
| 迭代次数 | ~100 |
| 关键点权重 | 双肩、髋部权重×2，面部和手部权重×0.5 |

**Stage 2 — 体型拟合**
| 参数 | 值 |
|------|------|
| 优化变量 | β（10维体型参数） |
| 固定变量 | θ（Stage 1结果） |
| 损失函数 | L = w_kp * ‖π(J(β,θ)) − kp_obs‖₂ + w_β * ‖β‖₂ + w_h * |max_y(vertices) − min_y(vertices) − height_cm/100| |
| 迭代次数 | ~50 |

**Stage 3 — 联合精调**
| 参数 | 值 |
|------|------|
| 优化变量 | β + θ + 全局平移t |
| 损失函数 | Stage 2损失 + 侧视图深度顺序正则项 |
| 迭代次数 | ~50 |

**输出**：
- `betas`: torch.Tensor (1, 10) — 体型参数
- `pose`: torch.Tensor (1, 165) — 姿态参数
- `vertices`: torch.Tensor (1, 10475, 3) — mesh顶点（单位：米）
- `faces`: torch.Tensor (20894, 3) — 三角面索引

---

### 3.4 尺寸测量提取 (src/measure/)

**依赖**：`trimesh`, `numpy`

**解剖标志点定位**：

| 标志点 | 定位策略 |
|--------|----------|
| 腋窝高度 | SMPL-X joint index 对应肩/肘关节连线的中点偏下 |
| 胸围平面 | 腋窝高度处的y轴截平面 |
| 腰围平面 | 遍历y轴，找躯干段内截面周长最小的y值 |
| 臀围平面 | 遍历y轴，找髋部段内截面周长最大的y值 |
| 肩峰点 | SMPL-X左右肩关节（joints[2], joints[5]） |
| 肘关节点 | SMPL-X左右肘关节 |
| 腕关节点 | SMPL-X左右腕关节 |
| 髋关节点 | SMPL-X左右髋关节 |
| 膝关节点 | SMPL-X左右膝关节 |
| 踝关节点 | SMPL-X左右踝关节 |

**围度计算**：
```
1. 在目标y值处，用trimesh的section_multiplane切割mesh
2. 获取截面多边形顶点集
3. 对顶点集计算凸包（scipy.spatial.ConvexHull）
4. 凸包周长 = 围度(cm)
```

**长度计算**：
```
1. 肩宽 = ‖shoulder_L − shoulder_R‖ （欧氏距离）
2. 袖长 = geodesic(shoulder_L, elbow_L) + geodesic(elbow_L, wrist_L)
3. 裤长 = geodesic(hip_L, knee_L) + geodesic(knee_L, ankle_L)
```
测地线采用trimesh的`geodesic_distance`或沿mesh表面Dijkstra最短路径。

---

### 3.5 输出模块 (src/output/)

**输出文件**：

`measurements.json`：
```
{
  "height_cm": 170.0,
  "measurements": {
    "chest_cm": 92.3,
    "waist_cm": 76.1,
    "hip_cm": 95.8,
    "shoulder_width_cm": 42.5,
    "sleeve_length_cm": 58.2,
    "pants_length_cm": 102.7
  },
  "confidence": "medium",
  "warnings": []
}
```

`mesh_render.png`：正面/侧面mesh渲染图，使用pyrender或matplotlib 3D。

---

## 4. 文件结构

```
3D-SmartTailor/
├── data/                          # gitignore
│   ├── body_models/               # SMPL-X模型文件 (.pkl)
│   ├── test_photos/               # 测试用自拍照片
│   │   ├── subject_001/
│   │   │   ├── front.jpg
│   │   │   ├── side.jpg
│   │   │   └── ground_truth.json  # {"height": 170, "chest": 92, ...}
│   └── bodym/                     # BodyM数据集（可选）
├── src/
│   ├── __init__.py
│   ├── pipeline.py                # 主入口
│   ├── input/
│   │   ├── __init__.py
│   │   └── preprocess.py
│   ├── keypoint/
│   │   ├── __init__.py
│   │   └── detect.py
│   ├── fitting/
│   │   ├── __init__.py
│   │   ├── smplify.py             # 优化主循环
│   │   ├── losses.py              # 损失函数
│   │   └── camera.py              # 相机参数估算
│   ├── measure/
│   │   ├── __init__.py
│   │   ├── extract.py             # 尺寸提取主逻辑
│   │   ├── landmarks.py           # 解剖标志点定位
│   │   └── geometry.py            # 截面/凸包/测地线
│   └── output/
│       ├── __init__.py
│       ├── export.py              # JSON输出
│       └── render.py              # mesh渲染
├── tests/
│   ├── __init__.py
│   ├── test_preprocess.py
│   ├── test_measure.py            # BodyM网格精度验证
│   └── test_pipeline.py           # 端到端测试
├── notebooks/
│   └── evaluation.ipynb           # 精度评估与误差分析
├── config.yaml                    # 配置
├── .gitignore
└── README.md
```

---

## 5. 配置 (config.yaml)

```yaml
model:
  smplx_path: "data/body_models/SMPLX_MALE.pkl"  # 或 SMPLX_FEMALE.pkl, SMPLX_NEUTRAL.pkl
  gender: "auto"  # auto | male | female | neutral

fitting:
  stage1_iterations: 100
  stage2_iterations: 50
  stage3_iterations: 50
  loss_weights:
    keypoint: 1.0
    reg_pose: 0.001
    reg_shape: 0.01
    height: 10.0

input:
  target_short_side: 1024
  min_resolution: 512
  crop_margin: 0.1

output:
  render_mesh: true
  render_sections: true

device: "cuda"  # or "cpu"
gpu_id: 0
```

---

## 6. 验证方案

### 6.1 模块4独立精度验证（BodyM数据集）

| 项目 | 内容 |
|------|------|
| 数据 | BodyM数据集中的SMPL网格 + 14项人体尺寸真值 |
| 操作 | 对每个SMPL网格运行 `src/measure/` 模块 |
| 指标 | 6项尺寸的MAE（cm） |
| 目标 | 每项 MAE < ±1.0cm |
| 原因 | 测量模块本身的误差应远小于端到端目标(±2cm)，否则无法定位瓶颈 |

### 6.2 端到端精度验证（自拍照片）

| 项目 | 内容 |
|------|------|
| 数据 | 至少3人自拍照片（正面+侧面） + 软尺手工测量真值 |
| 操作 | 运行完整管线 `python src/pipeline.py` |
| 指标 | 6项尺寸的MAE（cm） |
| 目标 | 每项 MAE < ±2.0cm |

### 6.3 可视化检查

- SMPL拟合结果mesh渲染图：目视确认姿态/体型拟合是否合理
- 截面切割图：确认胸/腰/臀切割平面位置是否正确
- 管线日志：记录每步耗时和中间结果

---

## 7. 技术依赖

```
Python >= 3.10
├── mediapipe >= 0.10.0     # 关键点检测
├── torch >= 2.0            # PyTorch (GPU)
├── smplx >= 0.1.28         # SMPL-X官方包
├── trimesh >= 4.0          # 网格处理
├── pyrender >= 0.1.45      # mesh渲染
├── opencv-python >= 4.8    # 图像处理
├── numpy >= 1.24
├── scipy >= 1.11
├── pyyaml >= 6.0
├── matplotlib >= 3.7
└── Pillow >= 10.0
```

**硬件**：NVIDIA GPU，6GB+ 显存

---

## 8. 范围界定

### 8.1 一期包含

- MediaPipe Pose封装
- SMPLify-X三阶段优化拟合
- trimesh几何尺寸提取（6项）
- JSON输出 + mesh可视化
- BodyM数据集精度验证
- 少量测试照片端到端验证

### 8.2 一期不包含

- 衣物表面重建（ECON/ICON）
- 手机端部署
- 任何Web/API/UI
- 大规模数据集采集
- 版型参数生成
- 实时推理优化
- 多GPU分布式
- 用户注册/登录/数据管理

### 8.3 后续阶段预留

- `src/fitting/` 模块接口预留ECON可选接入点（`use_econ: false`配置项）
- `src/output/` 模块JSON格式预留置信度和warning字段
- `src/measure/` 尺寸列表为列表参数，可扩展至更多测量项

---

## 9. 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| SMPL-X模型下载受阻（需学术邮箱注册） | 中 | 备选SMPL(SMPLify)模型，顶点点数少但算法一致 |
| SMPLify-X拟合发散 | 中 | 三阶段策略本身是成熟方案；加入中间结果保存，失败可分析 |
| 腰位/臀位自动定位不准确 | 中 | 添加可视化验证步骤；备用手工标注y值做对比 |
| BodyM的尺寸定义与我们的6项不完全对齐 | 低 | BodyM有"shoulder-breadth"和"arm-length"可直接对应 |
| trimesh截面计算在非流形mesh上失败 | 低 | mesh来自SMPL-X官方模型，流形良好 |

---

## 10. 成功标准

一期通过的硬性条件：
1. `python src/pipeline.py --front <f> --side <s> --height <h>` 可在本地GPU完成推理并输出JSON
2. BodyM数据集上模块4（mesh→尺寸）MAE < ±1.0cm（6项全部）
3. 至少3人自拍照片端到端MAE < ±2.0cm（6项全部）
4. 生成mesh渲染图可供人工评估拟合质量
