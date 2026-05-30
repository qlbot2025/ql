# 老人跌倒检测系统 (Senior Fall Detection System)

实时监测和识别老人跌倒的完整解决方案，支持**跌倒过程识别**和**跌倒状态识别**。

## 功能特性

### 1. 跌倒过程识别 (Fall Progression)
- ✅ **平衡失去阶段** - 身体开始失控
- ✅ **下落阶段** - 快速向地面移动
- ✅ **撞击阶段** - 与地面接触
- ✅ **静止阶段** - 倒在地面上

### 2. 跌倒状态识别 (Fall Postures)
- ✅ **仰卧位** - 背部朝下
- ✅ **俯卧位** - 脸朝下
- ✅ **侧卧位** - 身体侧躺
- ✅ **坐着** - 臀部着地
- ✅ **蜷缩** - 身体弯曲

## 系统架构

```
fall-detection-system/
├── src/
│   ├── pose_detector.py          # 姿态识别模块
│   ├── fall_classifier.py        # 跌倒分类模块
│   ├── fall_stage_detector.py    # 跌倒阶段检测
│   ├── posture_classifier.py     # 跌倒状态分类
│   └── alert_system.py           # 告警系统
├── models/                        # 预训练模型
├── data/                         # 数据集
├── tests/                        # 单元测试
├── requirements.txt              # 依赖管理
└── demo.py                       # 演示脚本
```

## 技术栈

- **姿态识别**: MediaPipe Pose / OpenPose
- **深度学习**: PyTorch / TensorFlow
- **视频处理**: OpenCV
- **时序分析**: LSTM / Transformer
- **多传感器融合**: IMU数据（可选）

## 安装与使用

### 1. 环境配置
```bash
pip install -r requirements.txt
```

### 2. 运行演示
```bash
python demo.py --video input.mp4 --output result.mp4
```

### 3. 实时监测
```bash
python demo.py --camera 0
```

## 工作流程

1. **视频输入** → 摄像头或视频文件
2. **姿态检测** → 提取人体关键点
3. **特征提取** → 计算关键点位置、速度、加速度
4. **跌倒分类** → 判断是否发生跌倒
5. **阶段识别** → 确定跌倒所处的阶段
6. **状态识别** → 识别最终的跌倒姿态
7. **告警系统** → 触发紧急通知

## 模型性能

目标指标：
- 检测准确率: >95%
- 假正率: <5%
- 实时处理: 30 FPS@1080p

## 许可证

MIT License
