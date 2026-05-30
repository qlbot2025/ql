"""毫米波雷达检测模块 - 基于mmWave Radar影像的跌倒检测"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List
from scipy.ndimage import label, center_of_mass


@dataclass
class RadarFrame:
    """毫米波雷达帧数据"""
    range_doppler_map: np.ndarray  # 距离-速度图
    azimuth_elevation_map: np.ndarray  # 方位-仰角图
    velocity_map: np.ndarray  # 速度分布图
    frame_id: int
    timestamp: float
    range_resolution: float  # 米/像素
    doppler_resolution: float  # m/s/像素


@dataclass
class RadarTarget:
    """雷达目标物体"""
    range_m: float  # 距离（米）
    azimuth_deg: float  # 方位角（度）
    elevation_deg: float  # 仰角（度）
    velocity_mps: float  # 速度（m/s）
    rcs: float  # 雷达截面积
    confidence: float  # 置信度


class RadarDetector:
    """基于毫米波雷达的人体检测和跌倒判断器"""

    def __init__(self, range_resolution: float = 0.04, doppler_resolution: float = 0.13):
        """
        初始化雷达检测器
        
        Args:
            range_resolution: 距离分辨率（米）
            doppler_resolution: 速度分辨率（m/s）
        """
        self.range_resolution = range_resolution
        self.doppler_resolution = doppler_resolution
        self.radar_history: List[RadarFrame] = []
        self.target_history: List[List[RadarTarget]] = []

    def process_frame(self, range_doppler_map: np.ndarray,
                     azimuth_elevation_map: np.ndarray = None,
                     velocity_map: np.ndarray = None,
                     frame_id: int = 0, timestamp: float = 0.0) -> Optional[RadarFrame]:
        """
        处理雷达帧数据
        
        Args:
            range_doppler_map: 距离-速度图 (range_bins, doppler_bins)
            azimuth_elevation_map: 方位-仰角图（可选）
            velocity_map: 速度分布图（可选）
            frame_id: 帧编号
            timestamp: 时间戳
            
        Returns:
            RadarFrame 或 None
        """
        if range_doppler_map.size == 0:
            return None
        
        # 归一化到0-1
        rd_normalized = (range_doppler_map - np.min(range_doppler_map)) / (
            np.max(range_doppler_map) - np.min(range_doppler_map) + 1e-6
        )
        
        if azimuth_elevation_map is None:
            azimuth_elevation_map = np.zeros_like(range_doppler_map)
        
        if velocity_map is None:
            velocity_map = np.zeros_like(range_doppler_map)
        
        frame = RadarFrame(
            range_doppler_map=rd_normalized,
            azimuth_elevation_map=azimuth_elevation_map,
            velocity_map=velocity_map,
            frame_id=frame_id,
            timestamp=timestamp,
            range_resolution=self.range_resolution,
            doppler_resolution=self.doppler_resolution
        )
        
        self.radar_history.append(frame)
        
        if len(self.radar_history) > 30:
            self.radar_history = self.radar_history[-30:]
        
        return frame

    def detect_targets(self, frame: RadarFrame, threshold: float = 0.3) -> List[RadarTarget]:
        """
        从雷达图中检测目标物体
        
        Args:
            frame: 雷达帧
            threshold: 检测阈值
            
        Returns:
            目标列表
        """
        # 二值化
        binary_map = frame.range_doppler_map > threshold
        
        # 连通分量标记
        labeled_array, num_features = label(binary_map)
        
        targets = []
        
        for i in range(1, num_features + 1):
            component = (labeled_array == i)
            
            # 计算连通分量的中心
            center_coords = center_of_mass(component)
            
            if center_coords[0] is None:
                continue
            
            range_idx, doppler_idx = int(center_coords[0]), int(center_coords[1])
            
            # 转换为物理量
            range_m = range_idx * frame.range_resolution
            
            # Doppler索引转换为速度
            doppler_bins = frame.range_doppler_map.shape[1]
            doppler_centered = doppler_idx - doppler_bins // 2
            velocity_mps = doppler_centered * frame.doppler_resolution
            
            # 方位角（从azimuth_elevation_map读取）
            azimuth_deg = 0.0  # 可从方位图读取
            elevation_deg = 0.0
            
            if frame.azimuth_elevation_map is not None and frame.azimuth_elevation_map.size > 0:
                az_idx, el_idx = int(center_coords[0]), int(center_coords[1])
                azimuth_deg = float(frame.azimuth_elevation_map[az_idx, el_idx])
            
            # 计算RCS（雷达截面积）- 使用分量大小估计
            component_size = np.sum(component)
            rcs = component_size / (frame.range_doppler_map.shape[0] * frame.range_doppler_map.shape[1])
            
            # 置信度基于强度
            intensity = np.mean(frame.range_doppler_map[component])
            confidence = min(intensity, 1.0)
            
            target = RadarTarget(
                range_m=range_m,
                azimuth_deg=azimuth_deg,
                elevation_deg=elevation_deg,
                velocity_mps=velocity_mps,
                rcs=rcs,
                confidence=confidence
            )
            
            targets.append(target)
        
        # 按强度排序
        targets.sort(key=lambda t: t.confidence, reverse=True)
        
        self.target_history.append(targets)
        if len(self.target_history) > 30:
            self.target_history = self.target_history[-30:]
        
        return targets

    def detect_fall(self, frame: RadarFrame) -> Tuple[bool, float, dict]:
        """
        基于雷达检测是否跌倒
        
        Args:
            frame: 雷达帧
            
        Returns:
            (是否跌倒, 置信度, 特征字典)
        """
        targets = self.detect_targets(frame)
        
        if len(targets) == 0:
            return False, 0.0, {}
        
        # 选择最强的目标（人体）
        main_target = targets[0]
        
        # 计算特征
        features = self._compute_radar_features(frame, main_target)
        
        # 判断是否跌倒
        fall_score = 0.0
        
        # 特征1: 距离增加（下降）
        range_trend = self._compute_range_trend()
        if range_trend > 0.3:  # 距离快速增加
            fall_score += 0.3
        
        # 特征2: 垂直速度大（向下）
        if features['vertical_velocity'] < -0.5:  # m/s 向下
            fall_score += 0.3
        
        # 特征3: 水平速度急剧变化
        horizontal_speed_change = self._compute_velocity_change()
        if horizontal_speed_change > 0.8:
            fall_score += 0.2
        
        # 特征4: 信号分布变化（躺着时分布更分散）
        signal_spread = self._compute_signal_spread(frame)
        if signal_spread > 0.7:
            fall_score += 0.2
        
        is_fall = fall_score > 0.6
        confidence = min(fall_score, 1.0)
        
        return is_fall, confidence, features

    def _compute_radar_features(self, frame: RadarFrame, target: RadarTarget) -> dict:
        """
        计算雷达特征
        
        Args:
            frame: 雷达帧
            target: 目标物体
            
        Returns:
            特征字典
        """
        features = {}
        
        # 目标属性
        features['range_m'] = target.range_m
        features['azimuth_deg'] = target.azimuth_deg
        features['elevation_deg'] = target.elevation_deg
        features['velocity_mps'] = target.velocity_mps
        features['rcs'] = target.rcs
        
        # 计算垂直速度分量（从仰角推断）
        if target.elevation_deg != 0:
            vertical_velocity = target.velocity_mps * np.sin(target.elevation_deg * np.pi / 180)
        else:
            vertical_velocity = 0.0
        features['vertical_velocity'] = vertical_velocity
        
        # 信号强度
        features['signal_strength'] = target.confidence
        
        return features

    def _compute_range_trend(self) -> float:
        """
        计算距离的变化趋势
        
        Returns:
            范围变化率
        """
        if len(self.target_history) < 2:
            return 0.0
        
        recent_targets = self.target_history[-5:]
        
        if len(recent_targets[0]) == 0 or len(recent_targets[-1]) == 0:
            return 0.0
        
        range_start = recent_targets[0][0].range_m
        range_end = recent_targets[-1][0].range_m
        
        trend = (range_end - range_start) / (range_start + 1e-6)
        return trend

    def _compute_velocity_change(self) -> float:
        """
        计算速度的急剧变化
        
        Returns:
            速度变化率
        """
        if len(self.target_history) < 2:
            return 0.0
        
        recent_targets = self.target_history[-5:]
        
        velocities = []
        for targets_frame in recent_targets:
            if len(targets_frame) > 0:
                velocities.append(abs(targets_frame[0].velocity_mps))
        
        if len(velocities) < 2:
            return 0.0
        
        max_vel = max(velocities)
        min_vel = min(velocities)
        
        change = (max_vel - min_vel) / (np.mean(velocities) + 1e-6)
        return change

    def _compute_signal_spread(self, frame: RadarFrame) -> float:
        """
        计算信号分布的分散度
        
        Args:
            frame: 雷达帧
            
        Returns:
            信号分散度 (0-1)
        """
        rd_map = frame.range_doppler_map
        
        # 计算能量分布
        threshold = np.mean(rd_map)
        active_pixels = rd_map > threshold
        
        # 计算活跃像素的分散度
        if np.sum(active_pixels) == 0:
            return 0.0
        
        # 计算空间标准差
        y_coords, x_coords = np.where(active_pixels)
        
        y_std = np.std(y_coords) / (rd_map.shape[0] + 1e-6)
        x_std = np.std(x_coords) / (rd_map.shape[1] + 1e-6)
        
        spread = (y_std + x_std) / 2
        
        return min(spread, 1.0)

    def get_target_description(self, target: RadarTarget) -> str:
        """
        获取目标描述
        
        Args:
            target: 目标物体
            
        Returns:
            描述文本
        """
        return (
            f"距离: {target.range_m:.2f}m, "
            f"方位: {target.azimuth_deg:.1f}°, "
            f"速度: {target.velocity_mps:.2f}m/s, "
            f"可信度: {target.confidence:.2%}"
        )
