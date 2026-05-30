#!/usr/bin/env python3
"""多模态融合演示脚本 - 支持RGB视频、点云和毫米波雷达"""

import cv2
import numpy as np
import argparse
from src import (
    PoseDetector, FallClassifier, FallStageDetector, PostureClassifier, AlertSystem,
    PointCloudDetector, RadarDetector, MultiModalFusion,
    ModalityType, FusionStrategy, ModalityResult
)


def main():
    parser = argparse.ArgumentParser(description="多模态融合跌倒检测系统")
    parser.add_argument("--video", type=str, help="输入视频文件路径")
    parser.add_argument("--camera", type=int, default=-1, help="摄像头编号")
    parser.add_argument("--output", type=str, help="输出视频文件路径")
    parser.add_argument("--fusion-strategy", type=str, default="hybrid",
                       choices=["early", "late", "hybrid"],
                       help="融合策略")
    parser.add_argument("--threshold", type=float, default=0.6,
                       help="跌倒置信度阈值")
    
    args = parser.parse_args()
    
    # 初始化RGB模块
    pose_detector = PoseDetector(model_complexity=1)
    fall_classifier = FallClassifier(window_size=10)
    fall_stage_detector = FallStageDetector(window_size=30)
    posture_classifier = PostureClassifier()
    alert_system = AlertSystem(fall_threshold=args.threshold)
    alert_system.register_handler(alert_system.print_alert)
    
    # 初始化其他模态模块
    point_cloud_detector = PointCloudDetector()
    radar_detector = RadarDetector()
    
    # 初始化融合器
    fusion_strategy = {
        "early": FusionStrategy.EARLY_FUSION,
        "late": FusionStrategy.LATE_FUSION,
        "hybrid": FusionStrategy.HYBRID_FUSION,
    }[args.fusion_strategy]
    
    fusion = MultiModalFusion(strategy=fusion_strategy)
    
    # 选择视频源
    if args.camera >= 0:
        cap = cv2.VideoCapture(args.camera)
    elif args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        print("错误：必须指定 --video 或 --camera")
        return
    
    if not cap.isOpened():
        print("错误：无法打开视频源")
        return
    
    frame_count = 0
    
    print(f"启动多模态融合检测系统")
    print(f"融合策略: {args.fusion_strategy}")
    print(f"跌倒置信度阈值: {args.threshold}")
    print("按 'q' 退出")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        timestamp = frame_count / cap.get(cv2.CAP_PROP_FPS)
        
        display_frame = frame.copy()
        
        # RGB视频检测
        pose_frame = pose_detector.detect(frame, frame_id=frame_count, timestamp=timestamp)
        
        if pose_frame is not None:
            is_fall, fall_confidence = fall_classifier.classify(pose_frame)
            fall_stage, stage_confidence = fall_stage_detector.update(pose_frame, is_fall)
            posture_type, posture_confidence = posture_classifier.classify(pose_frame)
            
            display_frame = pose_detector.draw_landmarks(frame, pose_frame)
            
            # 绘制结果
            y_offset = 30
            
            cv2.putText(display_frame, f"Frame: {frame_count}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            y_offset += 30
            fall_text = f"Fall: {'YES' if is_fall else 'NO'} ({fall_confidence:.2%})"
            fall_color = (0, 0, 255) if is_fall else (0, 255, 0)
            cv2.putText(display_frame, fall_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, fall_color, 2)
            
            y_offset += 30
            stage_text = f"Stage: {fall_stage.value} ({stage_confidence:.2%})"
            cv2.putText(display_frame, stage_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            y_offset += 30
            posture_text = f"Posture: {posture_type.value} ({posture_confidence:.2%})"
            cv2.putText(display_frame, posture_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # 显示视频
        cv2.imshow("Multi-Modal Fall Detection System", display_frame)
        
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 清理资源
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\\n处理完成!")
    print(f"总帧数: {frame_count}")


if __name__ == "__main__":
    main()
