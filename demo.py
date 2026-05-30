#!/usr/bin/env python3
"""老人跌倒检测系统演示脚本"""

import cv2
import argparse
from src import PoseDetector, FallClassifier, FallStageDetector, PostureClassifier, AlertSystem


def main():
    parser = argparse.ArgumentParser(description="老人跌倒检测系统")
    parser.add_argument("--video", type=str, help="输入视频文件路径")
    parser.add_argument("--camera", type=int, default=-1, help="摄像头编号（默认不使用）")
    parser.add_argument("--output", type=str, help="输出视频文件路径")
    parser.add_argument("--threshold", type=float, default=0.6, help="跌倒置信度阈值")
    
    args = parser.parse_args()
    
    # 初始化各模块
    pose_detector = PoseDetector(model_complexity=1)
    fall_classifier = FallClassifier(window_size=10)
    fall_stage_detector = FallStageDetector(window_size=30)
    posture_classifier = PostureClassifier()
    alert_system = AlertSystem(fall_threshold=args.threshold)
    
    # 注册告警处理器
    alert_system.register_handler(alert_system.print_alert)
    
    # 选择视频源
    if args.camera >= 0:
        cap = cv2.VideoCapture(args.camera)
    elif args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        print("错误: 必须指定 --video 或 --camera")
        return
    
    if not cap.isOpened():
        print("错误: 无法打开视频源")
        return
    
    # 准备输出视频（如果指定）
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    frame_count = 0
    
    print("开始处理视频...")
    print(f"跌倒置信度阈值: {args.threshold}")
    print("按 'q' 退出")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        timestamp = frame_count / cap.get(cv2.CAP_PROP_FPS)
        
        # 1. 姿态检测
        pose_frame = pose_detector.detect(frame, frame_id=frame_count, timestamp=timestamp)
        
        if pose_frame is None:
            # 未检测到人体
            display_frame = frame.copy()
            cv2.putText(display_frame, "No person detected", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # 2. 跌倒分类
            is_fall, fall_confidence = fall_classifier.classify(pose_frame)
            
            # 3. 跌倒阶段检测
            fall_stage, stage_confidence = fall_stage_detector.update(pose_frame, is_fall)
            
            # 4. 姿态分类
            posture_type, posture_confidence = posture_classifier.classify(pose_frame)
            
            # 5. 告警处理
            alert_event = alert_system.process(
                timestamp, is_fall, fall_confidence,
                fall_stage, stage_confidence,
                posture_type, posture_confidence
            )
            
            # 6. 绘制结果
            display_frame = pose_detector.draw_landmarks(frame, pose_frame)
            
            # 添加文字信息
            y_offset = 30
            cv2.putText(display_frame, f"Frame: {frame_count}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            y_offset += 30
            fall_text = f"Fall: {'YES' if is_fall else 'NO'} ({fall_confidence:.2%})"
            fall_color = (0, 0, 255) if is_fall else (0, 255, 0)
            cv2.putText(display_frame, fall_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, fall_color, 2)
            
            y_offset += 30
            stage_text = f"Stage: {fall_stage.value} ({stage_confidence:.2%})"
            cv2.putText(display_frame, stage_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            y_offset += 30
            posture_text = f"Posture: {posture_type.value} ({posture_confidence:.2%})"
            cv2.putText(display_frame, posture_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # 如果检测到跌倒，显示告警
            if alert_event:
                cv2.putText(display_frame, "ALERT!", (50, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        
        # 显示视频
        cv2.imshow("Fall Detection System", display_frame)
        
        # 保存输出视频
        if out:
            out.write(display_frame)
        
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 清理资源
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    # 打印统计信息
    print(f"\n处理完成!")
    print(f"总帧数: {frame_count}")
    print(f"检测到的跌倒告警: {len(alert_system.alert_history)}")
    
    if alert_system.alert_history:
        print("\n最近的告警:")
        for event in alert_system.alert_history[-3:]:
            print(f"  - {event.event_time}: {event.fall_confidence:.2%} 置信度")


if __name__ == "__main__":
    main()
