import cv2
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

def verify_model(model_path, image_path, save_result=True):
    """
    验证模型效果并输出实验要求的各项指标
    """
    # 1. 加载模型 (训练生成的 best.pt)
    if not os.path.exists(model_path):
        print(f"错误：未找到模型文件 {model_path}，请确认训练已完成并路径正确。")
        return
    
    print(f"--- 正在加载自定义模型: {model_path} ---")
    model = YOLO(model_path)

    # 2. 执行推理 (Task Input: 共享单车照片)
    # 使用 conf=0.3 过滤低置信度目标
    results = model.predict(source=image_path, conf=0.3, device='cpu') 

    # 3. 解析结果 (理解 特征提取 - 目标定位 - 分类判断 流程)
    for result in results:
        boxes = result.boxes
        num_bikes = len(boxes)
        print(f"\n[实验结果分析]:")
        print(f"检测状态: 完成")
        print(f"检测到共享单车数量: {num_bikes}")

        # 遍历每一个检测到的单车
        for i, box in enumerate(boxes):
            # 获取分类判断结果 (Class ID)
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            
            # 获取置信度 (Confidence Score)
            conf = box.conf[0].item()
            
            # 获取定位坐标 (Task Output: 位置)
            # xyxy 格式: [左上角x, 左上角y, 右下角x, 右下角y]
            coords = box.xyxy[0].tolist()
            
            print(f"目标 {i+1}: 类别={label}, 置信度={conf:.2f}, 位置={coords}")

        # 4. 可视化与保存
        # plot() 方法集成了特征可视化，能直观展示分类和定位结果
        res_plotted = result.plot()
        
        # 将 OpenCV 的 BGR 转换为 Matplotlib 的 RGB
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(res_rgb)
        plt.title(f"校园共享单车检测结果 (检测到: {num_bikes} 辆)")
        plt.axis('off')
        
        if save_result:
            output_name = 'campus_detection_result.png'
            plt.savefig(output_name, dpi=300)
            print(f"\n可视化结果已保存至: {output_name}")
        
        plt.show()

# --- 运行参数配置 ---
if __name__ == "__main__":
    # 指向训练阶段生成的最佳模型路径
    MY_MODEL = r"E:\学习\机器视觉\Machine Vision Lab\4\runs\detect\campus_small_run\weights\best.pt"
    
    # 待检测的校园共享单车图片 
    TEST_IMAGE = 'campus_bike.jpg' 

    # 开始执行
    verify_model(MY_MODEL, TEST_IMAGE)