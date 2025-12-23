# main.py
import matplotlib.pyplot as plt 
import numpy as np
from image_processor import ImageProcessor

# --- 配置 ---
INPUT_FILE = 'input_image.jpg'
OUTPUT_DIR = './results/'
# -------------

def visualize_histogram(hist):
    """使用 Matplotlib 可视化颜色直方图"""
    plt.figure(figsize=(10, 5)) 
    
    # 颜色通道映射
    colors = {'R': 'red', 'G': 'green', 'B': 'blue'}
    x = np.arange(256)
    
    for channel, data in hist.items():
        plt.bar(x, data, color=colors[channel], alpha=0.5, label=f'{channel} Channel', width=1.0)
    
    plt.title('Color Histogram')
    plt.xlabel('Intensity Value')
    plt.ylabel('Pixel Count')
    plt.legend()
    # 保存并显示
    plt.savefig(OUTPUT_DIR + 'color_histogram.png')
    plt.show()

if __name__ == '__main__':
    # 确保结果目录存在
    import os
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. 初始化处理器
    processor = ImageProcessor(INPUT_FILE)

    # 2. 任务一：Sobel 滤波
    sobel_img = processor.sobel_filter()
    sobel_img.save(OUTPUT_DIR + 'sobel_output.png')
    print(f"✅ Sobel 滤波结果已保存到: {OUTPUT_DIR}sobel_output.png")

    # 3. 任务二：给定卷积核滤波
    given_kernel_img = processor.given_kernel_filter()
    given_kernel_img.save(OUTPUT_DIR + 'given_kernel_output.png')
    print(f"✅ 给定卷积核滤波结果已保存到: {OUTPUT_DIR}given_kernel_output.png")

    # 4. 任务三：颜色直方图计算与可视化
    color_hist = processor.color_histogram()
    print("✅ 颜色直方图计算完成。")
    visualize_histogram(color_hist)
    print(f"✅ 颜色直方图图片已保存到: {OUTPUT_DIR}color_histogram.png")

    # 5. 任务四：纹理特征提取
    texture_features = processor.glcm_texture_features(distance=1, angle=0)
    print("\n--- 提取的纹理特征 ---")
    print(f"纹理特征结果: {texture_features}")
    
    # 将纹理特征保存为 .npy 格式
    np.save(OUTPUT_DIR + 'texture_features.npy', texture_features)
    print(f"✅ 纹理特征已保存到: {OUTPUT_DIR}texture_features.npy")
    