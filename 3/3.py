import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import cv2
import matplotlib.pyplot as plt
import os
import sys

np.random.seed(42)
tf.random.set_seed(42)

# --------------------------
# I. CNN 模型训练和加载函数
# --------------------------

def prepare_and_train_model(model_name='mnist_cnn_model.keras'):
    """
    加载 MNIST 数据集，构建并训练一个 CNN 模型。
    """
    if os.path.exists(model_name):
        print(f"模型文件 '{model_name}' 已存在，直接加载。")
        try:
            model = load_model(model_name)
            return model
        except Exception as e:
            print(f"加载模型失败: {e}. 将重新训练。")
    
    print("--- 1. 加载和预处理 MNIST 数据集 ---")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 预处理和归一化
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # 调整形状 (N, 28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    # One-hot 编码
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    input_shape = x_train.shape[1:]
    
    print("\n--- 2. 构建 CNN 模型 (用于加分项) ---")
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("模型开始训练...")
    # 训练模型
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=10,
              validation_data=(x_test, y_test),
              verbose=1)

    # 评估和保存
    score = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n测试集准确率: {score[1]*100:.2f}%")

    model.save(model_name)
    print(f"模型已保存为: {model_name}")
    
    return model

# --------------------------
# II. OpenCV 预处理和数字分割函数
# --------------------------

def preprocess_and_segment_student_id(image_path):
    """
    加载手机照片，进行预处理、分割、筛选和排序，提取出 28x28 的数字图像。
    """
    # 1. 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法加载图像文件 {image_path}。请检查路径。")
        return None, None
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊：减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. 自适应阈值处理：处理光照不均，实现黑底白字 (MNIST 风格)
    thresh = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, # 块大小
        2   # C值
    )
    
    # 3. 查找轮廓
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_data = [] # 存储 (图像, 边界框 x 坐标)
    
    # 4. 筛选、裁剪和归一化轮廓
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        aspect_ratio = w / float(h)
        
        # 经验性筛选条件
        # 过滤掉太小的噪音和太大的非数字区域
        if 200 < area < 20000 and 0.1 < aspect_ratio < 3.0:
            
            # 提取数字区域，并添加边距
            padding = 10
            x_pad = max(0, x - padding)
            y_pad = max(0, y - padding)
            # 确保裁剪区域不会超出原始图像边界
            x_end = min(thresh.shape[1], x + w + padding)
            y_end = min(thresh.shape[0], y + h + padding)
            
            digit_crop = thresh[y_pad:y_end, x_pad:x_end]
            
            # 归一化处理：将数字置于中央的正方形区域
            side = max(digit_crop.shape)
            square_digit = np.zeros((side, side), dtype=np.uint8)
            x_offset = int((side - digit_crop.shape[1]) / 2)
            y_offset = int((side - digit_crop.shape[0]) / 2)
            square_digit[y_offset:y_offset + digit_crop.shape[0], 
                         x_offset:x_offset + digit_crop.shape[1]] = digit_crop

            # 最终调整到 28x28 尺寸
            final_digit = cv2.resize(square_digit, (28, 28), interpolation=cv2.INTER_AREA)

            digit_data.append((final_digit, x)) # 存储图像和 x 坐标
            
            # 绘制边界框用于调试
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
    # 5. 排序：按 x 坐标从左到右排序
    digit_data.sort(key=lambda item: item[1]) # item[1] 是 x 坐标
    sorted_digit_images = [item[0] for item in digit_data]

    return sorted_digit_images, image


# --------------------------
# III. 完整的学号识别流程
# --------------------------

def recognize_student_id_from_photo(photo_path, model):
    """
    完整的学号识别流程：预处理 -> 分割 -> 识别 -> 组合学号
    """
    print(f"\n--- 开始处理学号照片: {photo_path} ---")
    
    # 步骤 1: 预处理和分割
    sorted_digit_images, annotated_img = preprocess_and_segment_student_id(photo_path)
    
    if sorted_digit_images is None:
        return "识别失败: 无法加载或处理图片。"

    if not sorted_digit_images:
        print("!!! 警告: 未检测到有效数字轮廓。请检查图片和筛选参数。")
        if annotated_img is not None:
             cv2.imshow('Annotated Image (Check Area)', annotated_img)
             cv2.waitKey(0)
             cv2.destroyAllWindows()
        return "未检测到数字"
        
    print(f"成功检测到并排序了 {len(sorted_digit_images)} 个数字。")
    
    # 步骤 2: 依次进行识别
    recognized_id = ""
    
    for i, digit_img in enumerate(sorted_digit_images):
        
        # 准备模型输入：归一化并添加维度 (1, 28, 28, 1)
        input_array = digit_img.astype('float32') / 255.0
        input_array = np.expand_dims(input_array, axis=0)
        input_array = np.expand_dims(input_array, axis=-1)

        # 模型预测
        predictions = model.predict(input_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        
        recognized_id += str(predicted_class)
        
    print("-" * 30)
    print(f"最终识别学号为: {recognized_id}")
    print("-" * 30)
    
    # 显示带有边界框的原始图像
    if annotated_img is not None:
        cv2.imshow('Annotated Student ID Photo', annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return recognized_id

# --------------------------
# IV. 主程序入口
# --------------------------

if __name__ == '__main__':
    # 1. 训练或加载 CNN 模型
    MODEL_FILE = 'mnist_cnn_model.keras'
    try:
        cnn_model = prepare_and_train_model(MODEL_FILE)
    except Exception as e:
        print(f"致命错误：训练或加载模型失败。请检查 TensorFlow 和 Keras 安装。错误信息: {e}")
        sys.exit(1)

    # 2. 设置学号照片路径
    STUDENT_PHOTO_PATH = 'my_student_id_photo.jpg' 
    
    if not os.path.exists(STUDENT_PHOTO_PATH):
        print("\n--- ❗ 警告：图片文件不存在 ❗ ---")
        print(f"请将学号照片重命名为 '{STUDENT_PHOTO_PATH}' 并放在代码同目录下。")
        print("程序退出。")
        sys.exit(1)
        
    # 3. 执行识别流程
    final_id = recognize_student_id_from_photo(STUDENT_PHOTO_PATH, cnn_model)
    
    print(f"\n最终报告结果：学号识别结果为 {final_id}")