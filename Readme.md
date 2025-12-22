# 实验一：图像滤波实验报告

## 一、实验目的

本实验旨在通过对数字图像进行滤波与特征提取，加深对图像处理基础理论与实际算法实现的理解。具体目标包括：

1. 理解并掌握 Sobel 算子 的边缘检测原理及其在图像中的应用；

2. 掌握 二维卷积运算 的实现方法，理解卷积核对图像特征的影响；

3. 学会手动计算图像的 RGB 颜色直方图，分析图像的颜色分布特性；

4. 掌握基于 灰度共生矩阵（GLCM） 的纹理特征提取方法；

5. 提升将理论算法转化为完整程序的能力，培养独立完成图像处理任务的实践能力。

   

## 二、实验原理

### 1. 图像卷积与滤波原理

图像滤波的核心思想是使用一个卷积核（Kernel）在图像上滑动，对局部像素进行加权求和，从而增强或抑制图像的某些特征。二维离散卷积公式为：
$$
g(x,y) = \sum_{i=-k}^{k}\sum_{j=-k}^{k} f(x+i, y+j)\cdot h(i,j)
$$
其中，(f(x,y)) 为原始图像，(h(i,j)) 为卷积核，(g(x,y)) 为输出图像。



### 2. Sobel 算子原理

Sobel 算子是一种常用的一阶梯度边缘检测算子，通过分别计算水平方向和垂直方向的灰度变化来检测图像边缘。

其卷积核为：
$$
G_x =
\begin{bmatrix}
-1 & 0 & 1 \
-2 & 0 & 2 \
-1 & 0 & 1
\end{bmatrix},\quad
G_y =
\begin{bmatrix}
-1 & -2 & -1 \
0 & 0 & 0 \
1 & 2 & 1
\end{bmatrix}
$$
最终的梯度幅值为：

$$
G = \sqrt{G_x^2 + G_y^2}
$$
梯度值越大，说明该位置灰度变化越剧烈，越可能是图像的边缘。



### 3. 给定卷积核滤波原理

实验中给定的卷积核为：
$$
\begin{bmatrix}
1 & 0 & -1 \
2 & 0 & -2 \
1 & 0 & -1
\end{bmatrix}
$$
该卷积核本质上是 Sobel 算子的一个方向算子，用于突出图像中某一方向上的边缘信息。通过卷积运算，可以增强图像中亮度变化明显的区域。



### 4. 颜色直方图原理

颜色直方图用于统计图像中不同颜色强度出现的频率。对 RGB 彩色图像而言，分别统计 R、G、B 三个通道中像素值（0–255）的分布情况，可用于分析图像整体的颜色特征和亮度分布。



### 5. 灰度共生矩阵（GLCM）与纹理特征

灰度共生矩阵描述了图像中某一灰度值与其邻域灰度值在特定空间关系下的联合分布情况。基于 GLCM 可提取多种纹理特征，本实验计算了以下三种：

- 对比度（Contrast）：反映纹理的清晰程度和灰度变化强弱；
- 能量（Energy）：反映纹理分布的均匀性；
- 同质性（Homogeneity）：反映相邻像素灰度的相似程度。



## 三、实验方法

1. 使用 Python 语言，基于 **Pillow、NumPy 和 Matplotlib** 等库完成实验；

2. 读取自己拍摄的图像，并分别转换为 RGB 图像和灰度图像；

3. 手动实现二维卷积函数，采用零填充方式处理图像边界；

   ```python
   def _apply_convolution(self, img_data, kernel):
           """
           核心函数：手动实现二维卷积操作。
           """
           # 获取图像尺寸
           W, H = self.width, self.height
           k_size = len(kernel)
           pad = k_size // 2
   
           # 创建一个与输入图像相同大小的空白结果矩阵
           output_data = [[0] * W for _ in range(H)]
   
           for y in range(H):
               for x in range(W):
                   # 遍历卷积核，执行乘加操作
                   sum_val = 0
                   for ky in range(k_size):
                       for kx in range(k_size):
                           # 计算输入图像中的对应坐标 (iy, ix)
                           iy = y + ky - pad
                           ix = x + kx - pad
   
                           # 处理边界：采用零填充 (zero padding)
                           if 0 <= ix < W and 0 <= iy < H:
                               sum_val += img_data[iy][ix] * kernel[ky][kx]
   
                   # 将结果限制在 [0, 255] 范围内
                   output_data[y][x] = max(0, min(255, int(abs(sum_val))))
   
           return output_data
   ```

4. 分别实现 Sobel 算子滤波和给定卷积核滤波，生成对应的滤波结果图像；

   ```python
   def sobel_filter(self):
           """
           实现 Sobel 算子滤波，返回滤波后的图像对象。
           """
           # 1. 将灰度图的像素值转换为列表 of 列表
           gray_data = list(self.img_gray.getdata())
           gray_matrix = [gray_data[i*self.width:(i+1)*self.width] for i in range(self.height)]
   
           # 2. 定义 Sobel 算子 Gx 和 Gy
           sobel_gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
           sobel_gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
   
           # 3. 计算 Gx 和 Gy 方向的梯度图像
           Gx_data = self._apply_convolution(gray_matrix, sobel_gx)
           Gy_data = self._apply_convolution(gray_matrix, sobel_gy)
   
           # 4. 计算梯度幅值 M = sqrt(Gx^2 + Gy^2)
           sobel_output_data = []
           for y in range(self.height):
               for x in range(self.width):
                   M = math.sqrt(Gx_data[y][x]**2 + Gy_data[y][x]**2)
                   sobel_output_data.append(max(0, min(255, int(M))))
   
           # 5. 创建新的图像对象
           result_img = Image.new('L', (self.width, self.height))
           result_img.putdata(sobel_output_data)
           return result_img
   
       def given_kernel_filter(self):
           """
           实现给定卷积核的滤波，返回滤波后的图像对象。
           """
           # 1. 将灰度图的像素值转换为列表 of 列表
           gray_data = list(self.img_gray.getdata())
           gray_matrix = [gray_data[i*self.width:(i+1)*self.width] for i in range(self.height)]
   
           # 2. 给定的卷积核
           kernel = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
   
           # 3. 应用卷积
           filtered_matrix = self._apply_convolution(gray_matrix, kernel)
   
           # 4. 转换回一维数据
           filtered_data = [pixel for row in filtered_matrix for pixel in row]
   
           # 5. 创建新的图像对象
           result_img = Image.new('L', (self.width, self.height))
           result_img.putdata(filtered_data)
           return result_img
   
       def color_histogram(self):
           """
           手动计算 RGB 三个颜色通道的直方图。
           """
           # 初始化 256 个 bin 的直方图（R, G, B）
           hist = {'R': [0] * 256, 'G': [0] * 256, 'B': [0] * 256}
   
           # 遍历图像的每个像素
           for pixel in self.img_rgb.getdata():
               R, G, B = pixel
               # 增加对应 bin 的计数
               hist['R'][R] += 1
               hist['G'][G] += 1
               hist['B'][B] += 1
           
           return hist
   ```

5. 手动统计 RGB 三个通道的像素值分布，绘制颜色直方图；

6. 对灰度图进行量化，构建灰度共生矩阵，并计算纹理特征；

7. 将提取到的纹理特征以 `.npy` 格式保存，便于后续分析与复用。



## 四、实验结果

1. Sobel 滤波结果
   实验结果显示，Sobel 算子能够有效突出图像中物体的轮廓和边缘，背景区域被明显抑制，边缘信息更加清晰。

   <img src="E:\学习\机器视觉\Machine Vision Lab\1\results\given_kernel_output.png" style="zoom:50%;" />

   

2. 给定卷积核滤波结果
   使用给定卷积核后，图像在特定方向上的边缘得到增强，与 Sobel 滤波结果相比，边缘方向性更加明显。

   <img src="E:\学习\机器视觉\Machine Vision Lab\1\results\sobel_output.png" alt="sobel_output" style="zoom:50%;" />

3. 颜色直方图结果
   颜色直方图清晰展示了图像中 RGB 三个通道的像素分布情况，可以直观判断图像整体的色彩倾向和亮度分布特征。

   ![](E:\学习\机器视觉\Machine Vision Lab\1\results\color_histogram.png)

4. 纹理特征结果
   成功提取了图像的对比度、能量和同质性等纹理特征，并以 `.npy` 文件形式保存，为后续图像分类或识别任务提供了数据基础。



## 五、实验体会

通过本次实验，我对图像滤波和特征提取的理论与实现过程有了更加深入的理解。相比直接调用现成函数，手动实现卷积、直方图统计和纹理特征计算的过程，使我更加清楚每一步计算的物理意义和算法细节。同时，在处理图像边界、数据归一化以及结果可视化时，也提升了实际编程和调试能力。

本实验不仅加深了我对图像处理基础算法的理解，也为后续更复杂的计算机视觉与模式识别任务打下了良好的基础。







# 实验二：基于霍夫变换的车道线检测

## 一、 实验目的

1. 掌握数字图像处理的基本流程，包括图像预处理、边缘检测等。

2. 深入理解霍夫变换（Hough Transform）的原理及其在直线检测中的应用。

3. 实现自动驾驶领域的基础模块——车道线检测，并能对校园实拍场景进行算法验证。

   

## 二、 实验原理

车道线检测主要依赖于图像中线条的几何特征。本实验采用以下核心技术：

1. **灰度化与边缘检测**：

   - 车道线通常与路面有较高的对比度。通过 Canny 算子 检测图像中亮度变化剧烈的边缘点，为后续直线提取减少计算量。

2. **霍夫变换（Hough Lines）**：

   - 在极坐标系中，直线方程表示为 
     $$
     r = x \cdot \cos(\theta) + y \cdot \sin(\theta)
     $$

   - 图像空间中的一个点在极坐标参数空间对应一条正弦曲线；图像空间中共线的点，其对应的正弦曲线在参数空间交于一点。

   - 通过在参数空间建立累加器，寻找交点最多的单元，即可确定原图中直线的参数 $(r, \theta)$。

     

## 三、 实验方法

本实验使用 Python 语言结合 OpenCV 库实现，具体步骤如下：

1. **图像预处理**：读取校园道路图像，将其从 BGR 彩色空间转换为 Gray 灰度空间，消除色彩干扰。

2. **边缘提取**：利用 `cv2.Canny()` 函数提取图像边缘，设置低阈值 50 和高阈值 150 以保留车道线轮廓。

3. **直线检测**：

   - 调用 `cv2.HoughLines()` 进行标准霍夫变换。
   - 参数设置：极径分辨率为 1 像素，累加器阈值设为 150（即至少有 150 个点共线才判定为直线）。

   ```python
   # 检查是否检测到直线
   if lines is not None:
       # 遍历所有检测到的直线
       for line in lines:
           
           r,theta = line[0]
           # Stores the value of cos(theta) in a
           a = np.cos(theta)
           # Stores the value of sin(theta) in b
           b = np.sin(theta)
           # x0 stores the value rcos(theta)
           x0 = a*r
           # y0 stores the value rsin(theta)
           y0 = b*r
           # 计算直线上的两个点 (x1, y1) 和 (x2, y2)
           # 直线方程为 x*cos(theta) + y*sin(theta) = r
           # 为了画出足够长的直线，我们从 (x0, y0) 沿垂直方向和水平方向延伸 1000 个单位
           x1 = int(x0 + 1000*(-b))
           y1 = int(y0 + 1000*(a))
           x2 = int(x0 - 1000*(-b))
           y2 = int(y0 - 1000*(a))
           
           # cv2.line 绘制直线
           # (0, 255, 255) 是黄色的 BGR 值
           cv2.line(img,(x1,y1), (x2,y2), (0,255,255), 2) # 将颜色改为黄色，并将线宽改为 2 以更清晰显示
   ```

4. **结果绘制**：

   - 将检测到的参数转换回笛卡尔坐标系下的两点。

   - 使用 `cv2.line()` 在原图上绘制黄色直线（BGR: 0, 255, 255）以标识车道线位置。

     

## 四、 实验结果

根据代码运行逻辑，实验输出了以下结果：

- **边缘检测图 (`edges`)**：清晰展示了校园道路的轮廓，尤其是车道线边缘。

- **最终识别图**：在原图基础上，使用黄色线条准确覆盖了检测到的直线区域。

- **文件保存**：结果已成功保存为 `campus_road_with_yellow_lines.jpg`。

  <img src="E:\学习\机器视觉\Machine Vision Lab\2\campus_road.jpg" style="zoom: 50%;" />

  <img src="E:\学习\机器视觉\Machine Vision Lab\2\campus_road_with_yellow_lines.jpg" style="zoom:50%;" />



## 五、 实验体会

1. **算法有效性**：霍夫变换对于结构化道路（如笔直的车道线）具有极佳的检测效果，鲁棒性较强。
2. **参数调试的重要性**：
   - Canny 阈值直接影响边缘点的质量。
   - Hough 阈值过高会导致漏检，过低则会引入路牙、建筑边缘等噪声直线。
3. **改进方向**：
   - ROI 过滤：在自动驾驶实际应用中，应只关注图像下方的三角区域（路面区域），以过滤掉上方干扰。
   - 概率霍夫变换：若要提高检测效率并获取直线段而非无限长直线，可考虑使用 `cv2.HoughLinesP()`。
   - 曲线适应性：对于弯道场景，标准霍夫变换难以应对，未来可尝试最小二乘法拟合或深度学习模型（如 LaneNet）。







# 实验三：基于卷积神经网络（CNN）的手写学号识别

## 一、 实验目的

1. 掌握手写体数字识别的基本流程，包括图像采集、预处理、字符分割和分类识别。

2. 学习并应用深度学习中的卷积神经网络（CNN）解决实际视觉任务。

3. 实践从“原始照片”到“结构化数据”的全栈处理过程。

   

## 二、 实验原理

本实验的核心是利用在 **MNIST** 数据集上训练好的深度学习模型来识别手机拍摄的学号照片。

1. **卷积神经网络 (CNN)**：

   - **卷积层 (Conv2D)**：通过卷积核提取图像的局部空间特征（如边缘、线条）。
   - **池化层 (MaxPooling)**：降低特征维度，增强模型对位置偏移的鲁棒性。
   - **全连接层 (Dense)**：将提取的特征映射到 0-9 的分类概率上。

   ```python
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
   ```

2. **图像预处理与分割**：

   - **自适应阈值处理**：将手机拍摄的复杂光照图像转换为黑底白字的二值图，以匹配 MNIST 的分布。

   - **轮廓检测 (Contours)**：寻找图像中的闭合区域，定位每个数字的具体位置并按从左到右的顺序排序。

     

## 三、 实验方法与过程

### 1. 代码环境配置

为确保实验运行，环境配置如下

- 创建环境：pyenv local 3.8.10

- 安装依赖：

  - `pip install tensorflow` (深度学习框架)

  - `pip install opencv-python` (图像处理工具)

  - `pip install numpy matplotlib` (科学计算与可视化)

    

### 2. 模型训练

- 数据集：使用 Keras 内置的 MNIST 手写数字数据集。

- 模型结构：包含两层卷积、两层最大池化及 Dropout 层（防止过拟合），最后通过 Softmax 输出 10 类概率。

- 准确率：模型在 MNIST 测试集上的识别准确率达到 99% 以上。

  

### 3. 学号图像处理流程

1. 灰度化与模糊：去除颜色干扰并平滑噪点。

2. 二值化：利用 `cv2.adaptiveThreshold` 将图像转为黑白。

3. 数字分割：通过 `cv2.findContours` 提取数字轮廓，并根据 `x` 坐标进行排序，确保学号顺序正确。

4. 归一化：将每个分割出的数字调整为 28x28 像素，作为 CNN 模型的输入。

   

## 四、 实验结果

- 任务输入：实拍的学号照片（如 `my_student_id_photo.jpg`）。

  <img src="E:\学习\机器视觉\Machine Vision Lab\3\my_student_id_photo.jpg" style="zoom: 33%;" />

- 识别输出：程序能够自动在照片中框选出数字区域，并在终端依次打印识别出的学号数字。

- 可视化结果：代码会显示带有绿色边界框（Bounding Box）的标注图像，验证分割的准确性。

  <img src="E:\学习\机器视觉\Machine Vision Lab\3\result.png" style="zoom:33%;" />



## 五、 实验体会

1. **现实场景挑战**：手机拍摄的照片与 MNIST 标准集存在差异。光照不均和背景干扰是主要难题。实验证明，自适应阈值处理比固定阈值更有效。
2. **参数调优**：在数字筛选阶段，通过限制轮廓的面积（`area`）和长宽比（`aspect_ratio`），可以有效过滤掉环境中的细小杂质。
3. **深度学习优势**：相比于传统的特征提取方法（如 HOG+SVM），CNN 对于数字的轻微形变和位移具有极强的容忍度，是目前处理此类任务的主流选择。







# 实验四：基于COCO训练集的校园共享单车目标检测

## 一、 实验目的

1. 深入理解计算机视觉中目标检测的完整流程：特征提取 - 目标定位 - 分类判断。

2. 掌握使用 COCO 数据集*进行特定类别（共享单车/自行车）提取与预处理的方法。

3. 学习并实践深度学习目标检测算法（如 YOLOv8）的训练、验证与推理流程。

4. 实现对校园道路、停车区图像中共享单车的自动化精准识别与定位。

   

## 二、 实验原理

目标检测是一项“定位 + 识别”的双重任务。本实验采用 YOLO（You Only Look Once）深度学习框架，其核心原理如下：

1. **特征提取 (Feature Extraction)**：通过骨干网络（Backbone）对输入的共享单车照片进行多层卷积，提取其轮廓、颜色（如哈啰单车的蓝色、美团单车的黄色）和结构特征。

2. **目标定位 (Localization)**：算法在图像中生成预测框，并通过回归损失函数精确计算共享单车的位置坐标（xyxy格式）。

3. **分类判断 (Classification)**：利用 Softmax 分类器判断预测框内的物体是否为“自行车（bicycle）”，并给出相应的置信度得分。

   

## 三、 实验方法与过程

#### 1. 实验环境构建

本实验严格遵循教学要求，创建了独立的python环境。该环境集成了 `ultralytics` 深度学习框架，利用 PyTorch 作为后端计算引擎，确保了算法运行的高效性与隔离性。



#### 2. 数据处理算法分析

本实验并未直接使用全量 COCO 数据集，而是通过脚本进行了精准的**子集重采样与格式转化**：

- **类别筛选逻辑**：利用 `pycocotools` 库，通过 `getCatIds(catNms=['bicycle'])` 建立索引，从庞大的 COCO 验证集中筛选出含有自行车的特定样本。这种方法极大缩短了 4 课时实验内的训练周转时间。

  ```python
  coco = COCO(ann_file)
      cat_ids = coco.getCatIds(catNms=[target_class])
      img_ids = coco.getImgIds(catIds=cat_ids)
  ```

- **数据动态切分**：代码中引入 `random.shuffle` 对样本进行随机打乱，并按 **8:2** 的比例划分为训练集与验证集。这种随机化处理能有效防止模型产生过拟合，提高对校园未知场景的泛化能力。

  ```python
  random.shuffle(img_ids)
      split_idx = int(len(img_ids) * 0.8)
      train_ids = img_ids[:split_idx]
      val_ids = img_ids[split_idx:]
  ```

- **坐标归一化处理**：将 COCO 的绝对像素坐标 `[x, y, w, h]` 转化为 YOLO 要求的相对中心点格式 `[xc, yc, wn, hn]`。这一步是算法实现**尺度不变性**的关键，使得模型能同时识别图像中远近不一、大小各异的单车。

  ```
  Campus_Bike_Detection/          # 项目根目录
  ├── bicycle_subset/             # 脚本自动生成的子集目录
  │   ├── images/
  │   │   ├── train/              # 训练用图片
  │   │   └── val/                # 验证用图片
  │   └── labels/
  │       ├── train/              # YOLO格式标注 (.txt)
  │       └── val/                # YOLO格式标注 (.txt)
  ├── COCO_Small/                 # 原始COCO Val集
  │   ├── annotations/
  │   │   └── instances_val2017.json
  │   └── val2017/                # 原始5000张图片
  ├── runs/                       # 训练过程中自动生成的文件夹
  │   └── detect/
  │       └── campus_small_run/
  │           └── weights/
  │               ├── best.pt     # 最终要提交的模型权重
  │               └── last.pt     # 最后一次迭代的权重
  ├── bicycle_data.yaml           # 数据集配置文件
  ├── train_all.py                # 数据提取与训练脚本
  ├── 4.py                        # 最终验证与结果输出脚本
  ├── campus_bike.jpg             # 校园单车测试图
  ```

  

#### 3. 深度学习模型架构分析

本实验采用 **YOLOv8n** 轻量化模型，其“特征提取-目标定位-分类判断”流程在代码中体现如下：

- **特征提取（Backbone）**：在 `model.train` 过程中，模型通过深层卷积网络提取校园单车的视觉特征。例如，代码中 `imgsz=640` 的设置确保了模型能捕捉到单车细长的车架和圆形的轮圈等关键语义信息。

- **分类判断（Classification）**：在推理脚本 `4.py` 中，通过 `box.cls` 获取分类结果，`box.conf` 获取置信度。算法不仅要圈出物体，还要给出一个概率值（如 `0.41`），只有高于阈值的目标才会被判定为 `bicycle`。

- **目标定位（Localization）**：算法通过边界框回归（Bounding Box Regression）输出 `xyxy` 格式的像素坐标。`4.py` 脚本中的 `result.plot()` 函数将这些数学坐标转化为直观的彩色检测框，完成了“定位”任务。

  

#### 4. 实验推理验证

在验证阶段，脚本 `4.py` 专门针对 `campus_bike.jpg` 进行单点推理。

- **逻辑闭环**：通过 `model.predict(source=image_path, conf=0.3)` 模拟了真实校园监控下的自动识别过程。

- **结果分析**：实验输出结果显示，算法能精准捕捉到具有特定品牌涂装（如蓝黄配色）的共享单车，证明了基于 COCO 预训练模型进行微调（Fine-tuning）方案的有效性。

  

## 四、 实验结果

#### 1. 任务输入

- **输入图像**：校园场景下的共享单车照片（`campus_bike.jpg`）。

#### 2. 任务输出与可视化

- **模型识别结果**：模型成功检测到目标，并输出分类为 `bicycle`。

- **置信度**：识别置信度约为 **0.41**。

- **位置坐标**：脚本 `4.py` 精确输出了共享单车在图中的像素坐标位置。

  

## 五、 实验体会

1. **算法鲁棒性**：通过实验发现，即使是校园中特定品牌（如黄色车筐的哈啰单车）的变体，经过在 COCO 通用自行车数据集上的训练后，模型依然能通过特征提取捕捉到其核心几何特征实现定位。
2. **深度学习优势**：相比传统视觉方法，深度学习能够自动学习特征，极大地简化了人工设计算子的过程，在复杂校园环境下的识别准确率显著提升。
3. **环境管理**：通过规范的环境配置，有效避免了库版本冲突，体现了计算机视觉工程化开发的严谨性。




