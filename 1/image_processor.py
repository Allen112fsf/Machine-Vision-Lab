from PIL import Image
import math
import numpy as np 

class ImageProcessor:
    def __init__(self, image_path):
        """初始化，加载图像"""
        # 使用 Pillow 库加载图像，并转换为灰度图和RGB图
        self.img_rgb = Image.open(image_path).convert("RGB")
        self.img_gray = Image.open(image_path).convert("L")
        self.width, self.height = self.img_gray.size

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

    def glcm_texture_features(self, distance=1, angle=0):
        """
        手动实现灰度共生矩阵 (GLCM) 和其衍生特征提取。
        """
        # 1. 预处理：量化灰度级
        bins = 8 
        max_val = 255
        
        gray_data = list(self.img_gray.getdata())
        gray_matrix = [gray_data[i*self.width:(i+1)*self.width] for i in range(self.height)]

        quantized_matrix = [[int(pixel / (max_val + 1) * bins) for pixel in row] for row in gray_matrix]

        # 2. 构造 GLCM 矩阵
        glcm = [[0] * bins for _ in range(bins)]
        
        # 定义方向 (angle=0: 水平向右)
        dx, dy = (distance, 0) # 角度0度，距离1

        for y in range(self.height):
            for x in range(self.width):
                i = quantized_matrix[y][x] # 当前像素的灰度级
                
                # 计算邻居像素坐标
                nx, ny = x + dx, y + dy
                
                # 检查边界
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    j = quantized_matrix[ny][nx] # 邻居像素的灰度级
                    glcm[i][j] += 1
                    # GLCM通常是对称的 (i, j) 和 (j, i) 都计数
                    if i != j:
                        glcm[j][i] += 1 

        # 3. 归一化 GLCM
        total_sum = sum(sum(row) for row in glcm)
        if total_sum == 0:
            return {'Contrast': 0, 'Energy': 0, 'Homogeneity': 0}

        P = [[glcm[i][j] / total_sum for j in range(bins)] for i in range(bins)]

        # 4. 计算纹理特征
        contrast = 0.0
        energy = 0.0
        homogeneity = 0.0

        for i in range(bins):
            for j in range(bins):
                p_ij = P[i][j]
                
                # 对比度 (Contrast): 反映图像局部灰度值的变化程度，纹理越粗，值越大
                contrast += (i - j)**2 * p_ij
                
                # 能量 (Energy) 或角二阶矩: 反映图像灰度分布的均匀程度，纹理越均匀，值越大
                energy += p_ij**2
                
                # 同质性 (Homogeneity): 反映图像局部相似性，值越大，纹理越均匀
                homogeneity += p_ij / (1 + abs(i - j))

        # 将结果以字典形式返回，作为纹理特征
        texture_features = {
            'Contrast': contrast,
            'Energy': energy,
            'Homogeneity': homogeneity
        }
        
        return texture_features