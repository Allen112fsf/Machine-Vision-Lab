import cv2
import numpy as np
'''图像中的直线检测
'''
# 读取图像
img = cv2.imread('campus_road.jpg')
 
# 转换为灰度图
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 边缘检测
edges = cv2.Canny(gray_img,50,150,apertureSize=3)

minLineLength = 30
maxLineGap = 5
print(np.pi/180)

# 标准霍夫变换检测直线
# 最后一个参数 150 是累加器的阈值
lines = cv2.HoughLines(edges,1,np.pi/180,150)

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
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        
        # cv2.line 绘制直线
        # (0, 255, 255) 是黄色的 BGR 值
        cv2.line(img,(x1,y1), (x2,y2), (0,255,255), 2) 

 
# 显示边缘图
cv2.imshow('edges',edges)
# 显示绘制了直线的图像
cv2.imshow('lines',img)

# 保存输出结果到文件
cv2.imwrite('campus_road_with_yellow_lines.jpg', img)
print("绘制了黄线的图像已保存为 'campus_road_with_yellow_lines.jpg'")

# 等待按键
cv2.waitKey(0) 
cv2.destroyAllWindows()