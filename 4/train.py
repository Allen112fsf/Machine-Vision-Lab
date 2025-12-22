import os
import shutil
import random
from pycocotools.coco import COCO
from tqdm import tqdm
from ultralytics import YOLO

def exert_coco_from_val(data_root, save_root, target_class='bicycle'):
    # 路径配置
    ann_file = os.path.join(data_root, 'annotations/instances_val2017.json')
    img_dir = os.path.join(data_root, 'val2017')
    
    coco = COCO(ann_file)
    cat_ids = coco.getCatIds(catNms=[target_class])
    img_ids = coco.getImgIds(catIds=cat_ids)
    
    # 随机打乱并切分：80% 拿来训练，20% 拿来测试
    random.shuffle(img_ids)
    split_idx = int(len(img_ids) * 0.8)
    train_ids = img_ids[:split_idx]
    val_ids = img_ids[split_idx:]

    def process_subset(ids, subset_name):
        img_save = os.path.join(save_root, 'images', subset_name)
        label_save = os.path.join(save_root, 'labels', subset_name)
        os.makedirs(img_save, exist_ok=True)
        os.makedirs(label_save, exist_ok=True)

        print(f"正在处理 {subset_name} 子集...")
        for img_id in tqdm(ids):
            img_info = coco.loadImgs(img_id)[0]
            file_name = img_info['file_name']
            
            # 1. 拷贝图片
            src = os.path.join(img_dir, file_name)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(img_save, file_name))

                # 2. 转换坐标
                ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
                anns = coco.loadAnns(ann_ids)
                labels = []
                for ann in anns:
                    x, y, w, h = ann['bbox']
                    # YOLO 归一化格式
                    xc, yc = (x + w/2)/img_info['width'], (y + h/2)/img_info['height']
                    wn, hn = w/img_info['width'], h/img_info['height']
                    labels.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
                
                with open(os.path.join(label_save, file_name.replace('.jpg', '.txt')), 'w') as f:
                    f.write('\n'.join(labels))

    process_subset(train_ids, 'train')
    process_subset(val_ids, 'val')

if __name__ == '__main__':
    COCO_PATH = r"E:\学习\机器视觉\Machine Vision Lab\4\COCO_Small" 
    SUBSET_PATH = r"E:\学习\机器视觉\Machine Vision Lab\4\bicycle_subset"
    
    # 1. 提取并切分数据
    exert_coco_from_val(COCO_PATH, SUBSET_PATH)

    # 2. 训练模型 (针对小数据集优化参数)
    model = YOLO('yolov8n.pt') 
    model.train(
        data='bicycle_data.yaml', 
        epochs=30,      
        imgsz=640, 
        batch=8,       
        name='campus_small_run'
    )