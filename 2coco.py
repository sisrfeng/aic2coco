import json
import os
import cv2

dataset = {'categories':[],'images':[],'annotations':[]}
# 根路径，里面包含images(图片文件夹)，annos.txt(bbox标注)，classes.txt(类别标签),以及annotations文件夹(如果没有则会自动创建，用于保存最后的json)
root_path = '/data2/lwf/COCO_my/c001/'
# 用于创建训练集或验证集 可以调成val
phase = 'instances_train2017'
# 训练集和验证集划分的界线  可以调
split =10000000
class_path=os.path.join(root_path, 'classes.txt')
print(class_path)
# 打开类别标签
with open(class_path) as f:
    classes = f.read().strip().split(',')

# 建立类别标签和数字id的对应关系
for i, cls in enumerate(classes, 1):
    dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})

# 读取train2017文件夹的图片名称
# 改了，加了sorted，保证图片按名字升序，使得<'image_id': k >里面，k=0,对应第一张图片，00...01.jpg
file_path=os.path.join(root_path, 'train2017')
file_list=sorted(os.listdir(file_path))
_names = [f for f in file_list]
# 判断是建立训练集还是验证集
if phase == 'instances_train2017':
    names = [line for i, line in enumerate(_names) if i <= split]
elif phase == 'instances_val2017':
    names = [line for i, line in enumerate(_names) if i > split]

# 读取Bbox信息
with open(os.path.join(root_path, 'annos.txt')) as tr:
    annos = tr.readlines()

# 以上数据转换为COCO所需要的
for k, name in enumerate(names,1):
    # 用opencv读取图片，得到图像的宽和高
    im = cv2.imread(os.path.join(file_path,name))
    height, width, _ = im.shape
    
    # 添加图像的信息到dataset中
    dataset['images'].append({'file_name': name,
                              'id': k,
                              'width': width,
                              'height': height})
                        
    index=str(int(name[:-4]))
    # 一张图多个框时需要判断
    for ii, anno in enumerate(annos,1):
        parts = anno.strip().split(',')
        # type(parts[0]) string
        # 如果图像的名称和标记的名称对上，则添加标记
        
        if parts[0] == index:
            # 类别
            cls_id = parts[1]
            # x_min
            x1 = float(parts[2])
            # y_min
            y1 = float(parts[3])
            width = float(parts[4])
            height = float(parts[5])
            x2 = x1 + width
            y2 = y1 + height
            dataset['annotations'].append({
                'area': width * height,
                'bbox': [x1, y1, width, height],
                # 'category_id': int(cls_id),
                'category_id': int(3),#我的数据集只有car
                'id': ii,
                'image_id': k,
                'iscrowd': 0,
                # mask, 矩形是从左上角点按顺时针的四个顶点
                'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
            })

# 保存结果
# # folder = os.path.join(root_path, 'annotations')
folder = '/data2/lwf/COCO_my/annotations'

if not os.path.exists(folder):
    os.makedirs(folder)
json_name = os.path.join(folder, '{}.json'.format(phase))
with open(json_name, 'w') as f:
    json.dump(dataset, f)
print('done')
