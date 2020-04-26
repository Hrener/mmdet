from mmdet.apis import init_detector, inference_detector, show_result
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# 首先下载模型文件https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth
# config_file = '/home1/rhhHD/rhh/mmdetection/configs/my_cascade_mask_rcnn_x101_64x4d_fpn_1x.py'
# checkpoint_file = "/home1/rhhHD/rhh/mmdetection/work_dirs/my_cascade_mask_rcnn_x101_64x4d_fpn_1x/epoch_9.pth"
config_file = '/home1/rhhHD/rhh/mmdetection/configs/dcn/cascade_rcnn_dconv_c3-c5_r50_fpn_1x.py'
checkpoint_file = "/home1/rhhHD/rhh/mmdetection/work_dirs/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_add_box/latest.pth"
# 初始化模型
model = init_detector(config_file, checkpoint_file)
#print(model.CLASSES)
# 测试一张图片
img = '/home1/rhhHD/rhh/mmdetection/data/coco/test/7257c1059408.png'
result = inference_detector(model, img)
class_name = tuple([str(i+1) for i in range(204)])
show_result(img, result, class_name, out_file='result_add.png')
 
# 测试一系列图片
#imgs = ['test1.jpg', 'test2.jpg']
#for i, result in enumerate(inference_detector(model, imgs, device='cuda:0')):
#    show_result(imgs[i], result, model.CLASSES, out_file='result_{}.jpg'.format(i))