# from mmdet.apis import init_detector, inference_detector, show_result
from mmdet.apis import init_detector, inference_detector
import os
import mmcv
import numpy as np
import json
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def s_result(img,
             result,
             class_names,
             score_thr=0.3,
             wait_time=0,
             show=True,
             out_file=None,
             predict=[]):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        wait_time (int): Value of waitKey param.
        show (bool, optional): Whether to show the image with opencv or not.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.

    Returns:
        np.ndarray or None: If neither `show` nor `out_file` is specified, the
            visualized image is returned, otherwise None is returned.
    """
    assert isinstance(class_names, (tuple, list))
    class_name = tuple([str(i+1) for i in range(205)])
    img = mmcv.imread(img)
    img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    for ann in range(len(labels)):
        image_result = {
            'image_id': out_file.split('/')[-1][:-4],
            'category_id': int(labels[ann])+1,
            'score': float(bboxes[ann][4]),
            'bbox': [bboxes[ann][0], bboxes[ann][1], bboxes[ann][2]-bboxes[ann][0], bboxes[ann][3]-bboxes[ann][1]],
        }

        predict.append(image_result)
    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        np.random.seed(42)
        color_masks = [
            np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            for _ in range(max(labels) + 1)
        ]
        for i in inds:
            i = int(i)
            color_mask = color_masks[labels[i]]
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # if out_file specified, do not show image in window
    if out_file is not None:
        show = False
    # draw bounding boxes
    mmcv.imshow_det_bboxes(
        img,
        bboxes,
        labels,
        class_names=class_name,
        score_thr=score_thr,
        show=show,
        wait_time=wait_time,
        out_file=out_file
        )
    return predict


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


# 首先下载模型文件https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth
#config_file = '/home1/rhhHD/mmdetection/configs/cascade_rcnn_r50_fpn_1x.py'
#checkpoint_file = '/home1/rhhHD/mmdetection/work_dirs/cascade_rcnn_r50_fpn_1x_0/latest.pth'
config_file = '/home1/rhhHD/rhh/mmdetection/configs/dcn/cascade_rcnn_dconv_c3-c5_r50_fpn_1x.py'
checkpoint_file = "/home1/rhhHD/rhh/mmdetection/work_dirs/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_3_images/latest_0.86.pth"
# config_file = '/home1/rhhHD/rhh/mmdetection/configs/my_cascade_mask_rcnn_x101_64x4d_fpn_1x.py'
# checkpoint_file = "/home1/rhhHD/rhh/mmdetection/work_dirs/my_4_ohem_cascade_mask_rcnn_x101_64x4d_fpn_1x/latest.pth"


# 初始化模型
model = init_detector(config_file, checkpoint_file)


# 测试一张图片"/home1/rhhHD/rhh/Datasets/iCartonnFace/personai_icartoonface_detval/"
# img = '/home1/rhhHD/rhh/Datasets/iCartonnFace/personai_icartoonface_detval/personai_icartoonface_detval_00393.jpg'
# result = inference_detector(model, img)
# # print(model.CLASSES)
# # predict = []
# s_result(img, result, model.CLASSES, out_file='cartoon_result.png')
# predict = s_result(img, result, model.CLASSES, out_file='cartoon_result.png')
# print(predict)
# with open('/home1/rhhHD/rhh/mmdetection/my_utils/cartoon_results/results_1000_600_400/results_1000_600_400.csv', 'w', encoding='utf-8') as fw:
    # for pred in predict:
        # fw.write(pred)


# # 测试一系列图片
test_json = "/home1/rhhHD/mmdetection_old/my_coco/coco/annotations/instances_test2017.json"
test_path = "/home1/rhhHD/mmdetection_old/my_coco/coco/test2017/"
#result = inference_detector(model, img)
#print(result)
save_path = '/home1/rhhHD/rhh/mmdetection/my_utils/results_end/'
images = []
predict = []
with open(test_json, 'r', encoding='utf-8') as fr:
    test_dict = json.load(fr)
    for image in test_dict['images']:
        images.append(test_path + image['file_name'])
num = 0
for image in images:
    image = str(image)
    result = inference_detector(model, image)
    print(num)
    predict = s_result(image, result, model.CLASSES, out_file=save_path + image.split('/')[-1], predict=predict)
    num += 1

with open('/home1/rhhHD/rhh/mmdetection/my_utils/results_end.json', 'w', encoding='utf-8') as fw:
    json.dump(predict, fw, ensure_ascii=False, cls=MyEncoder)
