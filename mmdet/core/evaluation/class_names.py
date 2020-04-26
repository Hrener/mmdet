import mmcv


def wider_face_classes():
    return ['face']


def voc_classes():
    return [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]


def imagenet_det_classes():
    return [
        'accordion', 'airplane', 'ant', 'antelope', 'apple', 'armadillo',
        'artichoke', 'axe', 'baby_bed', 'backpack', 'bagel', 'balance_beam',
        'banana', 'band_aid', 'banjo', 'baseball', 'basketball', 'bathing_cap',
        'beaker', 'bear', 'bee', 'bell_pepper', 'bench', 'bicycle', 'binder',
        'bird', 'bookshelf', 'bow_tie', 'bow', 'bowl', 'brassiere', 'burrito',
        'bus', 'butterfly', 'camel', 'can_opener', 'car', 'cart', 'cattle',
        'cello', 'centipede', 'chain_saw', 'chair', 'chime', 'cocktail_shaker',
        'coffee_maker', 'computer_keyboard', 'computer_mouse', 'corkscrew',
        'cream', 'croquet_ball', 'crutch', 'cucumber', 'cup_or_mug', 'diaper',
        'digital_clock', 'dishwasher', 'dog', 'domestic_cat', 'dragonfly',
        'drum', 'dumbbell', 'electric_fan', 'elephant', 'face_powder', 'fig',
        'filing_cabinet', 'flower_pot', 'flute', 'fox', 'french_horn', 'frog',
        'frying_pan', 'giant_panda', 'goldfish', 'golf_ball', 'golfcart',
        'guacamole', 'guitar', 'hair_dryer', 'hair_spray', 'hamburger',
        'hammer', 'hamster', 'harmonica', 'harp', 'hat_with_a_wide_brim',
        'head_cabbage', 'helmet', 'hippopotamus', 'horizontal_bar', 'horse',
        'hotdog', 'iPod', 'isopod', 'jellyfish', 'koala_bear', 'ladle',
        'ladybug', 'lamp', 'laptop', 'lemon', 'lion', 'lipstick', 'lizard',
        'lobster', 'maillot', 'maraca', 'microphone', 'microwave', 'milk_can',
        'miniskirt', 'monkey', 'motorcycle', 'mushroom', 'nail', 'neck_brace',
        'oboe', 'orange', 'otter', 'pencil_box', 'pencil_sharpener', 'perfume',
        'person', 'piano', 'pineapple', 'ping-pong_ball', 'pitcher', 'pizza',
        'plastic_bag', 'plate_rack', 'pomegranate', 'popsicle', 'porcupine',
        'power_drill', 'pretzel', 'printer', 'puck', 'punching_bag', 'purse',
        'rabbit', 'racket', 'ray', 'red_panda', 'refrigerator',
        'remote_control', 'rubber_eraser', 'rugby_ball', 'ruler',
        'salt_or_pepper_shaker', 'saxophone', 'scorpion', 'screwdriver',
        'seal', 'sheep', 'ski', 'skunk', 'snail', 'snake', 'snowmobile',
        'snowplow', 'soap_dispenser', 'soccer_ball', 'sofa', 'spatula',
        'squirrel', 'starfish', 'stethoscope', 'stove', 'strainer',
        'strawberry', 'stretcher', 'sunglasses', 'swimming_trunks', 'swine',
        'syringe', 'table', 'tape_player', 'tennis_ball', 'tick', 'tie',
        'tiger', 'toaster', 'traffic_light', 'train', 'trombone', 'trumpet',
        'turtle', 'tv_or_monitor', 'unicycle', 'vacuum', 'violin',
        'volleyball', 'waffle_iron', 'washer', 'water_bottle', 'watercraft',
        'whale', 'wine_bottle', 'zebra'
    ]


def imagenet_vid_classes():
    return [
        'airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car',
        'cattle', 'dog', 'domestic_cat', 'elephant', 'fox', 'giant_panda',
        'hamster', 'horse', 'lion', 'lizard', 'monkey', 'motorcycle', 'rabbit',
        'red_panda', 'sheep', 'snake', 'squirrel', 'tiger', 'train', 'turtle',
        'watercraft', 'whale', 'zebra'
    ]


def coco_classes():
    return [
        '瓜子壳', '核桃', '花生壳', '毛豆壳', '西瓜子', '枣核', '话梅核', '苹果皮', '柿子皮', '西瓜皮', '香蕉皮',
               '柚子皮', '荔枝壳', '芒果皮', '苹果核', '干果', '桔子皮', '饼干', '面包', '糖果', '宠物饲料', '风干食品',
               '蜜饯', '肉干', '冲泡饮料粉', '奶酪', '罐头', '糕饼', '薯片', '树叶', '杂草', '绿植', '鲜花', '豆类',
               '动物内脏', '绿豆饭', '谷类及加工物', '贝类去硬壳', '虾', '面食', '肉类', '五谷杂粮', '排骨-小肋排', '鸡',
               '鸡骨头', '螺蛳', '鸭', '鱼', '菜根', '菜叶', '菌菇类', '鱼鳞', '调料', '茶叶渣', '咖啡渣', '粽子', '动物蹄',
               '小龙虾', '蟹壳', '酱料', '鱼骨头', '蛋壳', '中药材', '中药渣', '镜子', '玻璃制品', '窗玻璃', '碎玻璃片',
               '化妆品玻璃瓶', '食品及日用品玻璃瓶罐', '保温杯', '玻璃杯', '图书期刊', '报纸', '食品外包装盒', '鞋盒',
               '利乐包', '广告单', '打印纸', '购物纸袋', '日历', '快递纸袋', '信封', '烟盒', '易拉罐', '金属制品', '吸铁石',
               '铝制品', '金属瓶罐', '金属工具', '罐头盒', '勺子', '菜刀', '叉子', '锅', '金属筷子', '数据线', '塑料玩具',
               '矿泉水瓶', '塑料泡沫', '塑料包装', '硬塑料', '一次性塑料餐盒餐具', '电线', '塑料衣架', '密胺餐具', '亚克力板',
               'PVC管', '插座', '化妆品塑料瓶', '篮球', '足球', 'KT板', '食品塑料盒', '食用油桶', '塑料杯', '塑料盆',
               '一次性餐盒', '废弃衣服', '鞋', '碎布', '书包', '床上用品', '棉被', '丝绸手绢', '枕头', '毛绒玩具', '皮带',
               '电路板', '充电宝', '木制品', '优盘', '灯管灯泡', '节能灯', '二极管', '纽扣电池', '手机电池', '镍镉电池',
               '锂电池', '蓄电池', '胶卷', '照片', '指甲油瓶', 'X光片', '农药瓶', '杀虫剂及罐', '蜡烛', '墨盒', '染发剂壳',
               '消毒液瓶', '油漆桶', '药品包装', '药瓶', '废弃针管', '输液管', '口服液瓶', '眼药水瓶', '水银温度计',
               '水银血压计', '胶囊', '药片', '固体杀虫剂', '甘蔗皮', '坚果壳', '橡皮泥', '毛发', '棉签', '创可贴', '口红',
               '笔', '纸巾', '胶带', '湿巾', '水彩笔', '打火机', '防碎气泡膜', '榴莲壳', '睫毛膏', '眼影', '仓鼠浴沙',
               '大骨棒', '旧毛巾', '竹制品', '粉笔', '一次性口罩', '一次性手套', '粉底液', '灰土', '尼龙制品', '尿片',
               '雨伞', '带胶制品', '牙膏皮', '狗尿布', '椰子壳', '粉扑', '破碗碟', '陶瓷', '卫生纸', '烟头', '假睫毛',
               '猫砂', '牙刷', '玉米棒'
    ]
    #return ['face']


def cityscapes_classes():
    return [
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle'
    ]


dataset_aliases = {
    'voc': ['voc', 'pascal_voc', 'voc07', 'voc12'],
    'imagenet_det': ['det', 'imagenet_det', 'ilsvrc_det'],
    'imagenet_vid': ['vid', 'imagenet_vid', 'ilsvrc_vid'],
    'coco': ['coco', 'mscoco', 'ms_coco'],
    'wider_face': ['WIDERFaceDataset', 'wider_face', 'WDIERFace'],
    'cityscapes': ['cityscapes']
}


def get_classes(dataset):
    """Get class names of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if mmcv.is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_classes()')
        else:
            raise ValueError('Unrecognized dataset: {}'.format(dataset))
    else:
        raise TypeError('dataset must a str, but got {}'.format(type(dataset)))
    return labels
