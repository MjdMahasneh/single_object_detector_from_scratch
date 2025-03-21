class Config:
    def __init__(self):
        self.image_size = (224, 224)
        self.lr = 1e-3
        self.use_scheduler = False
        self.loss_fn = "smoothl1"  # "iou", "giou", "ciou", "mse", "smoothl1", "WingLoss", "L1Loss"
        self.batch_size = 64#16
        self.shuffle = True
        self.epochs = 200
        self.backbone = "resnet34"  # "resnet34", "mobilenet_v2", "shufflenet", "efficientnet_b0"
        self.freeze_backbone = False
        self.eval_every = 4
        self.train_images = './data/train/images'
        self.train_labels = './data/train/labels'
        self.test_images = './data/test/images'
        self.test_labels = './data/test/labels'
        self.checkpoints_dir = "checkpoints"
        self.hflip = True
        self.vflip = True
        self.rotate = True
        self.color_jitter = True
        self.grayscale = False
        self.blur = False
        self.sharpness = False

