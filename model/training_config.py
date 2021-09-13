from util.matcher import HungarianMatcher


class TrainingConfig:
    def __init__(self,
                 lr = 1e-4,
                 batch_size=2,
                 weight_decay=1e-4,
                 epochs=100,
                 lr_drop=200,
                 clip_max_norm=0.1,
                 frozen_weights=None):
        self.lr = lr
        self.batch_size =batch_size
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.lr_drop = lr_drop
        self.clip_max_norm = clip_max_norm
        self.frozen_weights =frozen_weights


class CriterionConfig:
    def __init__(self,
                 num_classes=1,
                 cost_class=1,
                 cost_bbox=5,
                 cost_giou=2,
                 loss_ce=1,
                 loss_bbox=5,
                 loss_giou=2,
                 eos_coef=0.1,
                 losses=['labels', 'boxes']):#, 'cardinality']):
        self.num_classes = num_classes
        self.matcher = HungarianMatcher(cost_class=cost_class, cost_bbox=cost_bbox, cost_giou=cost_giou)
        weight_dict = {'loss_ce': loss_ce, 'loss_bbox': loss_bbox}
        weight_dict['loss_giou'] = loss_giou
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
