import torch
from model.training_config import CriterionConfig
from model.decoder_config import TableDetectionDecoderConfig


class MainArgs:
    def __init__(self,
                 eval=False,
                 seed=100,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 lr=1e-4,
                 weight_decay=1e-4,
                 lr_drop=200,
                 batch_size=1,
                 num_workers=0,
                 epochs=5,
                 clip_max_norm=0.1,
                 output_dir='/mnt/d/thesis/data/checkpoints/',
                 data_dir='/mnt/d/thesis/data/',
                 resume='',
                 start_epoch=0,
                 checkpoint_freq=1,
                 only_data_subset=True):

        """main() args"""
        self.eval = eval

        """configuration args"""
        self.seed = seed
        self.device = device
        self.criterion_cfg = CriterionConfig()
        self.model_cfg = TableDetectionDecoderConfig()

        """hyper-parameters"""
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_drop = lr_drop
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epochs = epochs
        self.clip_max_norm = clip_max_norm

        """checkpointing and directory args"""
        self.checkpoint_freq = checkpoint_freq
        self.resume = resume
        self.start_epoch = start_epoch
        self.output_dir = output_dir
        self.data_dir = data_dir
        self.only_data_subset = only_data_subset

