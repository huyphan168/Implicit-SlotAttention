from .cnn import ResNet, ResNet18
from .dvae import dVAE

def build_backbones(cfg):
    if cfg.backbone == "resnet":
        return ResNet()
    elif cfg.backbone == "resnet18":
        return ResNet18()
    elif cfg.backbone == "dvae":
        return dVAE(cfg.vocab_size, cfg.img_channels)
    else:
        raise NotImplementedError

