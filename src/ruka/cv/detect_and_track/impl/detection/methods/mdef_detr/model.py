import ruka.util.distributed_fs as dfs
import torch

from .backbone import Backbone, Joiner
from .inference import MDefDETRInference
from .mdef_detr_minus_language.deformable_transformer import DeformableTransformer as MDefDETRMinusLanguageTransformer
from .mdef_detr_minus_language.mdef_detr_minus_language import MDefDETRMinusLanguage
from .position_encoding import PositionEmbeddingSine


def make_mdef_detr_minus_language():
    dfs.set_dfs_v2()
    weights_path = dfs.cached_download(
        'aux_data/cv/weights/mdef_detr/MDef_DETR_minus_language_r101_epoch10.pth')

    model = _make_mdef_detr_minus_language(weights_path).eval()

    return MDefDETRInference(model)


def _make_backbone(backbone_name="resnet101", mask: bool = True):
    backbone = Backbone(backbone_name, train_backbone=True, return_interm_layers=mask, dilation=False)

    hidden_dim = 256
    pos_enc = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
    backbone_with_pos_enc = Joiner(backbone, pos_enc)
    backbone_with_pos_enc.num_channels = backbone.num_channels

    return backbone_with_pos_enc


def _make_mdef_detr(checkpoints_path):
    backbone = _make_backbone()
    transformer = MDefDETRTransformer(d_model=256, return_intermediate_dec=False, num_feature_levels=4,
                                      dim_feedforward=1024, text_encoder_type="roberta-base")
    model = MDefDETR(backbone=backbone, transformer=transformer, num_classes=255, num_queries=300,
                     num_feature_levels=4)
    checkpoint = torch.load(checkpoints_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    return model


def _make_mdef_detr_minus_language(checkpoints_path):
    backbone = _make_backbone()
    transformer = MDefDETRMinusLanguageTransformer(d_model=256, return_intermediate_dec=True, num_feature_levels=4,
                                                   dim_feedforward=1024)
    model = MDefDETRMinusLanguage(backbone=backbone, transformer=transformer, num_classes=1, num_queries=300,
                                  num_feature_levels=4)
    checkpoint = torch.load(checkpoints_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    return model
