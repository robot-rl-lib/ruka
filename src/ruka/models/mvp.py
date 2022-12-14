import os

import timm.models.vision_transformer

from functools import partial

import torch
import torch.nn as nn

import ruka_os.distributed_fs_v2 as dfs_v2
import ruka.util.distributed_fs as dfs

_MODELS = {
    "vits-mae-hoi": "https://berkeley.box.com/shared/static/m93ynem558jo8vltlads5rcmnahgsyzr.pth",
    "vits-mae-in": "https://berkeley.box.com/shared/static/qlsjkv03nngu37eyvtjikfe7rz14k66d.pth",
    "vits-sup-in": "https://berkeley.box.com/shared/static/95a4ncqrh1o7llne2b1gpsfip4dt65m4.pth",
    "vitb-mae-egosoup": "https://berkeley.box.com/shared/static/0ckepd2ja3pi570z89ogd899cn387yut.pth",
    "vitl-256-mae-egosoup": "https://berkeley.box.com/shared/static/6p0pc47mlpp4hhwlin2hf035lxlgddxr.pth",
}

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer
        referene:
            - MAE:  https://github.com/facebookresearch/mae/blob/main/models_vit.py
            - timm: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """
    def __init__(self, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        # remove the classifier
        if hasattr(self, "pre_logits"):
            del self.pre_logits
        del self.head

    def extract_feat(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = x[:, 0].detach().float()
        return x

    def forward_norm(self, x):
        return self.norm(x)

    def forward(self, x):
        return self.forward_norm(self.extract_feat(x))

    def freeze(self):
        self.pos_embed.requires_grad = False
        self.cls_token.requires_grad = False

        def _freeze_module(m):
            for p in m.parameters():
                p.requires_grad = False

        _freeze_module(self.patch_embed)
        _freeze_module(self.blocks)

        trainable_params = []
        for name, p in self.named_parameters():
            if p.requires_grad:
                trainable_params.append(name)


def vit_s16(pretrained, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    assert os.path.exists(pretrained) or pretrained.startswith("none")
    # load from checkpoint
    if not pretrained.startswith("none"):
        load_checkpoint(pretrained, model)
        print("Loaded encoder from: {}".format(pretrained))
    hidden_dim = 384
    return model, hidden_dim


def vit_b16(pretrained, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    assert os.path.exists(pretrained) or pretrained.startswith("none")
    # load from checkpoint
    if not pretrained.startswith("none"):
        load_checkpoint(pretrained, model)
        print("Loaded encoder from: {}".format(pretrained))
    hidden_dim = 768
    return model, hidden_dim


def vit_l16(pretrained, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    assert os.path.exists(pretrained) or pretrained.startswith("none")
    # load from checkpoint
    if not pretrained.startswith("none"):
        load_checkpoint(pretrained, model)
        print("Loaded encoder from: {}".format(pretrained))
    hidden_dim = 1024
    return model, hidden_dim

_MODEL_FUNCS = {
    "vits": vit_s16,
    "vitb": vit_b16,
    "vitl": vit_l16,
}

def unwrap_model(model):
    """Remove the DistributedDataParallel wrapper if present."""
    wrapped = isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel)
    return model.module if wrapped else model


def load_checkpoint(checkpoint_file, model):
    """Loads a checkpoint selectively based on the input options."""
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    state_dict = checkpoint["model"]
    r = unwrap_model(model).load_state_dict(state_dict, strict=False)
    if r.unexpected_keys or r.missing_keys:
        print(f"Loading weights, unexpected keys: {r.unexpected_keys}")
        print(f"Loading weights, missing keys: {r.missing_keys}")


def available_models():
    """Retrieves the names of available models."""
    return list(_MODELS.keys())

class ImageNetNormalizeWrapper(nn.Module):
    def __init__(self, mvp: nn.Module):
        super().__init__()
        self._mvp = mvp
        self.register_buffer('_img_mean',
                                torch.tensor([0.485, 0.456, 0.406],
                                        dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer('_img_std',
                                torch.tensor([0.229, 0.224, 0.225],
                                        dtype=torch.float32).view(1, -1, 1, 1))
        
    def forward(self, x):
        x.sub(self._img_mean).div(self._img_std)
        return self._mvp(x)

    def freeze(self):
        return self._mvp.freeze()



def load(name):
    """Loads a pre-trained model."""
    assert name in _MODELS.keys(), "Model {} not available".format(name)
    pretrained = dfs.cached_download(f'aux_data/{name}.pth', dfs_v2)
    model_func = _MODEL_FUNCS[name.split("-")[0]]
    img_size = 256 if "-256-" in name else 224
    model, _ = model_func(pretrained=pretrained, img_size=img_size)
    return ImageNetNormalizeWrapper(model)

