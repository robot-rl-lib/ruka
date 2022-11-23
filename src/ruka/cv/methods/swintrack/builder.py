from .drop_path import DropPathAllocator
import torch.nn as nn
from .modules.encoder.builder import build_encoder
from .modules.decoder.builder import build_decoder
from .positional_encoding.builder import build_position_embedding
from .modules.head.builder import build_head
from .network import SwinTrack
from ruka.cv.backbones import build_swin_transformer_backbone
import ruka.util.distributed_fs as dfs
import os
import torch


def build_swin_track_main_components(config):
    transformer_config = config['transformer']

    drop_path_config = transformer_config['drop_path']
    drop_path_allocator = DropPathAllocator(drop_path_config['rate'])

    backbone_dim = transformer_config['backbone']['dim']
    transformer_dim = transformer_config['dim']

    z_shape = transformer_config['backbone']['template']['shape']
    x_shape = transformer_config['backbone']['search']['shape']
    backbone_out_stage = transformer_config['backbone']['stage']

    z_input_projection = None
    x_input_projection = None
    if backbone_dim != transformer_dim:
        z_input_projection = nn.Linear(backbone_dim, transformer_dim)
        x_input_projection = nn.Linear(backbone_dim, transformer_dim)

    num_heads = transformer_config['num_heads']
    mlp_ratio = transformer_config['mlp_ratio']
    qkv_bias = transformer_config['qkv_bias']
    drop_rate = transformer_config['drop_rate']
    attn_drop_rate = transformer_config['attn_drop_rate']

    position_embedding_config = transformer_config['position_embedding']
    z_pos_enc, x_pos_enc = build_position_embedding(position_embedding_config, z_shape, x_shape, transformer_dim)

    with drop_path_allocator:
        encoder = build_encoder(config, drop_path_allocator,
                                transformer_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                                z_shape, x_shape)

        decoder = build_decoder(config, drop_path_allocator,
                                transformer_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                                z_shape, x_shape)

    out_norm = nn.LayerNorm(transformer_dim)

    return encoder, decoder, out_norm, backbone_out_stage, backbone_out_stage, z_input_projection, x_input_projection, z_pos_enc, x_pos_enc


def build_swin_track_by_config(config):
    backbone = build_swin_transformer_backbone(**config['backbone'], load_pretrained=False)
    encoder, decoder, out_norm, z_backbone_out_stage, x_backbone_out_stage, z_input_projection, x_input_projection, z_pos_enc, x_pos_enc = \
        build_swin_track_main_components(config)
    head = build_head(config)

    return SwinTrack(backbone, encoder, decoder, out_norm, head, z_backbone_out_stage, x_backbone_out_stage, z_input_projection, x_input_projection, z_pos_enc, x_pos_enc)


_config = dict({
    'Tiny': {
        'backbone': {
            'name': 'swin_tiny_patch4_window7_224',
            'output_layers': [2],
        },
        'transformer': {
            'position_embedding': {
                'enabled': False,
                'type': 'sine',
                'with_branch_index': True,
            },
            'untied_position_embedding': {
                'absolute': {
                    'enabled': True,
                },
                'relative': {
                    'enabled': True,
                },
            },
            'drop_path': {
                'rate': 0.1,
            },
            'dim': 384,
            'backbone': {
                'dim': 384,
                'stage': 2,
                'template': {
                    'shape': [7, 7],
                },
                'search': {
                    'shape': [14, 14],
                },
            },
            'num_heads': 8,
            'mlp_ratio': 4,
            'qkv_bias': True,
            'drop_rate': 0,
            'attn_drop_rate': 0,
            'encoder': {
                'type': 'concatenation_feature_fusion',
                'num_layers': 4,
            },
            'decoder': {
                'type': 'concatenation_feature_fusion',
                'num_layers': 1,
            },
        },
        'head': {
            'output_protocol': {
                'type': 'ResponseMap',
                'parameters': {
                    'label': {
                        'size': [14, 14],
                    }
                }
            },
            'type': 'Mlp',
            'parameters': {
                'dim': 384,
            },
        },
        'weights': 'aux_data/cv/weights/swin_track/SwinTrack-Tiny.pth',
    },
    'Base-384': {
        'backbone': {
            'name': 'swin_base_patch4_window12_384_in22k',
            'output_layers': [2],
        },
        'transformer': {
            'position_embedding': {
                'enabled': False,
                'type': 'sine',
                'with_branch_index': True,
            },
            'untied_position_embedding': {
                'absolute': {
                    'enabled': True,
                },
                'relative': {
                    'enabled': True,
                },
            },
            'drop_path': {
                'rate': 0.1,
            },
            'dim': 512,
            'backbone': {
                'dim': 512,
                'stage': 2,
                'template': {
                    'shape': [12, 12],
                },
                'search': {
                    'shape': [24, 24],
                },
            },
            'num_heads': 8,
            'mlp_ratio': 4,
            'qkv_bias': True,
            'drop_rate': 0,
            'attn_drop_rate': 0,
            'encoder': {
                'type': 'concatenation_feature_fusion',
                'num_layers': 8,
            },
            'decoder': {
                'type': 'concatenation_feature_fusion',
                'num_layers': 1,
            },
        },
        'head': {
            'output_protocol': {
                'type': 'ResponseMap',
                'parameters': {
                    'label': {
                        'size': [24, 24],
                    }
                }
            },
            'type': 'Mlp',
            'parameters': {
                'dim': 512,
            },
        },
        'weights': 'aux_data/cv/weights/swin_track/SwinTrack-Base-384.pth',
    },
})

def build_swin_track(swin_track_type):
    assert isinstance(swin_track_type, str), 'swin_track_type must be str'
    assert swin_track_type in _config, 'Unknown SwinTrack type'

    model = build_swin_track_by_config(_config[swin_track_type])
    # TODO: загружать не в pwd?
    weights_local = dfs.download_if_not_exists(_config[swin_track_type]['weights'])
    model.load_state_dict(torch.load(weights_local, map_location='cpu')['model'])

    return model.eval()
