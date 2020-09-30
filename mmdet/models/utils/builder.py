from mmcv.utils import Registry, build_from_cfg

POSITION_ENCODING = Registry('Position encoding')
TRANSFORMER = Registry('Transformer')


def build_position_encoding(cfg, default_args=None):
    """Builder of Position encoding."""
    return build_from_cfg(cfg, POSITION_ENCODING, default_args)


def build_transformer(cfg, default_args=None):
    """Builder of Transformer."""
    return build_from_cfg(cfg, TRANSFORMER, default_args)
