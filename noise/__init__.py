from noise.ladp import LADP
from noise.ddp import DDP
from noise.lavpn import LAVPN
from noise.dae import DAE, NLDAE

TRANSFORMS = [
        LADP,
        DDP,
        LAVPN,
        DAE,
        NLDAE
    ]
TRANSFORM_NAMES = [a().name.lower() for a in TRANSFORMS]


def create_noise_transform(name, **kwargs):
    """
    Denoising pretraining transform entrypoint, allows to create transform just with
    parameters, without using its class
    """

    trans_dict = {a().name.lower(): a for a in TRANSFORMS}
    try:
        model_class = trans_dict[name.lower()]
    except KeyError:
        raise KeyError(
            "Wrong transform type `{}`. Available options are: {}".format(
                name,
                list(trans_dict.keys()),
            )
        )
    return model_class(**kwargs)
