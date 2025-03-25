#################################################
# Copyright (c) 2021-present, xiaobing.ai, Inc. #
# All rights reserved.                          #
#################################################
# CV Research, DEV(USA) xiaobing.               #
# Written by wangduomin@xiaobing.ai             #
#################################################

import torch
import lib.models as models


def make_model(cfg):
    """
    Build and initialize the models based on the given configuration.

    Args:
        cfg: Configuration object containing model specifications.

    Returns:
        list: A list containing the initialized models [fd, ldmk, ldmk_3d].
    """
    return_list = []

    # Create face detection (fd) model
    fd = models.define_networks(cfg, cfg.model.fd.model_type, cfg.model.fd.model_cls)
    return_list.append(fd)

    # Create landmark (ldmk) model
    ldmk = models.define_networks(cfg, cfg.model.ldmk.model_type, cfg.model.ldmk.model_cls)
    return_list.append(ldmk)

    # Create 3D landmark (ldmk_3d) model
    ldmk_3d = models.define_networks(cfg, cfg.model.ldmk_3d.model_type, cfg.model.ldmk_3d.model_cls)
    return_list.append(ldmk_3d)

    return return_list