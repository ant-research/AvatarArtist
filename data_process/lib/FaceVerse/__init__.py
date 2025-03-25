from .FaceVerseModel_v3 import FaceVerseModel as FaceVerseModel_v3
import numpy as np


def get_recon_model(model_path=None, return_dict=False, **kargs):
    model_dict = np.load(model_path, allow_pickle=True).item()
    recon_model = FaceVerseModel_v3(model_dict, expr_52=False, **kargs)
    if return_dict:
        return recon_model, model_dict
    else:
        return recon_model
