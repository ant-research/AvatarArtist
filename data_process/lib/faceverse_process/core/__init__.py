from lib.faceverse_process.core.FaceVerseModel_v3 import FaceVerseModel as FaceVerseModel_v3
import numpy as np

def get_recon_model(model='meta', **kargs):
    if model == 'meta_simplify_v31':
        model_path = 'lib/faceverse_process/metamodel/v3/faceverse_v3_1.npy'
        model_dict = np.load(model_path, allow_pickle=True).item()
        recon_model = FaceVerseModel_v3(model_dict, expr_52=False, **kargs)
        return recon_model
    else:
        raise NotImplementedError()
