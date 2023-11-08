import torch
from SegRunLib.ml.unet3d import Unet3d, U_Net

def get_model(model_name, path_to_weights=None):
    if model_name == 'Unet3d_16ch':
        #return(Unet3d(channels=16, depth=4))
        model = U_Net(channels=16)
        if path_to_weights:
            model.load_state_dict(torch.load(path_to_weights))
        return(model)
    else:
        return None