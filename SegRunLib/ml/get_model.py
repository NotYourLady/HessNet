import torch
from SegRunLib.ml.unet3d import Unet3d, U_Net


def get_model(model_name, path_to_weights=None):
    tmp_path_pt='./tmp.pth'
    if model_name == 'Unet3d_16ch':
        #return(Unet3d(channels=16, depth=4))
        model = U_Net(channels=16)
        url_weights = 'https://github.com/NotYourLady/SegRunLib/blob/main/SegRunLib/pretrained_models/weights/Unet3d_16ch_weights?download=True'
        torch.hub.download_url_to_file(url_pt, tmp_path_pt)
        torch_model.load_state_dict(torch.load(tmp_path_pt))
        if path_to_weights:
            model.load_state_dict(torch.load(path_to_weights))
        return(model)
    else:
        return None