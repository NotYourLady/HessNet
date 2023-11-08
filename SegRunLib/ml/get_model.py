import os
import torch
from SegRunLib.ml.unet3d import Unet3d, U_Net


def get_model(model_name, path_to_weights=None, pretrained=False):
    if (path_to_weights is not None) and pretrained:
        raise RuntimeError("SegRunLib::ml::get model: please use path_to_weights OR pretrained")
    repo_url = 'https://github.com/NotYourLady/SegRunLib/raw/main/SegRunLib/pretrained_models/weights/'
    tmp_path_pth='./tmp.pth'
    if model_name == 'Unet3d_16ch':
        #return(Unet3d(channels=16, depth=4))
        model = U_Net(channels=16)
        if pretrained:
            url_weights = repo_url + 'Unet3d_16ch_weights.pth'
            torch.hub.download_url_to_file(url_weights, tmp_path_pth)
            model.load_state_dict(torch.load(tmp_path_pth))
            os.remove(tmp_path_pth)
        if path_to_weights:
            model.load_state_dict(torch.load(path_to_weights))
        return(model)
    else:
        return None