from SegRunLib.ml.unet3d import Unet3d, U_Net

def get_model(model_name):
    if model_name == 'Unet3d_16ch':
        #return(Unet3d(channels=16, depth=4))
        return(U_Net(channels=16))
    else:
        return None