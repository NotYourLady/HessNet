#from tqdm import tqdm
from tqdm import tqdm
import torch
import torchio as tio
from SegRunLib.ml.get_model import get_model
from SegRunLib.scripts.nifty_utils import save_vol_as_nii

class Runner:
    def __init__(self, settings):    
        self.settings = {
            "patch_shape" : (256, 256, 64),
            "overlap_shape" : (32, 32, 24),
            "batch_size" : 1,
            "device" : settings.get('device', 'cpu'),
            "path_to_weights" : settings.get('path_to_weights', None),
            "num_workers" : settings.get('num_workers', 4),
        }
        if self.settings["path_to_weights"] is None:
            raise RuntimeError('Need a weights path in config')
        
        self.model = get_model('Unet3d_16ch',
                               self.settings["path_to_weights"]).to(self.settings["device"])
        
        
    def fast_predict(self, patch_loader, grid_aggregator, thresh=0.5):
        for patches_batch in tqdm(patch_loader):
            patch_locations = patches_batch[tio.LOCATION]
            head_patches = patches_batch['head']['data'].to(self.settings['device'])
            with torch.no_grad():
                patch_seg = self.model(head_patches)
                grid_aggregator.add_batch(patch_seg.detach().cpu(), patch_locations)
        
        seg = grid_aggregator.get_output_tensor()
        if thresh is not None: 
            seg = torch.where(seg>thresh, 1, 0)
        return(seg)
    
    
    def single_predict(self, subject):
        grid_sampler = tio.GridSampler(subject,
                                       patch_size=self.settings["patch_shape"],
                                       patch_overlap=self.settings["overlap_shape"])
        grid_aggregator = tio.data.GridAggregator(sampler=grid_sampler, overlap_mode='hann')
        patch_loader = torch.utils.data.DataLoader(grid_sampler,
                                                   batch_size=self.settings["batch_size"],
                                                   num_workers=self.settings["num_workers"])
        seg = self.fast_predict(patch_loader, grid_aggregator)
        return(seg)
    
    
    def predict_and_save(self, in_path_nifty, out_path_nifty):
        subject = tio.Subject({'head' : tio.ScalarImage(in_path_nifty)})
        subject = tio.transforms.ZNormalization()(subject)
        seg = self.single_predict(subject)
        save_vol_as_nii(seg, subject.head.affine, out_path_nifty)
