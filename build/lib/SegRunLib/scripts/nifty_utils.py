import torch
import nibabel as nib


def save_vol_as_nii(arr, affine, path_to_save):
    if len(arr.shape) not in (3, 4):
        raise "Error::save_vol_as_nii: bad array shape"
    if len(arr.shape)==4:
        arr = arr[0]
    if type(arr) is torch.Tensor:
        arr = arr.numpy()
    empty_header = nib.Nifti1Header()
    Nifti1Image = nib.Nifti1Image(arr, affine, empty_header)
    nib.save(Nifti1Image, path_to_save)