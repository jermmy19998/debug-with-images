import os
import numpy as np
from PIL import Image
import torch
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import shutil

def dwi(images, var_name, save_dir='./img_debug', slice_index=None):
    """
    Save images to a specified directory, with subdirectories named after the input variable name.

    Parameters:
    images (Union[numpy.ndarray, torch.Tensor, pydicom.Dataset, PIL.Image.Image]): Image data.
    var_name (str): Name of the variable to use for subdirectory naming.
    save_dir (str): Directory path to save the images. Default is './img_debug'.
    slice_index (int or None): Index of the slice to save. If None, save the middle slice. Default is None.

    Returns:
    None
    """
    # Check if the save directory exists, create it if it doesn't
    if not os.path.exists(save_dir):
        os.makedirs(save_dir,exist_ok=True)
    
    # Determine the image type and convert to numpy array if necessary
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
        # Ensure the correct shape (H, W, C)
        if images.ndim == 4:
            images = np.squeeze(images, axis=0)  # Remove batch dimension if present
        if images.ndim == 3:
            images = np.transpose(images, (1, 2, 0))  # Transpose to (H, W, C)
        elif images.ndim == 4:
            images = np.transpose(images, (0, 2, 3, 1))
            images = np.squeeze(images, axis=0)  # Remove batch and channel dimensions if present
    elif isinstance(images, pydicom.Dataset):
        # Convert DICOM to numpy array
        if 'PixelData' in images:
            pixel_array = images.pixel_array
            if 'WindowWidth' in images and 'WindowCenter' in images:
                window_center = images.WindowCenter
                window_width = images.WindowWidth
                if not isinstance(window_center, list):
                    window_center = [window_center]
                if not isinstance(window_width, list):
                    window_width = [window_width]
                if len(window_center) == 0:
                    window_center = None
                if len(window_width) == 0:
                    window_width = None
                if window_center is not None and window_width is not None:
                    pixel_array = apply_voi_lut(pixel_array, window_center, window_width)
                images = pixel_array
    
    elif isinstance(images, Image.Image):
        images = np.array(images)
    
    # Convert the images to a NumPy array if necessary
    elif isinstance(images, np.ndarray):
        if len(images.shape) == 3:  # make sure images is 3d
            if images.shape[0] == 3:  # id frist shape is channel(C) (C, H, W)
                images = np.transpose(images, (1, 2, 0))  # transpose to (H, W, C)
    else:
        raise ValueError(f"Unsupported image type:{type(images)}. Supported types are numpy.ndarray, torch.Tensor, pydicom.Dataset, PIL.Image.Image.")
    
    # Determine the image shape
    if images.ndim == 2:
        images = np.expand_dims(images, axis=-1)  # Add channel dimension
    
    if images.ndim != 3:
        raise ValueError("Image array must be 3D with shape (H, W, C).")
    
    # Check slice_index validity
    if slice_index is None:
        slice_index = images.shape[2] // 2  # Default to the middle slice (0-based index)
    elif slice_index < 0 or slice_index >= images.shape[2]:
        raise ValueError(f"slice_index must be within the range [0, {images.shape[2] - 1}].")
    
    # Create a subdirectory for the input variable name
    save_subdir = os.path.join(save_dir, var_name)
    if not os.path.exists(save_subdir):
        os.makedirs(save_subdir)
    
    # Extract the selected image slice
    img = images[:, :, slice_index]
    
    # Convert the image to uint8 (if necessary)
    if img.dtype != np.uint8:
        img = img * 255
        img = img.astype(np.uint8)
    
    # Create an Image object from the array
    img_pil = Image.fromarray(img)
    
    # Save the image with shape information in the filename
    image_shape = f"{images.shape[0]}x{images.shape[1]}"
    img_pil.save(os.path.join(save_subdir, f'image_{image_shape}_{slice_index + 1}.png'))

def dd(self):
    """
    Delete the specified directory and its contents recursively, with confirmation from user.
    Returns:
    None
    """
    if os.path.exists(self.save_dir):
        confirm = input(f"Are you sure you want to delete {self.save_dir}? (y/n): ")
        if confirm.lower() == 'y':
            shutil.rmtree(self.save_dir)
            print(f"Deleted {self.save_dir}.")
        else:
            print(f"Deletion of {self.save_dir} cancelled.")
    else:
        print(f"{self.save_dir} does not exist.")
