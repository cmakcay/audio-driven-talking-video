import torch 

# util function to erode mask
def erode_mask(mask):
    erosion_factor = 0.6
    
    # up
    shift = int(erosion_factor * 8)
    mask_up = mask[:,:,shift:,:]
    mask_up = torch.cat([mask_up, torch.ones_like(mask[:,:,0:shift,:])], 2)
    mask = torch.logical_or(mask, mask_up)

    # down
    shift = int(erosion_factor * 40)
    mask_down = mask[:,:,0:-shift,:]
    mask_down = torch.cat([torch.ones_like(mask[:,:,0:shift:]), mask_down], 2)
    mask = torch.logical_or(mask, mask_down)

    # right
    shift = int(erosion_factor * 15)
    mask_right = mask[:,:,:,shift:]
    mask_right = torch.cat([mask_right, torch.zeros_like(mask[:,:,:,0:shift])], 3)
    mask = torch.logical_or(mask, mask_right)
    
    # left
    shift = int(erosion_factor * 15)
    mask_left = mask[:,:,:,0:-shift]
    mask_left = torch.cat([torch.zeros_like(mask[:,:,:,0:shift]), mask_left], 3)
    mask = torch.logical_or(mask, mask_left)

    mask[:,:,:int(erosion_factor * 40),:] = False
    mask[:,:,-int(erosion_factor * 8):,:] = False
    
    return mask 