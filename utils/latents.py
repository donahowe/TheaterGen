import torch
import numpy as np
from . import utils
from utils import torch_device
import matplotlib.pyplot as plt
import pickle
from PIL import Image, ImageDraw
import matplotlib.patches as patches

def draw_box(pil_img, bboxes, phrases):
    draw = ImageDraw.Draw(pil_img)
    for obj_bbox, phrase in zip(bboxes, phrases):
        x_min, y_min, w_box, h_box = obj_bbox[0], obj_bbox[1], obj_bbox[2], obj_bbox[3]
        x_max = x_min + w_box
        y_max = y_min + h_box
        draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=5)
def draw_box_v2(pil_img, bboxes, phrases):
    draw = ImageDraw.Draw(pil_img)
    for obj_bbox, phrase in zip(bboxes, phrases):
        x_min, y_min, x_max, y_max = obj_bbox[0], obj_bbox[1], obj_bbox[2], obj_bbox[3]
        draw.rectangle([64*x_min, 64*y_min, 64*x_max,64*y_max], outline='red', width=1)
def resize_tensor(tensor, refactor):
    l = int(refactor)
    n, m = tensor.shape
    new_n, new_m = n // l, m // l
    resized_tensor = torch.zeros((new_n, new_m), dtype=tensor.dtype)
    for i in range(new_n):
        for j in range(new_m):
            resized_tensor[i, j] = torch.max(tensor[i*l:(i+1)*l, j*l:(j+1)*l])
    return resized_tensor
def find_bounding_box(tensor):
    true_indices = torch.nonzero(tensor)
    if true_indices.shape[0] == 0:
        return None
    min_row = true_indices[:, 0].min().item()
    max_row = true_indices[:, 0].max().item()
    min_col = true_indices[:, 1].min().item()
    max_col = true_indices[:, 1].max().item()

    return min_row, min_col, max_row, max_col
def visualize_tensor_with_bbox(tensor, bbox):
    fig, ax = plt.subplots()
    ax.imshow(tensor, cmap='binary')
    rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()

def prepare_mid_image(basever, repeat_ind, mask_tensor_list_512, single_obj_img_list, bboxes):
    mask_tensor_512 = mask_tensor_list_512[0]
    m,n = mask_tensor_512.size()
    new_mask_tensor = np.zeros((m, n)).astype(np.uint8)
    white_image = Image.new('RGB', (m, n), (0, 0, 0))

    tag = 0
    for image,mask_tensor_512, box in zip(single_obj_img_list, mask_tensor_list_512, bboxes):
        tag += 1
        x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3]
        box_center = [(x_min+x_max)/2, (y_min+y_max)/2]

        box_w = abs(x_max-x_min)
        box_h = abs(y_max-y_min)
        m,n = mask_tensor_512.size()
        abs_box_w = box_w*n
        abs_box_h = box_h*m
        abs_box_x_min = int(x_min*n)
        abs_box_y_min = int(y_min*m)

        min_mask_y, min_mask_x, max_mask_y, max_mask_x = find_bounding_box(mask_tensor_512)
        mask_w = abs(max_mask_x-min_mask_x)
        mask_h = abs(max_mask_y-min_mask_y)

        image_array = np.array(image)
        cropped_image_array = image_array[min_mask_y:max_mask_y, min_mask_x:max_mask_x, :]
        located_mask_tensor = mask_tensor_512[min_mask_y:max_mask_y,min_mask_x:max_mask_x]

        cropped_image = Image.fromarray(cropped_image_array)

        mask_array = np.where(located_mask_tensor, 255, 0).astype(np.uint8)
        mask_image = Image.fromarray(mask_array, mode='L')

        refactor = max(mask_w/abs_box_w, mask_h/abs_box_h)
        print("refactor: ",refactor)
        new_w = int(mask_w/refactor)
        new_h = int(mask_h/refactor)
        resize_img = cropped_image.resize((new_w,new_h))
        resize_mask = mask_image.resize((new_w,new_h))

        resize_mask_tensor = np.array(resize_mask)
        resize_img_tensor = np.array(resize_img)
        resize_mask_tensor[resize_mask_tensor > 0] = 255
        re_m, re_n = len(resize_mask_tensor), len(resize_mask_tensor[0])
        resize_mask_tensor_normalized = resize_mask_tensor / 255
        resize_img_tensor = resize_img_tensor * np.expand_dims(resize_mask_tensor_normalized.astype(np.uint8), axis=2)

        
        small_mask_tensor = Image.fromarray(resize_mask_tensor, mode='L')
        resize_factor = 1 #parame
        new_mask_resized = small_mask_tensor.resize((int(re_n * resize_factor), int(re_m * resize_factor)), Image.BOX)
        final_mask = Image.new('L', (re_n, re_m), color=0)
        x_offset = (re_n - int(re_n * resize_factor)) // 2
        y_offset = (re_m - int(re_m * resize_factor)) // 2
        final_mask.paste(new_mask_resized, (x_offset, y_offset))
        
        resize_mask_tensor = np.array(final_mask)
        resize_mask_tensor[resize_mask_tensor > 0] = 255
        
        img_cover_tensor = ~new_mask_tensor.copy()
        img_cover_tensor = img_cover_tensor / 255 
        img_cover_tensor2 = new_mask_tensor.copy()
        img_cover_tensor2 = img_cover_tensor2 / 255 
        
        white_array = np.array(white_image)

        crop_m = len(white_array[abs_box_y_min : abs_box_y_min + re_m, abs_box_x_min : abs_box_x_min + re_n, :])
        cop_n = len(white_array[abs_box_y_min : abs_box_y_min + re_m, abs_box_x_min : abs_box_x_min + re_n, :][0])
        resize_img_tensor = resize_img_tensor[:crop_m, :cop_n]
        resize_mask_tensor = resize_mask_tensor[:crop_m, :cop_n]

        white_array[abs_box_y_min : abs_box_y_min + re_m, abs_box_x_min : abs_box_x_min + re_n, :] = resize_img_tensor
        white_array = white_array * np.expand_dims(img_cover_tensor.astype(np.uint8), axis=2)

        origin_white_array = np.array(white_image)
        final_array = white_array + (origin_white_array * np.expand_dims(img_cover_tensor2.astype(np.uint8), axis=2))

        white_image = Image.fromarray(final_array)

        #deal masks
        new_mask_tensor[abs_box_y_min : abs_box_y_min + re_m, abs_box_x_min : abs_box_x_min + re_n] = new_mask_tensor[abs_box_y_min : abs_box_y_min + re_m, abs_box_x_min : abs_box_x_min + re_n] + resize_mask_tensor
        new_mask_tensor[new_mask_tensor>255] = 255
        new_mask = Image.fromarray(~new_mask_tensor, mode='L')
        
    print("Down Inpainting Rreperation")
    white_image.save(f"visualization/{repeat_ind}vis_image.png")
    new_mask.save(f"visualization/{repeat_ind}vis_mask.png")
    return new_mask, white_image


def get_unscaled_latents(batch_size, in_channels, height, width, generator, dtype):
    """
    in_channels: often obtained with `unet.config.in_channels`
    """
    # Obtain with torch.float32 and cast to float16 if needed
    # Directly obtaining latents in float16 will lead to different latents
    latents_base = torch.randn(
        (batch_size, in_channels, height // 8, width // 8),
        generator=generator, dtype=dtype
    ).to(torch_device, dtype=dtype)
    
    return latents_base

def get_scaled_latents(batch_size, in_channels, height, width, generator, dtype, scheduler):
    latents_base = get_unscaled_latents(batch_size, in_channels, height, width, generator, dtype)
    latents_base = latents_base * scheduler.init_noise_sigma
    return latents_base 

def blend_latents(latents_bg, latents_fg, fg_mask, fg_blending_ratio=0.01):
    """
    in_channels: often obtained with `unet.config.in_channels`
    """
    assert not torch.allclose(latents_bg, latents_fg), "latents_bg should be independent with latents_fg"
    
    dtype = latents_bg.dtype
    latents = latents_bg * (1. - fg_mask) + (latents_bg * np.sqrt(1. - fg_blending_ratio) + latents_fg * np.sqrt(fg_blending_ratio)) * fg_mask
    latents = latents.to(dtype=dtype)

    return latents

@torch.no_grad()
def compose_latents(adapter, model_dict, latents_all_list, mask_tensor_list, num_inference_steps, overall_batch_size, height, width, latents_bg=None, bg_seed=None, compose_box_to_bg=True, use_fast_schedule=False, fast_after_steps=None):
    unet, scheduler, dtype = adapter.pipe.unet, adapter.pipe.scheduler, adapter.pipe.unet.dtype
    
    if latents_bg is None:
        generator = torch.manual_seed(bg_seed)  # Seed generator to create the inital latent noise
        latents_bg = get_scaled_latents(overall_batch_size, unet.config.in_channels, height, width, generator, dtype, scheduler)
    
    # Other than t=T (idx=0), we only have masked latents. This is to prevent accidentally loading from non-masked part. Use same mask as the one used to compose the latents.
    if use_fast_schedule:
        # If we use fast schedule, we only compose the frozen steps because the later steps do not match.
        composed_latents = torch.zeros((fast_after_steps + 1, *latents_bg.shape), dtype=dtype)
    else:
        # Otherwise we compose all steps so that we don't need to compose again if we change the frozen steps.
        composed_latents = torch.zeros((num_inference_steps + 1, *latents_bg.shape), dtype=dtype) #51*1*4*64*64
    composed_latents[0] = latents_bg 
    
    foreground_indices = torch.zeros(latents_bg.shape[-2:], dtype=torch.long) #64*64
    
    mask_size = np.array([mask_tensor.sum().item() for mask_tensor in mask_tensor_list])
    # Compose the largest mask first
    mask_order = np.argsort(-mask_size)
    
    if compose_box_to_bg:
        # This has two functionalities: 
        # 1. copies the right initial latents from the right place (for centered so generation), 2. copies the right initial latents (since we have foreground blending) for centered/original so generation.
        for mask_idx in mask_order:
            latents_all, mask_tensor = latents_all_list[mask_idx], mask_tensor_list[mask_idx]
            
            mask_array = np.array(mask_tensor, dtype=np.uint8)
            plt.imshow(mask_array, cmap='gray')
            plt.axis('off')  
            
            # Note: need to be careful to not copy from zeros due to shifting. 
            mask_tensor = utils.binary_mask_to_box_mask(mask_tensor, to_device=False)
            
            mask_tensor_expanded = mask_tensor[None, None, None, ...].to(dtype)
            composed_latents[0] = composed_latents[0] * (1. - mask_tensor_expanded) + latents_all[0] * mask_tensor_expanded

    # This is still needed with `compose_box_to_bg` to ensure the foreground latent is still visible and to compute foreground indices.
    for mask_idx in mask_order:
        latents_all, mask_tensor = latents_all_list[mask_idx], mask_tensor_list[mask_idx]
        foreground_indices = foreground_indices * (~mask_tensor) + (mask_idx + 1) * mask_tensor
        mask_tensor_expanded = mask_tensor[None, None, None, ...].to(dtype)
        if use_fast_schedule:
            composed_latents = composed_latents * (1. - mask_tensor_expanded) + latents_all[:fast_after_steps + 1] * mask_tensor_expanded
        else:
            composed_latents = composed_latents * (1. - mask_tensor_expanded) + latents_all * mask_tensor_expanded
        
    composed_latents, foreground_indices = composed_latents.to(torch_device), foreground_indices.to(torch_device)
    return composed_latents, foreground_indices  

def align_with_bboxes(latents_all_list, mask_tensor_list, bboxes, horizontal_shift_only=False):
    """
    Each offset in `offset_list` is `(x_offset, y_offset)` (normalized).
    """
    new_latents_all_list, new_mask_tensor_list, offset_list = [], [], []
    for latents_all, mask_tensor, bbox in zip(latents_all_list, mask_tensor_list, bboxes):
        x_src_center, y_src_center = utils.binary_mask_to_center(mask_tensor, normalize=True) 
        x_min_dest, y_min_dest, x_max_dest, y_max_dest = bbox
        x_dest_center, y_dest_center = (x_min_dest + x_max_dest) / 2, (y_min_dest + y_max_dest) / 2 
        x_offset, y_offset = x_dest_center - x_src_center, y_dest_center - y_src_center
        if horizontal_shift_only:
            y_offset = 0.
        offset = x_offset, y_offset
        latents_all = utils.shift_tensor(latents_all, x_offset, y_offset, offset_normalized=True)
        mask_tensor = utils.shift_tensor(mask_tensor, x_offset, y_offset, offset_normalized=True)

        new_latents_all_list.append(latents_all)
        new_mask_tensor_list.append(mask_tensor)
        offset_list.append(offset)

    return new_latents_all_list, new_mask_tensor_list, offset_list

@torch.no_grad()
def compose_latents_with_alignment(
    basever, adapter, repeat_ind, mask_tensor_list_512, single_obj_img_list, model_dict, latents_all_list, mask_tensor_list, num_inference_steps, overall_batch_size, height, width,
    align_with_overall_bboxes=True, overall_bboxes=None, horizontal_shift_only=False, **kwargs
):
    if align_with_overall_bboxes and len(latents_all_list):
        expanded_overall_bboxes = utils.expand_overall_bboxes(overall_bboxes)

        latents_all_list, mask_tensor_list, offset_list = align_with_bboxes(latents_all_list, mask_tensor_list, bboxes=expanded_overall_bboxes, horizontal_shift_only=horizontal_shift_only)
        inp_mask,inp_img = prepare_mid_image(basever, repeat_ind, mask_tensor_list_512, single_obj_img_list, bboxes=expanded_overall_bboxes)
    
    composed_latents, foreground_indices = compose_latents(adapter, model_dict, latents_all_list, mask_tensor_list, num_inference_steps, overall_batch_size, height, width, **kwargs)

    return composed_latents, foreground_indices, inp_mask, inp_img

def get_input_latents_list(model_dict, bg_seed, fg_seed_start, fg_blending_ratio, height, width, adapter, so_prompt_phrase_box_list=None, so_boxes=None, verbose=False):
    """
    Note: the returned input latents are scaled by `scheduler.init_noise_sigma`
    """
    unet, scheduler, dtype = adapter.pipe.unet, adapter.pipe.scheduler, adapter.pipe.unet.dtype
    
    generator_bg = torch.manual_seed(bg_seed)  # Seed generator to create the inital latent noise
    latents_bg = get_unscaled_latents(batch_size=1, in_channels=unet.config.in_channels, height=height, width=width, generator=generator_bg, dtype=dtype)

    input_latents_list = []
    
    if so_boxes is None:
        # For compatibility
        so_boxes = [item[-1] for item in so_prompt_phrase_box_list]
    
    # change this changes the foreground initial noise
    fg_seed_list = []
    for idx, obj_box in enumerate(so_boxes):
        H, W = height // 8, width // 8
        fg_mask = utils.proportion_to_mask(obj_box, H, W)

        if verbose:
            plt.imshow(fg_mask.cpu().numpy())
            plt.show()
            
        fg_seed = fg_seed_start
        fg_seed_list.append(fg_seed)
        generator_fg = torch.manual_seed(fg_seed)
        latents_fg = get_unscaled_latents(batch_size=1, in_channels=unet.config.in_channels, height=height, width=width, generator=generator_fg, dtype=dtype)
    
        input_latents = blend_latents(latents_bg, latents_fg, fg_mask, fg_blending_ratio=fg_blending_ratio)
    
        input_latents = input_latents * scheduler.init_noise_sigma
    
        input_latents_list.append(input_latents)
    
    latents_bg = latents_bg * scheduler.init_noise_sigma
    
    return input_latents_list, latents_bg, fg_seed_list

# ours
def get_input_latents_lne(idx, adapter, model_dict, bg_seed, fg_seed_start, fg_blending_ratio, height, width, so_prompt_phrase_box_list=None, so_boxes=None, verbose=False):
    """
    Note: the returned input latents are scaled by `scheduler.init_noise_sigma`
    """
    unet, scheduler, dtype = adapter.pipe.unet, adapter.pipe.scheduler, adapter.pipe.unet.dtype

    generator_bg = torch.manual_seed(bg_seed)  # Seed generator to create the inital latent noise
    latents_bg = get_unscaled_latents(batch_size=1, in_channels=unet.config.in_channels, height=height, width=width, generator=generator_bg, dtype=dtype)
    
    if so_boxes is None:
        # For compatibility
        so_boxes = [item[-1] for item in so_prompt_phrase_box_list]
    
    # change this changes the foreground initial noise
    obj_box = so_boxes[idx]
    H, W = height // 8, width // 8
    fg_mask = utils.proportion_to_mask(obj_box, H, W)
        
    fg_seed = fg_seed_start
        
    generator_fg = torch.manual_seed(fg_seed)
    latents_fg = get_unscaled_latents(batch_size=1, in_channels=unet.config.in_channels, height=height, width=width, generator=generator_fg, dtype=dtype)
    
    input_latents = blend_latents(latents_bg, latents_fg, fg_mask, fg_blending_ratio=fg_blending_ratio)
    
    input_latents = input_latents * scheduler.init_noise_sigma
    
    return input_latents

