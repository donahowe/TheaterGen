import torch
from tqdm import tqdm
from utils import guidance, schedule
import utils
from PIL import Image
import PIL
import gc
import numpy as np
from .attention import GatedSelfAttentionDense
from .models import process_input_embeddings, torch_device
import warnings
from typing import List
from diffusers import StableDiffusionPipeline,  StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from ip_adapter import IPAdapter
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import PIL_INTERPOLATION
from diffusers.models import ControlNetModel
import matplotlib.pyplot as plt
import cv2

DEFAULT_GUIDANCE_ATTN_KEYS = [("mid", 0, 0, 0), ("up", 1, 0, 0), ("up", 1, 1, 0), ("up", 1, 2, 0)]

def _preprocess_adapter_image(image, height, width):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        image = [np.array(i.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])) for i in image]
        image = [
            i[None, ..., None] if i.ndim == 2 else i[None, ...] for i in image
        ]  # expand [h, w] or [h, w, c] to [b, h, w, c]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        if image[0].ndim == 3:
            image = torch.stack(image, dim=0)
        elif image[0].ndim == 4:
            image = torch.cat(image, dim=0)
        else:
            raise ValueError(
                f"Invalid image tensor! Expecting image tensor with 3 or 4 dimension, but recive: {image[0].ndim}"
            )
    return image

def get_scaled_latents(batch_size, in_channels, height, width, generator, dtype, scheduler):
    latents_base = get_unscaled_latents(batch_size, in_channels, height, width, generator, dtype)
    latents_base = latents_base * scheduler.init_noise_sigma
    return latents_base 

def get_unscaled_latents(batch_size, in_channels, height, width, generator, dtype):
    latents_base = torch.randn(
        (batch_size, in_channels, height // 8, width // 8),
        generator=generator, dtype=dtype, device=torch_device
    )
    
    return latents_base

def latent_backward_guidance(adapter, scheduler, unet, cond_embeddings, index, bboxes, object_positions, t, latents, loss, loss_scale = 30, loss_threshold = 0.2, max_iter = 5, max_index_step = 10, cross_attention_kwargs=None, ref_ca_saved_attns=None, guidance_attn_keys=None, verbose=False, clear_cache=False, prompt_embeds=None, final=False, **kwargs):

    iteration = 0
    
    if index < max_index_step:
        if isinstance(max_iter, list):
            if len(max_iter) > index:
                max_iter = max_iter[index]
            else:
                max_iter = max_iter[-1]
        
        if verbose:
            print(f"time index {index}, loss: {loss.item()/loss_scale:.3f} (de-scaled with scale {loss_scale:.1f}), loss threshold: {loss_threshold:.3f}")
        
        while (loss.item() / loss_scale > loss_threshold and iteration < max_iter and index < max_index_step):
            saved_attn = {}
            full_cross_attention_kwargs = {
                'save_attn_to_dict': saved_attn,
                'save_keys': guidance_attn_keys,
            }
            
            if cross_attention_kwargs is not None:
                full_cross_attention_kwargs.update(cross_attention_kwargs)
            
            latents.requires_grad_(True)
            latent_model_input = latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            
            unet(latent_model_input, t, encoder_hidden_states=cond_embeddings, return_cross_attention_probs=False, cross_attention_kwargs=full_cross_attention_kwargs)

            # TODO: could return the attention maps for the required blocks only and not necessarily the final output
            # update latents with guidance
            loss = guidance.compute_ca_lossv3(saved_attn=saved_attn, bboxes=bboxes, object_positions=object_positions, guidance_attn_keys=guidance_attn_keys, ref_ca_saved_attns=ref_ca_saved_attns, index=index, verbose=verbose, **kwargs) * loss_scale

            if torch.isnan(loss):
                print("**Loss is NaN**")

            del full_cross_attention_kwargs, saved_attn
            # call gc.collect() here may release some memory

            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents])[0]

            latents.requires_grad_(False)
            
            if hasattr(scheduler, 'sigmas'):
                latents = latents - grad_cond * scheduler.sigmas[index] ** 2
            elif hasattr(scheduler, 'alphas_cumprod'):
                warnings.warn("Using guidance scaled with alphas_cumprod")
                # Scaling with classifier guidance
                alpha_prod_t = scheduler.alphas_cumprod[t]
                # Classifier guidance: https://arxiv.org/pdf/2105.05233.pdf
                # DDIM: https://arxiv.org/pdf/2010.02502.pdf
                scale = (1 - alpha_prod_t) ** (0.5)
                latents = latents - scale * grad_cond
            else:
                # NOTE: no scaling is performed
                warnings.warn("No scaling in guidance is performed")
                latents = latents - grad_cond
            iteration += 1
            
            if clear_cache:
                utils.free_memory()
            
            if verbose:
                print(f"time index {index}, loss: {loss.item()/loss_scale:.3f}, loss threshold: {loss_threshold:.3f}, iteration: {iteration}")
            
    return latents, loss

@torch.no_grad()
def encode(adapter, model_dict, image, generator):
    """
    image should be a PIL object or numpy array with range 0 to 255
    """
    
    vae, dtype = adapter.pipe.vae, adapter.pipe.unet.dtype
    
    if isinstance(image, Image.Image):
        w, h = image.size
        assert w % 8 == 0 and h % 8 == 0, f"h ({h}) and w ({w}) should be a multiple of 8"
        # w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8
        # image = np.array(image.resize((w, h), resample=Image.Resampling.LANCZOS))[None, :]
        image = np.array(image)
    
    if isinstance(image, np.ndarray):
        assert image.dtype == np.uint8, f"Should have dtype uint8 (dtype: {image.dtype})"
        image = image.astype(np.float32) / 255.0
        image = image[None, ...]
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    
    assert isinstance(image, torch.Tensor), f"type of image: {type(image)}"
    
    image = image.to(device=torch_device, dtype=dtype)
    latents = vae.encode(image).latent_dist.sample(generator)
    
    latents = vae.config.scaling_factor * latents

    return latents

@torch.no_grad()
def decode(vae, latents):
    # scale and decode the image latents with vae
    scaled_latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(scaled_latents).sample
        
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    
    return images

def generate_semantic_guidance(task, fg_seed_now, basever, ip_prompt, database_path, id, adapter,model_dict, latents, input_embeddings, num_inference_steps, bboxes, phrases, object_positions, guidance_scale = 7.5, semantic_guidance_kwargs=None, 
                           return_cross_attn=False, return_saved_cross_attn=False, saved_cross_attn_keys=None, return_cond_ca_only=False, return_token_ca_only=None, offload_guidance_cross_attn_to_cpu=False,
                           offload_cross_attn_to_cpu=False, offload_latents_to_cpu=True, return_box_vis=False, show_progress=True, save_all_latents=False, 
                           dynamic_num_inference_steps=False, fast_after_steps=None, fast_rate=2, use_boxdiff=False, use_adapter=False, obj_id=False, have_reffer = 0):

    '''
    Prepare Adapter parameter
    '''
    if obj_id: #not final
        try:
            img_path = database_path+str(obj_id)+".png"
            image = Image.open(img_path)
            # tag
            if task == 'editing':
                scale = 0.4
            else:
                scale = 0.4
            print(f"referrencing from {obj_id}")
            have_reffer = 1
        except:
            img_path = "model.png"
            image = Image.open(img_path)
            scale = 0
            print(f"New object, allocate id: {obj_id}") #without referrence image
            have_reffer = 0
        image.resize((512, 512))
        pil_image=image
        num_samples=1
         #parame
        adapter.set_scale(scale)
        clip_image_embeds=None
        image_prompt_embeds, uncond_image_prompt_embeds = adapter.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds
        ) 
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        
        print(f"Using IP Adapter {basever}")
        if task == 'editing':
            prompt = "single object, " + str(ip_prompt)
        else:
            prompt = "full-body picture of " + str(ip_prompt)
        print("\n",prompt,"\n")
        negative_prompt= "background, multiple objects, incomplete, lowres, bad anatomy, low quality, obscured"

        with torch.inference_mode():
            if basever == 'xl':
                (
                    prompt_embeds_,
                    negative_prompt_embeds_,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = adapter.pipe.encode_prompt(
                    prompt,
                    num_images_per_prompt=num_samples,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                )
            
            else:
                prompt_embeds_, negative_prompt_embeds_ = adapter.pipe.encode_prompt(
                    prompt,
                    device=adapter.device,
                    num_images_per_prompt=num_samples,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                )

            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1) #[4,81,768]
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1) #[4,81,768]


    '''
    Prepare Adapter
    '''
    # 0. Parametr

    height = adapter.pipe.unet.config.sample_size * adapter.pipe.vae_scale_factor
    width = adapter.pipe.unet.config.sample_size * adapter.pipe.vae_scale_factor    
    num_images_per_prompt = 1
    # 1. Check inputs. Raise error if not correct     
    prompt = None
    negative_prompt = None
    if basever == 'xl':
        adapter.pipe.check_inputs(
            prompt=prompt,
            prompt_2=None,
            height=height,
            width=width,
            callback_steps=1,
            negative_prompt=negative_prompt,
            negative_prompt_2=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        )
    else:
        adapter.pipe.check_inputs(
            prompt=prompt, height=height, width=width, callback_steps=1, negative_prompt=negative_prompt, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds
        )
    # 2. Define call parameters
    device = adapter.pipe._execution_device
    do_classifier_free_guidance = True
    # 3. Encode input prompt
    text_encoder_lora_scale = None

    if basever == 'xl':
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = adapter.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=None, 
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        adapter.pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = adapter.pipe.scheduler.timesteps
        generator = torch.Generator(device).manual_seed(fg_seed_now)
        num_channels_latents = adapter.pipe.unet.config.in_channels
        latents=None
        latents = adapter.pipe.prepare_latents( #[1,4128,128]
            1 * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        ) #[1,4,128,128]

    else:
        prompt_embeds, negative_prompt_embeds = adapter.pipe.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

    

    if basever == 'xl':

        extra_step_kwargs = adapter.pipe.prepare_extra_step_kwargs(generator, 0.0)
        original_size = (height, width)
        target_size =  (height, width)
        crops_coords_top_left = (0,0)
        negative_original_size = None
        negative_target_size = None
        negative_crops_coords_top_left = (0,0)

        add_text_embeds = pooled_prompt_embeds
        add_time_ids = adapter.pipe._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = adapter.pipe._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
            )
        else:
            negative_add_time_ids = add_time_ids

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(1 * num_images_per_prompt, 1)

    else:
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    scheduler = adapter.pipe.scheduler
    latents = latents.clone()
    
    if save_all_latents:
        # offload to cpu to save space
        if offload_latents_to_cpu:
            latents_all = [latents.cpu()]
        else:
            latents_all = [latents]
    
    scheduler.set_timesteps(num_inference_steps)
    if fast_after_steps is not None:
        scheduler.timesteps = schedule.get_fast_schedule(scheduler.timesteps, fast_after_steps, fast_rate)
    
    if dynamic_num_inference_steps:
        original_num_inference_steps = scheduler.num_inference_steps

    cross_attention_probs_down = []
    cross_attention_probs_mid = []
    cross_attention_probs_up = []
    loss = torch.tensor(10000.)
    guidance_cross_attention_kwargs = {
        'offload_cross_attn_to_cpu': offload_guidance_cross_attn_to_cpu,
        'enable_flash_attn': False
    }
    if return_saved_cross_attn:
        saved_attns = []
    main_cross_attention_kwargs = {
        'offload_cross_attn_to_cpu': offload_cross_attn_to_cpu,
        'return_cond_ca_only': return_cond_ca_only,
        'return_token_ca_only': return_token_ca_only,
        'save_keys': saved_cross_attn_keys,
    }

    for index, t in enumerate(tqdm(scheduler.timesteps, disable=not show_progress)):     

        with torch.no_grad():
            latent_model_input = torch.cat([latents] * 2) 
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
            
            main_cross_attention_kwargs['save_attn_to_dict'] = {}
            prompt_embeds=prompt_embeds.half()
            latent_model_input=latent_model_input.half()
            if basever == 'xl':
                add_text_embeds = add_text_embeds.half()
                add_time_ids = add_time_ids.half()
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                noise_pred = adapter.pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=None,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
            else:
                unet_output = adapter.pipe.unet(latent_model_input,t,encoder_hidden_states=prompt_embeds,cross_attention_kwargs=None,) 
                noise_pred = unet_output.sample 

            if return_cross_attn: 
                cross_attention_probs_down.append(unet_output.cross_attention_probs_down)
                cross_attention_probs_mid.append(unet_output.cross_attention_probs_mid)
                cross_attention_probs_up.append(unet_output.cross_attention_probs_up)
                
            if return_saved_cross_attn:
                saved_attns.append(main_cross_attention_kwargs['save_attn_to_dict'])              
                del main_cross_attention_kwargs['save_attn_to_dict']

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        if basever == 'xl':
            latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
        else:
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        if save_all_latents:
            if offload_latents_to_cpu:
                latents_all.append(latents.cpu())
            else:
                latents_all.append(latents)

    if dynamic_num_inference_steps:
        # Restore num_inference_steps to avoid confusion in the next generation if it is not dynamic
        scheduler.num_inference_steps = original_num_inference_steps

    latents=latents.half()
    if basever == 'xl':
        adapter.pipe.upcast_vae()
        latents = latents.to(next(iter(adapter.pipe.vae.post_quant_conv.parameters())).dtype)


    if basever == 'xl':
        latents = latents.to('cuda:1')
        adapter.pipe.vae.to('cuda:1')
    image = adapter.pipe.vae.decode(latents / adapter.pipe.vae.config.scaling_factor, return_dict=False)[0]
    adapter.pipe.vae.to(dtype=torch.float16)
    adapter.pipe.vae.to('cuda:0')
    image = image.detach()
    if basever == 'xl':
        images = adapter.pipe.image_processor.postprocess(image, output_type='pil')[0]
    else:
        images = adapter.pipe.image_processor.postprocess(image, output_type='pil', do_denormalize=[True])[0] #[1,3,512,512]
    if have_reffer == 0:
        images.save(database_path+str(obj_id)+".png") # add to database

    ret = [latents, images]
    if return_cross_attn:
        ret.append((cross_attention_probs_down, cross_attention_probs_mid, cross_attention_probs_up))
    if return_saved_cross_attn:
        ret.append(saved_attns)
    if return_box_vis:
        pil_images = images
        ret.append(pil_images)
    if save_all_latents:
        latents_all = torch.stack(latents_all, dim=0)
        ret.append(latents_all)
    return tuple(ret)

@torch.no_grad()
def generate(adapter, model_dict, latents, input_embeddings, num_inference_steps, guidance_scale = 7.5, no_set_timesteps=False, scheduler_key='scheduler'):
    vae, tokenizer, text_encoder, unet, scheduler, dtype = adapter.pipe.vae, adapter.pipe.tokenizer, adapter.pipe.text_encoder, adapter.pipe.unet, adapter.pipe.scheduler, adapter.pipe.unet.dtype
    text_embeddings, uncond_embeddings, cond_embeddings = input_embeddings
    
    if not no_set_timesteps:
        scheduler.set_timesteps(num_inference_steps)

    for t in tqdm(scheduler.timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)

        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    images = decode(vae, latents)
    
    ret = [latents, images]

    return tuple(ret)


def get_inverse_timesteps(inverse_scheduler, num_inference_steps, strength):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)

    # safety for t_start overflow to prevent empty timsteps slice
    if t_start == 0:
        return inverse_scheduler.timesteps, num_inference_steps
    timesteps = inverse_scheduler.timesteps[:-t_start]

    return timesteps, num_inference_steps - t_start


@torch.no_grad()
def invert(model_dict, latents, input_embeddings, num_inference_steps, guidance_scale = 7.5):
    """
    latents: encoded from the image, should not have noise (t = 0)
    
    returns inverted_latents for all time steps
    """
    vae, tokenizer, text_encoder, unet, scheduler, inverse_scheduler, dtype = model_dict.vae, model_dict.tokenizer, model_dict.text_encoder, model_dict.unet, model_dict.scheduler, model_dict.inverse_scheduler, model_dict.dtype
    text_embeddings, uncond_embeddings, cond_embeddings = input_embeddings
    
    inverse_scheduler.set_timesteps(num_inference_steps, device=latents.device)
    # We need to invert all steps because we need them to generate the background.
    timesteps, num_inference_steps = get_inverse_timesteps(inverse_scheduler, num_inference_steps, strength=1.0)

    inverted_latents = [latents.cpu()]
    for t in tqdm(timesteps[:-1]):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        if guidance_scale > 0.:
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = inverse_scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        else:
            latent_model_input = latents

            latent_model_input = inverse_scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred_uncond = unet(latent_model_input, t, encoder_hidden_states=uncond_embeddings).sample

            # perform guidance
            noise_pred = noise_pred_uncond

        # compute the previous noisy sample x_t -> x_t-1
        latents = inverse_scheduler.step(noise_pred, t, latents).prev_sample
        
        inverted_latents.append(latents.cpu())
    
    assert len(inverted_latents) == len(timesteps)
    # timestep is the first dimension
    inverted_latents = torch.stack(list(reversed(inverted_latents)), dim=0)
    
    return inverted_latents



def final_image_generation(basever, processor, controlnetpipe, tpipe, overall_prompt, overall_negative_prompt,  bg_prompt, single_obj_img_list, objects, repeat_ind, height, width, bg_seed, inp_mask, input_img, adapter, model_dict, latents_all, frozen_mask, bg_input_embeddings, input_embeddings, num_inference_steps, frozen_steps, guidance_scale = 7.5, bboxes=None, phrases=None, object_positions=None, semantic_guidance_kwargs=None, offload_guidance_cross_attn_to_cpu=False, use_boxdiff=False):
    vae, unet, scheduler, dtype = adapter.pipe.vae, adapter.pipe.unet, adapter.pipe.scheduler, adapter.pipe.unet.dtype
    generator = torch.Generator("cuda").manual_seed(bg_seed)
    text_embeddings, uncond_embeddings, cond_embeddings = input_embeddings

    #origin part
    scheduler.set_timesteps(num_inference_steps)
    adapter.pipe.scheduler.set_timesteps(num_inference_steps, device=torch_device)

    frozen_mask = frozen_mask.to(dtype=dtype).clamp(0., 1.) #[64,64]
    latents = latents_all[0] #[1,4,64,64]

    #prepare mask
    my_mask = inp_mask
    resized_my_mask = my_mask.resize((int(height/8), int(width/8)))
    resized_my_mask = resized_my_mask.convert('L')

    my_mask = np.array(resized_my_mask).astype(np.float32) / 255.0
    my_mask[my_mask > 0] = 1

    my_mask = 1 - my_mask  
    my_mask = torch.from_numpy(my_mask) 
    my_mask = my_mask.to(device=torch_device, dtype=dtype)

    #prepare latents
    inp_img = np.array(input_img).astype(np.float32) / 255.0
    inp_img = np.vstack([inp_img[None].transpose(0, 3, 1, 2)] * 1)
    inp_img = torch.from_numpy(inp_img)
    inp_img = 2.0 * inp_img - 1.0
    myimage = inp_img.to(device=torch_device, dtype=dtype)
    if basever == 'xl':
        vae = controlnetpipe.vae.to(torch_device)
    init_latent_dist = vae.encode(myimage).latent_dist
    init_latents = init_latent_dist.sample(generator=generator)
    init_latents = vae.config.scaling_factor * init_latents
    init_latents = torch.cat([init_latents] * 1, dim=0)

    noise = randn_tensor(init_latents.shape, generator=generator, device=torch.device(torch_device), dtype=dtype)
    init_latents = scheduler.add_noise(init_latents, noise, scheduler.timesteps)
    my_latents = init_latents.unsqueeze(1)
    my_bg = get_scaled_latents(1, unet.config.in_channels, height, width, generator, dtype, scheduler)

    #t2i:
    if basever == 'xl':
        t2i_img = np.array(input_img)
        control_image = processor(t2i_img, detect_resolution=384, image_resolution=1024)

        adapter_input = _preprocess_adapter_image(control_image, height, width).to('cuda:2')

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = controlnetpipe.encode_prompt(
            prompt=overall_prompt,
            prompt_2=None,
            device='cuda:2',
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt= overall_negative_prompt, #overall_negative_prompt
            negative_prompt_2=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
        )
        controlnetpipe.scheduler.set_timesteps(num_inference_steps, device='cuda:2')

        num_channels_latents = controlnetpipe.unet.config.in_channels

        generator = torch.Generator("cuda").manual_seed(bg_seed)
        controlnet_init_latents = controlnetpipe.prepare_latents(
            1 * 1,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device=torch.device('cuda:2'),
            generator=generator,
            latents=None,
        )
        
        extra_step_kwargs = controlnetpipe.prepare_extra_step_kwargs(generator, 0.0)

        adapter_input = adapter_input.type(latents.dtype)
        adapter_state = controlnetpipe.adapter(adapter_input)
        for k, v in enumerate(adapter_state):
            adapter_state[k] = v * 0.8 #adapter_conditioning_scale
        for k, v in enumerate(adapter_state):
            adapter_state[k] = torch.cat([v] * 2, dim=0) 

        original_size = (height, width)
        target_size =  (height, width)
        crops_coords_top_left = (0,0)

        add_text_embeds = pooled_prompt_embeds
        add_time_ids = controlnetpipe._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
        )
        negative_add_time_ids = add_time_ids
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)     
        add_time_ids = add_time_ids.repeat(1 * 1, 1)   

    #ip + controlnet
    else: 
        ip_embeds, add_text_embeds, add_time_ids = prepare_ip_embeds(height, width, generator, torch_device, basever, adapter, overall_prompt, overall_negative_prompt, single_obj_img_list[0], objects)
        adapter.set_scale(0.1)
        controlnet = controlnetpipe.controlnet
        guess_mode = False
        global_pool_conditions = (
                controlnet.config.global_pool_conditions
                if isinstance(controlnet, ControlNetModel)
                else controlnet.nets[0].config.global_pool_conditions
            )
        guess_mode = guess_mode or global_pool_conditions
        t2i_img = np.array(input_img)
        control_image = processor(t2i_img)
        imagetight = controlnetpipe.prepare_image(
            image=control_image,
            width=width,
            height=height,
            batch_size=1,
            num_images_per_prompt=1,
            device="cuda",
            dtype=controlnet.dtype,
            do_classifier_free_guidance=True,
            guess_mode=guess_mode,
        )
        height, width = imagetight.shape[-2:]
        timesteps = scheduler.timesteps
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip([0.0],[1.0])
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

    latents = my_bg
    if basever == 'xl':
        latents = controlnet_init_latents
    frozen_mask = my_mask
    latents_all[0] = latents
    latents_all[1:] = my_latents

    fin_img_latent = []

    for index, t in enumerate(tqdm(scheduler.timesteps)):

        with torch.no_grad():
            if basever == 'xl':
                latents = latents.to('cuda:2')
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = controlnetpipe.scheduler.scale_model_input(latent_model_input, timestep=t)

            if basever == 'xl':
                latent_model_input = latent_model_input.to('cuda:2')
                prompt_embeds = prompt_embeds.half().to('cuda:2')
                if index < int(num_inference_steps * 1.0):
                    down_block_additional_residuals = [state.clone().to('cuda:2') for state in adapter_state]
                else:
                    down_block_additional_residuals = None   

            else:
                #controlnet
                control_model_input = latent_model_input.half().to(torch_device)
                controlnet_prompt_embeds = text_embeddings.half().to(torch_device)
                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(1, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = 1
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]

                down_block_res_samples, mid_block_res_sample = controlnetpipe.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=imagetight,
                    conditioning_scale=cond_scale,
                    guess_mode=guess_mode,
                    return_dict=False,
                )       
                ip_embeds=ip_embeds.half()

            latent_model_input=latent_model_input.half()
            text_embeddings=text_embeddings.half()
            
            if basever == 'xl':
                added_cond_kwargs = {"text_embeds": add_text_embeds.to('cuda:2'), "time_ids": add_time_ids.to('cuda:2')}
                add_text_embeds = add_text_embeds.half()
                add_time_ids = add_time_ids.half()

                noise_pred = controlnetpipe.unet(
                    latent_model_input.to('cuda:2'),
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=None,
                    added_cond_kwargs=added_cond_kwargs,
                    down_block_additional_residuals=down_block_additional_residuals,
                    return_dict=False,
                )[0]     

            else:
                latent_model_input= latent_model_input.to(torch_device)
                if t<=5:
                    noise_pred = adapter.pipe.unet(
                        latent_model_input,
                        t, 
                        encoder_hidden_states=ip_embeds,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    )[0] 
                else:
                    noise_pred = adapter.pipe.unet(
                        latent_model_input,
                        t, 
                        encoder_hidden_states=ip_embeds,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    )[0] 


            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond) #7.5

            # compute the previous noisy sample x_t -> x_t-1

            if basever == 'xl':
                noise_pred = noise_pred.to('cuda:2')
                latents = latents.to('cuda:2')
                latents = controlnetpipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            else:
                latents = scheduler.step(noise_pred, t, latents).prev_sample
            latents = latents.to(torch_device)
            
            if index < frozen_steps:
                latents = latents_all[index+1] * frozen_mask + latents * (1. - frozen_mask)
            fin_img_latent.append(latents.cpu())

    if basever == 'xl':

        latents = latents.to('cuda:1')
        controlnetpipe.vae.to('cuda:1')
        image = controlnetpipe.vae.decode(latents / controlnetpipe.vae.config.scaling_factor, return_dict=False)[0]
        controlnetpipe.vae.to(dtype=torch.float16)
        controlnetpipe.vae.to('cuda:0')

        image = image.detach()
        images = controlnetpipe.image_processor.postprocess(image, output_type='pil')
    
    else:
        scaled_latents = (1 / 0.18215 * latents)
        with torch.no_grad():
            image = vae.decode(scaled_latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")

    ret = [latents, images]
    return tuple(ret)


def prepare_ip_embeds(height, width, generator, device, basever, adapter, bg_prompt, overall_negative_prompt, single_obj_img_list, objects):
    pil_image = single_obj_img_list
    prompt = bg_prompt
    negative_prompt = overall_negative_prompt
    image_prompt_embeds, uncond_image_prompt_embeds = adapter.get_image_embeds(pil_image=pil_image, clip_image_embeds=None)
    bs_embed, seq_len, _ = image_prompt_embeds.shape
    image_prompt_embeds = image_prompt_embeds.repeat(1, 1, 1)
    image_prompt_embeds = image_prompt_embeds.view(bs_embed * 1, seq_len, -1)
    uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, 1, 1)
    uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * 1, seq_len, -1)    
    with torch.inference_mode():
        if basever == 'xl':
            (
                prompt_embeds_, 
                negative_prompt_embeds_, 
                pooled_prompt_embeds, # 1, 1280
                negative_pooled_prompt_embeds, #1,1280
            ) = adapter.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )  
        else:      
            prompt_embeds_, negative_prompt_embeds_ = adapter.pipe.encode_prompt(
                prompt,
                device=adapter.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
        prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
        negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1) 
    
    if basever == 'xl':
        (
            prompt_embeds, #1,81,2048
            negative_prompt_embeds, #1,81,2048
            pooled_prompt_embeds, #1,1280
            negative_pooled_prompt_embeds, #1,1280
        ) = adapter.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
            negative_prompt_2=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=None,
        )
        
        original_size = (height, width)
        target_size =  (height, width)
        crops_coords_top_left = (0,0)
        negative_original_size = None
        negative_target_size = None
        negative_crops_coords_top_left = (0,0)

        add_text_embeds = pooled_prompt_embeds
        add_time_ids = adapter.pipe._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
        )   
        negative_add_time_ids = add_time_ids
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(1 * 1, 1)

    else:
        add_text_embeds = None
        add_time_ids = None
        prompt_embeds, negative_prompt_embeds = adapter.pipe.encode_prompt(
            negative_prompt=None,
            prompt=None,
            device = adapter.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=None,
        )

    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    return prompt_embeds, add_text_embeds, add_time_ids