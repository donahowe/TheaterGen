import os
import torch
import models
import utils
from utils.detector import detect
from models import pipelines, sam, model_dict
from utils import parse, guidance, latents, vis
from utils.latents import get_input_latents_lne
from prompt import (
    DEFAULT_SO_NEGATIVE_PROMPT,
    DEFAULT_OVERALL_NEGATIVE_PROMPT,
)
from easydict import EasyDict
import pickle
from PIL import Image
import numpy as np

# Hyperparams
height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion
H, W = height // 8, width // 8  # size of the latent
guidance_scale = 7.5  # Scale for classifier-free guidance
# batch size: set to 1
overall_batch_size = 1
# attn keys for semantic guidance
overall_guidance_attn_keys = pipelines.DEFAULT_GUIDANCE_ATTN_KEYS
# Start attention aggregation from t steps (take the mean over 50-t steps), used for latent masking
attn_aggregation_step_start = 10
# sigma for gaussian filtering the attn, different if we select point input or box input
gaussian_sigma_point_input = 1.5
gaussian_sigma_box_input = 0.1
# discourage masks with confidence below
discourage_mask_below_confidence = 0.85
# discourage masks with iou (with coarse binarized attention mask) below
discourage_mask_below_coarse_iou = 0.25
mask_th_for_box = 0.05
n_erode_dilate_mask_for_box = 1
offload_guidance_cross_attn_to_cpu = False


def generate_single_object_with_box(
    vis_location,
    task,
    fg_seed_now,
    basever,
    grounding_dino_model,
    ip_prompt,
    database_path,
    obj_id,
    repeat_ind,
    adapter,
    idx,
    prompt,
    box,
    phrase,
    word,
    input_latents,
    input_embeddings,
    semantic_guidance_kwargs,
    obj_attn_key,
    saved_cross_attn_keys,
    sam_refine_kwargs,
    num_inference_steps,
    verbose=False,
    visualize=False,
    use_adapter=False,
    **kwargs,
):
    
    #debug
    print("Generating on stage image of ,",word)
    bboxes, phrases, words = [box], [phrase], [word]

    if verbose:
        print(f"Getting token map (prompt: {prompt})")

    object_positions, word_token_indices = guidance.get_phrase_indices(
        tokenizer=adapter.pipe.tokenizer,
        prompt=prompt,
        phrases=phrases,
        words=words,
        return_word_token_indices=True,
        # Since the prompt for single object is from background prompt + object name, we will not have the case of not found
        add_suffix_if_not_found=False,
        verbose=verbose,
    )
    # phrases only has one item, so we select the first item in word_token_indices
    word_token_index = word_token_indices[0]

    if verbose:
        print("object positions:", object_positions)
        print("word_token_index:", word_token_index)

    is_exist = os.path.exists(database_path+str(obj_id)+".png")

    regen = 0
    while True:
        regen += 1
        # `offload_guidance_cross_attn_to_cpu` will greatly slow down generation
        (
            latents,
            single_object_images,
            saved_attns,
            single_object_pil_images_box_ann,
            latents_all,
        ) = pipelines.generate_semantic_guidance(
            task,
            fg_seed_now,
            basever,
            ip_prompt,
            database_path,
            idx,
            adapter,
            model_dict,
            input_latents,
            input_embeddings,
            num_inference_steps,
            bboxes,
            phrases,
            object_positions,
            guidance_scale=guidance_scale,
            return_cross_attn=False,
            return_saved_cross_attn=True,
            semantic_guidance_kwargs=semantic_guidance_kwargs,
            saved_cross_attn_keys=[obj_attn_key, *saved_cross_attn_keys],
            return_cond_ca_only=True,
            return_token_ca_only=word_token_index,
            offload_guidance_cross_attn_to_cpu=offload_guidance_cross_attn_to_cpu,
            offload_cross_attn_to_cpu=False,
            return_box_vis=True,
            save_all_latents=True,
            dynamic_num_inference_steps=True,
            use_adapter=use_adapter,
            obj_id=obj_id,
            **kwargs,
        )

        
        Detection, ok = detect(grounding_dino_model, words, single_object_images)
        input_boxes = [tuple(Detection)]
 
        if ok or regen >=3 :
            '''
            input_boxes = [tuple(Detection)]
            print(input_boxes)
            h, w, tmp = sam_refine_kwargs['height'], sam_refine_kwargs['width'], input_boxes[0]
            if tmp[0] >= 0.3 and tmp[1] >= 0.3 and w - tmp[2] >= 0.3 and h - tmp[3] >= 0.3:
                print("single obj is ok\n")
                break
            ''' 
            break
        print("regenerate now\n")
        arg_dict_l["fg_seed_start"] += 10
        arg_dict_l["bg_seed"] += 10

        if not is_exist:
            os.remove(database_path+str(obj_id)+".png")
        input_latents = get_input_latents_lne(idx, adapter, model_dict=model_dict, **arg_dict_l)

    token_attn_np = 1
    utils.free_memory()

   
    single_object_pil_image_box_ann = single_object_pil_images_box_ann
    visualize = False
    if visualize:
        print("Single object image")
        vis.display(single_object_pil_image_box_ann)

    single_object_images.save(f"visualization/{vis_location[0]}_{vis_location[1]}_{repeat_ind}single_object_images{obj_id}.jpg")
    
    mask_selected, conf_score_selected,mask_selected_512, conf_score_selected_512  = sam.sam_refine_attn(
        input_boxes_w = input_boxes,
        obj_id = obj_id,
        sam_input_image=single_object_images, 
        token_attn_np=token_attn_np,
        model_dict=model_dict,
        verbose=verbose,
        **sam_refine_kwargs,
    )

    mask_selected_tensor = torch.tensor(mask_selected)
    mask_selected_tensor_512 = torch.tensor(mask_selected_512)


    visualize=False
    if visualize:
         vis.visualize(repeat_ind, idx, mask_selected, "Mask(selected)after_resize")
         masked_latents = latents_all * mask_selected_tensor[None, None, None, ...]
         vis.visualize_masked_latents(repeat_ind, idx, latents_all, masked_latents, timestep_T=True, timestep_0=True ,visual_all = True)

    return (
        latents_all,
        mask_selected_tensor,
        mask_selected_tensor_512,
        saved_attns,
        single_object_pil_image_box_ann,
        single_object_images
    )


def get_masked_latents_all_list(vis_location, task, fg_seed_list, basever, grounding_dino_model, overall_phrases, database_path, obj_ids, repeat_ind, adapter,so_prompt_phrase_word_box_list, input_latents_list, so_input_embeddings, verbose=False, **kwargs,):
    latents_all_list, mask_tensor_list, mask_tensor_list_512, saved_attns_list, so_img_list, single_obj_img_list = [], [], [], [], [], []

    if not so_prompt_phrase_word_box_list:
        return latents_all_list, mask_tensor_list, saved_attns_list, so_img_list

    so_uncond_embeddings, so_cond_embeddings = so_input_embeddings

    now_prompt = None
    now_obj_id = None
    for idx, ((prompt, phrase, word, box), input_latents, obj_id, fg_seed_now) in enumerate(
        zip(so_prompt_phrase_word_box_list, input_latents_list, obj_ids, fg_seed_list)
    ):
        if now_prompt == prompt and now_obj_id == obj_id: 
                now_prompt = prompt
                now_obj_id = obj_id
                mask_tensor_list_512.append(mask_tensor_list_512[-1])
                latents_all_list.append(latents_all_list[-1])
                mask_tensor_list.append(mask_tensor_list[-1])
                saved_attns_list.append(saved_attns_list[-1]) 
                so_img_list.append(so_img_list[-1])
                single_obj_img_list.append(single_obj_img_list[-1])
                continue

        now_prompt = prompt
        now_obj_id = obj_id
        so_current_cond_embeddings = so_cond_embeddings[idx : idx + 1] 
        so_current_text_embeddings = torch.cat(
            [so_uncond_embeddings, so_current_cond_embeddings], dim=0
        ) 
        so_current_input_embeddings = (
            so_current_text_embeddings,
            so_uncond_embeddings,
            so_current_cond_embeddings,
        )


        ip_prompt = overall_phrases[idx]

        latents_all, mask_tensor, mask_selected_tensor_512, saved_attns, so_img ,single_object_images = generate_single_object_with_box(
            vis_location,
            task,
            fg_seed_now,
            basever,
            grounding_dino_model,
            ip_prompt,
            database_path,
            obj_id,
            repeat_ind,
            adapter,
            idx,
            prompt, 
            box, 
            phrase,
            word, 
            input_latents, 
            input_embeddings=so_current_input_embeddings, 
            verbose=verbose, 
            use_adapter = True,
            **kwargs,
        )

        mask_tensor_list_512.append(mask_selected_tensor_512)
        latents_all_list.append(latents_all)
        mask_tensor_list.append(mask_tensor)
        saved_attns_list.append(saved_attns) 
        so_img_list.append(so_img)
        single_obj_img_list.append(single_object_images)

    return latents_all_list, mask_tensor_list, mask_tensor_list_512,  saved_attns_list, so_img_list, single_obj_img_list




def run(
    vis_location,
    task,
    basever,
    grounding_dino_model,
    processor,
    controlnetpipe,
    database_path,
    repeat_ind,
    adapter,
    spec,
    bg_seed=1,
    overall_prompt_override="",
    fg_seed_start=20,
    frozen_step_ratio=0.5,
    num_inference_steps=30,
    loss_scale=5,
    loss_threshold=5.0,
    max_iter=[4] * 5 + [3] * 5 + [2] * 5 + [2] * 5 + [1] * 10,
    max_index_step=30,
    overall_loss_scale=5,
    overall_loss_threshold=5.0,
    overall_max_iter=[4] * 5 + [3] * 5 + [2] * 5 + [2] * 5 + [1] * 10,
    overall_max_index_step=30,
    fg_top_p=0.2,
    bg_top_p=0.2,
    overall_fg_top_p=0.2,
    overall_bg_top_p=0.2,
    fg_weight=1.0,
    bg_weight=4.0,
    overall_fg_weight=1.0,
    overall_bg_weight=4.0,
    ref_ca_loss_weight=2.0,
    so_center_box=True,
    fg_blending_ratio=0.01,
    so_negative_prompt=DEFAULT_SO_NEGATIVE_PROMPT,
    overall_negative_prompt=DEFAULT_OVERALL_NEGATIVE_PROMPT,
    mask_th_for_point=0.25,
    so_horizontal_center_only=False,
    align_with_overall_bboxes=True,
    horizontal_shift_only=False,
    use_fast_schedule=False,
    so_vertical_placement="floor_padding",
    so_floor_padding=0.2,
    use_box_input=False,
    use_ref_ca=True,
    use_autocast=False,
    verbose=False,
):
    if basever=='xl':
        height = 1024 
        width = 1024
        num_inference_steps = 30
    else:
        height = 512 
        width = 512  
        num_inference_steps = 50       
        
    frozen_step_ratio = min(max(frozen_step_ratio, 0.0), 1.0)
    frozen_steps = int(num_inference_steps * frozen_step_ratio)

    print("Key generation settings:",spec,bg_seed,fg_seed_start,frozen_step_ratio,max_index_step,overall_max_index_step,)

    (objects, bg_prompt, so_prompt_phrase_word_box_list,overall_prompt,overall_phrases_words_bboxes,obj_ids) = parse.convert_spec(spec, height=512, width=512, verbose=verbose)

    if overall_prompt_override and overall_prompt_override.strip():
        overall_prompt = overall_prompt_override.strip()

    overall_phrases, overall_words, overall_bboxes = ([item[1] for item in so_prompt_phrase_word_box_list], [item[2] for item in so_prompt_phrase_word_box_list],[[item[3]] for item in so_prompt_phrase_word_box_list])

    if so_center_box:
        centered_box_kwargs = dict(horizontal_center_only=so_horizontal_center_only,vertical_placement=so_vertical_placement,floor_padding=so_floor_padding,)
        so_prompt_phrase_word_box_list = [(prompt, phrase, word, utils.get_centered_box(bbox, **centered_box_kwargs)) for prompt, phrase, word, bbox in so_prompt_phrase_word_box_list]
        if verbose:
            print(
                f"centered so_prompt_phrase_word_box_list: {so_prompt_phrase_word_box_list}"
            )
    so_boxes = [item[-1] for item in so_prompt_phrase_word_box_list]
    
    if "extra_neg_prompt" in spec and spec["extra_neg_prompt"]:
        so_negative_prompt = spec["extra_neg_prompt"] + ", " + so_negative_prompt
        overall_negative_prompt = (
            spec["extra_neg_prompt"] + ", " + overall_negative_prompt
        )

    overall_negative_prompt = "incohesive, edge shadow, blurry, " + overall_negative_prompt

    gaussian_sigma = (
        gaussian_sigma_box_input if use_box_input else gaussian_sigma_point_input
    )

    semantic_guidance_kwargs = dict(loss_scale=loss_scale,loss_threshold=loss_threshold,max_iter=max_iter,max_index_step=max_index_step,fg_top_p=fg_top_p,bg_top_p=bg_top_p,fg_weight=fg_weight,bg_weight=bg_weight,use_ratio_based_loss=False,guidance_attn_keys=overall_guidance_attn_keys,verbose=True,)

    sam_refine_kwargs = dict(use_box_input=use_box_input,gaussian_sigma=gaussian_sigma, mask_th_for_box=mask_th_for_box,n_erode_dilate_mask_for_box=n_erode_dilate_mask_for_box,mask_th_for_point=mask_th_for_point,discourage_mask_below_confidence=discourage_mask_below_confidence,discourage_mask_below_coarse_iou=discourage_mask_below_coarse_iou,
        height=height,width=width,H=H,W=W,)


    with torch.autocast("cuda", enabled=use_autocast):
        so_prompts = [item[0] for item in so_prompt_phrase_word_box_list]
        if so_prompts:
            so_input_embeddings = models.encode_prompts(prompts=so_prompts,tokenizer=adapter.pipe.tokenizer,text_encoder=adapter.pipe.text_encoder,negative_prompt=so_negative_prompt,one_uncond_input_only=True,)
        else:
            so_input_embeddings = []

        input_latents_list, latents_bg, fg_seed_list = latents.get_input_latents_list(model_dict, adapter=adapter ,bg_seed=bg_seed,fg_seed_start=fg_seed_start,so_boxes=so_boxes,fg_blending_ratio=fg_blending_ratio,height=height,width=width,verbose=False)

        if use_fast_schedule:
            fast_after_steps = (
                max(frozen_steps, overall_max_index_step)
                if use_ref_ca
                else frozen_steps
            )
        else:
            fast_after_steps = None

        global arg_dict_l
        arg_dict_l = {"bg_seed":bg_seed, "fg_seed_start":fg_seed_start, "so_boxes":so_boxes, "fg_blending_ratio":fg_blending_ratio, "height":height, "width":width, "verbose":False}

        if use_ref_ca or frozen_steps > 0:

            (
                latents_all_list,
                mask_tensor_list,
                mask_tensor_list_512,
                saved_attns_list,
                so_img_list,
                single_obj_img_list
            ) = get_masked_latents_all_list(vis_location, task, fg_seed_list, basever, grounding_dino_model, overall_phrases, database_path, obj_ids, repeat_ind, adapter, so_prompt_phrase_word_box_list, input_latents_list,semantic_guidance_kwargs=semantic_guidance_kwargs,obj_attn_key=("down", 2, 1, 0),saved_cross_attn_keys=overall_guidance_attn_keys if use_ref_ca else [],sam_refine_kwargs=sam_refine_kwargs,
                so_input_embeddings=so_input_embeddings,num_inference_steps=num_inference_steps,fast_after_steps=fast_after_steps,fast_rate=2,verbose=verbose,
            ) 

        else:
            (latents_all_list, mask_tensor_list, saved_attns_list, so_img_list) = (
                [],
                [],
                [],
                [],
            )

        (
            composed_latents,
            foreground_indices,
            inp_mask,
            inp_img,
        ) = latents.compose_latents_with_alignment(basever, adapter, repeat_ind, mask_tensor_list_512, single_obj_img_list, model_dict,latents_all_list,mask_tensor_list,num_inference_steps,overall_batch_size,height, width,latents_bg=latents_bg,
            align_with_overall_bboxes=align_with_overall_bboxes,overall_bboxes=overall_bboxes,horizontal_shift_only=horizontal_shift_only,use_fast_schedule=use_fast_schedule,fast_after_steps=fast_after_steps,)
        # correlate this

        (
            overall_object_positions,
            overall_word_token_indices,
            overall_prompt1,
        ) = guidance.get_phrase_indices(tokenizer=adapter.pipe.tokenizer,prompt=overall_prompt,phrases=overall_phrases,words=overall_words,verbose=verbose,return_word_token_indices=True,add_suffix_if_not_found=True,)
        
        print(overall_prompt)

        overall_input_embeddings = models.encode_prompts(prompts=[overall_prompt],tokenizer=adapter.pipe.tokenizer,negative_prompt=overall_negative_prompt,text_encoder=adapter.pipe.text_encoder,)
        bg_input_embeddings = models.encode_prompts(prompts=[bg_prompt],tokenizer=adapter.pipe.tokenizer,negative_prompt=overall_negative_prompt,text_encoder=adapter.pipe.text_encoder,)

        # This is currently not-shared with the single object one.
        overall_semantic_guidance_kwargs = dict(loss_scale=overall_loss_scale,loss_threshold=overall_loss_threshold,max_iter=overall_max_iter,max_index_step=overall_max_index_step,fg_top_p=overall_fg_top_p,bg_top_p=overall_bg_top_p,fg_weight=overall_fg_weight,bg_weight=overall_bg_weight,
            ref_ca_word_token_only=True,
            ref_ca_last_token_only=True,ref_ca_saved_attns= None,word_token_indices=overall_word_token_indices,guidance_attn_keys=overall_guidance_attn_keys,ref_ca_loss_weight=ref_ca_loss_weight,use_ratio_based_loss=False,verbose=True,)

        frozen_mask = foreground_indices != 0
        image = Image.new("1", (int(height/8), int(width/8)))
        pixels = [int(value) * 255 for value in frozen_mask.flatten()]
        image.putdata(pixels)
        tpipe=1

        regen_latents, images = pipelines.final_image_generation(
            basever,
            processor,
            controlnetpipe,
            tpipe,
            overall_prompt,
            overall_negative_prompt,
            bg_prompt,
            single_obj_img_list,
            objects, 
            repeat_ind,
            height,
            width,
            bg_seed,
            inp_mask,
            inp_img,
            adapter,
            model_dict,
            composed_latents.cuda(),
            frozen_mask,
            bg_input_embeddings,
            overall_input_embeddings, 
            num_inference_steps,
            frozen_steps, 
            guidance_scale=guidance_scale,
            bboxes=overall_bboxes, 
            phrases=overall_phrases, 
            object_positions=overall_object_positions, 
            semantic_guidance_kwargs=overall_semantic_guidance_kwargs, 
        )
        
        print(
            f"Generation with spatial guidance from input latents and first {frozen_steps} steps frozen (directly from the composed latents input)"
        )
        print("Generation from composed latents (with semantic guidance)")




    utils.free_memory()
    return EasyDict(image=images[0], so_img_list=so_img_list)
