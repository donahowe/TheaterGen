import os
import time
import matplotlib.pyplot as  plt
import models
import traceback
import bdb
import time
import diffusers
import theatergen as generation
import argparse
import torch
import json

from sched import scheduler
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from utils import parse, vis
from utils.parse import show_boxes
from tqdm import tqdm
from typing import List
from models import sam
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
)
from controlnet_aux import LineartDetector
from GroundingDINO.groundingdino.util.inference import Model
from ip_adapter import IPAdapter, IPAdapterXL
from diffusers.models import UNet2DConditionModel
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL, StableDiffusionXLPipeline
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler

### In standard cases, the seed for each dialogue is fixed. If you want randomness, you can set freeze_dialogue_seed to None. ###
parser = argparse.ArgumentParser()
parser.add_argument("--task", default='story', type=str, help="Task type of CMIGBench")
parser.add_argument("--repeats", default=5, type=int, help="Number of samples for each prompt")
parser.add_argument("--regenerate", default=1, type=int, help="Number of regenerations. Different from repeats, regeneration happens after everything is generated")
parser.add_argument("--force_run_ind", default=None, type=int, help="If this is enabled, we use this run_ind and skips generated images. If this is not enabled, we create a new run after existing runs.")
parser.add_argument("--seed_offset", default=0, type=int, help="Offset to the seed (seed starts from this number)")
parser.add_argument("--sd_version", default='1.5', type=str, help="Base model version. Pick from [1.5, 1.5, xl]")
parser.add_argument("--database_path_base", default='database_1.5', type=str, help="Database path")
parser.add_argument("--base_save_dir", default='cmigbench_1.5', type=str, help="Database path")
parser.add_argument("--dataset_path", default='CMIGBench', type=str, help="Dataset path")
parser.add_argument("--frozen_step_ratio", default = 1, type=float, help="Latent replace ratio")
parser.add_argument("--freeze_dialogue_seed", default = 0, type=int, help="Use same seed for each dialogue for more consistency")
parser.add_argument("--is_notebook", default = False, type=bool, help="Is use notebook")

args = parser.parse_args()
device = "cuda"





print('Loading...\n')
if args.sd_version == '1.5':
    basever = '1.5'
    base_model_path = "runwayml/stable-diffusion-v1-5"
    vae_model_path = "stabilityai/sd-vae-ft-mse"
    unet_path = base_model_path + "/unet"
    image_encoder_path = "h94/IP-Adapter/models/image_encoder/"
    ip_ckpt = "h94/IP-Adapter/models/ip-adapter_sd15.bin"
    ctrn_ckpt = "ControlNet-1-1-preview/control_v11p_sd15_lineart"
    ctrn_processor_path = "lllyasviel/Annotators" 

    ## MainNet
    print('Using sd1.5 checkpoint\n')
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
    unet = UNet2DConditionModel.from_pretrained(base_model_path + "/unet").to(dtype=torch.float16)
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        unet=unet,
        feature_extractor=None,
        safety_checker=None
    )

    ## Using ControlNet
    print("Using ControlNet-1.5\n")
    controlnet = ControlNetModel.from_pretrained(ctrn_ckpt, torch_dtype=torch.float16)
    controlnetpipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path, controlnet=controlnet, torch_dtype=torch.float16
    )
    controlnetpipe.to(device)
    processor = LineartDetector.from_pretrained(ctrn_processor_path)
    
    # IP-Adapter
    print('Using IP-Adapter-1.5\n')
    global ip_model
    ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)

elif args.sd_version == 'xl':
    basever = 'xl'
    base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
    vae_path = "stabilityai/sdxl-vae"
    scheduler_path = "stabilityai/stable-diffusion-xl-base-1.0/scheduler"
    image_encoder_path = "h94/IP-Adapter/sdxl_models/image_encoder"
    ip_ckpt = "h94/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"
    ctrn_ckpt = "ControlNet-1-1-preview/control_v11p_sd15_lineart"
    ctrn_processor_path = "lllyasviel/Annotators"
    
    # MainNet
    print('Using xl checkpoint\n')
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        add_watermarker=False,
    )

    # T2I adapter
    print('Using T2I-Adapter-xl\n')
    adapter = T2IAdapter.from_pretrained(t2i_ckpt, torch_dtype=torch.float16, varient="fp16").to(device)
    euler_a = EulerAncestralDiscreteScheduler.from_pretrained(scheduler_path)
    vae=AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)
    controlnetpipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        base_model_path, vae=vae, adapter=adapter, scheduler=euler_a, torch_dtype=torch.float16, variant="fp16",
    ).to(device)

    # IP-Adapter
    print('Using IP-Adapter-xl\n')
    processor = LineartDetector.from_pretrained(ctrn_processor_path)
    ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)



# DINO
print('Using DINO\n')
GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "GroundingDINO/groundingdino_swint_ogc.pth"
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

# SAM
print('Using SAM')
sam_model_dict = sam.load_sam()
models.model_dict= {}
models.model_dict.update(sam_model_dict)

# CMIG-Benchmark
json_file = f'{args.dataset_path}/{args.task}.json'
with open(json_file, 'r', encoding='utf-8') as file:
    CMIG = json.load(file)

if __name__ == '__main__':

    # Init Params
    LARGE_CONSTANT = 123456789
    LARGE_CONSTANT2 = 56789
    LARGE_CONSTANT3 = 6789
    LARGE_CONSTANT4 = 7890

    task = args.task
    repeats = args.repeats
    run_ind = args.force_run_ind
    regenerate = args.regenerate
    is_notebook = args.is_notebook
    database_path_base = args.database_path_base
    save_dir = f"{args.base_save_dir}/{task}/run{run_ind}"
    seed_offset = args.seed_offset

    theatergen = generation.run

    plt.show = plt.clf
    use_time_list = []



    ind = 0
    print(f"Save dir: {save_dir}\n")
    for regenerate_ind in range(regenerate):
        print(f"regenerate_ind: {regenerate_ind}\n")

        for dialogue in CMIG:
            save_ind = 0
            os.makedirs(f"{save_dir}/{dialogue}", exist_ok=True)
            database_path = f"{database_path_base}/{task}/{dialogue}/"
            os.makedirs(database_path, exist_ok=True)
            vis.reset_save_ind()

            start_time = time.time()
            for turn in [f'turn {i+1}' for i in range(4)]:
                parse.img_dir = f"{save_dir}/{dialogue}/{turn}"
                if os.path.exists(parse.img_dir):
                    continue
                try:
                    data_now = CMIG[dialogue][turn]
                except:
                    continue
                os.makedirs(parse.img_dir, exist_ok=True) #make dir for each turn


                # start generating image
                print("Start Generation Image:\n")
                try:
                    prompt, bg_prompt, neg_prompt = data_now['caption'], data_now['background'], data_now['negative']
                    obj_ids, gen_boxes = [], []
                    if not is_notebook:
                        plt.clf()
                    if args.freeze_dialogue_seed != None: #freeze seed for each dialogue
                        original_ind_base = args.freeze_dialogue_seed
                    else:
                        original_ind_base = ind
                    

                    # save layout
                    for boundingbox in data_now['objects']:
                        gen_box = [boundingbox[0],tuple(boundingbox[1])]
                        obj_ids.append(boundingbox[2])
                        gen_boxes.append(tuple(gen_box))
                    spec = {
                        "prompt": prompt,
                        "gen_boxes": gen_boxes,
                        "bg_prompt": bg_prompt,
                        "extra_neg_prompt": neg_prompt,
                        "obj_ids":obj_ids,
                    }
                    print(f"spec: {spec}\n")
                    show_boxes(
                        gen_boxes,
                        bg_prompt=bg_prompt,
                        neg_prompt=neg_prompt,
                        show=is_notebook,
                    )

                    for repeat_ind in range(repeats):
                        ind_offset = repeat_ind * LARGE_CONSTANT3 + seed_offset

                        vis_location = [dialogue, turn]
                        output = theatergen(vis_location, task, basever, grounding_dino_model, 
                                            processor, controlnetpipe, database_path, repeat_ind, 
                                            adapter=ip_model, spec=spec,
                                            bg_seed =  original_ind_base + ind_offset, 
                                            fg_seed_start = original_ind_base + ind_offset + LARGE_CONSTANT, 
                                            frozen_step_ratio = args.frozen_step_ratio)

                        output = output.image
                        vis.display(output, "img", repeat_ind, save_ind_in_filename=False)


                except (KeyboardInterrupt, bdb.BdbQuit) as e:
                    print(e)
                    exit()
                except RuntimeError:
                    print("***RuntimeError: might run out of memory, skipping the current one***")
                    print(traceback.format_exc())
                    time.sleep(5)
                except Exception as e:
                    print(f"***Error: {e}***")
                    print(traceback.format_exc())
                ind += 1

            seed_offset += 1
            end_time = time.time()
            use_time = end_time - start_time
            use_time_list.append(use_time)

            print("single dialogue time:",use_time)

    print('average generate time:', sum(use_time_list) / len(use_time_list))
