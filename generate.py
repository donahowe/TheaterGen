import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from utils import parse, vis
from utils.parse import show_boxes
from tqdm import tqdm
from typing import List
import matplotlib.pyplot as  plt
import models
import traceback
import bdb
import time
import diffusers
from models import sam
import argparse
import torch
import json
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
)
from GroundingDINO.groundingdino.util.inference import Model
from ip_adapter import IPAdapter, IPAdapterXL
from diffusers.models import UNet2DConditionModel
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL, StableDiffusionXLPipeline
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler
import time

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
float_args = [
    "loss_threshold",
    "ref_ca_loss_weight",
    "fg_top_p",
    "bg_top_p",
    "overall_fg_top_p",
    "overall_bg_top_p",
    "fg_weight",
    "bg_weight",
    "overall_fg_weight",
    "overall_bg_weight",
    "overall_loss_threshold",
    "fg_blending_ratio",
    "mask_th_for_point",
    "so_floor_padding",
]
for float_arg in float_args:
    parser.add_argument("--" + float_arg, default=None, type=float)

int_args = [
    "loss_scale",
    "max_iter",
    "max_index_step",
    "overall_max_iter",
    "overall_max_index_step",
    "overall_loss_scale",
    "horizontal_shift_only",
    "so_horizontal_center_only",
    "use_autocast",
    "use_ref_ca"
]

for int_arg in int_args:
    parser.add_argument("--" + int_arg, default=None, type=int)

device = "cuda"
args = parser.parse_args()
database_path_base = args.database_path_base
task = args.task

print('Loading Adapter')
if args.sd_version == '1.5':
    basever = '1.5'
    print('Using sd1.5 checkpoint')
    print("Using IP Adapter")
    base_model_path = "pretrained_models/diffusion_1.5"
    #base_model_path = "pretrained_models/diffusion_1.5_comic"
    vae_model_path = "pretrained_models/vae_ft_mse"
    unet_path = base_model_path + "/unet"
    image_encoder_path = "pretrained_models/image_adapter/models/image_encoder/"
    ip_ckpt = "pretrained_models/image_adapter/models/ip-adapter_sd15.bin"

    #Control net
    print("Using Control Net")
    checkpoint = "pretrained_models/lineart_ckpt"
    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
    controlnetpipe = StableDiffusionControlNetPipeline.from_pretrained(
        "pretrained_models/diffusion_1.5", controlnet=controlnet, torch_dtype=torch.float16
    )
    controlnetpipe.to(device)
    from controlnet_aux import LineartDetector
    processor = LineartDetector.from_pretrained("pretrained_models/lineart_detector")
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
    global ip_model
    ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)

elif args.sd_version == 'xl':
    basever = 'xl'
    print('Using xl checkpoint')
    print("Using IP Adapterxl")
    base_model_path = "pretrained_models/diffusion_xl"
    image_encoder_path = "pretrained_models/image_adapter/sdxl_models/image_encoder"
    ip_ckpt = "pretrained_models/image_adapter/sdxl_models/ip-adapter_sdxl.bin"

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        add_watermarker=False,
    )

    from controlnet_aux import LineartDetector
    processor = LineartDetector.from_pretrained("pretrained_models/lineart_detector")
    ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)

    #T2I adapter
    print('Using T2I Adapter')
    adapter = T2IAdapter.from_pretrained("pretrained_models/lineart_xlckpt", torch_dtype=torch.float16, varient="fp16").to("cuda:2")
    euler_a = EulerAncestralDiscreteScheduler.from_pretrained("pretrained_models/diffusion_xl/scheduler")
    vae=AutoencoderKL.from_pretrained("pretrained_models//vae_sdxl", torch_dtype=torch.float16)
    controlnetpipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        'pretrained_models/diffusion_xl', vae=vae, adapter=adapter, scheduler=euler_a, torch_dtype=torch.float16, variant="fp16",
    ).to("cuda:2")

# DINO
print('Using DINO')
GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "GroundingDINO/groundingdino_swint_ogc.pth"
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

json_file = f'{args.dataset_path}/{args.task}.json'
with open(json_file, 'r', encoding='utf-8') as file:
    cmig = json.load(file)


print('Using SAM')
sam_model_dict = sam.load_sam()
models.model_dict= {}
models.model_dict.update(sam_model_dict)
import theatergen as generation
run = generation.run
plt.show = plt.clf
repeats = args.repeats
seed_offset = args.seed_offset
base_save_dir = f"{args.base_save_dir}/{task}"

run_kwargs = {}
argnames = float_args + int_args
for argname in argnames:
    argvalue = getattr(args, argname)
    if argvalue is not None:
        run_kwargs[argname] = argvalue
        print(f"**Setting {argname} to {argvalue}**")


scale_boxes_default = False
is_notebook = False


run_ind = args.force_run_ind
save_dir = f"{base_save_dir}/run{run_ind}"

print(f"Save dir: {save_dir}")

LARGE_CONSTANT = 123456789
LARGE_CONSTANT2 = 56789
LARGE_CONSTANT3 = 6789
LARGE_CONSTANT4 = 7890

ind = 0
use_time_list = []
for regenerate_ind in range(args.regenerate):
    print("regenerate_ind:", regenerate_ind)
    for dialogue in cmig:
       # if dialogue != "dialogue 35":
       #     continue
        save_ind = 0
        os.makedirs(f"{save_dir}/{dialogue}", exist_ok=True) #ouput dir for each dialogue
        database_path = f"{database_path_base}/{task}/{dialogue}/"
        os.makedirs(database_path, exist_ok=True) #database for each dialogue
        vis.reset_save_ind()

        start_time = time.time()

        for turn in [f'turn {i+1}' for i in range(4)]:

            parse.img_dir = f"{save_dir}/{dialogue}/{turn}"
            if os.path.exists(parse.img_dir):
                continue
            try:
                data_now = cmig[dialogue][turn]
            except:
                continue
            os.makedirs(parse.img_dir, exist_ok=True) #make dir for each turn

            #start generating image
            try:
                prompt = data_now['caption']
                bg_prompt = data_now['background']
                neg_prompt = data_now['negative']
                obj_ids = []
                gen_boxes = []
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

                print("spec:", spec)

                #save layout
                show_boxes(
                    gen_boxes,
                    bg_prompt=bg_prompt,
                    neg_prompt=neg_prompt,
                    show=is_notebook,
                )
                if not is_notebook:
                    plt.clf()

                if args.freeze_dialogue_seed != None: #freeze seed for each dialogue
                    original_ind_base = args.freeze_dialogue_seed
                else:
                    original_ind_base = ind

                for repeat_ind in range(repeats):
                    ind_offset = repeat_ind * LARGE_CONSTANT3 + seed_offset

                    vis_location = [dialogue, turn]
                    output = run(vis_location, task, basever, grounding_dino_model, processor, controlnetpipe, database_path, repeat_ind, adapter=ip_model,spec=spec, bg_seed =  original_ind_base + ind_offset, fg_seed_start = original_ind_base + ind_offset + LARGE_CONSTANT, frozen_step_ratio = args.frozen_step_ratio, **run_kwargs,)

                    output = output.image
                    vis.display(output, "img", repeat_ind, save_ind_in_filename=False)

            except (KeyboardInterrupt, bdb.BdbQuit) as e:
                print(e)
                exit()
            except RuntimeError:
                print(
                    "***RuntimeError: might run out of memory, skipping the current one***"
                )
                print(traceback.format_exc())
                time.sleep(10)
            except Exception as e:
                print(f"***Error: {e}***")
                print(traceback.format_exc())
            ind += 1

        end_time = time.time()
        use_time = end_time - start_time
        seed_offset += 1
        print("single dialogue time:",use_time)
        use_time_list.append(use_time)

print('average generate time:', sum(use_time_list) / len(use_time_list))
