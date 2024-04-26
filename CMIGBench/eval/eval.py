import argparse
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import re
import tqdm
import json
from transformers import AutoProcessor, CLIPModel
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
from PIL import Image
from torchvision.ops import box_convert
import torch.nn as nn
import torchvision.models as models  
import torchvision.transforms as transforms 
import torchvision
from PIL import Image  
import numpy as np  
from scipy.linalg import sqrtm  
import csv
import clip
import glob
from pytorch_fid import fid_score


def detector(model, 
             path, 
             objects,
             reference,
             turn,
             box_threshold,
             text_threshold,
             device):

    path = os.path.join(images_path, path[0])
    image_source, image = load_image(path)
    texts = ""
    detected_obj = []
    for object in objects:
        id = object[2]
        texts = object[0].split(' ')[-1]
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=texts,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        h, w, _ = image_source.shape
        boxes = boxes * torch.tensor([w, h, w, h])
        boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        if len(boxes) > 0:
            detected_obj.append([id, boxes[0], object[0]])
            for ref in reference:
                if ref[0] == turn and ref[1] == object[2]:
                    ref.append(path)
                    ref.append(boxes[0])
        else:
            detected_obj.append([id, None, object[0]])


    print(f'\ndetected objects:\n{detected_obj}')
    return detected_obj


def fid(name,
        model,
        generation,
        reference,
        device):
    generation_folder = f'{name}/generation/'
    reference_folder = f'{name}/reference/'

    target_size = (299, 299)

    for filename in os.listdir(generation_folder):
        image_path = os.path.join(generation_folder, filename)
        image = Image.open(image_path)
        resized_image = image.resize(target_size)
        resized_image.save(os.path.join(generation_folder, filename))

    for filename in os.listdir(reference_folder):
        image_path = os.path.join(reference_folder, filename)
        image = Image.open(image_path)
        resized_image = image.resize(target_size)
        resized_image.save(os.path.join(reference_folder, filename))

    fidScore = fid_score.calculate_fid_given_paths([reference, generation],
                                                 batch_size=10,
                                                 device=device,
                                                 dims=2048,
                                                 num_workers=0)
    
    return fidScore


def char2char_similarity(name,
                       model, 
                       processor, 
                       generation_path, 
                       reference,
                       turn,
                       detected_objects,
                       objects,
                       device):


    generation_path = os.path.join(images_path, generation_path[0])
    generation_image = Image.open(generation_path)

    for object in detected_objects:
        id = object[0]
        coordinate = object[1]
        if coordinate is not None:
            x1, y1, x2, y2 = coordinate[0], coordinate[1], coordinate[2], coordinate[3]

            object_image = generation_image.crop((x1, y1, x2, y2))
            for ref in reference:
                if ref[1] == id and len(ref)==4 and turn > ref[0]:
                    reference_image = Image.open(ref[2])
                    reference_image = reference_image.crop(ref[3])

                    existing_files = glob.glob(f'{name}/generation/' + '*.jpg')
                    if len(existing_files) == 0:
                        obj_path = os.path.join(f'{name}/generation/', '0.jpg')
                        ref_path = os.path.join(f'{name}/reference/', '0.jpg')
                    else:
                        max_number = max([int(os.path.splitext(os.path.basename(file))[0]) for file in existing_files])
                        obj_path = os.path.join(f'{name}/generation/', f'{max_number+1}.jpg')
                        ref_path = os.path.join(f'{name}/reference/', f'{max_number+1}.jpg')
                    
                    object_image.save(obj_path)
                    reference_image.save(ref_path)

                    image1_input = processor(object_image).unsqueeze(0).to(device)
                    image2_input = processor(reference_image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        image1_features = model.encode_image(image1_input)
                        image2_features = model.encode_image(image2_input)
                
                    cos = torch.nn.CosineSimilarity(dim=0)
                    similarity = cos(image1_features[0], image2_features[0]).item()

                    img_simi[id-1].append(similarity)

                elif ref[1] == id and len(ref) < 4 and turn > ref[0]:
                    orin_turn = int(ref[0].split()[1])
                    curr_turn = int(turn.split()[1])
                    diff = curr_turn - orin_turn

                    ref[0] = turn
                    ref.append(generation_path)
                    ref.append(coordinate)

                    img_simi[id-1].extend([0] * diff)

                    for i in range(diff):
                        existing_files = glob.glob(f'{name}/generation/' + '*.jpg')
                        if len(existing_files) == 0:
                            obj_path = os.path.join(f'{name}/generation/', '0.jpg')
                            ref_path = os.path.join(f'{name}/reference/', '0.jpg')
                        else:
                            max_number = max([int(os.path.splitext(os.path.basename(file))[0]) for file in existing_files])
                            obj_path = os.path.join(f'{name}/generation/', f'{max_number+1}.jpg')
                            ref_path = os.path.join(f'{name}/reference/', f'{max_number+1}.jpg')

                        reference_image = Image.open(ref[2])
                        reference_image = reference_image.crop(ref[3])
                        object_image = Image.fromarray(np.zeros_like(reference_image))
                        object_image.save(obj_path)
                        reference_image.save(ref_path)
                else:
                    continue

        else:
            for ref in reference:
                if ref[1] == id and turn > ref[0] and len(ref) == 4:
                    img_simi[id-1].append(0)

                    existing_files = glob.glob(f'{name}/generation/' + '*.jpg')
                    if len(existing_files) == 0:
                        obj_path = os.path.join(f'{name}/generation/', '0.jpg')
                        ref_path = os.path.join(f'{name}/reference/', '0.jpg')
                    else:
                        max_number = max([int(os.path.splitext(os.path.basename(file))[0]) for file in existing_files])
                        obj_path = os.path.join(f'{name}/generation/', f'{max_number+1}.jpg')
                        ref_path = os.path.join(f'{name}/reference/', f'{max_number+1}.jpg')

                    reference_image = Image.open(ref[2])
                    reference_image = reference_image.crop(ref[3])
                    object_image = Image.fromarray(np.zeros_like(reference_image))
                    object_image.save(obj_path)
                    reference_image.save(ref_path)



def text2img_similarity(model,
                        processor,
                        generation_path,
                        caption,
                        device):
    '''
    for each frame

    input:
    model: similarity calculation model
    processor: similarity calculation model
    generation_path: generation_image path 
    caption: frame caption

    output:
    similarity(should be a float)
    '''
    generation_path = os.path.join(images_path, generation_path[0])
    generation_image = Image.open(generation_path)

    image = processor(generation_image).unsqueeze(0).to(device)
    text = clip.tokenize([caption]).to(device)

    # 获取图像和文本的嵌入向量
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.cpu().numpy().tolist()

    text_simi.append(probs[0][0])


def evaluate(name,
             data, 
             generation_image, 
             reference,
             turn,
             dino_model, 
             box_threshold,
             text_threshold,
             clip_model, 
             clip_processor, 
             device):
    
    detected_objects = detector(model=dino_model,
                   path=generation_image,
                   objects=data['objects'],
                   reference=reference,
                   turn=turn,
                   box_threshold=box_threshold,
                   text_threshold=text_threshold,
                   device=device)


    simi_img = char2char_similarity(name,
                       model=clip_model,
                       processor=clip_processor,
                       generation_path=generation_image,
                       reference=reference,
                       turn=turn,
                       detected_objects=detected_objects,
                       objects=data['objects'],
                       device=device)
    

    simi_text = text2img_similarity(model=clip_model,
                        processor=clip_processor,
                        generation_path=generation_image,
                        caption=caption,
                        device=device)
    
    return simi_img, simi_text

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate for Image-Image similarity')  
    parser.add_argument('--image_path',  type=str, default='outputpath/')
    parser.add_argument('--annotation_path', type=str, default='CMIGBench/story.json')
    parser.add_argument('--model_name', type=str, default='story') 
    parser.add_argument('--box_threshold', type=float, default=0.5)
    parser.add_argument('--text_threshold', type=float, default=0.25)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    dino_model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
    clip_model, clip_processor = clip.load("ViT-B/32", device)
    inception_model = torchvision.models.inception_v3(pretrained=True).to(device)

    paths = os.listdir(args.image_path)
    paths = sorted(paths, key=lambda x: int(x.split()[1]))
    count = len(paths)

    with open(args.annotation_path, 'r') as f:
        data = json.load(f)

    name = args.model_name
    output_csv = f'story_result_{name}.csv'
    columns = ['dialogue_id', 'FID', 'CCS', 'TIS']
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(columns)

    folder_names = [f'{name}/reference', f'{name}/generation']
    for folder_name in folder_names:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    CCS = 0
    TIS = 0
    FID = 0

    real_count = count


    with tqdm.tqdm(total=count) as pbar:
        for dialog_id in paths:
            #try:
                folder_paths = [f'{name}/reference/', f'{name}/generation/']
                for folder_path in folder_paths:
                    file_paths = glob.glob(folder_path + '*')
                    for file_path in file_paths:
                        os.remove(file_path)

                group_data = data[dialog_id]
                max_id = -1
                for key, value in group_data.items():
                    if key.startswith('turn'):
                        for obj in value['objects']:
                            max_id = max(max_id, obj[2])
                reference_flag = [0] * max_id
                reference = []
                img_fid = [[] for _ in range(max_id)]
                img_simi = [[] for _ in range(max_id)]
                text_simi = []
                dialogCCS = 0
                dialogTIS = 0
                dialogFID = 0

                dialog_path = f'{dialog_id}'
                if dialog_path in paths:
                    images_path = os.path.join(args.image_path, dialog_path)
                    print(f'\npath:\n{images_path}')
                    images = os.listdir(images_path)
                
                for turn in group_data.keys():
                    if not turn.startswith('turn'):
                        continue
                    try:
                        testload = Image.open(f'{images_path}/{turn}.png')
                    except:
                        continue
                    #caption = group_data[turn]['caption']
                    caption =  group_data[turn]['background'] + " with "
                    for i in group_data[turn]['objects']:
                        caption = caption + i[0] + ","

                    print(f'\ncaption:\n{caption}')

                    generation_image = [image for image in images if image == f'{turn}.png']
                    objects = [object[0] for object in group_data[turn]['objects']]

                    for object in group_data[turn]['objects']:
                        id = object[2]
                        if not reference_flag[id-1]:
                            reference.append([turn, id])
                            reference_flag[id-1] = 1

                    evaluate(
                        name,
                        group_data[turn], 
                        generation_image, 
                        reference,
                        turn,
                        dino_model,
                        args.box_threshold,
                        args.text_threshold,
                        clip_model, 
                        clip_processor,
                        device)
                print(f'\nreference list:\n{reference}')
                print(f'\nsimi_img list:\n{img_simi}')
                print(f'\nsimi_text list:\n{text_simi}')
                flag = 0
                for simi_img in img_simi:
                    if simi_img:
                        dialogCCS += (sum(simi_img) / len(simi_img))
                        flag += 1
                dialogTIS = sum(text_simi)
                if flag > 0:
                    ccs_per_dialog = dialogCCS / flag
                    CCS += (dialogCCS / flag)

                    dialogFID = fid(
                        name,
                        inception_model,
                        f'{name}/generation',
                        f'{name}/reference',
                        device)
                    print(f'fid:\n{dialogFID}')
                    FID += dialogFID
                else:
                    ccs_per_dialog = None
                    dialogFID = None
                    real_count -= 1

                TIS += (dialogTIS / 4)

                with open(output_csv, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([dialog_id, dialogFID, ccs_per_dialog, (dialogTIS/4)])
                pbar.update(1)
            #except:
            #    continue
    ACCS = CCS / real_count
    ATIS = TIS / count
    AFID = FID / real_count

    print(f'Eval ACCS: {ACCS}')
    print(f'Eval ATIS: {ATIS}')
    print(f'Eval AFID: {AFID}')
