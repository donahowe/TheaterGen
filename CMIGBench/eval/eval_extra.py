import os
from groundingdino.util.inference import load_model, load_image, predict, annotate
from PIL import Image
from torchvision.ops import box_convert
import torch
import argparse
import json
import tqdm
import re
import math
import csv


def detector(model, 
             path, 
             objects,
             box_threshold,
             text_threshold,
             device,
             is_numer=False):

    path = os.path.join(images_path, path[0])
    image_source, image = load_image(path)

    detected_obj = []
    #texts = ""
    for object in objects:
        #texts = texts + object[0] + ". "
        texts =  object[0]
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
        if is_numer:
            if len(boxes) != 0:
                for i in range(len(boxes)):
                    detected_obj.append([object[0], boxes[i]])
        else:
            if len(boxes) != 0:
                detected_obj.append([object[0], boxes[0]]) #取第一个box 
        
    return detected_obj


def eval_spatial(objects,
                 caption,
                 generation_image,
                 box_threshold,
                 text_threshold):
    detected_obj = detector(model=dino_model,
        path=generation_image,
        objects=objects,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device)
    if len(detected_obj) != len(objects):
        return detected_obj, False
    
    pattern = r"(.+?)\sto the right of\s(.+)"
    match = re.search(pattern, caption)
    if match:
        item1 = match.group(1)
        item2 = match.group(2)

        words = item2.split()
        if len(words) >= 1:
            item2 = ' '.join(words[-1:])
        
        left_obj = min(detected_obj, key=lambda x: x[1][0])
        left_obj = left_obj[0]

        words = left_obj.split()
        if len(words) >= 1:
            left_obj = ' '.join(words[-1:])

        if item2 == left_obj:
            return detected_obj, True
        else:
            return detected_obj, False
    else:
        pattern = r"(.+?)\sto the left of\s(.+)"
        match = re.search(pattern, caption)
        if match:
            item1 = match.group(1)
            item2 = match.group(2)

            words = item2.split()
            if len(words) >= 1:
                item2 = ' '.join(words[-1:])
        
            right_obj = max(detected_obj, key=lambda x: x[1][0])
            right_obj = right_obj[0]

            words = right_obj.split()
            if len(words) >= 1:
                right_obj = ' '.join(words[-1:])

            if item2 == right_obj:
                return detected_obj, True
            else:
                return detected_obj, False
        else:
            pattern = r"(.+?)\sto the top of\s(.+)"
            match = re.search(pattern, caption)
            if match:
                item1 = match.group(1)
                item2 = match.group(2)

                words = item2.split()
                if len(words) >= 1:
                    item2 = ' '.join(words[-1:])
        
                down_obj = max(detected_obj, key=lambda x: x[1][1])
                down_obj = down_obj[0]

                words = down_obj.split()
                if len(words) >= 1:
                    down_obj = ' '.join(words[-1:])

                if item2 == down_obj:
                    return detected_obj, True
                else:
                    return detected_obj, False
            else:
                pattern = r"(.+?)\sto the down of\s(.+)"
                match = re.search(pattern, caption)
                if match:
                    item1 = match.group(1)
                    item2 = match.group(2)

                    words = item2.split()
                    if len(words) >= 1:
                        item2 = ' '.join(words[-1:])
        
                    top_obj = min(detected_obj, key=lambda x: x[1][1])
                    top_obj = top_obj[0]

                    words = top_obj.split()
                    if len(words) >= 1:
                        top_obj = ' '.join(words[-1:])

                    if item2 == top_obj:
                        return detected_obj, True
                    else:
                        return detected_obj, False
                else:
                    pattern = r"(.+?)\sbelow\s(.+)"
                    match = re.search(pattern, caption)
                    if match:
                        item1 = match.group(1)
                        item2 = match.group(2)

                        words = item2.split()
                        if len(words) >= 1:
                            item2 = ' '.join(words[-1:])
        
                        top_obj = min(detected_obj, key=lambda x: x[1][1])
                        top_obj = top_obj[0]

                        words = top_obj.split()
                        if len(words) >= 1:
                            top_obj = ' '.join(words[-1:])

                        if item2 == top_obj:
                            return detected_obj, True
                        else:
                            return detected_obj, False
                    else:
                        pattern = r"(.+?)\sin the middle of\s(.+)"
                        match = re.search(pattern, caption)
                        if match:
                            if len(detected_obj) < 2:
                                return detected_obj, False
                            elif len(detected_obj) == 2:
                                distance = calculate_distance(detected_obj[0][1], detected_obj[1][1])
                                if distance < 300:
                                    return detected_obj, True
                                else:
                                    return detected_obj, True


def calculate_center(rectangle):
    x, y, w, h = rectangle
    center_x = x + w/2
    center_y = y + h/2
    return center_x, center_y


def calculate_distance(rectangle1, rectangle2):
    x1, y1 = calculate_center(rectangle1)
    x2, y2 = calculate_center(rectangle2)
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def eval_attribute(attr_object,
                   generation_image,
                   box_threshold,
                   text_threshold):
    detected_obj = detector(model=dino_model,
             path=generation_image,
             objects=attr_object,
             box_threshold=box_threshold,
             text_threshold=text_threshold,
             device=device)
    
    if len(detected_obj) == 1:
        return detected_obj, True
    else:
        return detected_obj, False


def eval_negative(neg_object,
                  generation_image,
                  box_threshold,
                  text_threshold):
    detected_obj = detector(model=dino_model,
             path=generation_image,
             objects=neg_object,
             box_threshold=box_threshold,
             text_threshold=text_threshold,
             device=device)
    
    if len(detected_obj) == 0:
        return detected_obj, True
    else:
        return detected_obj, False


def eval_numeracy(objects,
                  generation_image,
                  box_threshold,
                  text_threshold):
    detected_obj = detector(model=dino_model,
             path=generation_image,
             objects=[objects[0]],
             box_threshold=box_threshold,
             text_threshold=text_threshold,
             device=device,
             is_numer=True)
    
    if len(detected_obj) == len(objects):
        return detected_obj, True
    else:
        return detected_obj, False



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate for Image-Image similarity')
    parser.add_argument('--image_path',  type=str, default='outputpath/')
    parser.add_argument('--annotation_path', type=str, default='CMIGBench/editing.json')
    parser.add_argument('--model_name', type=str, default='editing')    
    parser.add_argument('--box_threshold', type=float, default=0.5)
    parser.add_argument('--text_threshold', type=float, default=0.25)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dino_model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")

    paths = os.listdir(args.image_path)
    paths = sorted(paths, key=lambda x: int(x.split()[1]))
    count = len(paths)

    with open(args.annotation_path, 'r') as f:
        data = json.load(f)

    output_csv = f'extra_eval_{args.model_name}.csv'
    columns = ['dialogue_id', 'spatial', 'attribute', 'negative', 'numeracy']
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(columns)

    correct_spatial = 0
    correct_attribute = 0
    correct_negative = 0
    correct_numeracy = 0

    incorrect_spatial = 0
    incorrect_attribute = 0
    incorrect_negative = 0
    incorrect_numeracy = 0
    with tqdm.tqdm(total=count) as pbar:
        for dialog_id in paths:
            try:
                group_data = data[dialog_id]
                dialog_path = f'{dialog_id}'
                if dialog_path in paths:
                    images_path = os.path.join(args.image_path, dialog_path)
                    print(f'\npath:\n{images_path}')
                    images = os.listdir(images_path)
                
                for turn in group_data.keys():
                    if not turn.startswith('turn'):
                        continue

                    caption = group_data[turn]['caption']
                    print(f'\ncaption:{caption}')

                    generation_image = [image for image in images if image == f'{turn}.png']
                    objects = [object[0] for object in group_data[turn]['objects']]
                    #if args.model_name == 'ours':
                    #    generation_image = [f'{turn}/img_0.png']

                    if turn == 'turn 1':
                        detected, spatial = eval_spatial(group_data[turn]['objects'],
                                            caption,
                                            generation_image,
                                            0.35,
                                            args.text_threshold)
                        print(f'\nDetected objects:\n{detected}')
                        print(f'\nEval spatial: {spatial}')
                        if spatial:
                            correct_spatial += 1
                        else:
                            incorrect_spatial += 1

                    elif turn == 'turn 2':
                        caption = group_data[turn]['caption']
                        objects = [object[0] for object in group_data[turn]['objects']]

                        max_overlap_count = 0
                        max_overlap_object = ""

                        for object in objects:
                            overlap_count = sum(1 for word in object.split() if word in caption)
                            if overlap_count > max_overlap_count:
                                max_overlap_count = overlap_count
                                max_overlap_object = object

                        detected, attribute = eval_attribute([[max_overlap_object]],
                                    generation_image,
                                    args.box_threshold,
                                    args.text_threshold)
                        print(f'\nDetected objects:\n{detected}')
                        print(f'\nEval attribute: {attribute}')
                        if attribute:
                            correct_attribute += 1   
                        else:
                            incorrect_attribute += 1
                    
                    elif turn == 'turn 3':
                        detected, negative = eval_negative([[group_data[turn]['negative']]],
                                    generation_image,
                                    0.8,
                                    args.text_threshold)
                        print(f'\nDetected objects:\n{detected}')
                        print(f'\nEval negative: {negative}')
                        if negative:
                            correct_negative += 1     
                        else:
                            incorrect_negative += 1

                    elif turn == 'turn 4':
                        detected, numeracy = eval_numeracy(group_data[turn]['objects'],  #[group_data[turn]['objects'][0]], 8
                                    generation_image,
                                    0.35,
                                    args.text_threshold)
                        print(f'\nDetected objects:\n{detected}')
                        print(f'\nEval numeracy: {numeracy}')
                        if numeracy:
                            correct_numeracy += 1
                        else:
                            incorrect_numeracy += 1
                            
                with open(output_csv, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([dialog_id, spatial, attribute, negative, numeracy])
                count += 1
                pbar.update(1)
            except: 
                continue
    print(f'Spatial:{correct_spatial/(correct_spatial+incorrect_spatial)}\n')
    print(f'Attribute:{correct_attribute/(correct_attribute + incorrect_attribute)}\n')
    print(f'Negative:{correct_negative/(correct_negative + incorrect_negative)}\n')
    print(f'Numeracy:{correct_numeracy/(correct_numeracy + incorrect_numeracy)}')