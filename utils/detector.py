import cv2
import numpy as np
from GroundingDINO.groundingdino.util.inference import Model

def detect(grounding_dino_model, word, sam_input_image):  
    print("SAM Prompt Check", word)      
    ok = False
    numpy_image = np.array(sam_input_image)
    sam_input_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    Detection = grounding_dino_model.predict_with_classes(
        image=sam_input_image,
        classes=word,
        box_threshold=0.3,
        text_threshold=0.25
    )
    if len(Detection.xyxy) >= 1: 
        ok = True
        max_conf_index = np.argmax(Detection.confidence)
        Detection = Detection.xyxy[max_conf_index]
    return Detection, ok
