import cv2
import numpy as np
import glob
from resolucao.bounding_boxes import get_img_annotations
import os
import tools

# caminhios para os ficheiros de configuracao
MODEL_FILE = "config/frozen_inference_graph.pb"
CONFIG_FILE = "config/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt"
CLASS_FILE = "config/object_detection_classes_coco.txt"

# caminho para o video a analisar
images_dir = glob.glob("images\\*.jpg")

# valor de limiar miniar para considerar que as predicoes sao de fato objetos
CONFIDENCE_THRESHOLD = 0.4

# ler os nomes das classes
with open(CLASS_FILE, 'r') as f:
    class_names = f.read().split('\n')

# gerar cores aleatoriamente para cada uma das classes
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

# carregar o modelo (neste caso o SSD)
SSDmodel = cv2.dnn.readNet(model=MODEL_FILE, config=CONFIG_FILE, framework="TensorFlow")

images_iou_matches = {}

# ciclo de imagens no diretório
for img_file in images_dir:

    img = cv2.imread(img_file)
    img_height, img_width, channels = img.shape

    # Get file containing ground truth annotations, from current image
    annotation_file = os.path.basename(img_file)[:-4] + '.xml'

    # Get ground truth annotations, from current image
    img_gt_boxes = get_img_annotations("annotations\\" + annotation_file)

    # Establish comparison between each label previsioned
    # with labels from ground truth
    labels_equivalency = {}

    # normalizar com blobFromImage - 300x300 serao as dimensoes das imagens enviadas 'a rede
    blob = cv2.dnn.blobFromImage(image=img, size=(300, 300), swapRB=True)
    SSDmodel.setInput(blob)
    output = SSDmodel.forward()

    dict_key = 0

    for detection in output[0, 0, :, :]:

        # oter o indice de confianca na detecao
        confidence = detection[2]

        if confidence > CONFIDENCE_THRESHOLD:

            # obter a classe
            class_id = detection[1]
            class_name = class_names[int(class_id) - 1]
            color = COLORS[int(class_id)]

            # obter as coordenadas e dimensoes das bounding boxes, normalizadas para coordenadas da imagem
            bbox_x = detection[3] * img_width
            bbox_y = detection[4] * img_height
            bbox_width = detection[5] * img_width
            bbox_height = detection[6] * img_height

            # Information from box x and y
            previsioned_box = [bbox_x, bbox_y, bbox_x+bbox_width, bbox_y+bbox_height]

            # For labels classified "pizza"
            if class_name == "pizza":

                iou_results = tools.calc_iou_with_gt_boxes(previsioned_box, img_gt_boxes)

                # Add current previsioned box iot with ground truth boxes
                labels_equivalency[dict_key] = iou_results

                # Update key for next box
                dict_key += 1

            # colocar retangulos e texto a marcar os objetos identificados
            cv2.rectangle(img, (int(bbox_x), int(bbox_y)), (int(bbox_width), int(bbox_height)), color, thickness=2)
            cv2.putText(img, class_name, (int(bbox_x), int(bbox_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Associate file to resultant IoUs
    images_iou_matches[os.path.basename(img_file)] = labels_equivalency

    cv2.imshow('output', img)

    cv2.waitKey(0)

#print(images_iou_matches)

#for key, value in images_iou_matches.items():

 #   print(key, ':', value)

  #  for key2, value2 in value.items():

   #     print("\t", key, ':', value)


#print(images_iou_matches['06LasVegas-Pizza-HeadsUp-good-videoSixteenByNine3000.jpg'])

for key, value in images_iou_matches['alhofrito_57e575f2d901c.jpg'].items():

    print(key, ':', value)

  #  for key2, value2 in value.items():

   #     print("\t", key, ':', value)


# fechar o stream de video
cv2.destroyAllWindows()
