import glob
import cv2 as cv
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

# Directories
images_path = '../images'
annotations_path = '../annotations'

images_dir = glob.glob(images_path + "\\" + "*.jpg")
annotations_dir = glob.glob(annotations_path + "\\" + "*.xml")

sample_annotations_all = []


def get_img_annotations(image):
    tree = ET.parse(image)
    root = tree.getroot()

    sample_annotations = []

    for neighboor in root.iter('bndbox'):
        xmin = int(neighboor.find('xmin').text)
        ymin = int(neighboor.find('ymin').text)
        xmax = int(neighboor.find('xmax').text)
        ymax = int(neighboor.find('ymax').text)

        sample_annotations.append([xmin, ymin, xmax, ymax])

    return sample_annotations


def get_all_annotations():
    for path_ann in annotations_dir:
        tree = ET.parse(path_ann)
        root = tree.getroot()

        sample_annotations = []

        for neighboor in root.iter('bndbox'):
            xmin = int(neighboor.find('xmin').text)
            ymin = int(neighboor.find('ymin').text)
            xmax = int(neighboor.find('xmax').text)
            ymax = int(neighboor.find('ymax').text)

            sample_annotations.append([xmin, ymin, xmax, ymax])

        sample_annotations_all.append(sample_annotations)


def draw_bbox(annotations):
    i = 0
    for path in images_dir:
        img = Image.open(path).convert('RGB')

        img_annotated = img.copy()

        img_bbox = ImageDraw.Draw(img_annotated)

        for bbox in annotations[i]:
            print(bbox)
            img_bbox.rectangle(bbox, outline="green", width=5)

        img_annotated.save(path.title() + "_annotated.jpg")

        i += 1


get_all_annotations()
draw_bbox(sample_annotations_all)
