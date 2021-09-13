from util.s3_helpers import get_bucket
from multiprocessing import Pool
import json
import os
import time
from layoutlmft.models.layoutlmv2 import LayoutLMv2TokenizerFast
from layoutlmft.data.utils import load_image, normalize_bbox
import torch
import pickle


bucket = get_bucket()

prefixes_ocrs = ['intermediate/table-bank/ocr-data/latex/',
                 'intermediate/table-bank/ocr-data/word/']
prefixes_imgs = ['raw/table-bank/detection-data/Latex/images/',
                 'raw/table-bank/detection-data/Word/images/']
destination_prefixes = ['intermediate/table-bank/tokenized/latex/',
                        'intermediate/table-bank/tokenized/word/']

tokenizer = LayoutLMv2TokenizerFast.from_pretrained('microsoft/layoutlmv2-base-uncased')

all_token_ids = []
all_bbox_inputs = []
all_images = []
all_labels = []

f = open('../../files/Latex.json')
label_dict = json.load(f)

file_name = "../../files/ocrd_filenames.txt"
with open(file_name, "rb") as f:  # Unpickling
    ocrd_list = pickle.load(f)
    ocrd_list = ocrd_list[1:]

def myfunc(file):
    """download img"""
    image_key = prefixes_imgs[0] + file + '.jpg'
    bucket.download_file(image_key, 'files/local_img.jpg')
    # this loads the image as a torch tensor resized to (224,224) and returns the original size
    image, size = load_image('../../files/local_img.jpg')
    image = torch.reshape(image, (1, 3, 224, 224))
    """download ocr dict"""
    ocr_key = prefixes_ocrs[0] + file + '.json'
    bucket.download_file(ocr_key, 'files/local_ocr.json')

    """get token embeddings and bounding boxes"""
    # first create string from the dict
    f2 = open('files/local_ocr.json')
    ocr_dict = json.load(f2)
    n_boxes = len(ocr_dict["level"])
    level = 5  # level 5 corresponds to words
    tokens = []
    bboxes = []
    for i in range(n_boxes):
        if ocr_dict["level"][i] == level:
            l, t, w, h = ocr_dict["left"][i], ocr_dict["top"][i], ocr_dict["width"][i], ocr_dict["height"][i]
            box = [l, t, l + w, t + h]
            text = ocr_dict["text"][i]
            if text.strip() != '':
                tokens.append(text)
                normed_bbox = normalize_bbox(box, size)
                bboxes.append(normed_bbox)

    # now tokenize the string
    tokenized_inputs = tokenizer(tokens,
                                 padding="max_length",
                                 truncation=True,
                                 return_overflowing_tokens=True,
                                 is_split_into_words=True)
    token_ids = tokenized_inputs['input_ids'][0]  # this is a quick fix(may be wrong)
    # assign [0,0,0,0] bounding box to special tokens
    word_ids = tokenized_inputs.word_ids()
    bbox_inputs = []
    for word_idx in word_ids:
        if word_idx is None:
            bbox_inputs.append([0, 0, 0, 0])
        else:
            bbox_inputs.append(bboxes[word_idx])

    """get labels"""
    # get image id from file name
    for img in label_dict['images']:
        if img['file_name'] == file + '.jpg':
            image_id = img['id']
    box_labels = []
    class_labels = []
    for instance in label_dict['annotations']:
        if instance['image_id'] == image_id:
            label_bbox = instance["bbox"]
            normed_label_bbox = normalize_bbox(label_bbox, size)
            l, t, w, h = normed_label_bbox
            xc, yc, h, w = [l / 1000 + w / 2000, t / 1000 + h / 2000, h / 1000, w / 1000]
            box_labels.append([xc, yc, h, w])
            class_labels.append(instance['category_id'])

    class_labels = torch.tensor(class_labels)
    box_labels = torch.tensor(box_labels)
    label = {'labels': class_labels, 'boxes': box_labels}

    all_token_ids.append(token_ids)
    all_bbox_inputs.append(bbox_inputs)
    all_images.append(image)
    all_labels.append(label)

    """now remove local files"""
    os.remove('files/local_ocr.json')
    os.remove('../../files/local_img.jpg')


def pool_handler(mylist):
    if __name__ == '__main__':
        start = time.time()
        p = Pool(4)
        p.map(myfunc, mylist)
        p.close()
        p.join()
        end = time.time()
        print(str(len(mylist)) + " images OCR'd in ", end - start)

def download_files(file):
    """just downloads necessary files from s3"""
    prefixes_ocrs = ['intermediate/table-bank/ocr-data/latex/',
                     'intermediate/table-bank/ocr-data/word/']
    prefixes_imgs = ['raw/table-bank/detection-data/Latex/images/',
                     'raw/table-bank/detection-data/Word/images/']
    save_path = "D:/thesis/data"
    save_path = '/mnt/d/thesis/data'

    """download image, ocr dict"""
    image_key = prefixes_imgs[0] + file + '.jpg'
    image_path = save_path + '/images/' + file +'.jpg'
    ocr_key = prefixes_ocrs[0] + file + '.json'
    ocr_path = save_path + '/ocr_dicts/' + file +'.json'
    bucket.download_file(image_key, image_path)
    bucket.download_file(ocr_key, ocr_path)

#download_files(ocrd_list[0])
"""
start = time.time()
for file in ocrd_list[:100]:
    download_files(file)
end = time.time()
print(end - start)
"""

"""
print(os.getcwd())
os.chdir('/mnt/d/thesis/data')
print(os.getcwd())
"""

if __name__ == '__main__':
    start = time.time()
    p = Pool()
    todo = ocrd_list
    p.map(download_files, todo)
    p.close()
    p.join()
    end = time.time()
    print(str(len(todo)) + " done in: ", end - start)