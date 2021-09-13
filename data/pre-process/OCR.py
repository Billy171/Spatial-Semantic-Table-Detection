"""
The aim of this script is to perform OCR on the raw datasets and output tokens/bounding boxes
We upload these to our s3 bucket
"""
import multiprocessing
import sys
import boto3
import boto3.session
import pytesseract
from pytesseract import Output
from multiprocessing import Pool, current_process
import json
import time
import pandas as pd
import os
import cv2
from PIL import Image, ImageDraw, ImageFont

"""specify path to tesseract executable"""
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

prefixes = ['raw/table-bank/detection-data/Latex/images/',
            'raw/table-bank/detection-data/Word/images/']
destination_prefixes = ['intermediate/table-bank/ocr-data/latex/',
                        'intermediate/table-bank/ocr-data/word/']

url_files = ['url_latex.csv',
             'url_word.csv']

"""getting mfa code from authentication device"""
mfa_otp = input("Enter the MFA code for " + current_process().name + ": ")

"""getting temporary credentials"""
client = boto3.client('sts')
response = client.get_session_token(DurationSeconds=129600,
                                    SerialNumber='arn:aws:iam::536930143272:mfa/billy.ohara',
                                    TokenCode=mfa_otp)

"""starting session (max 36 hours)"""
my_session = boto3.session.Session(aws_access_key_id=response['Credentials']['AccessKeyId'],
                                   aws_secret_access_key=response['Credentials']['SecretAccessKey'],
                                   aws_session_token=response['Credentials']['SessionToken'])

"""connecting to s3 table identification bucket"""
s3 = my_session.resource('s3')
bucket = s3.Bucket('table-identification')



def ocrify(key_pair):
    """
    this function takes an s3 object and uploads an OCR'd version to the bucket
    """

    """download image from s3 bucket"""
    key, destination_key = key_pair
    temp_img = current_process().name+'local_img.jpg'
    bucket.download_file(key, temp_img)
    img = Image.open(temp_img)

    """OCR using tesseract"""
    config = f"--psm 3"
    ocr_dict = pytesseract.image_to_data(img, output_type=Output.DICT, config=config)
    temp_json = current_process().name+'local.json'
    f = open(temp_json, "w")
    json.dump(ocr_dict, f)
    f.close()

    """upload to s3 bucket"""
    bucket.upload_file(temp_json, Key=destination_key)

    os.remove(temp_img)
    os.remove(temp_json)

def pool_handler(i):
    """this takes care of the multiprocessing"""
    prefix = prefixes[i]
    destination_prefix = destination_prefixes[i]
    url_file = url_files[i]
    urls = pd.read_csv(url_file)
    keys = list(prefix + urls['image_name'])
    destination_keys = list(destination_prefix + urls['image_name'].str.replace('.jpg', '.json', regex=False))
    key_pairs = [(keys[i], destination_keys[i]) for i in range(len(keys))]
    if __name__ == '__main__':
        start = time.time()
        p = Pool()
        p.map(ocrify, key_pairs)
        p.close()
        p.join()
        end = time.time()
        print(str(len(key_pairs)) + " images OCR'd in ", end - start)

def ocr_markup(image,level):
    """takes an image and outputs an OCR markup at a desired level"""
    config = f"--psm 3"
    img_ocr = Image.open(image)
    img_draw = cv2.imread(image)
    ocr_dict = pytesseract.image_to_data(img_ocr, output_type=Output.DICT, config=config)
    n_boxes = len(ocr_dict["level"])
    for i in range(n_boxes):
        if ocr_dict["level"][i] == level:
            """
            3 for blocks
            4 for lines
            5 for words
            """
            l, t, w, h = ocr_dict["left"][i], ocr_dict["top"][i], ocr_dict["width"][i], ocr_dict["height"][i]
            cv2.rectangle(img_draw, (l, t), (l + w, t + h), 0, 1)
            if level == 5:
                cv2.putText(img_draw, ocr_dict["text"][i], (l, t), cv2.FONT_HERSHEY_SIMPLEX, 1/3, (0, 0, 0))
    cv2.imshow(str(level), img_draw)
    cv2.waitKey(0)

def counter(i):
    """counts number of objects in a folder"""
    pref = destination_prefixes[i]
    print(pref)
    objs = bucket.objects.filter(Prefix=pref)
    count = 0
    for obj in objs:
        count += 1
        if count % 10000 == 0:
            print(count)
    print('finished with ' + str(count))

#pool_handler(0)#198183 total (182021 done)
#pool_handler(1)#78064 total (59524 done)
#counter(0)
#counter(1)