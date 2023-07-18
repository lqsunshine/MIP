"""
The original code is from StegaStampG:
More details can be found here: https://github.com/tancik/StegaStamp
"""
import bchlib
import os
from PIL import Image
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #ignore warning
import tensorflow.compat.v1 as tf
tf.disable_eager_execution() #执行1.0
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
import argparse
import glob
from tqdm import tqdm
from pathlib import Path

parser = argparse.ArgumentParser(description='Generate StegaStampG triggers')
parser.add_argument('--model_path', type=str, default='backdoor/StegaStampG/ckpt/encoder_imagenet')
parser.add_argument('--dataset', type=str, default='cub200')
parser.add_argument('--source_path', type=str, default='Dataset')
parser.add_argument('--secret', type=str, default='a')
parser.add_argument('--secret_size', type=int, default=100)
args = parser.parse_args()

def ensure_3dim(img):
    if len(img.size) == 2:
        img = img.convert('RGB')
    return img



def Encode_image(model,image):

    # print(model_path)
    secret = 'a'
    secret_size = 100

    input_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['secret'].name
    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_secret = tf.get_default_graph().get_tensor_by_name(input_secret_name)
    input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

    output_stegastamp_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs[
        'stegastamp'].name
    output_residual_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs[
        'residual'].name
    output_stegastamp = tf.get_default_graph().get_tensor_by_name(output_stegastamp_name)
    output_residual = tf.get_default_graph().get_tensor_by_name(output_residual_name)

    BCH_POLYNOMIAL = 137
    BCH_BITS = 5
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    data = bytearray(secret + ' ' * (7 - len(secret)), 'utf-8')
    ecc = bch.encode(data)
    packet = data + ecc

    packet_binary = ''.join(format(x, '08b') for x in packet)
    secret = [int(x) for x in packet_binary]
    secret.extend([0, 0, 0, 0])

    # image = ensure_3dim(Image.open(image_path))
    image = np.array(image, dtype=np.float32) / 255.

    feed_dict = {
        input_secret: [secret],
        input_image: [image]
    }

    hidden_img, residual = sess.run([output_stegastamp, output_residual], feed_dict=feed_dict)

    hidden_img = (hidden_img[0] * 255).astype(np.uint8)
    residual = residual[0] + .5  # For visualization
    residual = (residual * 255).astype(np.uint8)

    # , Image.fromarray(residual)
    return hidden_img

model_path = args.model_path
dataset = args.dataset
source_path = args.source_path
secret = args.secret # lenght of secret less than 7
secret_size = args.secret_size

#load model
sess = tf.InteractiveSession(graph=tf.Graph())
model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], model_path)

datapath = source_path+'/'+dataset
image_sourcepath = datapath + '/images'
org_files = glob.glob(image_sourcepath + '/*/*.jpg')

bd_dataset = dataset + '_bd'
bd_save_path = source_path + '/' + bd_dataset

for img_path in tqdm(org_files,desc='Encoding {} images'.format(len(org_files))):
    image = ensure_3dim(Image.open(img_path))

    image = image.resize((224, 224), Image.ANTIALIAS)
    bd_image = Encode_image(model,image)

    name = os.path.basename(img_path).split('.')[0]
    bd_name = name+'_bd'
    bd_img_path = img_path.replace(dataset,bd_dataset).replace(name,bd_name)
    bd_dir = Path(bd_img_path).parent
    if not os.path.exists(bd_dir):
        os.makedirs(bd_dir)

    im = Image.fromarray(np.array(bd_image))
    im.save(bd_img_path)

