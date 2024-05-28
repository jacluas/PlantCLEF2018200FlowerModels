from argparse import ArgumentParser
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from keras.models import load_model
from keras.metrics import top_k_categorical_accuracy
import keras.applications as apps
import pickle
import cv2
import os
import functools
                       
from keras.applications.vgg16 import preprocess_input

# available architectures
models_list = [
    'efficientNetB0',
    'efficientNetB1',
    'efficientNetB2',
    'efficientNetB3',
    'efficientNetB4',
    'efficientNetB5',
    'efficientNetB6',
    'efficientNetB7'
]

parser = ArgumentParser()
parser.add_argument('model', help='which model to use', type=str, choices=models_list)
parser.add_argument('pathModel', help='path to load model (do not use the extension file)', type=str)
parser.add_argument('pathData', help='path to image (test)', type=str)
parser.add_argument('-TopK', type=int, help='Top K responses', default=5)
args = parser.parse_args()


modelPath = args.pathModel 	# /model/model'
imagePath = args.pathData 	# '/images/Test/16730/197886.jpg'
TopK = args.TopK            # 5

def getTopK(answer: np.array, class_list: list, K: int = 5):
    '''Get top N ordered answers'''
    top_answers = sorted([[i, val] for i, val in enumerate(answer)], key=lambda x: x[1], reverse=True)
    return [(class_list[i], val) for i, val in top_answers[:K]]


with open(modelPath + '.bin', 'rb') as class_file:
    modelName, classes = pickle.load(class_file)
if isinstance(classes, LabelBinarizer):
    classes = classes.classes_
elif isinstance(classes, OneHotEncoder):
    classes = classes.classes
else:
    raise TypeError('Classes object type is not supported ({}).'.format(type(classes).__name__))


# Top-1 metric
top1 = functools.partial(top_k_categorical_accuracy, k=1)
top1.__name__ = 'top1'
# Top-5 metric
top5 = functools.partial(top_k_categorical_accuracy, k=5)
top5.__name__ = 'top5'

#model
print('\nModel:' + args.model)

#image
print('\nTest image: ' + imagePath  + '\n')

#load model
print('Loading model: ' + modelPath  + '.h5\n')
model = load_model(os.path.abspath(modelPath  + '.h5'), custom_objects={"top1": top1,"top5": top5})

image_dim = 224 if args.model in ['efficientNetB0' ] else 0
image_dim = 240 if args.model in ['efficientNetB1' ] else image_dim
image_dim = 260 if args.model in ['efficientNetB2' ] else image_dim
image_dim = 300 if args.model in ['efficientNetB3' ] else image_dim
image_dim = 380 if args.model in ['efficientNetB4' ] else image_dim
image_dim = 456 if args.model in ['efficientNetB5' ] else image_dim
image_dim = 528 if args.model in ['efficientNetB6' ] else image_dim
image_dim = 600 if args.model in ['efficientNetB7' ] else image_dim

print('\nInput shape: ' + str(image_dim)  + '\n')

# setting inputs
input_shape = (image_dim, image_dim, 3)

#image dimensions
print('\nInput shape: ' + str(input_shape)  + '\n')

#read and preprocessing the image
img = cv2.imread(imagePath, cv2.IMREAD_COLOR)
if img.shape != (image_dim,image_dim):
    img = cv2.resize(img, (image_dim,image_dim))

img_array = np.expand_dims(img, axis=0)
img_array = preprocess_input(img_array)

####################### Prediction
y_pred1 = model.predict(img_array, steps=1)[0]
#pred = np.argmax(y_pred1, axis=1)

#model response
topk = getTopK(y_pred1, classes, TopK)
output = '\n'.join('{},\t{}'.format(*x) for x in topk)
print('\nPredictions:\n'+ output)
