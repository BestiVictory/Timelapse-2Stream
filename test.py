import os
import sys
import numpy as np
import random
import tensorflow as tf
from keras.utils import image_utils
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.models import Model
from models import ResearchModels

# 设置随机种子
os.environ['PYTHONHASHSEED'] = str(40)
np.random.seed(40)
random.seed(40)
tf.random.set_seed(40)

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# Model
base_model = InceptionResNetV2(weights='imagenet', include_top=True)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
model.trainable = False  # 确保模型在评估模式

class_limit = 1
seq_length = 40
saved_model = None
pwd = os.path.dirname(sys.argv[0])
rm = ResearchModels(2, 'lstm', seq_length, saved_model)
rm.model.load_weights(os.path.join(pwd, '2stream-multitype.1424-0.329_true.hdf5'), by_name=True)
rm.model.trainable = False  # 确保模型在评估模式
rm.model.compile()  # 重新编译模型

def rescale_list(input_list, size):
    assert len(input_list) >= size
    skip = len(input_list) // size
    output = [input_list[i] for i in range(0, len(input_list), skip)]
    return output[:size]

def extract_features(path):
    frames1 = sorted(os.listdir(path), key=lambda x: int(x[:-4]))  # 确保排序一致
    frames1 = rescale_list(frames1, seq_length)
    
    sequence = []
    for frame in frames1:
        img = image_utils.load_img(os.path.join(path, frame), target_size=(299, 299))
        x = image_utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        features = model.predict(x)
        sequence.append(features[0])
    
    sequence = np.array(sequence)
    np.save(r'output/now.npy', sequence)
    return sequence

def evaluate():
    path1 = os.path.join(pwd, 'output/now.npy')
    X1_test = np.load(path1)
    X1_test = X1_test.reshape(1, 40, 1536)
    predict_test = rm.model.predict(X1_test)
    print("--Prediction:", predict_test)
    print("--score:", predict_test[0][0][0])
    
    return predict_test

def evaluateall():
    scoremax = 0
    flag = 0
    i = 0
    paths = [
        './AblationResult/result3-all/scene1-point2/ame',
        './AblationResult/result3--1/scene1-point2/ame',
        './AblationResult/result3--2/scene1-point2/ame',
        './AblationResult/result3--3/scene1-point2/ame'
    ]
    
    for path in paths:
        extract_features(path)
        score = evaluate()
        if score[0][0][0] > scoremax:
            scoremax = score[0][0][0]
            flag = i
        i += 1
    
    return flag + 1

def main():
    print('\nbest:', evaluateall())

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')

