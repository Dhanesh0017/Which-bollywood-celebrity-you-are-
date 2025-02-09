# import os
# import pickle
# actors = os.listdir('faces')
#
# filenames =[]
#
# for actor in actors:
#     for file in os.listdir(os.path.join('faces', actor)):
#         filenames.append(os.path.join('faces', actor, file))
#
# # print(filenames)
# # print(len(filenames))
#
# pickle.dump(filenames,open('filenames.pkl', 'wb'))

from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from tqdm import tqdm

filenames = pickle.load(open('filenames.pkl','rb'))

model = VGGFace(model='resnet50', include_top=False, input_shape=(224,224,3),pooling='avg')

def feature_extractor(img_path,model):
    img= image.load_img(img_path, target_size=(224,224))
    img_array= image.img_to_array(img)
    expanded_img= np.expand_dims(img_array,axis=0)
    preprocessed_img= preprocess_input(expanded_img)

    result= model.predict(preprocessed_img).flatten()

    return result

features = []

for file in tqdm(filenames):
    # result= feature_extractor(file,model)
    # print(result.shape) #Every image is extracticing 2048 feautures "Now what are those features we as coder dont know but from one img 2048 features are getting extracted
    # break
    features.append(feature_extractor(file,model))

pickle.dump(features,open('embedding.pkl','wb'))





