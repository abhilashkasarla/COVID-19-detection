import matplotlib.pyplot as plt
from keras.preprocessing import image
import os
import numpy as np
from sklearn.metrics import confusion_matrix
VAL_PATH = VAL_PATH = "../chest_xray/test"

def eval(model, path=VAL_PATH):
    y_actual, y_test = [],[]
    for i in os.listdir(path + "/Normal/"):
        img=image.load_img(path + "/Normal/"+i,target_size=(224,224))
        img=image.img_to_array(img)/255.0
        img=np.expand_dims(img,axis=0)
        # pred=model.predict_classes(img)
        pred=model.predict(img)
        y_test.append(np.argmax(pred[0]))
#         if pred[0,1]>=0.5:
#             y_test.append(1)
#         else:
#             y_test.append(0)
#         # y_test.append(pred[0,0])
        y_actual.append(1)


    for i in os.listdir(path + "/Covid/"):
        img=image.load_img(path + "/Covid/"+i,target_size=(224,224))
        img=image.img_to_array(img)/255.0
        img=np.expand_dims(img,axis=0)
        # pred=model.predict_classes(img)
        pred=model.predict(img)
        y_test.append(np.argmax(pred[0]))
#         if pred[0,1]>=0.5:
#             y_test.append(1)
#         else:
#             y_test.append(0)
#         # y_test.append(pred[0,0])
        y_actual.append(0)
    
    for i in os.listdir(path + "/Pneumonia/"):
        img=image.load_img(path + "/Pneumonia/"+i,target_size=(224,224))
        img=image.img_to_array(img)/255.0
        img=np.expand_dims(img,axis=0)
        # pred=model.predict_classes(img)
        pred=model.predict(img)
        y_test.append(np.argmax(pred[0]))
#         if pred[0,1]>=0.5:
#             y_test.append(1)
#         else:
#             y_test.append(0)
#         # y_test.append(pred[0,0])
        y_actual.append(2)
        
    y_actual=np.array(y_actual)
    y_test=np.array(y_test)
    cn=confusion_matrix(y_actual,y_test)
    return cn, y_actual, y_test

def plot_training_metrics(history):
    plt.figure(figsize=(10,5))
    # summarize history for loss
    plt.subplot(121)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
#     plt.show()
    plt.subplot(122)
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()