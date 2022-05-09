import os as s
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
import numpy as np
#Her bir data klasörünü içindeki labellara bölerek okuyup return ediyoruz
def readImages(mainFolderName):
    folderNames=s.listdir(mainFolderName)
    images=[]
    for label in folderNames:
        imageFiles=s.listdir(str(mainFolderName+'/'+label))
        index=0
        for fileName in imageFiles:
            imageFiles[index]=mainFolderName+'/'+label+'/'+fileName
            index+=1
        images.append({label:imageFiles})
    return images
#Png dosyasını rgb arrayine çevirip return ediyoruz
def convertImageToArray(imagePath):
    img=load_img(imagePath,grayscale=True)
    imageArray=img_to_array(img)
    return imageArray
