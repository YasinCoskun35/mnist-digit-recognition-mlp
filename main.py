from utilFunctions import readImages,convertImageToArray
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report 


trainImages=readImages('mnist-train')
testImages=readImages('mnist-test')
x_train=[]
y_train=[]
x_test=[]
y_test=[]
index=0

for index in range(9):
    #0'dan 9a kadar tüm resimleri arraye çeviriyoruz
    columnImages=testImages[index]
    for image in columnImages[str(index)]:
        #Burada dosyanın ait oldugu label değerini kayıt ediyoruz
        y_test.append(str(index))
        imageArray=convertImageToArray(image)
        #resmin arraye dönüşmüş halini x_test arrayimize attık
        x_test.append(imageArray)
x_test=np.array(x_test)
y_test=np.array(y_test)
#label değerlerini kategorize edip sayısal değerlerini anlamsız kıldık
y_test = to_categorical(y_test,10)
#2 boyutlu olan rgb arrayini tek boyuta indirdik
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])


index=0
for index in range(9):
    columnImages=trainImages[index]
    for image in columnImages[str(index)]:
        y_train.append(str(index))
        imageArray=convertImageToArray(image)
        x_train.append(imageArray)
x_train=np.array(x_train)
y_train=np.array(y_train)
y_train = to_categorical(y_train,10)

x_train=x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])


x_train=x_train/255
x_test=x_test/255


mlp = MLPClassifier(
    hidden_layer_sizes=(40,),
    max_iter=20,
    solver="sgd",
    verbose=10,
    random_state=1,
    learning_rate_init=0.1,
)
mlp.fit(x_train,y_train)
y_pred=mlp.predict(x_train)

report=classification_report(y_train, y_pred)
test_pred=mlp.predict(x_test)

reportTest=classification_report(y_test, test_pred)
with open('sonuc.txt', 'w') as f:
    f.write('Train Sonuclari')
    f.write(report)
    f.write('Test Sonuclari')
    f.write(reportTest)
    
