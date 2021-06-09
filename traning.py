import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

from keras import models
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D


#####################################
path= 'myData'
pathLabels= 'labels.csv'
testRatio= 0.2
valRatio= 0.2
imageDimensions= (32,32,3)


#####################################
count = 0
images = []
classNo=[]
myList = os.listdir(path)
print("Total number of Class detected",len(myList))
noOfClasses = len(myList)
print("imoting classes")


for x in range (0,noOfClasses):
    myPicList = os.listdir(path+'/'+str(count))
    for y in myPicList:
        curImg = cv2.imread(path+'/'+str(count)+'/'+y)
        curImg = cv2.resize(curImg,(imageDimensions[0],imageDimensions[1]))
        images.append(curImg)
        classNo.append(count)
    print(count,end= " ")
    count +=1
print(" ")
print("Total Images in Images List= ",len(images))
print("Total IDS in classNo List= ",len(classNo))

images =np.array(images)
classNo=np.array(classNo)

print(images.shape)
print(classNo.shape)

#spliting data
x_train,x_test,y_train,y_test = train_test_split(images,classNo,test_size=testRatio)
x_train,x_validation,y_train,y_validation = train_test_split(x_train,y_train,test_size=valRatio)

print(x_train.shape)
print(x_test.shape)
print(x_validation.shape)

numOfSamples = []
for x in range(0,noOfClasses):
   #print(len(np.where(y_train==0)[0]))
    numOfSamples.append(len(np.where(y_train==0)[0]))
print(numOfSamples)

plt.figure(figsize=(10,5))
plt.bar(range(0,noOfClasses),numOfSamples)
plt.title("no of images for each class")
plt.xlabel("Class ID")
plt.ylabel("Number of Images")
plt.show()


def preProccesing(img):
        img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img=cv2.equalizeHist(img)
        img= img/255
        return img
#img = preProccesing(x_train[30])
#img = cv2.resize(img,(300,300))
#cv2.imshow("Preprocessed",img)
#cv2.waitKey(0)

x_train=np.array(list(map(preProccesing,x_train)))
x_test=np.array(list(map(preProccesing,x_test)))
x_validation=np.array(list(map(preProccesing,x_validation)))


x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
x_validation=x_validation.reshape(x_validation.shape[0],x_validation.shape[1],x_validation.shape[2],1)
#print(x_train.shape)

dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(x_train)

y_train = to_categorical(y_train,noOfClasses)
y_test = to_categorical(y_test,noOfClasses)
y_validation = to_categorical(y_validation,noOfClasses)

def myModel():
    noOfFilters = 60
    sizeofFilters1= (5,5)
    sizeofFilters2=(3,3)
    sizeofPool = (2,2)
    noOfNode = 500

    model =  Sequential()
    model.add((Conv2D(noOfFilters,sizeofFilters1,input_shape=(imageDimensions[0],
                                                              imageDimensions[1],
                                                             1),activation= 'relu'
                                                              )))
    model.add((Conv2D(noOfFilters, sizeofFilters1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeofPool))

    model.add((Conv2D(noOfFilters//2, sizeofFilters2, activation='relu')))
    model.add((Conv2D(noOfFilters//2, sizeofFilters2, activation='relu' )))

    model.add(MaxPooling2D(pool_size=sizeofPool))
    model.add(MaxPooling2D(pool_size=sizeofPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNode, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses,activation='softmax'))
    model.compile(Adam(lr=0.001),loss='categorical crossentropy',
                  metrics=['accuracy'])
    return model

model= myModel()
print(model.summary())

batchSizeVal =50
epochsVal = 10
stepsPerEpoch = 2000

history = model.fit_generator(dataGen.flow(x_train,y_train,
                                 batch_size=batchSizeVal),
                                steps_per_epoch=stepsPerEpoch,
                                epochs=epochsVal,
                                validation_data=(x_validation,y_validation),
                                   shuffle=1)


#img = x_train[30]
#img = cv2.resize(img,(300,300))
#cv2.imshow("Preprocessed",img)
#cv2.waitKey(0)

