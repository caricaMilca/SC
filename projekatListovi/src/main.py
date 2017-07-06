import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from pylab import *
import csv
import random
import cv2
from skimage.morphology import diamond
from skimage.morphology import erosion
from skimage.morphology import dilation
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.optimizers import SGD

file = open('rezultat.data','w')

def analizirajRegione(putanja, korijenSlike):
    img = cv2.imread(putanja,cv2.IMREAD_GRAYSCALE)
    plt.imshow(img,'gray')
    plt.show()
    #img_gray = rgb2gray(img)
  #  height, width = img.shape
    ret, thresh = cv2.threshold(img,0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    plt.imshow(thresh,'gray')
    plt.show()
    height, width = thresh.shape
    for x in range(0, height):
        for y in range(0, width):
            if thresh[x, y] == 255:
                thresh[x,y] = 0
            else:
                thresh[x, y] = 255
               # img[x, y] = [0, 0, 0]
    plt.imshow(thresh, 'gray')
    plt.show()
    thresh = dilation(thresh, selem=diamond(3))
    #plt.imshow(thresh, 'gray')
    #plt.show()
    thresh = erosion(thresh, selem=diamond(3))
    #plt.imshow(thresh, 'gray')
    #plt.show()
    #kernel = np.ones((5,5), np.uint8)
   # cv2.dilate(thresh, kernel, iterations = 7)
  #  plt.imshow(thresh, 'gray')
   # plt.show()
    derp, contours, hierarchy = cv2.findContours(thresh ,1, 2)
    print (len(contours))
    maxArea = 0
    x = 0
    for i in range(0, len(contours)):
        cnt = contours[i]
        M = cv2.moments(cnt)
        area = cv2.contourArea(cnt)
        if(area > maxArea):
            maxArea = area
            x = i

    print (maxArea, ' area ', x)
    cnt = contours[x]
    duzina = cv2.arcLength(cnt,False)
   # print(duzina, ' duzina ', i)
    rect = cv2.minAreaRect(cnt)
    print(rect, ' rect ', i)
    print(rect[0][0] , ' prvi clan ', i)
    print(rect[0][1], ' drugi clan ', i)


 #   print (cx)
  #  print (cy)
  #  print (hull)
 #   print (M)
    line = str(maxArea)+ ',' + str(rect[0][0]) + ',' + str(rect[0][1])+ ',' + korijenSlike
    return line



print  ('Krenulo je ' ,datetime.datetime.now().time())
pathRoot = 'C:/Users/FixMe/Desktop/listovi/leaf'
for i in range(1, 16):
    korijenSlike = 'l'+str(i)+'nr00'
    putanja = pathRoot+str(i)+'/'+korijenSlike+'1.tif'
    #print(putanja)
    #print(korijenSlike)
    for i in range(1, 76):
        dijelovi = putanja.split('/')
        if i >= 10:
            if dijelovi[5] == 'leaf10' or dijelovi[5] == 'leaf11' or dijelovi[5] == 'leaf12' or dijelovi[
                5] == 'leaf13' or dijelovi[5] == 'leaf14' or dijelovi[5] == 'leaf15':
                dijelovi[6] = korijenSlike[0:6] + str(i) + '.tif'
        else:
            dijelovi[6] = korijenSlike + str(i) + '.tif'

        putanja = dijelovi[0] + '/' + dijelovi[1] + '/' + dijelovi[2] + '/' + dijelovi[3] + '/' + dijelovi[4] + '/' + \
                  dijelovi[5] + '/' + dijelovi[6]
        dadatUFajl = analizirajRegione(putanja, korijenSlike)
        file.write(dadatUFajl + '\n')
file.close()

def loadDatasetForNM(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(3):  # da od stringa upisanog u fajl napravi float kad su posmatrana 3 parametra lista
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

def getAccuracy(accDec):
    accuracy = accDec*100
    return accuracy

def createLabels(className):
    labelWithL = className.split('n')[0]
    labelStr = labelWithL[1:len(labelWithL)]
    labelNum  =int(labelStr)
    retVal = np.zeros((1, 15), dtype='int')
    retVal[0,labelNum-1]=1
  #  print(retVal)
  #  print(retVal)
    return retVal#broj klasse

def mainNM():
    print ("Begin training:")
    print  (datetime.datetime.now().time())
    # prepare data
    trainingSet = []
    testSet = []
    split = 0.85
    loadDatasetForNM('rezultat.data', split, trainingSet, testSet)
    print ('Train set: ' + repr(len(trainingSet)))
    print ('Test set: ' + repr(len(testSet)))

    trainingSetValues = np.empty((len(trainingSet),3))
    trainingSetValues.astype(float)
    trainingSetLabels = np.empty((len(trainingSet),15))
    #print (trainingSetValues.shape)
    #print (trainingSetLabels.shape)
    testSetValues = np.empty((len(testSet),3))
    testSetValues.astype(float)
    testSetLabels = np.empty((len(testSet),15))

    #praviljenje liste numpy nizova koji su ulazi u nm
    for x in range(len(trainingSet)):
        trainingInstance = np.empty((1,3))
        trainingInstance.astype(float)
        for y in range(3):
            trainingInstance[0][y]=trainingSet[x][y]
        trainingSetValues[x]=trainingInstance
        retVal = createLabels(trainingSet[x][-1])
        trainingSetLabels[x]= retVal

    for x in range(len(testSet)):
        testInstance = np.empty((1,3))
        testInstance.astype(float)
        for y in range(3):
            testInstance[0][y]=testSet[x][y]
        testSetValues[x]=testInstance
        retVal = createLabels(testSet[x][-1])
        testSetLabels[x]=retVal
    """
    print ('test')
    print (testSetValues[0])
    print('train')
    print(trainingSetValues[0])
    """
    # prepare model
    model = Sequential()
    model.add(Dense(len(trainingSetValues), input_dim=3, init='uniform', activation='relu'))
    model.add(Dense(int(len(trainingSetValues)/2), init='uniform', activation='relu'))
    model.add(Dense(15, activation='softmax'))
    model.build()
    # compile model with optimizer
    sgd = SGD(lr=0.01, decay=0.001, momentum=0.9) # isto i kad optimizer bude adam, i isto kad je lr = 0.01 isto i za momentum=0.7
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # training
    training = model.fit(trainingSetValues, trainingSetLabels, epochs=30, batch_size=200, verbose=0)

    scores = model.evaluate(testSetValues, testSetLabels, verbose=0)
    #print ('\n test', scores)
    testAccuracyDecimal = scores[1]
    testAccuracyProcent = getAccuracy(testAccuracyDecimal)
    print("Procentualna tacnost za test skup: ", testAccuracyProcent)
    neka = np.zeros((1, 3), dtype='float')
    neka[0, 0] = 493967.5
    neka[0, 1] = 333.67584228515625
    neka[0, 2] = 695.5736694335938
    #397228.5,351.7366943359375,599.2153930664062,l15nr00
    #493967.5,333.67584228515625,695.5736694335938,l4nr00
    #562185.0,528.0906372070312,755.585693359375
    rezultat = model.predict(testSetValues)
    model.save('model.h5')
mainNM()
print ("End :")
print  (datetime.datetime.now().time())