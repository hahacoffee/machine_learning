import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.callbacks import History
from sklearn.preprocessing import StandardScaler

with open('train.csv', newline='') as csvfile1 , open('test.csv', newline='') as csvfile2:
    CSVDATA1 = csv.reader(csvfile1)
    CSVDATA2 = csv.reader(csvfile2)
    traindata = list(CSVDATA1)
    testdata = list(CSVDATA2) 

test_target = np.zeros((54,3))
test_feature = np.zeros((54,13))
train_target = np.zeros((124,3))
train_feature = np.zeros((124,13))

for i in range(0,54) :
    idx = testdata[i][0]
    if idx=='1':
        test_target[i][0] = 1
    elif idx=='2':
        test_target[i][1] = 1
    elif idx=='3':
        test_target[i][2] = 1
    for j in range(1,13):
        test_feature[i][j-1]=testdata[i][j]

for i in range(0,124) :
    idx = traindata[i][0]
    if idx=='1':
        train_target[i][0] = 1
    elif idx=='2':
        train_target[i][1] = 1
    elif idx=='3':
        train_target[i][2] = 1
    for j in range(1,13):
        train_feature[i][j-1]=traindata[i][j]

scaler = StandardScaler()
scaler = scaler.fit(train_feature)
train_feature[:] = scaler.transform(train_feature)

scaler = scaler.fit(test_feature)
test_feature[:] = scaler.transform(test_feature)

feature_num = 13
class_num = 3
train_num = 124
test_num = 54

input_layer = feature_num
hidden_layer_1 = 20
hidden_layer_2 = 20
output_layer = class_num

model = Sequential()
model.add(Dense(hidden_layer_1,activation = 'relu',input_shape = (feature_num,)))
model.add(Dense(hidden_layer_2,activation = 'relu'))
model.add(Dense(class_num,activation = 'softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = History()
model.fit(train_feature, train_target, epochs=30, callbacks=[history])

predictions = model.predict(test_feature)

print('\nPosterior probabilities of test data: \n')
print(predictions) # Posterior probabilities of test data
print('\n')

results = model.evaluate(test_feature, test_target)

print('\nEvaluate on test data \n\n(loss), (accuracy) :\n{}'.format(results))


dataframe = pd.DataFrame(np.argmax(predictions,1), columns=['Prediction'])
dataframe['Target'] = np.argmax(test_target, 1)
dataframe['Hit'] = np.equal(dataframe.Target, dataframe.Prediction)
print('\n\nPrinting results :\n\n', dataframe)

# Plot the visualized result of testing data

plt.figure(1)

blue_patch = mpatches.Patch(color='blue', label='Train Accuracy [Maximize]')
plt.legend(handles=[blue_patch])

plt.plot(history.history['accuracy'], color='blue')
plt.ylabel('score')

plt.figure(2)

green_patch = mpatches.Patch(color='green', label='Avg Loss [Minimize]')
plt.legend(handles=[green_patch])

plt.plot(history.history['loss'], color='green')

plt.xlabel('epochs')
plt.ylabel('score')

plt.show()