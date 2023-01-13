import os,cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.testing._private.utils import clear_and_catch_warnings
from scipy.sparse.sputils import isintlike
from sklearn.decomposition import PCA
from numpy import exp, array, random, dot

def read_image(file_path):
    file_path=str(file_path)
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=img.flatten()
    img=np.reshape(img,(ROWS*COLS,1))
    img = img.transpose()
    return img

def prepare_data(i1,i2,i3):
    m = len(i1)+len(i2)+len(i3)
    X = np.zeros((m,PCA_NUM), dtype=np.uint8)
    y = np.zeros((3, m))
    pca = PCA(PCA_NUM)

    temp1 = np.zeros((len(i1),ROWS*COLS), dtype=np.uint8)
    for i, image_file in enumerate(i1):
        temp1[i,:] = read_image(image_file)
        y[0, i] = 1
    
    img_transformed = pca.fit_transform(temp1)

    for i in range(0,len(i1)):
        X[i,:] = img_transformed[i,:]

    temp2 = np.zeros((len(i2),ROWS*COLS), dtype=np.uint8)
    for i, image_file in enumerate(i2):
        temp2[i,:] = read_image(image_file)
        y[1, len(i1)+i] = 1
    
    img_transformed = pca.fit_transform(temp2)

    for i in range(0,len(i2)):
        X[len(i1)+i,:] = img_transformed[i,:]

    temp3 = np.zeros((len(i3),ROWS*COLS), dtype=np.uint8)
    for i, image_file in enumerate(i3):
        temp3[i,:] = read_image(image_file)
        y[2, len(i1)+len(i2)+i] = 1
    
    img_transformed = pca.fit_transform(temp3)

    for i in range(0,len(i3)):
        X[len(i1)+len(i2)+i,:] = img_transformed[i,:]
    return X, y

class NeuralNetwork():
    def __init__(self):
        random.seed(1)

        l2 = 15

        self.synaptic_weights1 = 2 * random.random((3, l2)) -1
        self.synaptic_weights2 = 2 * random.random((l2, 3)) -1
		
    def __sigmoid(self, x):
        return 1/(1+exp(-x))

    def __sigmoid_derivative(self, x):
        return x*(1-x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        
        for iteration in range(number_of_training_iterations):

            idx=int(np.random.rand()*len(training_set_inputs))
            T_input=np.reshape(training_set_inputs[idx,:],(1,3))
            T_output=np.reshape(training_set_outputs[idx,:],(1,3))

            a2 = self.__sigmoid(dot(T_input, self.synaptic_weights1))

            output = self.__sigmoid(dot(a2, self.synaptic_weights2))

            del3 = (T_output - output)*self.__sigmoid_derivative(output)

            del2 = dot(self.synaptic_weights2, del3.T)*(self.__sigmoid_derivative(a2).T)

            adjustment2 = dot(a2.T, del3)
            adjustment1 = dot(T_input.T, del2.T)

            self.synaptic_weights1 += adjustment1
            self.synaptic_weights2 += adjustment2

    def forward_pass(self, inputs):

	    a2 = self.__sigmoid(dot(inputs, self.synaptic_weights1))
	    output = self.__sigmoid(dot(a2, self.synaptic_weights2)) 
	    return output

if __name__ == "__main__":

    ROWS = 32
    COLS = 32
    PCA_NUM = 2

    TRAIN_DIR_1 = "Data_train\\Carambula\\"
    TRAIN_DIR_2 = "Data_train\\Lychee\\"
    TRAIN_DIR_3 = "Data_train\\Pear\\"

    TEST_DIR_1 = "Data_test\\Carambula\\"
    TEST_DIR_2 = "Data_test\\Lychee\\"
    TEST_DIR_3 = "Data_test\\Pear\\"

    train_images_1 = [TRAIN_DIR_1+i for i in os.listdir(TRAIN_DIR_1)]
    train_images_2 = [TRAIN_DIR_2+i for i in os.listdir(TRAIN_DIR_2)]
    train_images_3 = [TRAIN_DIR_3+i for i in os.listdir(TRAIN_DIR_3)]

    test_images_1 = [TEST_DIR_1+i for i in os.listdir(TEST_DIR_1)]
    test_images_2 = [TEST_DIR_2+i for i in os.listdir(TEST_DIR_2)]
    test_images_3 = [TEST_DIR_3+i for i in os.listdir(TEST_DIR_3)]

    neural_network = NeuralNetwork()

    train_set_x, train_set_y = prepare_data(train_images_1,train_images_2,train_images_3)
    test_set_x, test_set_y = prepare_data(test_images_1,test_images_2,test_images_3)


    Bias = np.ones((len(train_set_x),1))
    training_set_inputs = np.concatenate((train_set_x, Bias), axis = 1)
    training_set_outputs = train_set_y.T

    Bias_t = np.ones((len(test_set_x),1))
    test_set_inputs = np.concatenate((test_set_x, Bias_t), axis = 1)
    test_set_outputs = test_set_y.T

    neural_network.train(training_set_inputs, training_set_outputs, 100000)
    
    hit = 0.0
    count = 0.0
    
    for i in range(len(test_set_x)):
        count += 1.0
        XX=neural_network.forward_pass(test_set_inputs[i,:])
        YY=neural_network.forward_pass(test_set_outputs[i,:])
        if np.argmax(XX) == np.argmax(YY):
            hit += 1.0

    accuracy = round(hit/count,3)
    print("Accuracy:",accuracy*100,"%")