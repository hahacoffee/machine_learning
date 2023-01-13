'''
NTHU EE Machine Learning HW2
Author: 黃俊維
Student ID: 109033591
'''
import numpy as np
import pandas as pd
import math
import scipy.stats
import argparse


# do not change the name of this function
def BLR(train_data, test_data_feature, O1=2, O2=2):  # remember to set best choice O1 and O2 as default
    '''
    output: ndarray with size (length of test_data, )
    '''
    O1=2
    O2=2

    Beta = 100
    
    P=O1*O2+2
    W=np.zeros([P,1])[:,0]
    W=np.reshape(W,(P,1))
    M0=np.zeros([P,1])[:,0]
    M0=np.reshape(M0,(P,1))
    S0=np.diag(np.ones(P))
    
    X1=train_data[:, 0]
    X1=np.reshape(X1,(np.size(X1,0),1))
    X2=train_data[:, 1]
    X2=np.reshape(X2,(np.size(X2,0),1))
    X3=train_data[:, 2]
    X3=np.reshape(X3,(np.size(X3,0),1))
    T=train_data[:, 3]
    T=np.reshape(T,(np.size(T,0),1))
    
    X11=test_data_feature[:, 0]
    X11=np.reshape(X11,(np.size(X11,0),1))
    X22=test_data_feature[:, 1]
    X22=np.reshape(X22,(np.size(X22,0),1))
    X33=test_data_feature[:, 2]
    X33=np.reshape(X33,(np.size(X33,0),1))
    
    S1=(X1.max(0)-X1.min(0))/(O1-1)
    S1=np.reshape(S1,(1,1))
    S2=(X2.max(0)-X2.min(0))/(O2-1)
    S2=np.reshape(S2,(1,1))
    
    S11=(X11.max(0)-X11.min(0))/(O1-1)
    S11=np.reshape(S11,(1,1))
    S22=(X22.max(0)-X22.min(0))/(O2-1)
    S22=np.reshape(S22,(1,1))
    
    P=O1*O2+2
    
    Phi=np.zeros([np.size(X1,0),P])
    PhiP=np.zeros([np.size(X11,0),P])
    
    for i in range(1,O1+1):
        for j in range(1,O2+1):
            U1=S1*(i-1)+X1.min(0)
            U2=S2*(j-1)+X2.min(0)
            
            U11=S11*(i-1)+X11.min(0)
            U22=S22*(j-1)+X22.min(0)
            
            k=O2*(i-1)+j
            
            for n1 in range(0,np.size(Phi,0)):
                Phi[n1,k]=math.exp(-(((X1[n1,0]-U1)**2)/(2*(S1**2)))-(((X2[n1,0]-U2)**2)/(2*(S2**2))))
            
            for n2 in range(0,np.size(PhiP,0)):
                PhiP[n2,k]=math.exp(-(((X11[n2,0]-U11)**2)/(2*(S11**2)))-(((X22[n2,0]-U22)**2)/(2*(S22**2))))
                
    Phi[:,P-2]=X3[:,0]
    Phi[:,P-1]=np.ones([np.size(X1,0),1])[:,0]
    
    PhiP[:,P-2]=X33[:,0]
    PhiP[:,P-1]=np.ones([np.size(X11,0),1])[:,0]
    
    
    Sn=np.linalg.pinv((np.linalg.pinv(S0)+Beta*np.dot(Phi.transpose(),Phi)))
    Mn=np.dot(Sn,(np.dot(np.linalg.pinv(S0),M0))+Beta*np.dot(Phi.transpose(),T))
    

    W=scipy.stats.multivariate_normal.rvs(Mn.flatten(),Sn)    
    
    y_BLRprediction = np.dot(PhiP,W)
    return y_BLRprediction 


# do not change the name of this function
def MLR(train_data, test_data_feature, O1=2, O2=2):  # remember to set best choice O1 and O2 as default
    '''
    output: ndarray with size (length of test_data, )
    '''
    O1=2
    O2=2

    X1=train_data[:, 0]
    X1=np.reshape(X1,(np.size(X1,0),1))
    X2=train_data[:, 1]
    X2=np.reshape(X2,(np.size(X2,0),1))
    X3=train_data[:, 2]
    X3=np.reshape(X3,(np.size(X3,0),1))
    T=train_data[:, 3]
    T=np.reshape(T,(np.size(T,0),1))
    
    X11=test_data_feature[:, 0]
    X11=np.reshape(X11,(np.size(X11,0),1))
    X22=test_data_feature[:, 1]
    X22=np.reshape(X22,(np.size(X22,0),1))
    X33=test_data_feature[:, 2]
    X33=np.reshape(X33,(np.size(X33,0),1))
    
    S1=(X1.max(0)-X1.min(0))/(O1-1)
    S1=np.reshape(S1,(1,1))
    S2=(X2.max(0)-X2.min(0))/(O2-1)
    S2=np.reshape(S2,(1,1))
    
    S11=(X11.max(0)-X11.min(0))/(O1-1)
    S11=np.reshape(S11,(1,1))
    S22=(X22.max(0)-X22.min(0))/(O2-1)
    S22=np.reshape(S22,(1,1))
    
    P=O1*O2+2
    
    Phi=np.zeros([np.size(X1,0),P])
    PhiP=np.zeros([np.size(X11,0),P])
    
    for i in range(1,O1+1):
        for j in range(1,O2+1):
            U1=S1*(i-1)+X1.min(0)
            U2=S2*(j-1)+X2.min(0)
            
            
            U11=S11*(i-1)+X11.min(0)
            U22=S22*(j-1)+X22.min(0)
            
            k=O2*(i-1)+j
            #print(k)
            for n1 in range(0,np.size(Phi,0)):
                Phi[n1,k]=math.exp(-(((X1[n1,0]-U1)**2)/(2*(S1**2)))-(((X2[n1,0]-U2)**2)/(2*(S2**2))))
            
            for n2 in range(0,np.size(PhiP,0)):
                PhiP[n2,k]=math.exp(-(((X11[n2,0]-U11)**2)/(2*(S11**2)))-(((X22[n2,0]-U22)**2)/(2*(S22**2))))
                
    Phi[:,P-2]=X3[:,0]
    Phi[:,P-1]=np.ones([np.size(X1,0),1])[:,0]
    
    PhiP[:,P-2]=X33[:,0]
    PhiP[:,P-1]=np.ones([np.size(X11,0),1])[:,0]
    
    W=np.dot(np.dot(np.linalg.pinv(np.dot(Phi.transpose(),Phi)),Phi.transpose()),T)
    y_MLLSprediction = np.dot(PhiP,W).flatten()
    
    return y_MLLSprediction 


def CalMSE(data, prediction):

    squared_error = (data - prediction) ** 2
    sum_squared_error = np.sum(squared_error)
    mean__squared_error = sum_squared_error/prediction.shape[0]

    return mean__squared_error


def main():
    global data_train,data_test,data_test_feature,data_test_label,predict_MLR
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-O1', '--O_1', type=int, default=5)
    parser.add_argument('-O2', '--O_2', type=int, default=5)
    args = parser.parse_args()
    O_1 = args.O_1
    O_2 = args.O_2
    
    data_train = pd.read_csv('Training_set.csv', header=None).to_numpy()
    data_test = pd.read_csv('Validation_set.csv', header=None).to_numpy()
    data_test_feature = data_test[:, :3]
    data_test_label = data_test[:, 3]
    
    
    
    predict_BLR = BLR(data_train, data_test_feature, O1=O_1, O2=O_2)
    predict_MLR = MLR(data_train, data_test_feature, O1=O_1, O2=O_2)

    print('MSE of BLR = {e1}, MSE of MLR= {e2}.'.format(e1=CalMSE(predict_BLR, data_test_label), e2=CalMSE(predict_MLR, data_test_label)))


if __name__ == '__main__':
    main()

