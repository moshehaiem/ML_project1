import csv
import pandas
import numpy as np
import time

class SGDSolver():
    weight = np.random.rand(1, 7)
    bias = 0

    def __init__(self, path):
        
        df = pandas.read_csv(path)
        self.y = np.asarray(df['Chance of Admit '].values)
        self.x = np.asarray(df.drop(['Chance of Admit ', 'Serial No.'], axis=1).values)

        

    def training(self, alpha, lam, nepoch, epsilon):

        
        cur_iter = 1
        a = 0
        l = 0
        mse = float('inf')
        step_size = 10000
        curr_error=0
        alpha1 = int(alpha[0])
        alpha2 = int(alpha[1])
        lam1 = int(lam[0])
        lam2 = int(lam[1])
        for l in np.arange(lam1, lam2, step_size):
            for a in np.arange(alpha1, alpha2, step_size):
                w_gradient = np.random.rand(1, 7) 
                curr_weight = np.random.rand(1, 7) * .004
                curr_bias = 0
                while(cur_iter <= nepoch):
                    for i in range (360):
                        prediction = np.dot(self.x[i], np.transpose(self.weight)) + self.bias
                        w_gradient = w_gradient + (-1/180)*(self.x[i]* (self.y[i] - (prediction))) 
                        b_gradient = (-1/180)*(self.y[i] - prediction) 
                        curr_weight =(1-a*l)*curr_weight - a * w_gradient
                        curr_bias = (1-a*l)*curr_bias - a * b_gradient
                    a /= (nepoch+1)**.5
                    cur_iter +=1
                curr_error = ((self.y - (np.dot(self.x, np.transpose(curr_weight)) + curr_bias))**2).mean(axis=None)
                if curr_error < mse:
                    self.weight = curr_weight
                    self.bias = curr_bias
                    mse = curr_error
                elif curr_error < epsilon:
                    return self.weight, self.bias
        return self.weight, self.bias
    




    def testing(self, testX):
        return np.dot(testX, np.transpose(self.weight)) + self.bias


model = SGDSolver('tests/train.csv')
start = time.time()
model.training([10**-10, 10], [1, 1e10], 300, 0.00001)
end = time.time()
