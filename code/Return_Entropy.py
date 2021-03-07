import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class Return_Entropy_Portfolio_Optimization():
    def __init__(self,df,u,var):
        ## 主觀設定 expected portfolio return and the tolerate portfolio variance
        df=df.pct_change()
        df=df.dropna()
        self.initial_weight = np.ones((df.shape[1],1))
        self.bounds = [(0,1000),(0,1000),(0,1000),(0,1000),(0,1000)]
        self.mean = df.mean()
        self.cov = df.cov() * 252
        self.returns= df.mean().values*252
        self. u = u
        self. var = var
    def objective_function(self,w):
        w_tp=w.transpose()
        return np.dot(np.dot(w_tp,self.cov),w)
    def equality_constraint_1(self,w):
        w_tp=w.transpose()
        return 1-np.dot(w_tp,self.initial_weight)
    def equality_constraint_2(self,w):
        w_tp=w.transpose()
        return self.u-np.dot(w_tp,self.returns)
    def inequality_constraint(self,w):
        w_tp= w.transpose()
        return self.var -np.dot(np.dot(w_tp,self.cov),w)
    def optimization(self):
        constraint_1={'type': 'eq','fun':  self.equality_constraint_1}
        constraint_2={'type': 'eq','fun':  self.equality_constraint_2}
        constraint_3={'type':'ineq','fun': self.inequality_constraint}
        constraint=[constraint_1,constraint_2,constraint_3]
        result=minimize(self.objective_function,self.initial_weight,method='SLSQP',bounds=self.bounds,constraints=constraint)
        self.final_weight = result['x']
        return result['x']
    def print_result(self):
        for i in range(len(self.final_weight)):
            print('weights'+ str(i+1) + ': ',"{:.19f}".format(float(self.final_weight[i])))
        return self.final_weight




