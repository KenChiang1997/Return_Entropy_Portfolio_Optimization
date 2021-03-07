import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def objective_function(w):
    w_tp=w.transpose()
    return np.dot(np.dot(w_tp,cov),w)

def equality_constraint_1(w):
    w_tp=w.transpose()
    return 1-np.dot(w_tp,ones)

def equality_constraint_2(w):
    w_tp=w.transpose()
    return u-np.dot(w_tp,returns)

path=r'/Users/chen-lichiang/Desktop/2020HW/數金作業/return_all.xlsx'
df=pd.read_excel(path,index_col='Date')
df=df.pct_change()
df=df.dropna()
mean=df.mean()
cov=df.cov()
returns=mean.values*252
cov=cov.values*252
ones=np.ones((cov.shape[0],1))
u=0.01
bounds=[(0,1000),(0,1000),(0,1000),(0,1000),(0,1000)]
w0=[1,1,1,1,1]
constraint_1={'type': 'eq','fun':equality_constraint_1}
constraint_2={'type': 'eq','fun':equality_constraint_2}
constraint=[constraint_1,constraint_2]
result=minimize(objective_function,w0,method='SLSQP',bounds=bounds,constraints=constraint)
w=result['x']

print("------------------mean-varianve-portfolio-optimization-approach----------------------------")
for i in range(len(w)):
    print('weights'+ str(i+1) + ': ',"{:.19f}".format(float(w[i])))
w_tp=w.transpose()
portfolio_variance=np.dot(np.dot(w_tp,cov),w)
portfolio_return=np.dot(w_tp,returns)
print("portfolio risk : " ,np.sqrt(portfolio_variance))
print("portfolio return : ",portfolio_return)



trial_numbers=20000
all_weights=np.zeros((trial_numbers,len(w)))
all_risk=np.zeros((trial_numbers))
all_returns=np.zeros((trial_numbers))
for i in range(trial_numbers):
    #weights
    weights=np.random.uniform(size=len(w))
    weights=weights/np.sum(weights)
    all_weights[i]=weights
    #risk
    variance=np.dot(np.dot(weights.T,cov),weights)
    risk=np.sqrt(variance)
    all_risk[i]=risk
    #returns
    port_ret=np.dot(weights,returns)
    all_returns[i]=port_ret

plt.figure()
plt.title("mean-variance-approach")
plt.scatter(all_risk,all_returns,marker="o",alpha=0.2,color='b',label="simulation")
plt.scatter(np.sqrt(portfolio_variance),portfolio_return,marker="x",color="r",label="MVPO")
plt.xlabel("risk(sigma)")
plt.ylabel("return")
plt.legend()
plt.grid()
plt.show()