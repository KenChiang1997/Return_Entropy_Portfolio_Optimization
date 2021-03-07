import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def objective_function(w):
    w_tp=w.transpose()
    entropy_part=np.dot(w_tp,np.log(w))
    return entropy_part 

def equality_constraint_1(w):
    w_tp=w.transpose()
    return 1-np.dot(w_tp,ones)

def equality_constraint_2(w):
    w_tp=w.transpose()
    return u-np.dot(w_tp,returns)

def inequality_constraint(w):
    w_tp=w.transpose()
    return 0.05950352128795654-np.dot(np.dot(w_tp,cov),w)


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
constraint_3={'type':'ineq','fun':inequality_constraint}
constraint=[constraint_1,constraint_2,constraint_3]
result=minimize(objective_function,w0,method='SLSQP',bounds=bounds,constraints=constraint)
w=result['x']

print("------------------objective fnc 是 entropy part----------------------------")
print("variance 主觀設定不能大於 0.05950352128795654")
for i in range(len(w)):
    print('weights'+ str(i+1) + ': ',"{:.19f}".format(float(w[i])))

w_tp=w.transpose()
portfolio_variance=np.dot(np.dot(w_tp,cov),w)
portfolio_return=np.dot(w_tp,returns)
portfolio_entropy=-1*np.dot(w_tp,np.log(w))
print("portfolio risk : ",np.sqrt(portfolio_variance))
print("portfolio return : ",portfolio_return)
print("portfolio entropy : ",portfolio_entropy)

trial_numbers=20000
all_weights=np.zeros((trial_numbers,len(w)))
all_risk=np.zeros((trial_numbers))
all_returns=np.zeros((trial_numbers))
all_entropy=np.zeros((trial_numbers))

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
    #entropy
    entropy_part=np.dot(weights.transpose(),np.log(weights))
    all_entropy[i]=-1*entropy_part


plt.figure()
plt.title("mean-variance-approach modified by entropy")
plt.scatter(all_risk,all_returns,marker="o",alpha=0.2,color='b',label="simulation")
plt.scatter(np.sqrt(portfolio_variance),portfolio_return,marker="x",color="r",label="REPO")
plt.xlabel("risk(sigma)")
plt.ylabel("return")
plt.legend()
plt.grid()




plt.figure()
plt.title("entropy/return")
plt.scatter(all_returns,all_entropy,color="b",alpha=0.2)
plt.scatter(portfolio_return,portfolio_entropy,color="r",label="max_entropy")
plt.xlabel("return")
plt.ylabel("entropy")
plt.grid()
plt.legend()
plt.show()


