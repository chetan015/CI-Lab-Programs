import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

def cost_func(beta, X, y): 
    ''' 
    cost function, J 
    '''
    log_func_v = logistic_func(beta, X) 
    y = np.squeeze(y) 
    step1 = y * np.log(log_func_v) 
    step2 = (1 - y) * np.log(1 - log_func_v) 
    final = -step1 - step2 
    return np.mean(final) 

def log_gradient(beta,x,y):
    first_calc = logistic_func(beta,x) - y.reshape(x.shape[0],-1)
    final_calc = np.dot(first_calc.T,x)
    return final_calc

def logistic_func(beta,x):
    z = np.dot(x,beta.T)
    return 1.0/(1+np.exp(-z))

def cost_func(beta,x,y):
    log_func_v = logistic_func(beta,x)
    y = np.squeeze(y)
    step1 = y*np.log(log_func_v)
    step2 = (1-y)*np.log(1-log_func_v)
    final = -step1 - step2
    return np.mean(final)

    
def grad_desc(x,y,beta,lr=0.01,converge_change=0.001):
    cost = cost_func(beta,x,y)
    change_cost = 1
    num_iter = 1
    while(change_cost>converge_change):
        old_cost = cost
        beta = beta - (lr*log_gradient(beta,x,y))
        cost = cost_func(beta,x,y)
        change_cost = old_cost - cost
        num_iter += 1
    return beta, num_iter

def pred_values(beta, X): 
    ''' 
    function to predict labels 
    '''
    pred_prob = logistic_func(beta, X) 
    pred_value = np.where(pred_prob >= .5, 1, 0) 
    return np.squeeze(pred_value) 
  
  
def plot_reg(x, y, beta): 
    ''' 
    function to plot decision boundary 
    '''
    # labelled observations 
    x_0 = x[np.where(y == 0.0)] 
    x_1 = x[np.where(y == 1.0)] 
      
    # plotting points with diff color for diff label 
    plt.scatter([x_0[:, 1]], [x_0[:, 2]], c='b', label='Iris_setosa') 
    plt.scatter([x_1[:, 1]], [x_1[:, 2]], c='r', label='y = Iris_versicolor') 
      
    # plotting decision boundary 
    x1 = np.arange(0, 1, 0.1) 
    x2 = -(beta[0,0] + beta[0,1]*x1)/beta[0,2] 
    plt.plot(x1, x2, c='k', label='reg line') 
  
    plt.xlabel('Sepal Length') 
    plt.ylabel('Sepal Width') 
    plt.legend() 
    plt.show() 

def main():
    df = pd.read_csv("iris.csv")
    df = df.head(100)
    df = df.replace('Iris-setosa',0)
    df = df.replace('Iris-versicolor',1)
    df = (df-df.min())/(df.max()-df.min())
    data = np.array(df)
    x = data[:,:2]
    y = data[:,-1]
    x = np.append(np.ones([100,1]),x,axis=1)
    beta = np.matrix(np.zeros(x.shape[1]))
    beta, num_iter = grad_desc(x, y, beta) 
    print("Estimated regression coefficients:", beta) 
    print("No. of iterations:", num_iter)
    y_pred = pred_values(beta, x) 
          
    # number of correctly predicted labels 
    print("Correctly predicted labels:", np.sum(y == y_pred)) 
      
    # plotting regression line 
    plot_reg(x, y, beta)     
                  
if __name__ == "__main__":
    main()