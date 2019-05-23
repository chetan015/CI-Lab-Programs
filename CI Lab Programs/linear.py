import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def estimate_coef(x,y):
    n = np.size(x)
    m_x, m_y = np.mean(x), np.mean(y)
    SS_xy = np.sum(x*y - n*m_x*m_y)
    SS_xx = np.sum(x*x - n*m_x*m_x)
    
    b_1 = SS_xy/SS_xx
    b_0 = m_y - b_1*m_x
    return (b_0,b_1)
def plot_regression_line(x,y,b):
    plt.scatter(x,y,c = 'm',marker="o", s = 10)
    y_pred = b[0] + b[1]*x
    plt.plot(x,y_pred,c = 'b')    
    plt.xlabel('Area')
    plt.ylabel('Cost')
    plt.show()
def predict(x,y,b):
    h = b[0] + b[1]*x
    print("Plot_size Predicted_Price Actual_Price Error")
    for i in range(x.size):
        print(str(x[i])+"\t"+str(h[i])+"\t"+str(y[i])+"\t"+str(y[i]-h[i]))

def main():
    houseData = pd.read_csv("Housing.csv")
    lotsize = houseData.lotsize
    price = houseData.price
    train_x = np.array(lotsize.head(500))
    train_y = np.array(price.head(500))
    test_x = np.array(lotsize.tail(46))
    test_y = np.array(price.tail(46))
    b = estimate_coef(train_x,train_y)
    print("Estimated coefficients:\nb_0="+str(b[0])+"\nb_1="+str(b[1]))
    plot_regression_line(train_x,train_y,b)
    predict(test_x,test_y,b)
if __name__ == "__main__":
    main()