import numpy as np

x = np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])
y = np.array([[1],[1],[0]])

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

epoch = 5000
lr = 0.1
inputlayer_neurons = x.shape[1]
hiddenlayer_neurons = 3
outputlayer_neurons = 1

wh = np.random.uniform(size = (inputlayer_neurons,hiddenlayer_neurons))
bh = np.random.uniform(size = (1,hiddenlayer_neurons))
wout = np.random.uniform(size=(hiddenlayer_neurons,outputlayer_neurons))
bout = np.random.uniform(size=(1,outputlayer_neurons))
a2 = []
for i in range(epoch):
    z1 = np.dot(x,wh)+bh
    a1 = sigmoid(z1)
    
    z2 = np.dot(a1,wout)+bout
    a2 = sigmoid(z2)
    
    output_error = y - a2
    d_output = output_error * sigmoid_derivative(a2)
    
    wout += a1.dot(d_output)*lr
    bout+= np.sum(d_output,axis=0,keepdims=True)*lr
    
    
    hidden_error = d_output.dot(wout.T)
    d_hidden = hidden_error*sigmoid_derivative(a1)
    wh += x.T.dot(d_hidden)*lr
    bh +=np.sum(d_hidden,axis=0,keepdims=True)*lr
print a2