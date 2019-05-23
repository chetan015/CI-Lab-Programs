import numpy as np

X = np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])

y = np.array([[1],[1],[0]])

def sigmoid (x):
    return 1/(1+np.exp(-x))

def derivatives_sigmoid(x):
    return x*(1-x)

epoch = 5000
lr = 0.1
inputlayer_neurons = X.shape[1]
hiddenlayer_neurons = 3
output_neurons = 1

wh = np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh = np.random.uniform(size=(1,hiddenlayer_neurons))
wout = np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout = np.random.uniform(size=(1,output_neurons))

for i in range(epoch):
    hidden_layer_input  = np.dot(X,wh) + bh
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hiddenlayer_activations,wout) + bout
    output = sigmoid(output_layer_input)
    
    E = y - output
    slope_output_layer = derivatives_sigmoid(output)
    slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
    d_output = E * slope_output_layer
    error_at_hidden_layer = d_output.dot(wout.T)
    d_hiddenlayer = error_at_hidden_layer*slope_hidden_layer
    
    wout += hiddenlayer_activations.T.dot(d_output) *lr
    bout += np.sum(d_output, axis=0,keepdims=True) *lr
    wh += X.T.dot(d_hiddenlayer) *lr
    bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr

print output