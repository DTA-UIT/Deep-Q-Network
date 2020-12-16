from d2l import tensorflow as d2l
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

#define our model a linear regression model
def linear_regression(x, w, b):
    return tf.matmul(x, w) + b

#Self-create a dataset for LR-model
def generate_dataset(w, b, N):
    x = tf.zeros((N, w.shape[0]))
    #take the shape of w and generate a N-by-w.shape zero matrix 
    
    x += tf.random.normal(shape = x.shape)
    #push Gaussian-distributionally generated values into matrix x 

    y = tf.matmul(x, tf.reshape(w, (-1, 1))) + b
    #create a 2-by-1 matrix y by multiplying x to w and adding to b

    y += tf.random.normal(shape = y.shape, stddev = 0.1)
    #make noise to data by adding y to a normal distribution with a standard deviation of 0.01

    y = tf.reshape(y, (-1, 1))
    #reshape y

    return x, y

true_w = tf.constant([2, -3.4])
true_b = 4.2

#generate input
#we called our model has feature w and label y
features, labels = generate_dataset(true_w, true_b, 1000)

d2l.set_figsize()
d2l.plt.scatter(d2l.numpy(features[:, 1]), d2l.numpy(labels), 1)

def read_data(batch_size, features, labels):
    N = len(features)
    #save number of features

    indices = list(range(N))
    #create a list of N features we take from features dataset 

    random.shuffle(indices)
    #randomize data to ensure conditions in machine learning

    for i in range(0, N, batch_size):
    #loop from 0 to N with a step of batch_size

        j = tf.constant(indices[i: min(i + batch_size, N)])
        #just take a certain number of features to run at a time
        #in order to reduce model-training time

        yield tf.gather(features, j), tf.gather(labels, j)
        #yield is like return but return generators instead

w = tf.Variable(tf.random.normal(shape = (2, 1), mean = 0, stddev = 0.01), trainable = True)
#w is initialized by using a normal distribution with the mean of 0 and 
#the standard deviation of 0.01

b = tf.Variable(tf.zeros(1), trainable = True)
#initialize b with zero value

def loss_function(y_hat, y):
    return (y_hat - tf.reshape(y, y_hat.shape))**2/2


def stochastic_gradient_descent(parameters, gradient, learning_rate, batch_size):
    for param, grad in zip(parameters, gradient):
        #zip function is used to take iterator from variable "parameters" and 
        #"gradient" as iterators and loop 

        param.assign_sub(learning_rate * grad / batch_size)
        #update parameters with learning_rate * gradient / batch_size

learning_rate = 0.03
num_epochs = 5
batch_size = 10

for epoch in range(num_epochs):
    for x, y in read_data(batch_size, features, labels):
        #tf.GradientTape is used to modify bias according to the direction
        #of the gradient in order to minimize the parameters for the model 
        with tf.GradientTape() as g:
            linreg = linear_regression(x, w, b)
            loss = loss_function(linreg, y)
            
        dw, db = g.gradient(loss, [w, b])
        
        stochastic_gradient_descent([w, b], [dw, db], learning_rate, batch_size)
        #update parameters in the same direction as the gradient to reduce loss

    train_l = loss_function(linear_regression(features, w, b), labels)
    #estimate the loss of model throughout sgd 
    
    print('Epoch ' + str(epoch + 1) + ': ' + str(float(tf.reduce_mean(train_l))))

print(f'w error: {true_w - tf.reshape(w, true_w.shape)}')
print(f'b error: {true_b - b}')