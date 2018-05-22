# -*- coding: utf-8 -*-
"""
Created on Tue May  8 23:05:43 2018

@author: joy820411
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#%%
"""

train_function -> main function to train a linear function

create_function -> create a linear function then return the train_y_list and model

parameter_list -> the coefficient of One-time multiple equation

result : show a image with original train data and new data which got after train

learning rate -> using decay learning rate to avoid divergence

"""


        
#%%
            
def train_function():
    #initialize
    n_observations = 100
    x = tf.placeholder(tf.float32)
    # Reducing power
    parameter_list = [3., 2., 1., 3.]
    
    W = tf.Variable([np.random.uniform(-1, 1) for w in parameter_list], tf.float32)
    xs = np.linspace(-3, 3, n_observations)
    ys, model = create_function(parameter_list, n_observations, xs, x, W)
    y = tf.placeholder(tf.float32)
    loss = tf.reduce_sum(tf.pow(model - y,2))/n_observations
    init_learning_rate = 0.001
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(init_learning_rate, 
                                               global_step=global_step,
                                               decay_steps=100,
                                               decay_rate=0.96)
    
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    for i in range(5000):
#        sess.run(train, feed_dict={x:x_train, y:y_train})
        if i%500 == 0:
            curr_W, curr_loss, curr_learn = sess.run([W, loss, learning_rate], feed_dict={x:xs, y:ys})
            print('step = {}, W = {}, Loss = {}, learning_rate ={}'.format(i, curr_W, curr_loss, curr_learn))
        _, l = sess.run([optimizer, loss], feed_dict={x: xs, y:ys}) 
    curr_W, curr_loss = sess.run([W, loss], feed_dict={x:xs, y:ys})
    
    Y, model = create_function(curr_W, 1000, xs, x, W)
    plt.plot(xs, Y,'r-', xs, ys, 'g^')
    plt.show()
    
    print ('W = {}, loss = {}'.format(curr_W, curr_loss))       
            
            
#%%
    
def create_function(parameter_list, n_observations, x_list, x, W):
    y_list = []
    parameter_len = len(parameter_list)-1
    #create linear model
    for index, value in  enumerate(parameter_list):
        if index == 0:
            model =  tf.multiply(tf.pow(x, index), W[parameter_len-index])
        else:
            model = model + tf.multiply(tf.pow(x, index), W[parameter_len-index])
    #create training data
    for x in x_list:
        y = 0
        for i, para in enumerate(parameter_list):
            y = y + para*np.power(x,len(parameter_list)-i -1)
        
        y_list.append(y)
    
    return y_list, model
    

            
            
            
            
            
            
            