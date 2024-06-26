#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
from sklearn.utils import shuffle
from sklearn.preprocessing import minmax_scale
import time

def read_data(filename):
    df = pd.read_csv(filename)
    df = shuffle(df)
    label = np.array(df.iloc[:,0])
    feature = np.array(df.iloc[:,1:])
    onehot_label = np.zeros([label.shape[0],class_num])
    for i in range(len(label)):
        tmp_label = label[i]
        count_each_class[int(tmp_label-1)] +=1
        onehot_label[i][int(tmp_label-1)] =1
    label = label.reshape(-1,1)
    return feature, onehot_label, label

def normalize_data(data):
    data = minmax_scale(data,axis=0)
    return data

def get_mean(data):
    array1=np.array(data)
    means=np.mean(array1,axis=0) # 每列平均值
    return means

def get_specific_class(C):
    result=[]
    for i in range(len(data_X)): # len(data_X):行数
        if data_Y[i][C-1]==1:
            result.append(data_X[i])
    return result

def next_batch(data_X,data_Y,i):
    batch=i% n_batch
    batch_xs = data_X[batch * mb_size:(batch + 1) * mb_size]
    batch_ys = data_Y[batch * mb_size:(batch + 1) * mb_size]
    return batch_xs,batch_ys

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

class_num=2
count_each_class=[0]*class_num

win=79
input_data = "../S-nitrosylation/train_test/Nitro_train_pcc07.csv"
data_X, data_Y, class_label=read_data(input_data)

loss_file_name = 'cwgan_loss/cwgan_loss_'+'pcc07'+'.csv'
time_cost_file_name = 'cwgan_loss/cwgan_time_'+'pcc07'+'.txt'
res_file_name = 'cwgan_augmented_data/cwgan_augmented_data_'+'pcc07'+'.csv'
data_X=normalize_data(data_X)
iter_num = 26000
mb_size = 64
Z_dim = 100
h_dim = 128
X_dim =len(data_X[1])
y_dim = len(data_Y[1])

n_batch = len(data_X) // mb_size

""" Discriminator Net model """
X = tf.placeholder(tf.float32, shape=[None, X_dim])
y = tf.placeholder(tf.float32, shape=[None, y_dim])

D_W1 = tf.Variable(xavier_init([X_dim + y_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
D_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W3 = tf.Variable(xavier_init([h_dim, 1]))
D_b3 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]


""" Generator Net model """
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

G_W1 = tf.Variable(xavier_init([Z_dim + y_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

G_W3 = tf.Variable(xavier_init([h_dim, X_dim]))
G_b3 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]


def generator(z, y):
    inputs = tf.concat(axis=1, values=[z, y])
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    G_log_prob = tf.matmul(G_h2, G_W3) + G_b3
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob

def discriminator(x, y):
    inputs = tf.concat(axis=1, values=[x, y])
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3

    return D_logit

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

G_sample = generator(Z, y)
D_real = discriminator(X, y)
D_fake = discriminator(G_sample, y)

D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
G_loss = -tf.reduce_mean(D_fake)

D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-5)
            .minimize(-D_loss, var_list=theta_D))
G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-5)
            .minimize(G_loss, var_list=theta_G))
clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

start_time = time.time()
loss_result=[]
for it in range(iter_num):
    X_mb, y_mb = next_batch(data_X,data_Y,it)
    Z_sample = sample_Z(mb_size, Z_dim)
    _, D_loss_curr,_ = sess.run([D_solver, D_loss, clip_D], feed_dict={X: X_mb, Z: Z_sample, y:y_mb})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: Z_sample, y:y_mb})

    if (it+1) % 200 == 0:
        print('Iter: {}'.format(it+1))
        print('D_loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))

        loss_result.append([it+1,D_loss_curr,G_loss_curr])
pd.DataFrame(loss_result,columns=['iterations','D_loss','G_loss']).to_csv(loss_file_name,header = True, index=None)

res_data = np.concatenate((data_X, class_label),axis=1)

for i in range(class_num):
    now_class = i+1
    sample_num = np.max(count_each_class)-count_each_class[now_class-1]
    if sample_num == 0:
        continue
    Z_sample = sample_Z(sample_num, Z_dim)
    y_sample = np.zeros(shape=[sample_num, y_dim])
    y_sample[:, 0] = 1
    samples = sess.run(G_sample, feed_dict={Z: Z_sample, y: y_sample})
    tmp_label = np.ones(sample_num)
    tmp_label = tmp_label.reshape(-1,1)
    tmp_data = np.concatenate((samples,tmp_label),axis=1)
    res_data = np.concatenate((res_data,tmp_data),axis=0)

res_data = shuffle(res_data)
pd.DataFrame(res_data).to_csv(res_file_name,index=None, header=True)
end_time = time.time()
cost = end_time-start_time
f = open(time_cost_file_name,'w')
f.write('time cost: '+str(cost)+' s')
f.close()
