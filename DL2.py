#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
@author: Gengj
@license: (C) Copyright 2013-2017.
@contact: 35285770@qq.com
@software: NONE
@file: DL2.py
@time: 2018/1/12 下午4:02
@desc:反向传播算法神经网络解决二分类
'''
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()

    sess.run(init_op)

    print(sess.run(w1))
    # [[-0.81131822  1.48459876  0.06532937]
    #  [-2.4427042   0.0992484   0.59122431]]
    print(sess.run(w2))
    # [[-0.81131822]
    #  [1.48459876]
    #  [0.06532937]]

    STEPS = 5000
    # 训练次数5000次

    # batch_size是训练使用的每组数据大小----8
    # data_size是训练使用的全部数据大小-----128
    # 所以，一共分为16个batch

    for i in range(STEPS):
        # 假设现在运行到i = 17，即训练的第17次
        # 那么，X数组的开始行 = （17 * 8）% 128  = 1
        # 结束行 = min（1 + 8，128）= 9
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)


        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})


        if i % 1000 == 0:

            total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X,y_:Y})
            print("After %d training step(s), cross entropy on all data is %g"%(i,total_cross_entropy))

            # After 0 training step(s), cross entropy on all data is 0.0674925
            # After 1000 training step(s), cross entropy on all data is 0.0163385
            # After 2000 training step(s), cross entropy on all data is 0.00907547
            # After 3000 training step(s), cross entropy on all data is 0.00714436
            # After 4000 training step(s), cross entropy on all data is 0.00578471
            # 从上可见交叉熵逐渐变小

    print(sess.run(w1))
    # [[-1.96182752  2.58235407  1.68203771]
    #  [-3.46817183  1.06982315  2.11788988]]
    print(sess.run(w2))
    # [[-1.82471502]
    #  [2.68546653]
    #  [1.41819501]]



