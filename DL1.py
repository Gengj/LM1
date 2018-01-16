#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
@author: Gengj
@license: (C) Copyright 2013-2017.
@contact: 35285770@qq.com
@software: NONE
@file: DL1.py
@time: 2018/1/9 下午7:30
@desc:
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

node1 = tf.constant(3.0,dtype=tf.float32)
node2 = tf.constant(4.0,dtype=tf.float32)

print(node1)
print(node2)
print(node1 + node2)

sess = tf.Session()
result = sess.run([node1 + node2])
print(result)

node3 = tf.add(node1,node2)
print('node3:',node3)
print("session run node 3 :",sess.run(node3))

