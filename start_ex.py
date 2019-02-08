"""
	初步初步尝试:一个简单的线性回归的训练
	2019.2.7
"""



import tensorflow as tf
import numpy as np

x_data = np.float32(np.random.rand(2,100))  
#np.random.rand():生成一个shape为参数的tensor,各element值为[0,1)中随机
#这里生成的为2x100的矩阵

y_data = np.dot([0.100,.0200],x_data) + 0.300
#np.dot,将作为参数的两个矩阵点乘
#矩阵 + 值 == 矩阵中全部elements + 值


b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1,2],-1.0,1.0))
#随训练修改的参数定义为 tf.Variable

y = tf.matmul(W,x_data) + b
#矩阵点乘.......这里很有趣的是,W为tf.Variable,x_data为np.float32,类型居然没报错
#因为这里Class Variable中的矩阵以np.ndarray的形式存储

loss = tf.reduce_mean(tf.square(y - y_data))
#以方差作为误差

optimizer = tf.train.GradientDescentOptimizer(0.5)
#采用梯度下降优化算法,每次步长为0.5
train = optimizer.minimize(loss)
#以loss最小为优化目标

init = tf.global_variables_initializer()
#初始化全部变量
#部分老版书中这里是tf.initialize_all_variable()

sess = tf.Session()
#会话开始

sess.run(init)
#先初始化再开始训练
for step in range(0,201):
	sess.run(train)
	if step %20 == 0:
		print (step,sess.run(W),sess.run(b))
		
