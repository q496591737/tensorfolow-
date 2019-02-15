"""
	通过MNIST学习卷积神经网络
		2019.2.11
"""

#step0.准备
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
#变量初始化:
#tf.Variable()均接受一个initializer实例为参数,tf.global_variables_initializer()会由此初始化这些变量
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	#生成的值服从具有指定平均值和标准偏差的正态分布，如果生成的值大于平均值2个标准偏差的值则丢弃重新选择,默认均值为0
	return tf.Variable(initial)
def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

#卷积层:
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#x为输入矩阵,W为卷积核
#x,W维数均为4,strides决定每一维上的步长
#x的四个维度分别为:[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]
#通道数:比如灰度图通道数是1,RGB图通道数是3
#W的四个维度分别为:[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
#padding = 'SAME'即输出与输入为相同规模,若padding = 'VALID'则zero padding = 0

#池化层
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
#x为输入,ksize为池化窗口的大小,四个值分别对应:[batch, height, width, channels]
#strides为各维度步长
#padding与上面一样

#Step1.载入数据
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
x = tf.placeholder("float",shape = [None,784])
y_ = tf.placeholder("float",shape = [None,10])


#Step2.搭建网络
#输入层
x_image = tf.reshape(x,[-1,28,28,1])
#分别对应[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]

#第一层卷积
W_conv1 = weight_variable([5,5,1,32])
#W的四个维度分别为:[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
#卷积核个数对应着生成的feature_map的数量
b_conv1 = bias_variable([32])
#每个feature_map各元素共享一个bias
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1) #卷积
h_pool1 = max_pool_2x2(h_conv1)                      #池化

#第二层卷积
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#全连接层
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
#舍弃原本的像素间相对位置的信息
#因为经过卷积后图片尺寸不变,经过池化后(由于池化窗口的尺寸),每次图片尺寸/2,因此两次池化后尺寸由28*28缩为7*7
#每幅图64个feature_map,因此输入为7*7*64
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = weight_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
#防止过拟合,以概率屏蔽某些神经元
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#输出层
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)+b_fc2)


#Step3.训练
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step= tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#仍以交叉熵为训练目标,用Adam算法优化
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(1000):
	batch = mnist.train.next_batch(50)
	sess.run(train_step,feed_dict = {x:batch[0],y_:batch[1],keep_prob: 0.5})
	
#Step4.测试
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
print ("test accuracy %g"%sess.run(accuracy,feed_dict = {x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))

