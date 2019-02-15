"""
	Mnist: Hello tensorflow
	2019.2.8
"""
import tensorflow as tf

#Step0.将数据集读入
#但是read_from_sets函数不久后将被淘汰
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
#每一张图片784像素,于是可以用一个784维的向量表示一张图
#one_hot = True 意味着label为一个10维向量,且仅有一维为1,其余为0

#Step1.搭建神经网络

# x 表示图的输入,第一个维度检索图片,第二个维度检索像素点
#由于图片数量不确定,所以第一个维度为None,表示不定
#像素的强度值介于0和1之间,因此类型为float
x = tf.placeholder("float",[None,784])

#用最简单的单层全连接: 输出y = F(XW+b)
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b)
#softmax(X)[i] = exp(X[i]) / sum(exp(X)) 


#Step2.定义loss与优化算法
#先导入正确的输出
y_ = tf.placeholder("float",[None,10])
#这些placeholder将来都要通过run()的feed_dict参数传入

#用交叉熵作为loss
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#tf.log()对各元素单独取log
#矩阵间 * 将对应元素相乘
#tf.reduce_sum() 对全体元素求和

#用梯度下降算法进行优化,学习速率设为0.01,优化目标为最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#Step3.开始训练
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict = {x: batch_xs, y_ : batch_ys})

#Step4.评估模型
#虽然我们用交叉熵来作为训练标准,但是我们并不能用它来做评价标准
#评价标准只有正确与不正确的布尔值以及总体看来的正确率

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
#tf.argmax会返回y在第1维上的最大值,(第0维检索图片,第1维检索one_hot值)
#tf.equal()将两个矩阵中的相应元素比较,并将相应布尔值记在返回矩阵的相应位置

accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
#True => 1.0 Flase => 0.0 然后全体取平均即为正确率

print (sess.run(accuracy,feed_dict = {x:mnist.train.images, y_:mnist.train.labels}))
#这里的run与train_step节点无关,因此不会改变W和b,是纯粹的测试