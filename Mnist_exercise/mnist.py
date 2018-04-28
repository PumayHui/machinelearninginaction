import input_data
import tensorflow as tf

# 为了不在建立模型的时候反复做初始化操作，定义两个函数用于初始化
def weights_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 卷积使用1步长（stride size），0边距（padding size）的模板，保证输出和输入是同一个大小
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 池化用简单传统的2x2大小的模板做max pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')

def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # 784是一张展平的MNIST图片的维度(28*28),None表示其值大小不定，在这里作为第一个维度值，用以指代batch的大小，意即x的数量不定
    # 输出类别值y_也是一个2维张量，其中每一行为一个10维的one-hot向量,用于代表对应某一MNIST图片的类别。
    # 利用placeholder, TensorFlow能够自动捕捉因数据维度不一致导致的错误
    x = tf.placeholder("float", shape=[None, 784])
    y_ = tf.placeholder("float", shape=[None, 10])

    # 模型参数一般用Variable来表示
    # W是一个784x10的矩阵(有784个特征和10个输出值)
    # b是一个10维的向量(有10个分类)
    # TODO：事实上，权重在初始化时应该加入少量的噪声来打破对称性以及避免0梯度
    # W = tf.Variable(tf.zeros([784, 10]))
    # b = tf.Variable(tf.zeros([10]))
    #
    #
    #
    # # 计算每个分类的softmax概率值
    # y = tf.nn.softmax(tf.matmul(x, W) + b)
    # 为训练过程指定最小化误差函数

    '''
    第一层卷积
    '''
    # 卷积在每个5x5的patch中计算32个特征
    # 前两个维度是patch的大小，第三个是输入的通道数目，最后一个表示输出的通道数目
    W_conv1 = weights_variable([5, 5, 1, 32])
    # 对于每个输出通道都有一个对应的偏置量
    b_conv1 = bias_variable([32])

    # 为了使用第一层卷积，把x变成一个四维向量，2、3维对应图片的宽、高，最后一维代表图片的通道数（灰色即1，rgb即3）
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    # 把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    '''
    由于采用了zero-padding，因此：
    input:28x28
    conv1 output:28x28
    pool1 output:14x14((28-2)/2+1)
    '''

    '''
    第二层卷积
    '''
    # 第二层中，每个5x5的patch会得到64个特征
    W_conv2 = weights_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    '''
    input:14x14
    conv1 output:14x14
    pool1 output:7*7
    '''

    '''
    密集连接层（全连接层）
    '''
    W_fc1 = weights_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # 为了减少过拟合，在输出层之前加dropout
    # 用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率
    # 这样可以在训练过程中启用dropout，在测试过程中关闭dropout
    # TensorFlow的tf.nn.dropout操作除了可以屏蔽神经元的输出外，还会自动处理神经元输出值的scale，所以用dropout的时候可以不用考虑scale。
    keep_prop = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prop)

    '''
    输出层
    '''
    W_fc2 = weights_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    '''
    模型评估
    '''
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))

    # 用最速下降法让交叉熵下降，步长为0.01
    # 这一行代码实际上是用来往计算图上添加一个新操作，其中包括计算梯度，计算每个参数的步长变化，并且计算出新的参数值
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    for i in range(20000):
        # 每一步迭代，我们都会加载50个训练样本，然后执行一次train_step，并通过feed_dict将x 和 y_张量占位符用训练训练数据替代
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prop: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prop: 0.5})

    print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prop: 1.0}))

if __name__ == '__main__':
    main()