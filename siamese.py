import os
import random
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

'''
The output of Siamese network would be similarity score, which indicates a pair of gait cycle are from the same user or not (binary classification)
'''
def get_data(data_path, batch_size):

    data = np.load(data_path, allow_pickle = True)
    num_class = len(data)

    pairs = []
    pair_idx = []
    labels = []

    for user_idx, user_data in enumerate(data):

        user_label = np.zeros(batch_size)
        user_label[:len(user_label)//2] = 1

        num_data = len(user_data)

        for idx in range(batch_size):

            batch = []
            batch_idx = [user_idx]
            random_idx = random.randint(0, num_data-1)
            batch.append(user_data[random_idx])

            if idx < len(user_label)//2:
                random_idx = random.randint(0, num_data-1)
                batch.append(user_data[random_idx])
                batch_idx.append(user_idx)
            else:
                random_user = (user_idx + random.randint(1,num_class-1)) % num_class
                random_user_data = data[random_user]

                random_idx = random.randint(0, len(random_user_data)-1)
                batch.append(random_user_data[random_idx])
                batch_idx.append(random_user)

            pairs.append(batch)
            pair_idx.append(batch_idx)
        labels.append(user_label)


    return np.array(pairs), np.array(pair_idx), np.array(labels).ravel().reshape(-1, 1)

def shuffle_data(data, index, label, ratio):

    data, index, label = shuffle(data, index, label)

    train_data, train_index, train_label = data[:int(len(data)*ratio)], index[:int(len(data)*ratio)], label[:int(len(data)*ratio)]
    test_data, test_index, test_label = data[int(len(data)*ratio):], index[int(len(data)*ratio):], label[int(len(data)*ratio):]

    return train_data, train_index, train_label, test_data, test_index, test_label

def train_test_split(data, index, label, ratio):

    train_data, train_index, train_label, test_data, test_index, test_label = shuffle_data(data, index, label, ratio)

    while np.any(np.unique(train_index) != np.unique(index)):

        train_data, train_index, train_label, test_data, test_index, test_label = shuffle_data(data, index, label, ratio)

    return train_data, train_index, train_label, test_data, test_index, test_label

def Layer(X, num_output, initializer, keep_prob, W_name, b_name):

    _, num_feature = X.shape

    W = tf.get_variable(W_name, shape = [num_feature, num_output], dtype = tf.float32, initializer = initializer)
    b = tf.Variable(tf.random_normal([num_output]), name = b_name)
    L = tf.matmul(X, W) + b
    L = tf.nn.relu(L)
    L = tf.nn.dropout(L, keep_prob = keep_prob)

    return L

def siamese(input_data, keep_prob, reuse = False):

    l1_dim = 2000
    l2_dim = 3000
    l3_dim = 3000

    initializer = tf.contrib.layers.xavier_initializer()

    with tf.variable_scope('Layer1', reuse = reuse) as scope:
        model = Layer(input_data, l1_dim, initializer, keep_prob, 'W1', 'b1')

    with tf.variable_scope('Layer2', reuse = reuse) as scope:
        model = Layer(model, l2_dim, initializer, keep_prob, 'W2', 'b2')

    with tf.variable_scope('Layer3', reuse = reuse) as scope:
        model = Layer(model, l3_dim, initializer, keep_prob, 'W3', 'b3')

    return model

walk_data_path = 'data/walk/filtered_interpolation.pkl'
batch_size = 80
num_iter = 500
initial_learning_rate = 10**(-4)

pairs, pair_idx, labels = get_data(walk_data_path, batch_size)
num_pairs, pair_size, cycle_length, num_feature = pairs.shape
print(labels.shape)
train_data, train_index, train_label, test_data, test_index, test_label = train_test_split(pairs, pair_idx, labels, 0.7)
print(train_data.shape, train_index.shape, train_label.shape)

left = tf.placeholder(tf.float32, shape = [None, cycle_length, num_feature], name = 'left')
right = tf.placeholder(tf.float32, shape = [None, cycle_length, num_feature], name = 'right')

new_left = tf.reshape(left, [-1, cycle_length*num_feature])
new_right = tf.reshape(right, [-1, cycle_length*num_feature])

Y = tf.placeholder(tf.float32, shape = [None, 1])

keep_prob = tf.placeholder(tf.float32)


left_model = siamese(new_left, keep_prob, False)
right_model = siamese(new_right, keep_prob, True)

with tf.variable_scope('Difference'):
    difference = tf.math.abs(left_model - right_model)

with tf.variable_scope('Dense'):

    W = tf.get_variable('W', shape = [difference.shape[-1], 1], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.random_normal([1]), name = 'b')
    L = tf.matmul(difference, W) + b

with tf.name_scope('Training'):

    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = L, labels = Y))

    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, 100, 0.9)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    similarity_score = tf.nn.sigmoid(L)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(similarity_score), Y), tf.float32))

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for iter in range(num_iter):

        c, a, _ = sess.run([cost, accuracy, optimizer], feed_dict = {left: train_data[:, 0, :, :], right: train_data[:, 1, :, :], Y: train_label, keep_prob: 0.7})
        print('Cost: {}, Accuracy: {}'.format(c, a))

    acc = sess.run(accuracy, feed_dict = {left: test_data[:, 0, :, :], right: test_data[:, 1, :, :], Y: test_label, keep_prob: 1.0})
    print(acc)


























#end
