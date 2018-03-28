import tensorflow as tf
import numpy as np
import collections
import os
import argparse
import datetime as dt
import pickle 
import time 
import codecs
import warnings

warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

data_path = "./Files/five/"

num_l = 2

num_e = 25 

batch_s = 128 

num_s = 20

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def read_words(filename):
    with codecs.open(filename,"r",encoding = 'latin-1',errors = 'ignore') as f:
        return list(f.read())

def build_vocab(filename):
    data = read_words(filename)

    cnts = collections.Counter(data)
    count = cnts.most_common()
    word_to_id = dict()
    
    for word, _ in count:
        word_to_id[word] = len(word_to_id)
        
    print "Number of words in data: " + str(len(data))
    print "Vocabulary size : " + str(len(word_to_id.keys())) 
    
    fp = open('word_to_id','w')
    pickle.dump(word_to_id,fp)
    fp.close()
    
    return word_to_id

def file_to_word_ids(filename, word_to_id):
    
    data = read_words(filename)
    qwert = [] 
    
    for word in data:
        if word in word_to_id:
            qwert.append(word_to_id[word])
            
    return qwert

def load_data():
    # get the data paths
    train_path = os.path.join(data_path, "train_sent.txt")
    valid_path = os.path.join(data_path, "dev_sent.txt")
    test_path = os.path.join(data_path, "test_sent.txt")
    
    # build the complete vocabulary, then convert text data to list of integers
    word_to_id = build_vocab(train_path)
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))

    return train_data, valid_data, test_data, vocabulary, reversed_dictionary

def batch_producer(raw_data, batch_size, num_steps):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0: batch_size * batch_len],
                      [batch_size, batch_len])

   # epoch_size = (batch_len - 1) // num_steps
    epoch_size = batch_len - num_steps + 1
    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = data[:, i :(i + 1) * num_steps]
    x.set_shape([batch_size, num_steps])
    y = data[:, i * num_steps + 1: (i + 1) * num_steps + 1]
    y.set_shape([batch_size, num_steps])
    return x, y


class Input(object):
    def __init__(self, batch_size, num_steps, data):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = batch_producer(data, batch_size, num_steps)


# create the main model
class Model(object):
    def __init__(self, input, is_training, hidden_size, vocab_size, num_layers,
                 dropout=0.5, init_scale=0.05):
        self.is_training = is_training
        self.input_obj = input
        self.batch_size = input.batch_size
        self.num_steps = input.num_steps
        self.hidden_size = hidden_size

        # create the word embeddings
        with tf.device("/cpu:0"):
            embedding = tf.Variable(tf.random_uniform([vocab_size, self.hidden_size], -init_scale, init_scale))
            inputs = tf.nn.embedding_lookup(embedding, self.input_obj.input_data)

        if is_training and dropout < 1:
            inputs = tf.nn.dropout(inputs, dropout)

        # set up the state storage / extraction
        self.init_state = tf.placeholder(tf.float32, [num_layers, 2, self.batch_size, self.hidden_size])

        state_per_layer_list = tf.unstack(self.init_state, axis=0)
        rnn_tuple_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
             for idx in range(num_layers)]
        )

        # create an LSTM cell to be unrolled
        cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=1.0)
        # add a dropout wrapper if training
        if is_training and dropout < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)], state_is_tuple=True)

        output, self.state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, initial_state=rnn_tuple_state)
        # reshape to (batch_size * num_steps, hidden_size)
        output = tf.reshape(output, [-1, hidden_size])

        softmax_w = tf.Variable(tf.random_uniform([hidden_size, vocab_size], -init_scale, init_scale))
        softmax_b = tf.Variable(tf.random_uniform([vocab_size], -init_scale, init_scale))
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        # Reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])
        self.logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

        target = tf.reshape(self.input_obj.targets, [-1])
        target = tf.one_hot(target, vocab_size)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=target))
        
        # Update the cost
        self.cost = loss

        # get the prediction accuracy
        self.softmax_out = tf.nn.softmax(tf.reshape(logits, [-1, vocab_size]))
        self.predict = tf.cast(tf.argmax(self.softmax_out, axis=1), tf.int32)
        correct_prediction = tf.equal(self.predict, tf.reshape(self.input_obj.targets, [-1]))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        if not is_training:
           return
        self.learning_rate = tf.Variable(0.0, trainable=False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        # optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())
        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

        self.new_lr = tf.placeholder(tf.float32, shape=[])
        self.lr_update = tf.assign(self.learning_rate, self.new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})


def train(train_data, vocabulary, num_layers, num_epochs, batch_size, model_save_name,
          learning_rate=1.0, max_lr_epoch=10, lr_decay=0.93, print_iter=50):
    # setup data and models
    training_input = Input(batch_size=batch_size, num_steps=num_s, data=train_data)
    m = Model(training_input, is_training=True, hidden_size=650, vocab_size=vocabulary,
              num_layers=num_layers)
    init_op = tf.global_variables_initializer()
    orig_decay = lr_decay
    with tf.Session() as sess:
        # start threads
        sess.run([init_op])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver = tf.train.Saver()
        for epoch in range(num_epochs):
            new_lr_decay = orig_decay ** max(epoch + 1 - max_lr_epoch, 0.0)
            m.assign_lr(sess, learning_rate * new_lr_decay)
            # m.assign_lr(sess, learning_rate)
            # print(m.learning_rate.eval(), new_lr_decay)
            current_state = np.zeros((num_layers, 2, batch_size, m.hidden_size))
            curr_time = dt.datetime.now()
            for step in range(training_input.epoch_size):
                # cost, _ = sess.run([m.cost, m.optimizer])
                if step % print_iter != 0:
                    cost, _, current_state = sess.run([m.cost, m.train_op, m.state],
                                                      feed_dict={m.init_state: current_state})
                else:
                    seconds = (float((dt.datetime.now() - curr_time).seconds) / print_iter)
                    curr_time = dt.datetime.now()
                    cost, _, current_state, acc = sess.run([m.cost, m.train_op, m.state, m.accuracy],
                                                           feed_dict={m.init_state: current_state})
                    print("Epoch {}, Step {}, cost: {:.3f}, accuracy: {:.3f}, Seconds per step: {:.3f}".format(epoch,
                            step, cost, acc, seconds))

            # save a model checkpoint
            saver.save(sess, data_path  + model_save_name, global_step=epoch)
        # do a final save
        saver.save(sess, data_path + model_save_name + '-final')
        # close threads
        coord.request_stop()
        coord.join(threads)
def test(model_path, test_data, reversed_dictionary):
    test_input = Input(batch_size=batch_s, num_steps=num_s, data=test_data)
    m = Model(test_input, is_training=False, hidden_size=650, vocab_size=vocabulary,
              num_layers=num_l)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # start threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        current_state = np.zeros((2, 2, m.batch_size, m.hidden_size))
        # restore the trained model
        saver.restore(sess, model_path)
        # get an average accuracy over num_acc_batches
        num_acc_batches = 30
        check_batch_idx = 8
        acc_check_thresh = 10
        accuracy = 0
        for batch in range(num_acc_batches):
            if batch == check_batch_idx:
                true_vals, pred, current_state, acc,cost = sess.run([m.input_obj.targets, m.predict, m.state, m.accuracy,m.cost],
                                                               feed_dict={m.init_state: current_state})
                            
                pred_string = [reversed_dictionary[x] for x in pred[:m.num_steps]]
                true_vals_string = [reversed_dictionary[x] for x in true_vals[0]]
                
                print("True values (1st line) vs predicted values (2nd line):")
                print("".join(true_vals_string))
                print("".join(pred_string))
            else:
                acc, current_state = sess.run([m.accuracy, m.state], feed_dict={m.init_state: current_state})
        print("Average accuracy: {:.3f}".format(accuracy / (num_acc_batches-acc_check_thresh)))
        # close threads
        coord.request_stop()
        coord.join(threads)


train_data, valid_data, test_data, vocabulary, reversed_dictionary = load_data()
#train(train_data, vocabulary, num_layers = num_l, num_epochs=num_e, batch_size=batch_s,
s#          model_save_name='monish')
trained_model = data_path + "monish-11"
test(trained_model, test_data, reversed_dictionary)

