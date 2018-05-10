#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import time
import os, sys
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import input_data
import argparse
from tensorflow.python.platform import gfile
import math

FLAGS = None


#训练最大轮次
num_epochs = 10000

num_hidden = 64
num_layers = 1

#初始化学习速率
INITIAL_LEARNING_RATE = 1e-3
DECAY_STEPS = 5000
REPORT_STEPS = 100
LEARNING_RATE_DECAY_FACTOR = 0.9  # The learning rate decay factor
MOMENTUM = 0.9
DIGITS='abcdefghijklmnopqrstuvwxyz_ '
phones = []


num_classes = 218 + 1 + 1 #218 chinese phones + blank + ctc blank

BATCHES = 100

data_dir = '/dfsdata/heli3_data/data'
audio_processor = None
def get_max_length():
    max_time_duration = 0
    search_path = os.path.join(FLAGS.data_dir, '*','*','*','*.wav')
    for wav_path in gfile.Glob(search_path):
      length_path = "%s%s"%(wav_path,"_time.txt")
      print("length_path %s wav_path %s"%(length_path,wav_path))
      fd = open(length_path)
      time_duration = 0
      while 1:
        lines = fd.read().splitlines()
        if not lines:
            break
        for line in lines:
          if "Length (seconds):" in line:
            last_position=-1
            while True:
                position=line.find(" ",last_position+1)
                if position==-1:
                    break
                last_position=position
            b=float(line[last_position:])
            time_duration = math.ceil(b) * 1000
            if time_duration > max_time_duration:
                max_time_duration = time_duration
            print("Length: length_path %s %s %s %s"%(length_path,line[last_position:],b,time_duration))

    search_path = os.path.join(FLAGS.data_dir2, '*.wav')
    for wav_path in gfile.Glob(search_path):
      length_path = "%s%s"%(wav_path,"_time.txt")
      print("length_path %s wav_path %s"%(length_path,wav_path))
      fd = open(length_path)
      time_duration = 0
      while 1:
        lines = fd.read().splitlines()
        if not lines:
            break
        for line in lines:
          if "Length (seconds):" in line:
            last_position=-1
            while True:
                position=line.find(" ",last_position+1)
                if position==-1:
                    break
                last_position=position
            b=float(line[last_position:])
            time_duration = math.ceil(b) * 1000
            if time_duration > max_time_duration:
                max_time_duration = time_duration
            print("Length: length_path %s %s %s %s"%(length_path,line[last_position:],b,time_duration))
    return max_time_duration

def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count):
  """Calculates common settings needed for all models.

  Args:
    label_count: How many classes are to be recognized.
    sample_rate: Number of audio samples per second.
    clip_duration_ms: Length of each audio clip to be analyzed.
    window_size_ms: Duration of frequency analysis window.
    window_stride_ms: How far to move in time between frequency windows.
    dct_coefficient_count: Number of frequency bins to use for analysis.

  Returns:
    Dictionary containing common settings.
  """
  if FLAGS.max_clip_duration_ms != 0:
      max_clip_duration_ms = FLAGS.max_clip_duration_ms
      desired_samples = int(sample_rate * max_clip_duration_ms / 1000)
  else:
      desired_samples = int(sample_rate * clip_duration_ms / 1000)
  window_size_samples = int(sample_rate * window_size_ms / 1000)
  window_stride_samples = int(sample_rate * window_stride_ms / 1000)
  length_minus_window = (desired_samples - window_size_samples)
  if length_minus_window < 0:
    spectrogram_length = 0
  else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
  fingerprint_size = dct_coefficient_count * spectrogram_length

  return {
      'desired_samples': desired_samples,
      'window_size_samples': window_size_samples,
      'window_stride_samples': window_stride_samples,
      'spectrogram_length': spectrogram_length,
      'dct_coefficient_count': dct_coefficient_count,
      'fingerprint_size': fingerprint_size,
      'label_count': label_count,
      'sample_rate': sample_rate,
      'other_validation': FLAGS.other_validation,
      'other_validation_dir': FLAGS.other_validation_dir,
  }

def decode_sparse_tensor(sparse_tensor):
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)
    result = []
    for index in decoded_indexes:
        result.append(decode_a_seq(index, sparse_tensor))
    return result

def decode_a_seq(indexes, spars_tensor):
    decoded = []
    for m in indexes:
        str = phones[spars_tensor[1][m]]
        str = str.split(' ')

        decoded.append(str[0])
    # Replacing space label to space
    return decoded

def report_accuracy(decoded_list, test_targets):
    original_list = decode_sparse_tensor(test_targets)
    detected_list = decode_sparse_tensor(decoded_list)
    true_numer = 0
    """
    for i in range(0,len(original_list)):
        print("original_list %s %s"%(i,original_list[i]))
    for i in range(0,len(detected_list)):
        print("detected_list %s %s"%(i,detected_list[i]))
    """
    if len(original_list) != len(detected_list):
        return
    print("T/F: original(length) <-------> detectcted(length)")
    for idx, number in enumerate(original_list):
      if len(detected_list) > idx:
        detect_number = detected_list[idx]
        hit = (number == detect_number)
        print((hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")"))
        if hit:
            true_numer = true_numer + 1
    print(("Test Accuracy:", true_numer * 1.0 / len(original_list)))

def sparse_tuple_from2(sequences, dtype=np.int32):
    """
    Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []
    for n, seq in enumerate(sequences):
        indices.extend(list(zip([n] * len(seq), list(range(len(seq))))))

        for i in range(0,len(seq)):
            l = 0
            for line in phones:
                if line == "%s %s"%(seq[i],seq[i]):
                    values.append(l)
                l = l + 1

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=np.dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
    return indices, values, shape

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.5)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, stride=(1, 1), padding='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1],padding=padding)

def max_pool(x, ksize=(2, 2), stride=(2, 2)):
    return tf.nn.max_pool(x, ksize=[1, ksize[0], ksize[1], 1],strides=[1, stride[0], stride[1], 1], padding='SAME')

def avg_pool(x, ksize=(2, 2), stride=(2, 2)):
    return tf.nn.avg_pool(x, ksize=[1, ksize[0], ksize[1], 1],strides=[1, stride[0], stride[1], 1], padding='SAME')

def get_train_model(inputs):
    #定义ctc_loss需要的稀疏矩阵
    targets = tf.sparse_placeholder(tf.int32)

    #1维向量 序列长度 [batch_size,]
    seq_len = tf.placeholder(tf.int32, [None])

    #定义LSTM网络
    cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    outputs, _ = tf.nn.dynamic_rnn(cell, inputs, seq_len, dtype=tf.float32)
    shape = tf.shape(inputs)
    batch_s, max_timesteps = shape[0], shape[1]

    outputs = tf.reshape(outputs, [-1, num_hidden])
    W = tf.Variable(tf.truncated_normal([num_hidden,
                                          num_classes],
                                         stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0., shape=[num_classes]), name="b")

    logits = tf.matmul(outputs, W) + b
    logits = tf.reshape(logits, [batch_s, -1, num_classes])
    logits = tf.transpose(logits, (1, 0, 2))

    return logits, targets, seq_len, W, b

def train():
    fd = open( "lexicon.txt")
    while 1:
      lines = fd.read().splitlines()
      if not lines:
          break
      for line in lines:
          phones.append('%s'%(line))
    fd.close()
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                                global_step,
                                                DECAY_STEPS,
                                                LEARNING_RATE_DECAY_FACTOR,
                                                staircase=True)

    #inputs = tf.placeholder(tf.float32, [None, None, OUTPUT_SHAPE[0]]) flex
    inputs = tf.placeholder(tf.float32, [None, None, FLAGS.dct_coefficient_count])
    logits, targets, seq_len, W, b = get_train_model(inputs)

    loss = tf.nn.ctc_loss(labels=targets,inputs=logits, sequence_length=seq_len)
    cost = tf.reduce_mean(loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step)
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False,beam_width=10)

    predict = tf.sparse_to_dense(decoded[0].indices, decoded[0].dense_shape, decoded[0].values)

    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

    init = tf.global_variables_initializer()
    def do_report():
        test_inputs,test_targets,test_seq_len = get_data(FLAGS.batch_size,"validation")
        test_feed = {inputs: test_inputs,
                     targets: test_targets,
                     seq_len: test_seq_len}

        dd, log_probs, accuracy, predict_test= session.run([decoded[0], log_prob, acc, predict], test_feed)
        report_accuracy(dd, test_targets)


    def get_data(batch_size,mode):
        #inputs = np.zeros([batch_size, OUTPUT_SHAPE[1],OUTPUT_SHAPE[0]]) flex
        codes = []

        data,text = audio_processor.get_data(batch_size, 0, model_settings, FLAGS.background_frequency,
        FLAGS.background_volume, int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000), mode, session)

        #inputs = np.reshape(data,[batch_size,OUTPUT_SHAPE[1],OUTPUT_SHAPE[0]]) flex
        inputs = np.reshape(data,[batch_size,-1,FLAGS.dct_coefficient_count])

        for i in range(0,len(text)):
            codes.append(text[i].split(' '))
        targets = [np.asarray(i) for i in codes]

        sparse_targets = sparse_tuple_from2(targets)
        #1 indices [0,0][0,1]...[63,17]
        #2 values [all the value]
        #3 shape [64(batch size),18]
        #(batch_size,) 值都是256
        #seq_len = np.ones(inputs.shape[0]) * OUTPUT_SHAPE[1] flex
        seq_len = np.ones(inputs.shape[0]) * len(inputs[0])

        return inputs, sparse_targets, seq_len

    def do_batch():
        start = time.time()
        train_inputs, train_targets, train_seq_len = get_data(FLAGS.batch_size,"training")
        print("get_data cost %s"%(time.time()-start))
        feed = {inputs: train_inputs, targets: train_targets, seq_len: train_seq_len}
        b_loss,b_targets, b_logits, b_seq_len,b_cost, steps, _ = session.run([loss, targets, logits, seq_len, cost, global_step, optimizer], feed)
        if steps > 0 and steps % REPORT_STEPS == 0:
            do_report()
        return b_cost, steps

    with tf.Session() as session:
        session.run(init)

        saver = tf.train.Saver(tf.global_variables())
        if FLAGS.start_checkpoint:
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(session, FLAGS.start_checkpoint)

        tf.train.write_graph(session.graph_def,'train','graph.pbtxt')
        for curr_epoch in range(num_epochs):
            train_cost = train_ler = 0
            for batch in range(BATCHES):
                start = time.time()
                c, steps = do_batch()
                train_cost += c * FLAGS.batch_size
                seconds = time.time() - start
                print(("Step:", steps, ", batch seconds:", seconds))

            train_cost /= (BATCHES * FLAGS.batch_size)#TRAIN_SIZE

            train_inputs, train_targets, train_seq_len = get_data(FLAGS.batch_size,"validation")
            print("epoch: train_inputs %s"%(train_inputs.size))
            val_feed = {inputs: train_inputs,
                        targets: train_targets,
                        seq_len: train_seq_len}

            val_cost, val_ler, lr, steps = session.run([cost, acc, learning_rate, global_step], feed_dict=val_feed)

            log = "Epoch {}/{}, steps = {}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}s, learning_rate = {}"
            print((log.format(curr_epoch + 1, num_epochs, steps, train_cost, train_ler, val_cost, val_ler,time.time() - start, lr)))
            print("===== epoch %s ====="%(curr_epoch))
            if curr_epoch != 0 and curr_epoch % 5 == 0:
                print("======save checkpoint=======")
                checkpoint_path = os.path.join(FLAGS.train_dir,'checkpoint.ckpt')
                saver.save(session, checkpoint_path, global_step=curr_epoch)
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
      '--data_url',
      type=str,
      # pylint: disable=line-too-long
      default='http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz',
      # pylint: enable=line-too-long
      help='Location of speech training data archive on the web.')
    parser.add_argument(
      '--data_dir',
      type=str,
      default='/dfsdata/heli3_data/Audio_files/',
      help="""\
      Where to download the speech training data to.
      """)
    parser.add_argument(
      '--data_dir2',
      type=str,
      default='/dfsdata/heli3_data/THCHS30/data/',
      help="""\
      Where to download the speech training data to.
      """)
    parser.add_argument(
      '--data_dir3',
      type=str,
      #default='/dfsdata/heli3_data/train',
      default='',
      help="""\
      Where to download the speech training data to.
      """)
    parser.add_argument(
      '--background_volume',
      type=float,
      default=0.1,
      help="""\
      How loud the background noise should be, between 0 and 1.
      """)
    parser.add_argument(
      '--background_frequency',
      type=float,
      default=0.8,
      help="""\
      How many of the training samples have background noise mixed in.
      """)
    parser.add_argument(
      '--silence_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be silence.
      """)
    parser.add_argument(
      '--unknown_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be unknown words.
      """)
    parser.add_argument(
      '--time_shift_ms',
      type=float,
      default=100.0,
      help="""\
      Range to randomly shift the training audio by in time.
      """)
    parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a test set.')
    parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a validation set.')
    parser.add_argument(
      '--sample_rate',
      type=int,
      default=16000,
      help='Expected sample rate of the wavs',)
    parser.add_argument(
      '--clip_duration_ms',
      type=int,
      default=1000,
      help='Expected duration in milliseconds of the wavs',)
    parser.add_argument(
      '--max_clip_duration_ms',
      type=int,
      default=17000,
      help='Expected duration in milliseconds of the wavs',)
    parser.add_argument(
      '--window_size_ms',
      type=float,
      default=40.0,
      help='How long each spectrogram timeslice is',)
    parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=20.0,
      help='How long each spectrogram timeslice is',)
    parser.add_argument(
      '--dct_coefficient_count',
      type=int,
      default=10,
      help='How many bins to use for the MFCC fingerprint',)
    parser.add_argument(
      '--how_many_training_steps',
      type=str,
      default='15000,3000',
      help='How many training loops to run',)
    parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=50,
      help='How often to evaluate the training results.')
    parser.add_argument(
      '--learning_rate',
      type=str,
      default='0.001,0.0001',
      help='How large a learning rate to use when training.')
    parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='How many items to train with at once',)
    parser.add_argument(
      '--summaries_dir',
      type=str,
      default='/tmp/retrain_logs',
      help='Where to save summary logs for TensorBoard.')
    parser.add_argument(
      '--wanted_words',
      type=str,
      default='n i3 h ao3 l ian2 x iang3',
      help='Words to use (others will be added to an unknown label)',)
    parser.add_argument(
      '--train_dir',
      type=str,
      default='/dfsdata/heli3_data/train_result/',
      help='Directory to write event logs and checkpoint.')
    parser.add_argument(
      '--save_step_interval',
      type=int,
      default=100,
      help='Save model checkpoint every save_steps.')
    parser.add_argument(
      '--start_checkpoint',
      type=str,
      default='',
      help='If specified, restore this pretrained model before any training.')
    parser.add_argument(
      '--model_architecture',
      type=str,
      default='dnn',
      help='What model architecture to use')
    parser.add_argument(
      '--model_size_info',
      type=int,
      nargs="+",
      default=[128,128,128],
      help='Model dimensions - different for various models')
    parser.add_argument(
      '--check_nans',
      type=bool,
      default=False,
      help='Whether to check for invalid numbers during processing')
    parser.add_argument(
      '--other_validation',
      type=bool,
      default=False,
      help='Whether to use other validation file')

    parser.add_argument(
      '--other_validation_dir',
      type=str,
      default='/media/yy/9a19ad59-dbd6-40b3-8b68-4589aea51b4a1/yy/workspace/kws/hello_lenovo_traindata/file/pc/Aispeech2_PC/audio',
      help="""\
      Other validation file path.
      """)

    FLAGS, unparsed = parser.parse_known_args()
    model_settings = prepare_model_settings(
      len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
      FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
      FLAGS.window_stride_ms, FLAGS.dct_coefficient_count)
    audio_processor = input_data.AudioProcessor(FLAGS.data_url, FLAGS.data_dir, FLAGS.data_dir2, FLAGS.data_dir3, FLAGS.silence_percentage,
      FLAGS.unknown_percentage,
      FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
      FLAGS.testing_percentage, model_settings)

    train()
