# # Working with TF commit 24466c2e6d32621cd85f0a78d47df6eed2c5c5a6

# import math

# import numpy as np
# import tensorflow as tf
# import tensorflow.contrib.seq2seq as seq2seq
# from tensorflow.contrib.layers import safe_embedding_lookup_sparse as embedding_lookup_unique
# from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell

# import helpers


# class Seq2SeqModel():
#     """Seq2Seq model usign blocks from new `tf.contrib.seq2seq`.
#     Requires TF 1.0.0-alpha"""

#     PAD = 0
#     EOS = 1

#     def __init__(self, encoder_cell, decoder_cell, vocab_size, embedding_size,
#                  bidirectional=True,
#                  attention=False,
#                  debug=False):
#         self.debug = debug
#         self.bidirectional = bidirectional
#         self.attention = attention

#         self.vocab_size = vocab_size
#         self.embedding_size = embedding_size

#         self.encoder_cell = encoder_cell
#         self.decoder_cell = decoder_cell

#         self._make_graph()

#     @property
#     def decoder_hidden_units(self):
#         # @TODO: is this correct for LSTMStateTuple?
#         return self.decoder_cell.output_size

#     def _make_graph(self):
#         if self.debug:
#             self._init_debug_inputs()
#         else:
#             self._init_placeholders()

#         self._init_decoder_train_connectors()
#         self._init_embeddings()

#         if self.bidirectional:
#             self._init_bidirectional_encoder()
#         else:
#             self._init_simple_encoder()

#         self._init_decoder()

#         self._init_optimizer()

#     def _init_debug_inputs(self):
#         """ Everything is time-major """
#         x = [[5, 6, 7],
#              [7, 6, 0],
#              [0, 7, 0]]
#         xl = [2, 3, 1]
#         self.encoder_inputs = tf.constant(x, dtype=tf.int32, name='encoder_inputs')
#         self.encoder_inputs_length = tf.constant(xl, dtype=tf.int32, name='encoder_inputs_length')

#         self.decoder_targets = tf.constant(x, dtype=tf.int32, name='decoder_targets')
#         self.decoder_targets_length = tf.constant(xl, dtype=tf.int32, name='decoder_targets_length')

#     def _init_placeholders(self):
#         """ Everything is time-major """
#         self.encoder_inputs = tf.placeholder(
#             shape=(None, None),
#             dtype=tf.int32,
#             name='encoder_inputs',
#         )
#         self.encoder_inputs_length = tf.placeholder(
#             shape=(None,),
#             dtype=tf.int32,
#             name='encoder_inputs_length',
#         )

#         # required for training, not required for testing
#         self.decoder_targets = tf.placeholder(
#             shape=(None, None),
#             dtype=tf.int32,
#             name='decoder_targets'
#         )
#         self.decoder_targets_length = tf.placeholder(
#             shape=(None,),
#             dtype=tf.int32,
#             name='decoder_targets_length',
#         )

#     def _init_decoder_train_connectors(self):
#         """
#         During training, `decoder_targets`
#         and decoder logits. This means that their shapes should be compatible.
#         Here we do a bit of plumbing to set this up.
#         """
#         with tf.name_scope('DecoderTrainFeeds'):
#             sequence_size, batch_size = tf.unstack(tf.shape(self.decoder_targets))

#             EOS_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.EOS
#             PAD_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.PAD

#             self.decoder_train_inputs = tf.concat([EOS_SLICE, self.decoder_targets], axis=0)
#             self.decoder_train_length = self.decoder_targets_length + 1

#             decoder_train_targets = tf.concat([self.decoder_targets, PAD_SLICE], axis=0)
#             decoder_train_targets_seq_len, _ = tf.unstack(tf.shape(decoder_train_targets))
#             decoder_train_targets_eos_mask = tf.one_hot(self.decoder_train_length - 1,
#                                                         decoder_train_targets_seq_len,
#                                                         on_value=self.EOS, off_value=self.PAD,
#                                                         dtype=tf.int32)
#             decoder_train_targets_eos_mask = tf.transpose(decoder_train_targets_eos_mask, [1, 0])

#             # hacky way using one_hot to put EOS symbol at the end of target sequence
#             decoder_train_targets = tf.add(decoder_train_targets,
#                                            decoder_train_targets_eos_mask)

#             self.decoder_train_targets = decoder_train_targets

#             self.loss_weights = tf.ones([
#                 batch_size,
#                 tf.reduce_max(self.decoder_train_length)
#             ], dtype=tf.float32, name="loss_weights")

#     def _init_embeddings(self):
#         with tf.variable_scope("embedding") as scope:

#             # Uniform(-sqrt(3), sqrt(3)) has variance=1.
#             sqrt3 = math.sqrt(3)
#             initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

#             self.embedding_matrix = tf.get_variable(
#                 name="embedding_matrix",
#                 shape=[self.vocab_size, self.embedding_size],
#                 initializer=initializer,
#                 dtype=tf.float32)

#             self.encoder_inputs_embedded = tf.nn.embedding_lookup(
#                 self.embedding_matrix, self.encoder_inputs)

#             self.decoder_train_inputs_embedded = tf.nn.embedding_lookup(
#                 self.embedding_matrix, self.decoder_train_inputs)

#     def _init_simple_encoder(self):
#         with tf.variable_scope("Encoder") as scope:
#             (self.encoder_outputs, self.encoder_state) = (
#                 tf.nn.dynamic_rnn(cell=self.encoder_cell,
#                                   inputs=self.encoder_inputs_embedded,
#                                   sequence_length=self.encoder_inputs_length,
#                                   time_major=True,
#                                   dtype=tf.float32)
#                 )

#     def _init_bidirectional_encoder(self):
#         with tf.variable_scope("BidirectionalEncoder") as scope:

#             ((encoder_fw_outputs,
#               encoder_bw_outputs),
#              (encoder_fw_state,
#               encoder_bw_state)) = (
#                 tf.nn.bidirectional_dynamic_rnn(cell_fw=self.encoder_cell,
#                                                 cell_bw=self.encoder_cell,
#                                                 inputs=self.encoder_inputs_embedded,
#                                                 sequence_length=self.encoder_inputs_length,
#                                                 time_major=True,
#                                                 dtype=tf.float32)
#                 )

#             self.encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

#             if isinstance(encoder_fw_state, LSTMStateTuple):

#                 encoder_state_c = tf.concat(
#                     (encoder_fw_state.c, encoder_bw_state.c), 1, name='bidirectional_concat_c')
#                 encoder_state_h = tf.concat(
#                     (encoder_fw_state.h, encoder_bw_state.h), 1, name='bidirectional_concat_h')
#                 self.encoder_state = LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

#             elif isinstance(encoder_fw_state, tf.Tensor):
#                 self.encoder_state = tf.concat((encoder_fw_state, encoder_bw_state), 1, name='bidirectional_concat')

#     def _init_decoder(self):
#         with tf.variable_scope("Decoder") as scope:
#             def output_fn(outputs):
#                 return tf.contrib.layers.linear(outputs, self.vocab_size, scope=scope)

#             if not self.attention:
#                 decoder_fn_train = seq2seq.simple_decoder_fn_train(encoder_state=self.encoder_state)
#                 decoder_fn_inference = seq2seq.simple_decoder_fn_inference(
#                     output_fn=output_fn,
#                     encoder_state=self.encoder_state,
#                     embeddings=self.embedding_matrix,
#                     start_of_sequence_id=self.EOS,
#                     end_of_sequence_id=self.EOS,
#                     maximum_length=tf.reduce_max(self.encoder_inputs_length) + 3,
#                     num_decoder_symbols=self.vocab_size,
#                 )
#             else:

#                 # attention_states: size [batch_size, max_time, num_units]
#                 attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])

#                 (attention_keys,
#                 attention_values,
#                 attention_score_fn,
#                 attention_construct_fn) = seq2seq.prepare_attention(
#                     attention_states=attention_states,
#                     attention_option="bahdanau",
#                     num_units=self.decoder_hidden_units,
#                 )

#                 decoder_fn_train = seq2seq.attention_decoder_fn_train(
#                     encoder_state=self.encoder_state,
#                     attention_keys=attention_keys,
#                     attention_values=attention_values,
#                     attention_score_fn=attention_score_fn,
#                     attention_construct_fn=attention_construct_fn,
#                     name='attention_decoder'
#                 )

#                 decoder_fn_inference = seq2seq.attention_decoder_fn_inference(
#                     output_fn=output_fn,
#                     encoder_state=self.encoder_state,
#                     attention_keys=attention_keys,
#                     attention_values=attention_values,
#                     attention_score_fn=attention_score_fn,
#                     attention_construct_fn=attention_construct_fn,
#                     embeddings=self.embedding_matrix,
#                     start_of_sequence_id=self.EOS,
#                     end_of_sequence_id=self.EOS,
#                     maximum_length=tf.reduce_max(self.encoder_inputs_length) + 3,
#                     num_decoder_symbols=self.vocab_size,
#                 )

#             (self.decoder_outputs_train,
#              self.decoder_state_train,
#              self.decoder_context_state_train) = (
#                 seq2seq.dynamic_rnn_decoder(
#                     cell=self.decoder_cell,
#                     decoder_fn=decoder_fn_train,
#                     inputs=self.decoder_train_inputs_embedded,
#                     sequence_length=self.decoder_train_length,
#                     time_major=True,
#                     scope=scope,
#                 )
#             )

#             self.decoder_logits_train = output_fn(self.decoder_outputs_train)
#             self.decoder_prediction_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_prediction_train')

#             scope.reuse_variables()

#             (self.decoder_logits_inference,
#              self.decoder_state_inference,
#              self.decoder_context_state_inference) = (
#                 seq2seq.dynamic_rnn_decoder(
#                     cell=self.decoder_cell,
#                     decoder_fn=decoder_fn_inference,
#                     time_major=True,
#                     scope=scope,
#                 )
#             )
#             self.decoder_prediction_inference = tf.argmax(self.decoder_logits_inference, axis=-1, name='decoder_prediction_inference')

#     def _init_optimizer(self):
#         logits = tf.transpose(self.decoder_logits_train, [1, 0, 2])
#         targets = tf.transpose(self.decoder_train_targets, [1, 0])
#         self.loss = seq2seq.sequence_loss(logits=logits, targets=targets,
#                                           weights=self.loss_weights)
#         self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

#     def make_train_inputs(self, input_seq, target_seq):
#         inputs_, inputs_length_ = helpers.batch(input_seq)
#         targets_, targets_length_ = helpers.batch(target_seq)
#         return {
#             self.encoder_inputs: inputs_,
#             self.encoder_inputs_length: inputs_length_,
#             self.decoder_targets: targets_,
#             self.decoder_targets_length: targets_length_,
#         }

#     def make_inference_inputs(self, input_seq):
#         inputs_, inputs_length_ = helpers.batch(input_seq)
#         return {
#             self.encoder_inputs: inputs_,
#             self.encoder_inputs_length: inputs_length_,
#         }


# def make_seq2seq_model(**kwargs):
#     args = dict(encoder_cell=LSTMCell(10),
#                 decoder_cell=LSTMCell(20),
#                 vocab_size=10,
#                 embedding_size=10,
#                 attention=True,
#                 bidirectional=True,
#                 debug=False)
#     args.update(kwargs)
#     return Seq2SeqModel(**args)


# def train_on_copy_task(session, model,
#                        length_from=3, length_to=8,
#                        vocab_lower=2, vocab_upper=10,
#                        batch_size=100,
#                        max_batches=5000,
#                        batches_in_epoch=1000,
#                        verbose=True):

#     batches = helpers.random_sequences(length_from=length_from, length_to=length_to,
#                                        vocab_lower=vocab_lower, vocab_upper=vocab_upper,
#                                        batch_size=batch_size)
#     loss_track = []
#     try:
#         for batch in range(max_batches+1):
#             batch_data = next(batches)
#             fd = model.make_train_inputs(batch_data, batch_data)
#             _, l = session.run([model.train_op, model.loss], fd)
#             loss_track.append(l)

#             if verbose:
#                 if batch == 0 or batch % batches_in_epoch == 0:
#                     print('batch {}'.format(batch))
#                     print('  minibatch loss: {}'.format(session.run(model.loss, fd)))
#                     for i, (e_in, dt_pred) in enumerate(zip(
#                             fd[model.encoder_inputs].T,
#                             session.run(model.decoder_prediction_train, fd).T
#                         )):
#                         print('  sample {}:'.format(i + 1))
#                         print('    enc input           > {}'.format(e_in))
#                         print('    dec train predicted > {}'.format(dt_pred))
#                         if i >= 2:
#                             break
#                     print()
#     except KeyboardInterrupt:
#         print('training interrupted')

#     return loss_track


# if __name__ == '__main__':
#     import sys

#     if 'fw-debug' in sys.argv:
#         tf.reset_default_graph()
#         with tf.Session() as session:
#             model = make_seq2seq_model(debug=True)
#             session.run(tf.global_variables_initializer())
#             session.run(model.decoder_prediction_train)
#             session.run(model.decoder_prediction_train)

#     elif 'fw-inf' in sys.argv:
#         tf.reset_default_graph()
#         with tf.Session() as session:
#             model = make_seq2seq_model()
#             session.run(tf.global_variables_initializer())
#             fd = model.make_inference_inputs([[5, 4, 6, 7], [6, 6]])
#             inf_out = session.run(model.decoder_prediction_inference, fd)
#             print(inf_out)

#     elif 'train' in sys.argv:
#         tracks = {}

#         tf.reset_default_graph()

#         with tf.Session() as session:
#             model = make_seq2seq_model(attention=True)
#             session.run(tf.global_variables_initializer())
#             loss_track_attention = train_on_copy_task(session, model)

#         tf.reset_default_graph()

#         with tf.Session() as session:
#             model = make_seq2seq_model(attention=False)
#             session.run(tf.global_variables_initializer())
#             loss_track_no_attention = train_on_copy_task(session, model)

#         import matplotlib.pyplot as plt
#         plt.plot(loss_track)
#         print('loss {:.4f} after {} examples (batch_size={})'.format(loss_track[-1], len(loss_track)*batch_size, batch_size))

#     else:
#         tf.reset_default_graph()
#         session = tf.InteractiveSession()
#         model = make_seq2seq_model(debug=False)
#         session.run(tf.global_variables_initializer())

#         fd = model.make_inference_inputs([[5, 4, 6, 7], [6, 6]])

# inf_out = session.run(model.decoder_prediction_inference, fd)
# 
# 
# from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length
num_layers = 3

def generateData():
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0

    x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    y = y.reshape((batch_size, -1))

    return (x, y)

batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

init_state = tf.placeholder(tf.float32, [num_layers, 2, batch_size, state_size])

state_per_layer_list = tf.unpack(init_state, axis=0)
rnn_tuple_state = tuple(
    [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
     for idx in range(num_layers)]
)

W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

# Unpack columns
inputs_series = tf.split(1, truncated_backprop_length, batchX_placeholder)
labels_series = tf.unpack(batchY_placeholder, axis=1)

# Forward passes
cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
states_series, current_state = tf.nn.rnn(cell, inputs_series, initial_state=rnn_tuple_state)

logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels) for logits, labels in zip(logits_series,labels_series)]
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

def plot(loss_list, predictions_series, batchX, batchY):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)

    for batch_series_idx in range(5):
        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])

        plt.subplot(2, 3, batch_series_idx + 2)
        plt.cla()
        plt.axis([0, truncated_backprop_length, 0, 2])
        left_offset = range(truncated_backprop_length)
        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

    plt.draw()
    plt.pause(0.0001)


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    for epoch_idx in range(num_epochs):
        x,y = generateData()

        _current_state = np.zeros((num_layers, 2, batch_size, state_size))

        print("New data, epoch", epoch_idx)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length

            batchX = x[:,start_idx:end_idx]
            batchY = y[:,start_idx:end_idx]

            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={
                    batchX_placeholder: batchX,
                    batchY_placeholder: batchY,
                    init_state: _current_state
                })


            loss_list.append(_total_loss)

            if batch_idx%100 == 0:
                print("Step",batch_idx, "Batch loss", _total_loss)
                plot(loss_list, _predictions_series, batchX, batchY)

plt.ioff()
plt.show()