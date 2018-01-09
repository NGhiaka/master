""" Dynamic Recurrent Neural Network.
TensorFlow implementation of a Recurrent Neural Network (LSTM) that performs
dynamic computation over sequences with variable length. This example is using
a toy dataset to classify linear sequences. The generated sequences have
variable length.
Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell
import tensorflow.contrib.seq2seq as seq2seq
import numpy as np
import config as cf
import random
# ====================
#  TOY DATA GENERATOR
# ====================

# ==========
#   MODEL
# ==========

class Seq2SeqModel(object):
	"""docstring for Seq2seq"""
	
	GO = 1 #Bat dau cau
	EOS = 0 #Ket Thuc Cau
	def __init__(self,encoder_cell, decoder_cell, sequence_length ,input_dim, hidden_dim, output_dim, nlayer = 1,bilingual = True):
		
		self.encoder_cell = encoder_cell
		self.decoder_cell = decoder_cell
		
		self.sequence_length = sequence_length  #Do dai cua chuoi
		self.input_dim = input_dim				#Khong gian input, Tai moi tu vector 1 x input_dim
		self.hidden_dim = hidden_dim			#Khong gian hidden, Tai moi LSTM (trong so: input_dim x hidden_dim)
		self.output_dim = output_dim			#Khong gian output, Tai moi out 1 x output_dim
		
		self.num_layers = nlayer  				#So lop LSTM trong tung giai doan
		self.bilingual = bilingual				#Su dung song ngu hay don ngu


		self._make_graph()

	@property
	def decoder_hidden_units(self):
		# @TODO: is this correct for LSTMStateTuple?
		return self.decoder_cell.output_size
	def _make_graph(self):
		self._init_placeholders()
		self._init_embeddings()
		self._init_bidirectional_encoder()

		self._init_attention()
		self._init_decoder()

	def _init_embeddings(self):
		with tf.variable_scope("embedding") as scope:
			
			num_init = 6/(self.input_dim + self.hidden_dim)
			initializer = tf.random_uniform_initializer(-num_init, num_init)

			self.hidden_matrix = tf.get_variable(
	        	name="hidden_matrix",
				shape=[self.input_dim, self.hidden_dim],
				initializer=initializer,
				dtype=tf.float32)

			self.encoder_hidden_state = tf.nn.embedding_lookup(
	            self.hidden_matrix, self.encoder_inputs)

			self.decoder_hidden_state = tf.nn.embedding_lookup(
	            self.hidden_matrix, self.decoder_train_inputs)

	def _init_placeholders(self):
		""" Everything is time-major """
		# encoder_input: ma tran input cua moi cau (x), cac ma tran nay da duoc xu ly them PAD
		self.encoder_inputs = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='encoder_inputs',
        )
        # Do dai thuc su cua moi cau
		self.encoder_inputs_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='encoder_inputs_length',
        )

		# required for training, not required for testing
		# gia tri output (y) cua mo hinh
		self.decoder_targets = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='decoder_targets'
        )
        # do dai cua chuoi output
		self.decoder_targets_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='decoder_targets_length',
		)
	def _init_bidirectional_encoder(self):
		with tf.variable_scope("Encoder") as scope:
			cell = tf.nn.rnn_cell.DropoutWrapper(self.encoder_cell, output_keep_prob=0.5)  #giam hien tuong overfitting
			cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_layers, state_is_tuple=True)
			# cell_bw = tf.nn.rnn_cell.MultiRNNCell([self.encoder_cell] * self.num_layers, state_is_tuple=True)
			((encoder_fw_outputs, encoder_bw_outputs),(encoder_fw_states, encoder_bw_states)) = (
				tf.nn.bidirectional_dynamic_rnn(cell_fw = cell,
					cell_bw = self.cell,
					inputs = self.encoder_inputs_embedded,
					sequence_length=self.encoder_inputs_length,
					time_major = True,
					dtype = tf.float32
					))
			self.encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

			if isinstance(encoder_fw_state, LSTMStateTuple):
				encoder_state_c = tf.concat(
					(encoder_fw_state.c, encoder_bw_state.c), 1, name='bidirectional_concat_c')
				encoder_state_h = tf.concat(
					(encoder_fw_state.h, encoder_bw_state.h), 1, name='bidirectional_concat_h')
				self.encoder_state = LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
			
			elif isinstance(encoder_fw_state, tf.Tensor):
				self.encoder_state = tf.concat((encoder_fw_state, encoder_bw_state), 1, name='bidirectional_concat')

	def attention_monolinguage(self):
		with tf.variable_scope("attention") as scope:
			
			attention_mechanism = tf.contrib.seq2seq.MahdanauAttention(
				num_units = self.hidden_dim, 
				menory = self.encoder_outputs, 
				memory_sequence_length=self.encoder_inputs_length)
			
			cell = tf.nn.rnn_cell.MultiRNNCell([self.decoder_cell] * self.num_layers, state_is_tuple=True)
			
			atten_cell = tf.contrib.seq2seq.AttentionWrapper(
				cell, attention_mechanism, attention_layer_size = self.self.hidden_dim/2)

			output_cell = tf.contrib.rnn.OutputProjectionWrapper(
				atten_cell, )

	def attention_bilinguage(self):
		pass
	def _init_attention(self):
		if self.bilingual:
			return self.attention_bilinguage()
		else:
			return self.attention_monolinguage()
			
		
	def _init_decoder(self):
		with tf.variable_scope("Decoder") as scope:
			attention_mechanism = tf.contrib.seq2seq.MahdanauAttention(
				num_units = self.hidden_dim, 
				menory = self.encoder_outputs, 
				memory_sequence_length=self.encoder_inputs_length)
			cell = tf.nn.rnn_cell.MultiRNNCell([self.decoder_cell] * self.num_layers, state_is_tuple=True)

	def softmax(ar):
		with tf.variable_scope("softmax") as scope:
			output = tf.nn.softmax(ar)