# encoding: utf-8

import tensorflow as tf


class Seq2seq():
    def __init__(self, args, word_ids_dict):
        self.args = args
        self.word_ids_dict = word_ids_dict
        self.sequence_input = tf.placeholder(dtype=tf.int32, shape=(self.args.batch_size, None), name='seq_input')
        self.sequence_length = tf.placeholder(dtype=tf.int32, shape=(self.args.batch_size,), name='sequence_length')
        self.target_input = tf.placeholder(dtype=tf.int32, shape=(self.args.batch_size, None), name='target_input')
        self.target_length = tf.placeholder(dtype=tf.int32, shape=(self.args.batch_size,), name='target_length')

        self.bulid_model()

    def bulid_model(self):
        with tf.variable_scope('encoder'):
            # sou_emd = tf.get_variable(name='source_embedding', shape=[self.args.sou_wd_num, self.args.emd_size], initializer=tf.random_uniform_initializer(), trainable=True)
            sou_emd = tf.Variable(tf.random_uniform([self.args.sou_wd_num, self.args.emd_size]), dtype=tf.float32, name='encoder_embedding')
            seq_embedded = tf.nn.embedding_lookup(sou_emd, self.sequence_input)
            fw_lstm = tf.nn.rnn_cell.GRUCell(self.args.hidden_size)
            bk_lstm = tf.nn.rnn_cell.GRUCell(self.args.hidden_size)

            ((ec_fw_output, ec_bk_output), (fal_fw_h, fal_bk_h)) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw_lstm, cell_bw=bk_lstm, inputs=seq_embedded, sequence_length=self.sequence_length, dtype=tf.float32)
            ecd_output = tf.add(ec_fw_output, ec_bk_output)
            ecd_h = tf.add(fal_fw_h, fal_bk_h)

        with tf.variable_scope('decoder'):
            # tag_emd = tf.get_variable(name='target_embedding', shape=[self.args.tag_wd_num, self.args.emd_size], initializer=tf.random_uniform_initializer(), trainable=True)
            tag_emd = tf.Variable(tf.random_uniform([self.args.tag_wd_num, self.args.emd_size]), dtype=tf.float32, name='decoder_embedding')
            begin_token = tf.ones(shape=(self.args.batch_size), dtype=tf.int32, name='begin_token') * self.word_ids_dict['_GO']

            if self.args.train:
                input_ids = tf.concat([tf.reshape(begin_token, (self.args.batch_size, 1)), self.target_input], axis=-1)
                helper = tf.contrib.seq2seq.TrainingHelper(tf.nn.embedding_lookup(tag_emd, input_ids), self.target_length)

            else:
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(tag_emd, begin_token, self.word_ids_dict['_EOS'])

            decoder_cell = tf.nn.rnn_cell.GRUCell(self.args.hidden_size)

            if self.args.attention:
                if self.args.beam_search_num > 1:
                    tiled_ecd_output = tf.contrib.seq2seq.tile_batch(ecd_output, multiplier=self.args.beam_search_num)
                    tiled_ecd_h = tf.contrib.seq2seq.tile_batch(ecd_h, multiplier=self.args.beam_search_num)
                    tiled_seq_length = tf.contrib.seq2seq.tile_batch(self.sequence_length, multiplier=self.args.beam_search_num)
                    attention_model = tf.contrib.seq2seq.BahdanauAttention(num_units=self.args.hidden_size,
                                                                           memory=tiled_ecd_output,
                                                                           memory_sequence_length=tiled_seq_length)
                    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_model)
                    tiled_initial_h = decoder_cell.zero_state(batch_size=self.args.batch_size * self.args.beam_search_num, dtype=tf.float32)
                    tiled_initial_h = tiled_initial_h.clone(cell_state=tiled_ecd_h)
                    intital_h = tiled_initial_h
                else:
                    attention_model = tf.contrib.seq2seq.BahdanauAttention(num_units=self.args.hidden_size,
                                                                           memory=ecd_output,
                                                                           memory_sequence_length=self.sequence_length)
                    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_model)
                    initial_h = decoder_cell.zero_state(batch_size=self.args.batch_size, dtype=tf.float32)
                    intital_h = initial_h.clone(cell_state=ecd_h)
            else:
                if self.args.beam_search_num > 1:
                    intital_h = tf.contrib.seq2seq.tile_batch(ecd_h, multiplier=self.args.beam_search_num)
                else:
                    intital_h = ecd_h

            if self.args.beam_search_num > 1:
                decoder = tf.contrib.seq2seq.BeamSearchDecoder(decoder_cell, tag_emd, begin_token,
                                                               self.word_ids_dict["_EOS"], intital_h,
                                                               beam_width=self.args.beam_search_num,
                                                               output_layer=tf.layers.Dense(self.args.tag_wd_num))
            else:
                decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, intital_h,
                                                          output_layer=tf.layers.Dense(self.args.tag_wd_num))

            decoder_outputs, decoder_state, final_sequence_lengths = \
                tf.contrib.seq2seq.dynamic_decode(decoder,  maximum_iterations=tf.reduce_max(self.target_length))

            if self.args.beam_search_num > 1:
                self.out = decoder_outputs.predicted_ids[:, :, 0]
            else:
                decoder_logits = decoder_outputs.rnn_output
                self.out = tf.argmax(decoder_logits, 2)
                sequence_mask = tf.sequence_mask(self.target_length, dtype=tf.float32)
                self.loss = tf.contrib.seq2seq.sequence_loss(logits=decoder_logits, targets=self.target_input,
                                                             weights=sequence_mask)
                self.train_op = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate).minimize(self.loss)

