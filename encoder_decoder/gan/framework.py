"""GAN model with attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import tensorflow as tf

from encoder_decoder.gan.cnn_discriminator import CNNDiscriminator
from encoder_decoder.gan.utils.model_optimization import create_dis_pretrain_op
from encoder_decoder.seq2seq.seq2seq_model import Seq2SeqModel
from encoder_decoder import graph_utils

class GANModel(graph_utils.NNModel):
    def __init__(self, params, buckets=None):
        super(GANModel, self).__init__(params, buckets)

        # just do discriminator pretrain
        self.train_decoder = False
        self.train_discriminator = True

        params['forward_only'] = True
        self.generator = Seq2SeqModel(params, buckets)
        self.define_discriminator()

    def define_discriminator(self, dim, ):
        """
        Construct QANet discriminator.
        """
        self.discriminator = CNNDiscriminator(
            hyperparameters=self.hypermarams,
            scope='discriminator')

    # override
    def define_graph(self):
        self.encoder_inputs = []
        self.encoder_attn_masks = []
        self.decoder_inputs = []
        for i in range(self.max_source_length):
            self.encoder_inputs.append(
                tf.placeholder(
                    tf.int32, shape=[None], name="encoder{0}".format(i)))
            self.encoder_attn_masks.append(
                tf.placeholder(
                    tf.float32, shape=[None], name="attn_alignment{0}".format(i)))

        for j in range(self.max_target_length + 1):
            self.decoder_inputs.append(
                tf.placeholder(
                    tf.int32, shape=[None], name="decoder{0}".format(j)))
            self.target_weights.append(
                tf.placehoder(
                    tf.float32, shape=[None], name="weight{0}".format(j)))

        if self.copynet:
            for i in range(self.max_source_length):
                self.encoder_copy_inputs.append(
                    tf.placeholder(
                        tf.int32, shape=[None], name="encoder_copy{0}".format(i)))
            for j in range(self.max_target_length):
                self.targets.append(
                    tf.placeholder(
                        tf.int32, shape=[None], name="copy_target{0}".format(j)))

        self.output_symbols = []
        self.dis_losses = []
        for bucket_id, bucket in enumerate(self.buckets):
            with tf.variable_scope(tf.get_variable_scope(),
                                   reuse=True if bucket_id > 0 else None):
                print("creating bucket {} ({}, {})...".format(
                    bucket_id, bucket[0], bucket[1]))
                # encode_decode
                encode_decode_outputs = \
                    self.generator.encode_decode(
                        [self.encoder_inputs[:bucket[0]]],
                        self.encoder_attn_masks[:bucket[0]],
                        self.decoder_inputs[:bucket[1]],
                        self.targets[:bucket[1]],
                        self.target_weights[:bucket[1]],
                        encoder_copy_inputs=self.generator.encoder_copy_inputs[:bucket[0]])
                self.output_symbols.append(encode_decode_outputs['output_symbols'])

                # get encoded
                q_state = encode_decode_outputs["encoder_hidden_states"]
                a_state = 

                # discriminate
                discriminator
_outputs = \
                    self.discriminator.define_graph(
                        q_state,
                        a_state,
                        q_mask,
                        a_mask)
                self.dis_losses.append(discriminator_outputs["loss"])


        # pretrain discriminator
        self.gradient_norms, self.updates = create_dis_pretrain_op(
            self.dis_losses, self.buckets)

        self.saver = tf.train.Saver(tf.global_variables())

    def step(self, session, formatted_example, bucket_id=-1, forward_only=False):
        """Run a step of the model feeding the given inputs.
        :param session: tensorflow session to use.
        :param encoder_inputs: list of numpy int vectors to feed as encoder inputs.
        :param attn_alignments: list of numpy int vectors to feed as the mask
            over inputs about which tokens to attend to.
        :param decoder_inputs: list of numpy int vectors to feed as decoder inputs.
        :param target_weights: list of numpy float vectors to feed as target weights.
        :param bucket_id: which bucket of the model to use.
        :param forward_only: whether to do the backward step or only forward.
        :param return_rnn_hidden_states: if set to True, return the hidden states
            of the two RNNs.
        :return (gradient_norm, average_perplexity, outputs)
        """

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = self.generator.feed_input(formatted_example)

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            if bucket_id == -1:
                output_feed = {
                    'updates': self.updates,                    # Update Op that does SGD.
                    'gradient_norms': self.gradient_norms,      # Gradient norm.
                    'losses': self.losses}                      # Loss for this batch.
            else:
                output_feed = {
                    'updates': self.updates[bucket_id],         # Update Op that does SGD.
                    'gradient_norms': self.gradient_norms[bucket_id],  # Gradient norm.
                    'losses': self.losses[bucket_id]}           # Loss for this batch.
        else:
            if bucket_id == -1:
                output_feed = {
                    'output_symbols': self.generator.output_symbols,      # Loss for this batch.
                    'sequence_logits': self.generator.sequence_logits,        # Batch output sequence
                    'losses': self.generator.losses}                      # Batch output scores
            else:
                output_feed = {
                    'output_symbols': self.generator.output_symbols[bucket_id],  # Loss for this batch.
                    'sequence_logits': self.generator.sequence_logits[bucket_id],   # Batch output sequence
                    'losses': self.generator.losses[bucket_id]}           # Batch output logits

        if self.tg_token_use_attention:
            if bucket_id == -1:
                output_feed['attn_alignments'] = self.generator.attn_alignments
            else:
                output_feed['attn_alignments'] = self.generator.attn_alignments[bucket_id]

        if bucket_id != -1:
            assert(isinstance(self.generator.encoder_hidden_states, list))
            assert(isinstance(self.generator.decoder_hidden_states, list))
            output_feed['encoder_hidden_states'] = \
                self.generator.encoder_hidden_states[bucket_id]
            output_feed['decoder_hidden_states'] = \
                self.generator.decoder_hidden_states[bucket_id]
        else:
            output_feed['encoder_hidden_states'] = self.generator.encoder_hidden_states
            output_feed['decoder_hidden_states'] = self.generator.decoder_hidden_states

        if self.use_copy:
            output_feed['pointers'] = self.generator.pointers

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if extra_update_ops and not forward_only:
            outputs, extra_updates = session.run(
                [output_feed, extra_update_ops], input_feed)
        else:
            outputs = session.run(output_feed, input_feed)

        O = Output()
        if not forward_only:
            # Gradient norm, loss, no outputs
            O.gradient_norms = outputs['gradient_norms']
            O.losses = outputs['losses']
        else:
            # No gradient loss, output_symbols, sequence_logits
            O.output_symbols = outputs['output_symbols']
            O.sequence_logits = outputs['sequence_logits']
            O.losses = outputs['losses']
        # [attention_masks]
        if self.tg_token_use_attention:
            O.attn_alignments = outputs['attn_alignments']

        O.encoder_hidden_states = outputs['encoder_hidden_states']
        O.decoder_hidden_states = outputs['decoder_hidden_states']

        if self.use_copy:
            O.pointers = outputs['pointers']

        return O



class Example(object):
    """
    Input data to the neural network (batched when mini-batch training is used).
    """
    def __init__(self):
        self.encoder_inputs = None
        self.encoder_attn_masks = None
        self.decoder_inputs = None
        self.target_weights = None
        self.encoder_copy_inputs = None     # Copynet
        self.copy_targets = None            # Copynet


class Output(object):
    """
    Data output from the neural network (batched when mini-batch training is used).
    """
    def __init__(self):
        self.updates = None
        self.gradient_norms = None
        self.losses = None
        self.output_symbols = None
        self.sequence_logits = None
        self.attn_alignments = None
        self.encoder_hidden_states = None
        self.decoder_hidden_states = None
        self.pointers = None
