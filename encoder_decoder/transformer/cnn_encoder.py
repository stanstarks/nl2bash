import tensorflow as tf

from encoder_decoder.encoder import Encoder
from . import model_utils
from . import cnn


class CNNEncoder(Encoder):
    def __init__(self, params, input_keep, output_keep):
        super(CNNEncoder, params, input_keep, output_keep)
        self.encoder_stack = cnn.EncoderStack(params, not self.forward_only)

    def define_graph(self, encoder_channel_inputs, input_embeddings=None, attention_bias=None):
        if input_embeddings is None:
            input_embeddings = self.token_representations(encoder_channel_inputs)
        with tf.variable_scope("encoder_cnn"):
            inputs_padding = model_utils.get_padding(encoder_channel_inputs)
            length = tf.shape(input_embeddings)[1]
            pos_encoding = model_utils.get_position_encoding(
                length, self.dim)
            encoder_inputs = input_embeddings + pos_encoding

        if not self.forward_only:
            encoder_inputs = tf.nn.dropout(
                encoder_inputs, self.input_keep)

        return self.encoder_stack(encoder_inputs, attention_bias, inputs_padding)
