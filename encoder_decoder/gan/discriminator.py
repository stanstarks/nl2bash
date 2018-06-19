import tensorflow as tf

from encoder_decoder import graph_utils


class Discriminator(graph_utils.NNModel):
    def __init__(self, hyperparams, buckets=None):
        super(Discriminator, self).__init__(hyperparams, buckets)
        self.learning_rate = tf.Variable(
            float(hyperparams["learning_rate_d"]), trainable=False)

        self.global_epoch = tf.Variable(0, trainable=False)

        # Encoder
        self.define_encoder(self.sc_input_keep, self.sc_output_keep)
