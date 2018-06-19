import tensorflow as tf

from encoder_decoder import graph_utils
from encoder_decoder.cnn import residual_block, optimized_trilinear_for_attention, mask_logits, conv

class CNNDiscriminator(graph_utils.NNModel):
    def __init__(self, hyperparameters, scope):
        super(CNNDiscriminator, self).__init__(hyperparameters)

        # 

    def define_graph(self, q_state, a_state, q_mask, a_mask):
        with tf.variable_scope("discriminator_encoder"):
            q = residual_block(q_state,
                               num_blocks=1,
                               num_conv_layers=4,
                               kernel_size=7,
                               mask=q_mask,
                               num_filters=self.d,
                               num_heads=self.nh,
                               seq_len=self.q_len,
                               scope='encoder_residual_block',
                               bias=False,
                               dropout=self.dropout_d)
            a = residual_block(a_state,
                               num_blocks=1,
                               num_conv_layers=4,
                               kernel_size=7,
                               mask=a_mask,
                               num_filters=self.d,
                               num_heads=self.nh,
                               seq_len=self.a_len,
                               scope='encoder_residual_block',
                               reuse=True,  # Share the weights?
                               bias=False,
                               dropout=self.dropout_d)

        with tf.variable_scope("discriminator_attention"):
            S = optimized_trilinear_for_attention([q, a], self.q_maxlen, self.a_maxlen,
                                                  input_keep_prob=1.0 - self.dropout_d)
            mask_a = tf.expand_dims(a_mask, 1)
            S_ = tf.nn.softmax(mask_logits(S, mask=mask_a))
            mask_q = tf.expand_dims(q_mask, 2)
            S_T = tf.transpose(tf.nn.softmax(mask_logits(S, mask=mask_q), dim=1), (0, 2, 1))
            self.q2a = tf.matmul(S_, a)
            self.a2q = tf.matmul(tf.matmul(S_, S_T), q)
            attention_outputs = [q, self.q2a, q * self.q2a, q * self.a2q]

        with tf.variable_scope("discriminator_model"):
            inputs = tf.concat(attention_outputs, axis=-1)
            self.enc = [conv(inputs, self.d, name='input_projection')]
            for i in range(2):
                if i % 2 == 0: # dropout every 2 blocks
                    self.enc[i] = tf.nn.dropout(self.enc[i], 1.0 - self.dropout_d)
                self.enc.append(
                    residual_block(self.enc[i],
                                   num_blocks=7,
                                   num_conv_layers=2,
                                   kernel_size=5,
                                   mask=q_mask,
                                   num_filters=self.d,
                                   num_heads=self.nh,
                                   seq_len=self.q_len,
                                   scope="model_encoder",
                                   bias=False,
                                   reuse=True if i > 0 else None,
                                   dropout=self.dropout_d))

        with tf.variable_scope("discriminator_output"):
            self.logits = tf.squeeze(conv(tf.concat([self.enc[1], self.enc[2]],
                                                    axis=-1),
                                          1, bias=False, name="pointer"))

