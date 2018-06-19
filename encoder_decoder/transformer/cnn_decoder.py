import tensorflow as tf

from encoder_decoder import decoder
from . import beam_search
from . import model_utils
from . import cnn


class CNNDecoder(decoder.Decoder):
    def __init__(self, params, scope, dim, embedding_dim, use_attention,
                 attention_function, input_keep, output_keep, decoding_algorithm):
        super(CNNDecoder, self).__init__(
            params, scope, dim, embedding_dim, use_attention, attention_function,
            input_keep, output_keep, decoding_algorithm)
        print("{}  dimention = {}".format(scope, dim))
        print("{} decoding_algorithm = {}".format(scope, decoding_algorithm))
        self.params = {}
        self.params["hidden_size"]
        self.decoder_stack = cnn.DecoderStack(self.params, not self.forward_only)

    def define_graph(self, encoding_state, decoder_inputs, input_embeddings=None,
                     attention_bias=None):
        # don't use attention
        if input_embeddings is None:
            input_embeddings = self.embeddings()

        if self.force_reading_input:
            print("Warning: reading ground truth decoder inputs at decoding time.")

        with tf.variable_scope(self.scope + "_decoder_cnn"):
            decoder_inputs = tf.nn.embedding_lookup(input_embeddings, decoder_inputs)
            with tf.name_scope("shift_targets"):
                # shifting targets, adding positional encoding
                decoder_inputs = tf.pad(
                    decoder_inputs, [[0, 0], [1, 0], [0, 0]])[:, -1, :]
            with tf.name_scope("add_pos_encoding"):
                length = tf.shape(decoder_inputs)[1]
                decoder_inputs += model_utils.get_position_encoding(
                    length, self.dim)
            if not self.forward_only:
                decoder_inputs = tf.nn.dropout(
                    decoder_inputs, self.output_keep)

            decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(
                length)
            outputs = self.decoder_stack(
                decoder_inputs, encoding_state, decoder_self_attention_bias,
                attention_bias)
            logits = self.embedding(outputs)
            return output_symbols, sequence_logits, past_output_logits,\
                states, attn_alignments, pointers

    def predict(self, encoder_outputs, encoder_decoder_attention_bias):
        """Return predicted sequence."""
        batch_size = tf.shape(encoder_outputs)[0]
        input_length = tf.shape(encoder_outputs)[1]
        max_decode_length = input_length + self.params["extra_decode_length"]

        symbols_to_logits_fn = self._get_symbols_to_logits_fn(max_decode_length)

        initial_ids = tf.zeros([batch_size], dtype=tf.int32)

        cache = {
            "layer_%d" % layer: {
                "k": tf.zeros([batch_size, 0, self.params["hidden_size"]]),
                "v": tf.zeros([batch_size, 0, self.params["hidden_size"]]),
            } for layer in range(self.params["num_hidden_layers"])
        }

        cache["encoder_outputs"] = encoder_outputs
        cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

        decoded_ids, scores = beam_search.sequence_beam_search(
            symbols_to_logits_fn=symbols_to_logits_fn,
            initial_ids=initial_ids,
            initial_cache=cache,
            vocab_size=self.params["vocab_size"],
            beam_size=self.params["beam_size"],
            alpha=self.params["alpha"],
            max_decode_length=max_decode_length,
            eos_id=EOS_ID)


