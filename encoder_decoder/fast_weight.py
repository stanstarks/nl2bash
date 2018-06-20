import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops


class GraphCellWrapper(tf.nn.rnn_cell.RNNCell):
    def __init__(self, cell):
        self.cell = cell

    def __call__(self, input_embedding, state, scope=None):
        node_state = self.structure_update(state)
        cell_output, state = self.cell(input_embedding, node_state)
        return cell_output, self.global_update(state)


class DiagonalFastWeight(tf.nn.rnn_cell.RNNCell):
    def __init__(self, cell):
        self.cell = cell

    def __call__(self, input_embedding, state, scope=None):
        edge_state, node_state = state
        cell_output, state = self.cell(input_embedding, state, scope)
        tf.multiply(edge_state, )
        global_state = None


class FastGRUCell(tf.nn.rnn_cell.GRUCell):
    def __init__(self, num_units):
        super(FastGRUCell, self).__init__(num_units)

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("inputs.shape[-1] not known")

        input_depth = inputs_shape[1].value
        if input_depth != self._num_units:
            raise ValueError("input_depth: %d not match with num_units %d"
                             % (input_depth, self._num_units))
        self._gate_kernel = self.add_variable(
            "gates/%s" % "kernel",
            shape=[3 * self._num_units, 2 * self._num_units],
            initializer=self._kernel_initializer)
        self._gate_bias = self.add_variable(
            "gates/%s" % "bias",
            shape=[2 * self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)))
        self._candidate_kernel = self.add_variable(
            "candidate/%s" % "kernel",
            shape=[2 * self._num_units, self._num_units],
            initializer=self._kernel_initializer)
        self._candidate_bias = self.add_variable(
            "candidate/%s" % "bias",
            shape=[self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype)))

        self.built = True

    def call(self, inputs, state):
        fast_weight = math_ops.multiply(inputs, state)
        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state, fast_weight], 1), self._gate_kernel
        )
        gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

        value = math_ops.sigmoid(gate_inputs)
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state

        candidate = math_ops.matmul(
            array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        candidate = nn_ops.bias_add(candidate, self._candidate_bias)

        c = self._activation(candidate)
        new_h = u * state + (1 - u) * c
        return new_h, new_h

