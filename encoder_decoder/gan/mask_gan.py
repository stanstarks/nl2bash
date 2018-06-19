from encoder_decoder import data_utils, graph_utils


class GANModel(graph_utils.NNModel):

    def __init__(self, hyperparams, buckets=None):
        """
        params:
          buckets: list of pairs (I, O) indicating the I/O length
        """
        super(GANModel, self).__init__(hyperparams, buckets)
        self.generator
