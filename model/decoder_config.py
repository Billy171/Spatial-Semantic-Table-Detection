class TableDetectionDecoderConfig:
    def __init__(self, num_queries=10, d_model=561, nhead=11,
                 dropout=0.1, dim_feedforward=2048, activation="relu",
                 num_layers=6, num_classes=1):

        self.num_queries = num_queries
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward
        self.activation = activation
        self.num_layers = num_layers
        self.num_classes = num_classes






