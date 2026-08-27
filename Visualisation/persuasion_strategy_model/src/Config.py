class Config(object):
    def __init__(self, embedding_size = 128, hidden_size = 256, feature_size = 7, output_size = 1):
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.output_size = output_size