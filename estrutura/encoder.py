from estrutura.attention import SelfAttention
from estrutura.feed_forward import FeedForward
from estrutura.layer_norm import layer_norm


class EncoderLayer:

    def __init__(self, d_model):
        self.attention = SelfAttention(d_model)
        self.ffn = FeedForward(d_model)

    def forward(self, X):

        X_att = self.attention.forward(X)

        X_norm1 = layer_norm(X + X_att)

        X_ffn = self.ffn.forward(X_norm1)

        X_out = layer_norm(X_norm1 + X_ffn)

        return X_out


class TransformerEncoder:

    def __init__(self, d_model, num_layers=6):

        self.layers = [EncoderLayer(d_model) for _ in range(num_layers)]

    def forward(self, X):

        for layer in self.layers:
            X = layer.forward(X)

        return X