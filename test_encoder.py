import numpy as np
import pandas as pd
from estrutura.encoder import TransformerEncoder


# vocabulário
vocab = {
    "o":0,
    "banco":1,
    "bloqueou":2,
    "cartao":3
}

df = pd.DataFrame(list(vocab.items()), columns=["palavra","id"])

print(df)


sentence = ["o","banco","bloqueou","cartao"]

ids = [vocab[w] for w in sentence]

print("IDs:", ids)


# parâmetros
d_model = 64
vocab_size = len(vocab)


# embeddings
embedding_table = np.random.randn(vocab_size, d_model)

X = embedding_table[ids]

X = np.expand_dims(X, axis=0)

print("Shape entrada:", X.shape)


encoder = TransformerEncoder(d_model)

Z = encoder.forward(X)

print("Shape saída:", Z.shape)