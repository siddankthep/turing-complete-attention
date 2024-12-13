import numpy as np
from architecture.Attention import Attention
from architecture.Embedding import Embedding

class Encoder(Attention):
    def __init__(self, embed: Embedding):
        self.embed = embed
        super().__init__(embed.d)

        # Hardcoded weight matrices
        self.W_k[self.d - 3][self.d - 4] = 1
        self.W_k[self.d - 4][self.d - 3] = -1

        self.W_v[self.d - 14][self.d - 14] = 1
        self.W_v[self.d - 13][self.d - 13] = 1
        self.W_v[self.d - 12][self.d - 12] = 1
        self.W_v[self.d - 11][self.d - 11] = 1
        self.W_v[self.d - 3][self.d - 10] = 1

    def encode(self, X: np.ndarray) -> np.ndarray:
        assert isinstance(X, np.ndarray), "Input must be a NumPy array"
        assert X.ndim == 2, "Input must be a matrix"
        assert X.shape[1] == self.d, "Input must have the correct embedding dimension"

        K = self.K(X)  # This will contain the positional encoding of the input
        V = self.V(X)  # This will contain the information of the input symbols

        self.K_e = K
        self.V_e = V

        return K, V