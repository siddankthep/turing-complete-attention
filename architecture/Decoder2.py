import numpy as np
from architecture.Attention import Attention
from architecture.Embedding import Embedding

class SecondDecoder(Attention):
    def __init__(self, transitions: dict, embed: Embedding):
        super().__init__(embed.d)
        self.transitions = transitions
        self.embed = embed
        self.num_symbols, self.num_states = len(embed.symbols), len(embed.states)

        # W_v2 = np.zeros((embed.d, embed.d))
        self.W_v[embed.d - 18][embed.d - 16] = 1
        self.W_v[embed.d - 17][embed.d - 15] = 1

        # self.W_v2 = W_v2

    # def V_2(self, X: np.ndarray) -> np.ndarray:
    #     assert isinstance(X, np.ndarray), "Input must be a NumPy array"

    #     return np.matmul(X, self.W_v2)

    # Decode to calculate the position of the head
    def decode(self, Z_1: np.ndarray) -> np.ndarray:
        assert isinstance(Z_1, np.ndarray), "Input must be a NumPy array"

        Z_2 = []
        KV_d = []

        Z_i = np.array([Z_1[0]])
        # print(f"Z_i shape: {Z_i.shape}")
        for i in range(len(Z_1)):
            K_i, V_i = self.K(Z_i), self.V(Z_i)  # K is zero matrix
            KV_d.append((K_i, V_i))  # K and V up until y_i

            # print(f"Z_i shape: {Z_1[i].shape}, K_i shape: {K_i.shape}, V_i shape: {V_i.shape}")
            # print(f"Z_1[i]: {Z_1[i]}")

            p_i = self.attention(self.Q(Z_1[i]), K_i, V_i) + Z_1[i]  # Lemma 9: Self attention to calculate c_i+1/i+1 and c_i/i+1
            a_i = p_i  # Residual connection only, O(.) is null function

            # print(f"a_i:  {a_i}")
            z_i = a_i

            Z_2.append(z_i)

            if i > 0:
                Z_i = np.vstack([Z_i, Z_2[i]])

        return np.array(Z_2)
