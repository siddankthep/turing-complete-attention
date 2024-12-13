import numpy as np
from architecture.Attention import Attention
from architecture.Embedding import Embedding

class ThirdDecoder(Attention):
    def __init__(self, transitions: dict, embed: Embedding):
        super().__init__(embed.d)
        self.transitions = transitions
        self.embed = embed
        self.num_symbols, self.num_states = len(embed.symbols), len(embed.states)

        # W_q3 = np.zeros((embed.d, embed.d))
        self.W_q[embed.section_indices["x4"]][embed.section_indices["x9"]] = 1
        self.W_q[embed.section_indices["x10"]][embed.section_indices["x10"]] = 1
        self.W_q[embed.section_indices["x11"]][embed.section_indices["x11"]] = 1 / 3

        # self.W_q3 = W_q3

        # W_k3 = np.zeros((embed.d, embed.d))
        self.W_k[embed.section_indices["x10"]][embed.section_indices["x9"]] = 1
        self.W_k[embed.section_indices["x5"]][embed.section_indices["x10"]] = -1
        self.W_k[embed.section_indices["x11"]][embed.section_indices["x11"]] = 1

        # self.W_k3 = W_k3

        # W_v3 = np.zeros((embed.d, embed.d))
        self.W_v[embed.section_indices["s2"]][embed.section_indices["s4"]] = 1
        self.W_v[embed.section_indices["s2"] + 1][embed.section_indices["s4"] + 1] = 1
        self.W_v[embed.section_indices["s2"] + 2][embed.section_indices["s4"] + 2] = 1
        self.W_v[embed.section_indices["s2"] + 3][embed.section_indices["s4"] + 3] = 1
        self.W_v[embed.section_indices["x9"]][embed.section_indices["x7"]] = 1
        self.W_v[embed.section_indices["x8"]][embed.section_indices["x7"]] = -1

        # self.W_v3 = W_v3

    # def Q_3(self, X: np.ndarray) -> np.ndarray:
    #     assert isinstance(X, np.ndarray), "Input must be a NumPy array"

    #     return np.matmul(X, self.W_q3)

    # def K_3(self, X: np.ndarray) -> np.ndarray:
    #     assert isinstance(X, np.ndarray), "Input must be a NumPy array"

    #     return np.matmul(X, self.W_k3)

    # def V_3(self, X: np.ndarray) -> np.ndarray:
    #     assert isinstance(X, np.ndarray), "Input must be a NumPy array"

    #     return np.matmul(X, self.W_v3)

    def decode(self, Z_2: np.ndarray) -> np.ndarray:
        assert isinstance(Z_2, np.ndarray), "Input must be a NumPy array"

        Z_3 = []
        KV_d = []

        Z_i = np.array([Z_2[0]])
        # print(f"Z_i shape: {Z_i.shape}")
        for i in range(len(Z_2)):
            K_i, V_i = self.K(Z_i), self.V(Z_i)
            KV_d.append((K_i, V_i))
            # print(f"Z_i shape: {Z_2[i].shape}, K_i shape: {K_i.shape}, V_i shape: {V_i.shape}")
            # print(f"Z_2[i]: {Z_2[i]}")
            p_i = self.attention(self.Q(Z_2[i]), K_i, V_i) + Z_2[i]
            a_i = p_i

            # print(f"a_i:  {a_i}")
            z_i = a_i

            Z_3.append(z_i)

            if i > 0:
                Z_i = np.vstack([Z_i, Z_3[i]])

        return np.array(Z_3)