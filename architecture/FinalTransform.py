from typing import Tuple
import numpy as np
from architecture.Attention import Attention
from architecture.Embedding import Embedding


class FinalTransform(Attention):
    def __init__(self, embed: Embedding):
        super().__init__(embed.d)
        self.embed = embed
        self.num_symbols, self.num_states = len(embed.symbols), len(embed.states)

    # This should be implemented as a matrix multiplication, but I was out of time so I did it manually
    def f_1(self, z_3: np.ndarray) -> np.ndarray:
        q_next_hot = z_3[self.embed.section_indices["q2"] : self.embed.section_indices["s2"]]  # q_i+1
        m_curr = np.array([1, 0]) if z_3[self.embed.section_indices["x2"]] == 1 else np.array([0, 1])  # m_i
        alpha = z_3[self.embed.section_indices["s3"] : self.embed.section_indices["x6"]]  # alpha_i+1
        r_plus1 = z_3[self.embed.section_indices["x9"]]  # r_i+1
        beta = z_3[self.embed.section_indices["x6"]]  # beta_i+1
        v_l_r_plus1 = z_3[self.embed.section_indices["s4"] : self.embed.section_indices["x7"]]  # v_l_r+1
        blank = self.embed.hot_syms["#"]  # One-hot encoding of blank symbol
        l_r_plus1 = z_3[self.embed.section_indices["x7"]]  # l_r+1

        return np.concatenate(
            [
                q_next_hot,
                m_curr,
                alpha,
                np.array([r_plus1 - beta]),
                v_l_r_plus1,
                blank,
                np.array([l_r_plus1 - r_plus1 - 2]),
            ]
        )

    # Again, this should be implemented as a matrix multiplication but I didn't have enough time
    def f_2(self, f_1: np.ndarray) -> np.ndarray:
        linear_f_1 = self.linear_sigmoid(f_1)

        b_1_index = self.num_states + 2 + self.num_symbols
        b_2 = linear_f_1[-1]

        if b_2 == 0:
            return linear_f_1[: b_1_index + 1 + self.num_symbols], b_2
        else:
            return (
                np.concatenate(
                    [
                        linear_f_1[: b_1_index + 1],
                        linear_f_1[b_1_index + 1 + self.num_symbols : -1],
                    ]
                ),
                b_2,
            )

    # Should also be implemented as a matrix multiplication
    def f_3(self, f_2b_2: Tuple[np.ndarray, int]) -> np.ndarray:
        b_1_index = self.num_states + 2 + self.num_symbols

        f_2, b_2 = f_2b_2

        b_1 = f_2[b_1_index]

        if b_2 == 0 and b_1 == 0:
            return f_2[:b_1_index]
        elif b_2 == 0 and b_1 == 1:
            return np.concatenate(
                [
                    f_2[: b_1_index - self.num_symbols],
                    f_2[b_1_index + 1 : b_1_index + self.num_symbols + 1],
                ]
            )
        elif b_2 == 1 and b_1 == 0:
            return f_2[:b_1_index]
        else:
            return np.concatenate(
                [
                    f_2[: b_1_index - self.num_symbols],
                    f_2[b_1_index + self.num_symbols + 1 : -1],
                ]
            )

    # Should also be implemented as a matrix multiplication
    def f_4(self, f_3: np.ndarray) -> np.ndarray:
        hot_m_next = f_3[self.num_states : self.num_states + 2]
        m_next_index = np.argmax(hot_m_next)
        m_next = 1 if m_next_index == 0 else -1

        return np.concatenate([f_3[: self.num_states], f_3[self.num_states + 2 :], np.array([m_next])])  # (q_i+1, s_i+1, m_i+1)

    #
    def F_final(self, Z_3: np.ndarray) -> np.ndarray:
        F_out = []
        for i in range(len(Z_3)):
            z_3 = Z_3[i]
            f_1 = self.f_1(z_3)
            f_2b_2 = self.f_2(f_1)
            f_3 = self.f_3(f_2b_2)
            f_4 = self.f_4(f_3)
            zeroes = np.zeros(self.embed.d - len(f_4))  # Pad with 0 to match embedding dimension
            F_out.append(np.concatenate([f_4, zeroes]))  # y_i+1

        return np.array(F_out)
