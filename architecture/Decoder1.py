import numpy as np
from architecture.Attention import Attention
from architecture.Embedding import Embedding

class FirstDecoder(Attention):
    def __init__(self, transitions: dict, embed: Embedding):
        super().__init__(embed.d)
        self.transitions = transitions
        self.embed = embed
        self.num_symbols, self.num_states = len(embed.symbols), len(embed.states)

        # Initialize M_delta for f_2
        M_delta = np.zeros((self.num_symbols * self.num_states, 2 * self.num_symbols * self.num_states))
        for key, value in self.transitions.items():
            q_curr, s_curr = key  # q, s
            q_next, s_write, move_prev = value  # q', s', m

            pi_curr = self.pi(q_curr, s_curr)  # pi(q, s)
            pi_prime_next = self.pi_prime(q_next, s_write, move_prev)  # pi'(q', s', m)

            M_delta[pi_curr][pi_prime_next] = 1

        self.M_delta = M_delta

        # Initialize A for f_3
        A = np.zeros(
            (
                2 * self.num_symbols * self.num_states,
                self.num_symbols + self.num_states + 1,
            )
        )
        for q_next, s_write, move_prev in self.transitions.values():
            pi_prime_next = self.pi_prime(q_next, s_write, move_prev)
            hot_sym, hot_state = embed.hot_syms[s_write], embed.hot_states[q_next]
            hot_vec_qsm = np.concatenate([hot_state, hot_sym, np.array([move_prev])])

            A[pi_prime_next] = hot_vec_qsm

        self.A = A

    # g_1 in proof of Lemma 8
    def g_1(self, q: str, s: str) -> np.ndarray:
        assert s in self.embed.symbols, "Symbol must be a valid symbol"
        assert q in self.embed.states, "State must be a valid state"
        # State and symbol are strings
        hot_sym, hot_state = self.embed.hot_syms[s], self.embed.hot_states[q]

        # Create all S_i
        S = []
        for i in range(self.num_symbols):
            S_i = np.zeros((self.num_symbols, self.num_states))
            S_i[i] = np.ones(self.num_states)
            S.append(S_i)

        # Calculate v_qs
        v_qs = np.zeros((self.num_symbols, self.num_states))
        for i in range(self.num_symbols):
            v_qs[i] = np.matmul(hot_sym, S[i]) + hot_state

        return v_qs.flatten() - np.ones(self.num_symbols * self.num_states)

    # f_1 in proof of Lemma 8
    def f_1(self, q: str, s: str) -> np.ndarray:
        assert s in self.embed.symbols, "Symbol must be a valid symbol"
        assert q in self.embed.states, "State must be a valid state"

        g_1 = self.g_1(q, s)  # g_1(q, s)

        # Output: [[(q,s)]] contains 1 index i(s-1)|Q| + i(q) where i(.) is the index of number 1 in the symbol/state one hot vector
        return self.linear_sigmoid(g_1)

    # Calculate argmax() of one-hot of [symbol and state] from one-hot of symbol and one-hot of state
    def pi(self, q: str, s: str) -> np.ndarray:
        assert s in self.embed.symbols, "Symbol must be a valid symbol"
        assert q in self.embed.states, "State must be a valid state"
        # q and s are strings
        hot_sym, hot_state = self.embed.hot_syms[s], self.embed.hot_states[q]
        index_q = np.argmax(hot_state) + 1  # 1-indexed
        index_s = np.argmax(hot_sym) + 1  # 1-indexed
        return (index_s - 1) * len(hot_state) + index_q - 1  # 0-indexed to insert into one hot vector

    # Calculate argmax() of one-hot of [symbol and state and previous move] from one-hot of symbol and one-hot of state
    def pi_prime(self, q: str, s: str, m: int) -> np.int64:
        assert s in self.embed.symbols, "Symbol must be a valid symbol"
        assert q in self.embed.states, "State must be a valid state"
        # q and s are strings

        if m == -1:  # pi'(q,s,m) = pi(q,s)
            return self.pi(q, s)
        elif m == 1:  # pi'(q,s,m) = pi(q,s) + 1
            return self.pi(q, s) + self.num_states * self.num_symbols

    # f_2 in proof of Lemma 8
    def f_2(self, q: str, s: str) -> np.ndarray:
        assert s in self.embed.symbols, "Symbol must be a valid symbol"
        assert q in self.embed.states, "State must be a valid state"

        qs = self.f_1(q, s)  # v_qs

        return np.matmul(qs, self.M_delta)  # delta(q, s) as one-hot vector

    # f_3 in proof of Lemma 8
    def f_3(self, q: str, s: str) -> np.ndarray:
        assert s in self.embed.symbols, "Symbol must be a valid symbol"
        assert q in self.embed.states, "State must be a valid state"

        delta_qs = self.f_2(q, s)  # delta(q, s)

        return np.matmul(delta_qs, self.A)  # [[q], [s], m]

    # Output of first layer of decoder
    # Outputs (q, s, m) as one-hot vectors
    # Simulate a transition function

    def O_1(self, a_i: np.ndarray) -> np.ndarray:
        assert isinstance(a_i, np.ndarray), "Input must be a NumPy array"

        q_hot = a_i[self.embed.section_indices["q1"] : self.embed.section_indices["s1"]]
        s_hot = a_i[self.embed.section_indices["s1"] : self.embed.section_indices["x1"]]
        m_prev = a_i[self.embed.section_indices["x1"]]

        q = self.embed.get_state_from_one_hot(q_hot)
        s = self.embed.get_sym_from_one_hot(s_hot)

        print(f"q: {q}, s: {s}")

        f_3 = self.f_3(q, s)

        h_4 = np.concatenate([-q_hot, -s_hot, -np.array([m_prev]), f_3, np.array([m_prev])])
        zeroes = np.zeros(self.embed.d - len(h_4))
        return np.concatenate([h_4, zeroes])

    def decode(self, Y: np.ndarray, K_e: np.ndarray, V_e: np.ndarray) -> np.ndarray:
        assert isinstance(Y, np.ndarray), "Input must be a NumPy array"

        Z_1 = []
        KV_d = []

        Y_i = np.array([Y[0]])
        # print(f"Y_i shape: {Y_i.shape}")
        for i in range(len(Y)):
            K_i, V_i = self.K(Y_i), self.V(Y_i)
            KV_d.append((K_i, V_i))  # K and V up until y_i
            # print(f"Y_i shape: {Y[i].shape}, K_i shape: {K_i.shape}, V_i shape: {V_i.shape}")
            # print(f"Y[i]: {Y[i]}")
            p_i = self.embed.pos(i) + Y[i]  # Self attention to add positional encoding
            a_i = self.attention(p_i, K_e, V_e) + p_i  # Cross attention with encoder TODO: alpha and beta wrong

            # print(f"a_i:  {a_i}")
            z_i = self.O_1(a_i) + a_i

            Z_1.append(z_i)

            if i > 0:
                Y_i = np.vstack([Y_i, Y[i]])

        return np.array(Z_1)