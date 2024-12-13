import numpy as np


class Embedding:
    def __init__(self, symbols: list, states: list):
        self.symbols = symbols
        self.states = states
        self.hot_syms = self.one_hot(symbols)
        self.hot_states = self.one_hot(states)
        self.section_indices = self.calculate_embedding_indices()
        self.d = (
            2 * len(self.states) + 4 * len(self.symbols) + 11
        )  # The vector stores 2 states, 4 symbols, and 11 numerical values, which will all be utilized in the embedding

    def get_sym_from_one_hot(self, one_hot: np.ndarray) -> str:
        assert isinstance(one_hot, np.ndarray), "Input must be a numpy array"
        assert one_hot.shape[0] == len(self.symbols), "Input must be a one-hot encoding of a symbol"

        print(f"One hot symbol: {one_hot}")
        for sym, hot in self.hot_syms.items():
            if np.array_equal(hot, one_hot):
                return sym

    def get_state_from_one_hot(self, one_hot: np.ndarray) -> str:
        assert isinstance(one_hot, np.ndarray), "Input must be a numpy array"
        assert one_hot.shape[0] == len(self.states), "Input must be a one-hot encoding of a state"

        print(f"One hot state: {one_hot}")
        for state, hot in self.hot_states.items():
            if np.array_equal(hot, one_hot):
                return state

    # One hot
    def one_hot(self, items: list) -> np.ndarray:
        assert isinstance(items, list), "Input must be an list of strings"

        one_hot_mat = np.eye(len(items))
        result = {}
        for i in range(len(items)):
            result[items[i]] = one_hot_mat[i]

        return result

    # Embedding indices
    def calculate_embedding_indices(self) -> dict:
        num_symbols = len(self.symbols)
        num_states = len(self.states)

        embedded_format = {
            "q1": num_states,
            "s1": num_symbols,
            "x1": 1,
            "q2": num_states,
            "s2": num_symbols,
            "x2": 1,
            "x3": 1,
            "x4": 1,
            "x5": 1,
            "s3": num_symbols,
            "x6": 1,
            "s4": num_symbols,
            "x7": 1,
            "x8": 1,
            "x9": 1,
            "x10": 1,
            "x11": 1,
        }

        current_index = 0
        section_indices = {}
        for key, length in embedded_format.items():
            section_indices[key] = current_index
            current_index += length

        return section_indices

    # Create embedding for input string, contains both the symbol and the position of the input
    def create_embedding_x(self, input_w: str) -> np.ndarray:
        """
        Embedding function for the Turing machine

        Args:
            input_w (str): Input string of length n

        Returns:
            np.ndarray: Embedded input string of size (n x d)
        """

        assert isinstance(input_w, str), "Input must be a string"
        assert [sym in self.symbols for sym in input_w], "Input must be a valid sequence of symbols"

        x = np.zeros((len(input_w), self.d))

        for i, sym in enumerate(input_w):
            i += 1  # 1-indexed
            s_i = self.hot_syms[sym]

            # X
            x_i = np.zeros(self.d)
            x_i[self.section_indices["s3"] : self.section_indices["x6"]] = s_i  # One-hot encoding of symbol
            x_i[self.section_indices["x8"] : self.section_indices["x11"] + 1] = np.array([1, i, 1 / i, 1 / i**2])  # Positional encoding
            x[i - 1] = x_i

        return x

    # Create embedding for computational step, contains the state, symbol, and move
    def create_embedding_y(self, state: str, sym: str, move: str) -> np.ndarray:
        """
        Embedding function for the Turing machine

        Args:
            input_w (str): Input string of length n

        Returns:
            np.ndarray: Embedded input string of size (n x d)
        """

        moves = {"L": -1, "R": 1, "N": 0}
        assert isinstance(move, str), "Move must be a string"
        assert move in moves.keys(), "Input must be a valid move"
        assert isinstance(sym, str), "Symbol must be a string"
        assert sym in self.symbols, "Input must be a valid symbol"
        assert isinstance(state, str), "State must be a string"
        assert state in self.states, "State must be a valid state"

        s_i = self.hot_syms[sym]
        q_i = self.hot_states[state]

        # y_i
        y_i = np.zeros(self.d)
        y_i[self.section_indices["q1"] : self.section_indices["s1"]] = q_i  # One-hot encoding of state
        y_i[self.section_indices["s1"] : self.section_indices["x1"]] = s_i  # One-hot encoding of symbol
        y_i[self.section_indices["x1"]] = moves[move]  # Move at previous step i.e how did we get here

        return y_i

    # Function to compute positional encoding for each computational step
    def pos(self, i: int) -> np.ndarray:
        i += 1  # 1-indexed
        out = np.zeros(self.d)
        out[self.section_indices["x8"] : self.section_indices["x11"] + 1] = np.array([1, i + 1, 1 / (i + 1), 1 / (i + 1) ** 2])  # Positional encoding
        return out
