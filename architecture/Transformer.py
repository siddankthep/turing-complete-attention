import numpy as np
from architecture.Embedding import Embedding
from architecture.Encoder import Encoder
from architecture.Decoder1 import FirstDecoder
from architecture.Decoder2 import SecondDecoder
from architecture.Decoder3 import ThirdDecoder
from architecture.FinalTransform import FinalTransform
from utils import print_in_box


from typing import List, Tuple


class TuringCompleteTransformers:
    def __init__(
        self,
        states: List[str],
        symbols: List[str],
        transitions: dict,
        accept_states: List[str],
    ):
        self.embed = Embedding(symbols, states)
        self.encoder = Encoder(self.embed)
        self.decoder_1 = FirstDecoder(transitions, self.embed)
        self.decoder_2 = SecondDecoder(transitions, self.embed)
        self.decoder_3 = ThirdDecoder(transitions, self.embed)
        self.final_transform = FinalTransform(self.embed)
        self.hot_accept_states = [self.embed.hot_states[accept_state] for accept_state in accept_states]

    def generate(self, input_w: str) -> np.ndarray:
        infinite_input = "#" + input_w + "#"  # Add blank symbols to the input
        X = self.embed.create_embedding_x(infinite_input)
        K, V = self.encoder.encode(X)  # Encode the input
        Y = np.vstack([self.embed.create_embedding_y("q_init", "#", "N")])  # Static initial state and symbol
        last_state = self.embed.hot_states["q_init"]  # Latest state to check if we have reached an accept state

        # print(f"Hot accept states: {self.hot_accept_states}")

        while not np.any(np.all(self.hot_accept_states == last_state, axis=1)):
            Z_1 = self.decoder_1.decode(Y, K, V)
            Z_2 = self.decoder_2.decode(Z_1)
            Z_3 = self.decoder_3.decode(Z_2)
            y_next = self.final_transform.F_final(Z_3)[-1]  # The final embedding vector, which is y_i+1

            Y = np.vstack([Y, y_next])
            last_state = y_next[self.embed.section_indices["q1"] : self.embed.section_indices["s1"]]
            last_symbol = y_next[self.embed.section_indices["s1"] : self.embed.section_indices["x1"]]
            print_in_box(f"Next state: {self.embed.get_state_from_one_hot(last_state)}\nNext symbol: {self.embed.get_sym_from_one_hot(last_symbol)}")
            print("\n-----------------------------------\n")

        return Y

    # Pretty print only
    def Y_to_steps(self, Y: np.ndarray) -> List[Tuple[str, str, int]]:
        steps = []
        for y in Y:
            q = self.embed.get_state_from_one_hot(y[self.embed.section_indices["q1"] : self.embed.section_indices["s1"]])
            s = self.embed.get_sym_from_one_hot(y[self.embed.section_indices["s1"] : self.embed.section_indices["x1"]])
            step = (q, s)
            steps.append(step)

        return steps

    # Main function to simulate a Turing machine
    def simulate_turing_machine(self, input_w: str) -> List[Tuple[str, str, int]]:
        assert isinstance(input_w, str), "Input must be a string"
        assert all([sym in self.embed.symbols for sym in input_w]), "Input must be a valid sequence of symbols"

        Y = self.generate(input_w)
        steps = self.Y_to_steps(Y)
        for i in range(len(steps)):
            q, s = steps[i]
            print(f"Step {i + 1}: ({q} {s})")
        return self