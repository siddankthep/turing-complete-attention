from architecture.Transformer import TuringCompleteTransformers

# Define the Turing machine
symbols = ["a", "b", "X", "#"]
states = ["q_init", "q_read", "q1", "q2", "q3", "q4", "q5"]

# Define Turing machine transition function
# delta(q,s) = (q', s', m)
transitions = {
    # q_init
    ("q_init", "#"): ("q_read", "#", 1),
    # q_read
    ("q_read", "a"): ("q_read", "a", 1),
    ("q_read", "b"): ("q_read", "b", 1),
    ("q_read", "X"): ("q_read", "b", 1),
    ("q_read", "#"): ("q1", "#", -1),
    # q1
    ("q1", "X"): ("q1", "X", -1),
    ("q1", "a"): ("q2", "X", -1),
    ("q1", "b"): ("q3", "X", -1),
    ("q1", "#"): ("q5", "#", -1),
    # q2
    ("q2", "a"): ("q2", "a", -1),
    ("q2", "X"): ("q2", "X", -1),
    ("q2", "b"): ("q4", "X", 1),
    # q3
    ("q3", "b"): ("q3", "b", -1),
    ("q3", "X"): ("q3", "X", -1),
    ("q3", "a"): ("q4", "X", 1),
    # q4
    ("q4", "X"): ("q4", "X", 1),
    ("q4", "a"): ("q4", "a", 1),
    ("q4", "b"): ("q4", "b", 1),
    ("q4", "#"): ("q1", "#", -1),
}
accept_states = ["q5"]
trans = TuringCompleteTransformers(states, symbols, transitions, accept_states)

if __name__ == "__main__":
    input_str = "ab"
    trans.simulate_turing_machine(input_str)
