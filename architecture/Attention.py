import numpy as np


class Attention:
    def __init__(self, d: int):
        self.d = d
        self.W_k = np.zeros((d, d))
        self.W_v = np.zeros((d, d))
        self.W_q = np.zeros((d, d))

    # Q = XW_q
    def Q(self, x: np.ndarray) -> np.ndarray:
        assert isinstance(x, np.ndarray), "Input must be a NumPy array"

        q = np.matmul(x, self.W_q)

        return q

    # K = XW_k
    def K(self, X: np.ndarray) -> np.ndarray:
        assert isinstance(X, np.ndarray), "Input must be a NumPy array"

        K = np.matmul(X, self.W_k)

        return K

    # V = XW_v
    def V(self, X: np.ndarray) -> np.ndarray:
        assert isinstance(X, np.ndarray), "Input must be a NumPy array"

        V = np.matmul(X, self.W_v)

        return V

    # Standard ReLU activation function
    def relu(self, x: np.ndarray) -> np.ndarray:
        assert isinstance(x, np.ndarray), "Input must be a NumPy array"

        return np.maximum(x, 0)

    # Piecewise-linear sigmoidal activation function
    # Output is either 0 or 1
    def linear_sigmoid(self, x: np.ndarray) -> np.ndarray:
        assert isinstance(x, np.ndarray), "Input must be a NumPy array"

        return self.relu(x) - self.relu(x - 1)

    # Calculate the negative of the absolute value of the dot product of two vectors
    def score(self, u: np.ndarray, v: np.ndarray) -> float:
        assert isinstance(u, np.ndarray), "Input must be a NumPy array"
        assert isinstance(v, np.ndarray), "Input must be a NumPy array"

        return -abs(np.dot(u, v))

    # Hardmax function
    def hardmax(self, u: np.ndarray) -> np.ndarray:
        assert isinstance(u, np.ndarray), "Input must be a NumPy array"
        assert u.ndim == 1, "u must a 1D array"

        max_value = np.max(u)
        max_count = np.sum(u == max_value)

        if max_count > 1:
            u = np.where(u == max_value, 1 / max_count, 0)
        else:
            u = np.where(u == max_value, 1, 0)

        return u

    # Attention function as used by the author
    # Instead of using matrix multiplication, this calculate the attention of 1 row of the query matrix Q at a time.
    # Repeating this for all rows of Q gives the same result as matrix multiplication

    def attention(self, q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        assert isinstance(K, np.ndarray), "Input must be a NumPy array"
        assert isinstance(V, np.ndarray), "Input must be a NumPy array"
        assert isinstance(q, np.ndarray), "Input must be a NumPy array"

        s = np.zeros(len(K))
        a = 0

        # (s1,s2,...sn) = (score(q,K1),score(q,K2),...score(q,Kn))
        for i in range(len(K)):
            # print(f"q shape: {q.shape}, K[i] shape: {K[i].shape}")
            dot = self.score(q, K[i])

            # print("dot: ", dot)
            s[i] = dot

        # hardmax(s1,s2,...sn) = (a1,a2,...an)
        s = self.hardmax(s)
        # print("s: ", s)

        # a = sum(ai * Vi)
        for i in range(len(V)):
            a += s[i] * V[i]

        # print("a: ", a)

        return a
