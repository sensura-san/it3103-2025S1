import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Weights and biases for all gates and cell state
        # (W_xf, W_xi, W_xo, W_xc, W_hf, W_hi, W_ho, W_hc, b_f, b_i, b_o, b_c)
        # For simplicity, initialized similarly to RNN
        self.W_combined = np.random.randn(input_size + hidden_size, 4 * hidden_size) * 0.01
        self.b_combined = np.zeros((1, 4 * hidden_size))

        self.W_ho = np.random.randn(hidden_size, output_size) * 0.01
        self.b_o = np.zeros((1, output_size))

    def forward(self, inputs):
        h = np.zeros((1, self.hidden_size)) # Hidden state
        c = np.zeros((1, self.hidden_size)) # Cell state
        outputs = []

        for x in inputs:
            x = x.reshape(1, -1)
            concat_input = np.concatenate((x, h), axis=1)

            # Compute gate activations and candidate cell state
            gates_and_candidate = np.dot(concat_input, self.W_combined) + self.b_combined
            f_gate = self._sigmoid(gates_and_candidate[:, :self.hidden_size])
            i_gate = self._sigmoid(gates_and_candidate[:, self.hidden_size:2*self.hidden_size])
            o_gate = self._sigmoid(gates_and_candidate[:, 2*self.hidden_size:3*self.hidden_size])
            c_candidate = np.tanh(gates_and_candidate[:, 3*self.hidden_size:])

            # Update cell state and hidden state
            c = f_gate * c + i_gate * c_candidate
            h = o_gate * np.tanh(c)

            y = np.dot(h, self.W_ho) + self.b_o
            outputs.append(y)
        return np.array(outputs)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))