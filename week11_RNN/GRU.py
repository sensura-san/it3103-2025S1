import numpy as np

class GRU:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Weights and biases for gates and candidate hidden state
        # (W_xz, W_xr, W_xh_tilde, W_hz, W_hr, W_hh_tilde, b_z, b_r, b_h_tilde)
        # For simplicity, initialized similarly to RNN
        self.W_combined = np.random.randn(input_size + hidden_size, 3 * hidden_size) * 0.01
        self.b_combined = np.zeros((1, 3 * hidden_size))

        self.W_ho = np.random.randn(hidden_size, output_size) * 0.01
        self.b_o = np.zeros((1, output_size))

    def forward(self, inputs):
        h = np.zeros((1, self.hidden_size)) # Hidden state
        outputs = []

        for x in inputs:
            x = x.reshape(1, -1)
            
            # Reset and Update gates
            concat_input_gates = np.concatenate((x, h), axis=1)
            gates = np.dot(concat_input_gates, self.W_combined[:, :2*self.hidden_size]) + self.b_combined[:, :2*self.hidden_size]
            z_gate = self._sigmoid(gates[:, :self.hidden_size])
            r_gate = self._sigmoid(gates[:, self.hidden_size:])

            # Candidate hidden state
            h_tilde_input = np.concatenate((x, r_gate * h), axis=1)
            h_tilde = np.tanh(np.dot(h_tilde_input, self.W_combined[:, 2*self.hidden_size:]) + self.b_combined[:, 2*self.hidden_size:])

            # Update hidden state
            h = (1 - z_gate) * h + z_gate * h_tilde

            y = np.dot(h, self.W_ho) + self.b_o
            outputs.append(y)
        return np.array(outputs)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))