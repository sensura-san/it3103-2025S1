import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Weights and biases for input-to-hidden, hidden-to-hidden, and hidden-to-output
        self.W_xh = np.random.randn(input_size, hidden_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_h = np.zeros((1, hidden_size))

        self.W_ho = np.random.randn(hidden_size, output_size) * 0.01
        self.b_o = np.zeros((1, output_size))

    def forward(self, inputs):
        h = np.zeros((1, self.hidden_size))  # Initial hidden state
        outputs = []
        hidden_states = []

        for x in inputs:
            x = x.reshape(1, -1) # Reshape input
            h = np.tanh(np.dot(x, self.W_xh) + np.dot(h, self.W_hh) + self.b_h)
            y = np.dot(h, self.W_ho) + self.b_o
            outputs.append(y)
            hidden_states.append(h)
        return np.array(outputs), np.array(hidden_states)

    # Backward pass (BPTT) for training would be significantly more complex
    # and involves calculating gradients for all weights and biases.