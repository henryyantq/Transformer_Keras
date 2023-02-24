from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np

# Define the sequence length and number of input features
seq_length = 50
num_features = 1

# Generate a random sequence of values
sequence = np.random.rand(1000)

# Split the sequence into input/output pairs
inputs = []
outputs = []
for i in range(len(sequence) - seq_length):
    inputs.append(sequence[i:i+seq_length])
    outputs.append(sequence[i+seq_length])
inputs = np.array(inputs)
outputs = np.array(outputs)

# Reshape the inputs to be compatible with the LSTM layer
inputs = np.reshape(inputs, (inputs.shape[0], seq_length, num_features))

# Define the autoregression model
model = Sequential()
model.add(LSTM(128, input_shape=(seq_length, num_features)))
model.add(Dense(1))

# Compile and train the model
model.compile(optimizer='adam', loss='mse')
model.fit(inputs, outputs, epochs=100)

# Generate predictions for the next 10 steps
preds = []
current_input = inputs[-1]
for i in range(10):
    pred = model.predict(current_input.reshape((1, seq_length, num_features)))
    preds.append(pred[0][0])
    current_input = np.concatenate((current_input[1:], pred), axis=0)

# Print the predicted sequence
print(preds)
