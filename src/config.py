import os

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#path
ROOT_DATA = os.path.join(BASE_DIR, "data")

ROOT_INPUT = os.path.join(BASE_DIR, "input")

ROOT_OUTPUT = os.path.join(BASE_DIR, "output")

ROOT_MODEL = os.path.join(BASE_DIR, "model")

ROOT_SOURCE = os.path.join(BASE_DIR, "src")



# Training Parameters
learning_rate = 0.001
training_steps = 10000
mini_batch_size = 10
display_step = 200

# Network Parameters
input_dim = 100 # không gian đầu vào
# timesteps = 28 # timesteps
hidden_dim = 128 # hidden layer num of features

