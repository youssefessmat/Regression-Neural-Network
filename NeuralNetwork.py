import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_excel('concrete_data.xlsx')

##Normalize the data
# for column in df.columns:
#     if column != 'concrete strength':  # assuming 'concrete strength' is your target column

#         df[column] = df[column] / df[column].sum()

features = df.drop('concrete_compressive_strength', axis=1).values  
targets = df['concrete_compressive_strength'].values  


num_samples = features.shape[0]  
num_train = int(0.75 * num_samples)  

indices = np.arange(num_samples)
np.random.shuffle(indices)

train_indices = indices[:num_train]
test_indices = indices[num_train:]

features_train = features[train_indices]
targets_train = targets[train_indices]
features_test = features[test_indices]
targets_test = targets[test_indices]


# print("Training Features:\n", features_train)
# print("Training Targets:\n", targets_train)
# print("Testing Features:\n", features_test)
# print("Testing Targets:\n", targets_test)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, epochs=100, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.weights_input_to_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_to_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))#intialize el bias hena be zero 
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def set_hyperparameters(self, epochs, learning_rate):
        self.epochs = epochs
        self.learning_rate = learning_rate

