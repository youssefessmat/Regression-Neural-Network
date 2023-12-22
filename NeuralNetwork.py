import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

try:
    df = pd.read_excel('concrete_data.xlsx')
except FileNotFoundError:
    print("Error: File not found.")
    exit()
except pd.errors.EmptyDataError:
    print("Error: Empty data in the file.")
    exit()
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

def normalize_features(features, mean, std):
    return (features - mean) / std


mean_train = features_train.mean(axis=0)
std_train = features_train.std(axis=0)
features_train = normalize_features(features_train, mean_train, std_train)
features_test = normalize_features(features_test, mean_train, std_train)  

mean_target = targets_train.mean()
std_target = targets_train.std()
targets_train = (targets_train - mean_target) / std_target
targets_test = (targets_test - mean_target) / std_target  

class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size, epochs=100, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.epochs = epochs            
        self.learning_rate = learning_rate
        self.weights_input_to_hidden = np.random.randn(self.input_size, self.hidden_size) / np.sqrt(self.input_size)
        self.weights_hidden_to_output = np.random.randn(self.hidden_size, self.output_size) / np.sqrt(self.hidden_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def setParameters(self, epochs, learning_rate):
        self.epochs = epochs
        self.learning_rate = learning_rate

    def forward_propagation(self, X):
        self.hidden_layer_input = np.dot(X, self.weights_input_to_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_to_output) + self.bias_output
        return self.output_layer_input
    
    def backward_propagation(self, feature, target, output):
        prediction_error = target - output
        output_layer_delta = self.calculateHiddenError(prediction_error, output)
        hidden_layer_error = self.calculateHiddenDelta(output_layer_delta)
        hidden_layer_delta = self.calculateMse(hidden_layer_error)
        self.update_weight_bias(feature, output_layer_delta, hidden_layer_delta)
    
    def calculateOutputDelta(self, error, output):
        return error * self.sigmoid_derivative(output)
    
    def calculateHiddenError(self, output_delta):
        return np.dot(output_delta, np.transpose(self.weights_hidden_to_output))
    def calculateHiddenDelta(self, hidden_error):
        return hidden_error * self.sigmoid_derivative(self.hidden_layer_output)
    def calculateMse(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))
    
    def update_weight_bias(self, feature, output_layer_delta, hidden_layer_delta):
        self.weights_hidden_to_output += np.dot(np.transpose(self.hidden_layer_output), output_layer_delta) * self.learning_rate
        self.bias_output += np.sum(output_layer_delta, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_to_hidden += np.dot(feature.T, hidden_layer_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_layer_delta, axis=0, keepdims=True) * self.learning_rate
        
    def update_weights(self, X_batch, y_batch):
        train_output = self.forward_propagation(X_batch)
        self.backward_propagation(X_batch, y_batch, train_output)
    

    def train_batch(self, X_train, y_train, X_test, y_test, batch_size):
            acceptable_error = 0.01  
            prev_test_mse = float('inf')
            for epoch in range(self.epochs):
                for i in range(0, len(X_train), batch_size):
                    X_batch, y_batch = X_train[i:i + batch_size], y_train[i:i + batch_size]
                    self.update_weights(X_batch, y_batch)
                test_output = self.forward_propagation(X_test)
                test_mse = self.calculateMse(y_test, test_output)
                print(f'Epoch {epoch+1}, Test MSE: {test_mse}')
                if test_mse < acceptable_error:
                    print("Stopping training, error is less than acceptable value...")
                    break
                elif epoch > 0 and test_mse > prev_test_mse:
                    print("Early stopping...")
                    break
                prev_test_mse = test_mse
    def predict(self, X):
        return self.forward_propagation(X)
input_size = features_train.shape[1]
hidden_size = 16
output_size = 1
epochs = 1000 
learning_rate = 0.001  
model = NeuralNetwork(input_size, hidden_size, output_size, epochs, learning_rate)
model.train_batch(features_train, targets_train.reshape(-1, 1), features_test, targets_test.reshape(-1, 1), batch_size=16)

predictions = model.predict(features_test)

test_mae = mean_absolute_error(targets_test, predictions)
test_r2 = model.calculateMse(targets_test, predictions)

print(f"Mean Absolute Error on Test Set: {test_mae}")
print(f"Mean squared Error on Test Set: {test_r2}")
def get_user_input():
    cement = float(input("Enter cement: "))
    water = float(input("Enter water: "))
    superplasticizer = float(input("Enter superplasticizer: "))
    age = float(input("Enter age: "))
    user_data = np.array([cement, water, superplasticizer, age])
    user_data_normalized = normalize_features(user_data, mean_train, std_train)
    user_data_reshaped = user_data_normalized.reshape(1, -1)   
    return user_data_reshaped
user_data = get_user_input()
predicted_strength = model.predict(user_data)
predicted_strength_actual = (predicted_strength * std_target) + mean_target
print(f"The predicted cement strength is: {predicted_strength}")
print(f"The predicted actualncement strength is: {predicted_strength_actual}")