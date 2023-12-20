import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Load data from the file
try:
    df = pd.read_excel('concrete_data.xlsx')
except FileNotFoundError:
    print("Error: File not found.")
    exit()
except pd.errors.EmptyDataError:
    print("Error: Empty data in the file.")
    exit()

# Separate features and targets
features = df.drop('concrete_compressive_strength', axis=1).values
targets = df['concrete_compressive_strength'].values

# Split data into training (75%) and testing (25%) sets
num_samples = features.shape[0]
num_train = int(0.75 * num_samples)

indices = np.arange(num_samples)
np.random.shuffle(indices)

train_indices = indices[:num_train]
test_indices = indices[num_train:]

# Shuffle both features and targets based on shuffled indices
features_train = features[train_indices]
targets_train = targets[train_indices]
features_test = features[test_indices]
targets_test = targets[test_indices]

# Normalize the features using training set statistics
def normalize_features(features, mean, std):
    return (features - mean) / std

# Feature scaling (standardization)
mean_train = features_train.mean(axis=0)
std_train = features_train.std(axis=0)
features_train = normalize_features(features_train, mean_train, std_train)
features_test = normalize_features(features_test, mean_train, std_train)  # Use mean and std from training set for normalization

# Normalize targets
mean_target = targets_train.mean()
std_target = targets_train.std()
targets_train = (targets_train - mean_target) / std_target
targets_test = (targets_test - mean_target) / std_target  # Use mean and std from training set for normalization

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, epochs=100, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        # Use more sophisticated weight initialization
        self.weights_input_to_hidden = np.random.randn(self.input_size, self.hidden_size) / np.sqrt(self.input_size)
        self.weights_hidden_to_output = np.random.randn(self.hidden_size, self.output_size) / np.sqrt(self.hidden_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def set_hyperparameters(self, epochs, learning_rate):
        self.epochs = epochs
        self.learning_rate = learning_rate

    def forward_propagation(self, X):
        self.hidden_layer_input = np.dot(X, self.weights_input_to_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_to_output) + self.bias_output
        return self.output_layer_input  

    def backward_propagation(self, X, y, output):
        self.error = y - output
        self.output_delta = self.error * self.sigmoid_derivative(output)
        self.hidden_error = self.output_delta.dot(self.weights_hidden_to_output.T)
        self.hidden_delta = self.hidden_error * self.sigmoid_derivative(self.hidden_layer_output)

        self.weights_hidden_to_output += self.hidden_layer_output.T.dot(self.output_delta) * self.learning_rate
        self.bias_output += np.sum(self.output_delta, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_to_hidden += X.T.dot(self.hidden_delta) * self.learning_rate
        self.bias_hidden += np.sum(self.hidden_delta, axis=0, keepdims=True) * self.learning_rate

    def update_weights(self, X_batch, y_batch):
        # Calculate gradients for a batch and update weights
        train_output = self.forward_propagation(X_batch)
        self.backward_propagation(X_batch, y_batch, train_output)

    def train_batch(self, X_train, y_train, X_test, y_test, batch_size):
        prev_test_mse = float('inf')
        for epoch in range(self.epochs):
            for i in range(0, len(X_train), batch_size):
                X_batch, y_batch = X_train[i:i + batch_size], y_train[i:i + batch_size]
                self.update_weights(X_batch, y_batch)

            # Testing phase
            test_output = self.forward_propagation(X_test)
            test_mse = np.mean(np.square(y_test - test_output))

            print(f'Epoch {epoch+1}, Test MSE: {test_mse}')

            # Early stopping if the testing error increases
            if epoch > 0 and test_mse > prev_test_mse:
                print("Early stopping...")
                break

            prev_test_mse = test_mse

    def predict(self, X):
        return self.forward_propagation(X)

    def calculate_error(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

input_size = features_train.shape[1]
hidden_size = 8
output_size = 1
epochs = 500
learning_rate = 0.0001
model = NeuralNetwork(input_size, hidden_size, output_size, epochs, learning_rate)

# Perform cross-validation using Scikit-Learn
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_index, test_index in kf.split(features):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = targets[train_index], targets[test_index]

    # Feature scaling (standardization)
    X_train = normalize_features(X_train, X_train.mean(axis=0), X_train.std(axis=0))
    X_test = normalize_features(X_test, X_train.mean(axis=0), X_train.std(axis=0))  # Use mean and std from the training set for normalization

    # Normalize targets
    y_train = (y_train - y_train.mean()) / y_train.std()
    y_test = (y_test - y_train.mean()) / y_train.std()  # Use mean and std from the training set for normalization

    # Train the model on the current fold
    model.train_batch(X_train, y_train.reshape(-1, 1), X_test, y_test.reshape(-1, 1), batch_size=32)

# Train the model on the full training set
model.train_batch(features_train, targets_train.reshape(-1, 1), features_test, targets_test.reshape(-1, 1), batch_size=32)

# Make predictions on the test set
predictions = model.predict(features_test)

# Calculate and print additional error metrics on the test set
test_mae = mean_absolute_error(targets_test, predictions)
test_r2 = r2_score(targets_test, predictions)
print(f"Mean Absolute Error on Test Set: {test_mae}")
print(f"R-squared on Test Set: {test_r2}")

def get_user_input():
    # Get input from user
    cement = float(input("Enter cement: "))
    water = float(input("Enter water: "))
    superplasticizer = float(input("Enter superplasticizer: "))
    age = float(input("Enter age: "))

    # Create a numpy array from the input
    user_data = np.array([cement, water, superplasticizer, age])

    # Normalize the data using the mean and standard deviation from the training set
    user_data_normalized = normalize_features(user_data, mean_train, std_train)

    # Reshape the data to match the input shape of the neural network
    user_data_reshaped = user_data_normalized.reshape(1, -1)

    return user_data_reshaped

# Get user input
user_data = get_user_input()

# Use the neural network to predict the cement strength
predicted_strength = model.predict(user_data)
predicted_strength_actual = (predicted_strength * std_target) + mean_target

print(f"The predicted cement strength is: {predicted_strength}")
print(f"The predicted actualncement strength is: {predicted_strength_actual}")
