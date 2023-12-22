import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

# Attempting to Read Data from an Excel File:
try:
    # Try reading data from the 'concrete_data.xlsx' Excel file using Pandas.
    df = pd.read_excel('concrete_data.xlsx')

except FileNotFoundError:
    # If the file is not found, handle the FileNotFoundError.
    print("Error: File not found.")
    exit()

except pd.errors.EmptyDataError:
    # If the file is found but contains no data, handle the EmptyDataError.
    print("Error: Empty data in the file.")
    exit()

# Extracting Features and Targets from the DataFrame:
# - Features are obtained by dropping the 'concrete_compressive_strength' column.
# - Targets are obtained by selecting only the 'concrete_compressive_strength' column.
features = df.drop('concrete_compressive_strength', axis=1).values
targets = df['concrete_compressive_strength'].values

# Counting the Number of Samples:
# - 'num_samples' is the total number of samples in the dataset.
num_samples = features.shape[0]

# Calculate the number of training samples by taking 75% of the total number of samples
num_train = int(0.75 * num_samples)

# Generate an array of indices representing the samples
indices = np.arange(num_samples)

# Shuffle the indices randomly to create a randomized order of samples
np.random.shuffle(indices)

# Select the first 'num_train' indices for training and the rest for testing
train_indices = indices[:num_train]
test_indices = indices[num_train:]

# Extract the corresponding features and targets for training and testing sets
features_train = features[train_indices]
targets_train = targets[train_indices]
features_test = features[test_indices]
targets_test = targets[test_indices]

# Normalize the features using the mean and standard deviation of the training set
# we use normalization for:

# 1. Leveling the Playing Field:
#    - Normalization ensures that all input factors have an equal say in the model's learning process. It prevents one feature from overshadowing others just because it has a larger scale.

# 2. Smoother Learning Process:
#    - For our model to learn effectively, we want it to focus on the patterns in our data, not be distracted by differences in scale. Normalization smoothens the learning journey, making it more efficient and accurate.
def normalize_features(features, mean, std):
    return (features - mean) / std

# Calculate the mean and standard deviation of the training features
mean_train = features_train.mean(axis=0)
std_train = features_train.std(axis=0)

# Normalize the training and testing features using the mean and standard deviation of the training set
features_train = normalize_features(features_train, mean_train, std_train)
features_test = normalize_features(features_test, mean_train, std_train)

# Calculate the mean and standard deviation of the training targets
mean_target = targets_train.mean()
std_target = targets_train.std()

# Normalize the training and testing targets using the mean and standard deviation of the training set
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
        # Calculate the sigmoid activation function for a given input 'x'
        return 1 / (1 + np.exp(-x))

                                                        # - 'x' is the input to the sigmoid function.
                                                        # - 'np.exp(-x)' calculates the exponent of the negative input.
                                                        # - '1 / (1 + np.exp(-x))' computes the sigmoid activation function, transforming 'x' to a value between 0 and 1.
    def sigmoid_derivative(self, x):
        # Calculate the derivative of the sigmoid activation function for a given input 'x'
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
                                                        # - 'x' is the input to the sigmoid derivative.
                                                        # - 'self.sigmoid(x)' computes the sigmoid function for the given input.
                                                        # - '1 - self.sigmoid(x)' calculates (1 - sigmoid(x)), representing the complementary part of the sigmoid function.
                                                        # - The product of the sigmoid function and its complementary part gives the derivative of the sigmoid.

    def setParameters(self, epochs, learning_rate):
        # Set the number of training epochs and learning rate for the neural network
        self.epochs = epochs
        self.learning_rate = learning_rate                                                        # - 'epochs' is the number of training epochs, which represents the number of times the entire training dataset is passed through the network.
                                                        # - 'learning_rate' is the step size used during gradient descent to update the model parameters.
                                                        # - 'self.epochs = epochs' sets the number of training epochs for the neural network object.
                                                        # - 'self.learning_rate = learning_rate' sets the learning rate for the neural network object.

    def forward_propagation(self, X):
        # Calculate the input to the hidden layer by performing a dot product of input features and weights,
        # then adding the bias for the hidden layer
        self.hidden_layer_input = np.dot(X, self.weights_input_to_hidden) + self.bias_hidden

        # Apply the sigmoid activation function to the hidden layer input to obtain the hidden layer output
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)

        # Calculate the input to the output layer by performing a dot product of hidden layer output and weights,
        # then adding the bias for the output layer
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_to_output) + self.bias_output

        # Return the calculated input to the output layer
        return self.output_layer_input

    
    def backward_propagation(self, feature, target, output):
        # Calculate the error in the prediction by finding the difference between the target and the actual output
        prediction_error = target - output      # - 'prediction_error' represents the difference between the predicted output and the actual target.

        # Calculate the delta (sensitivity to change) at the output layer using the prediction error
        output_layer_delta = self.calculateOutputDelta(prediction_error, output)    # how much a change in the output layer's input contributes to the change in the loss.

        # Calculate the error at the hidden layer by propagating the output layer delta backward
        hidden_layer_error = self.calculateHiddenError(output_layer_delta)  # the output layer delta through the weights connecting the hidden and output layers.

        # Calculate the delta at the hidden layer using the hidden layer error
        hidden_layer_delta = self.calculateHiddenDelta(hidden_layer_error)  # how much a change in the hidden layer's input contributes to the change in the loss.

        # Update the weights and biases based on the calculated deltas and the input feature
        self.update_weight_bias(feature, output_layer_delta, hidden_layer_delta)    #   through the 'update_weight_bias' method, which contributes to the model's learning process.


    def calculateOutputDelta(self, error, output):
        # Calculate the delta at the output layer using the provided error and the derivative of the sigmoid function
        return error * self.sigmoid_derivative(output)
    
    def calculateHiddenError(self, output_delta):
        # Calculate the error at the hidden layer by propagating the output delta backward
        return np.dot(output_delta, np.transpose(self.weights_hidden_to_output))

    def calculateHiddenDelta(self, hidden_error):
        # Calculate the delta at the hidden layer using the provided hidden layer error
        return hidden_error * self.sigmoid_derivative(self.hidden_layer_output)

    def calculateMse(self, y_true, y_pred):
        # Calculate the Mean Squared Error (MSE) between true and predicted values
        return np.mean(np.square(y_true - y_pred))
    
    def update_weight_bias(self, feature, output_layer_delta, hidden_layer_delta):
        # Update weights and biases connecting the hidden layer to the output layer
        self.weights_hidden_to_output += np.dot(np.transpose(self.hidden_layer_output), output_layer_delta) * self.learning_rate # - The weights connecting the hidden layer to the output layer are updated based on the product
                                                                                                                                #   of the transpose of the hidden layer's output and the output layer delta, scaled by the learning rate.

        # Update biases at the output layer
        self.bias_output += np.sum(output_layer_delta, axis=0, keepdims=True) * self.learning_rate  #The biases at the output layer are updated by summing the output layer delta along the samples

        # Update weights and biases connecting the input layer to the hidden layer
        self.weights_input_to_hidden += np.dot(feature.T, hidden_layer_delta) * self.learning_rate  #   updated based on the product
                                                                                                    #   of the transpose of the input feature and the hidden layer delta, scaled by the learning rate.

        # Update biases at the hidden layer
        self.bias_hidden += np.sum(hidden_layer_delta, axis=0, keepdims=True) * self.learning_rate  # updated by summing the hidden layer delta along the samples
                                                                                                    #   axis (axis=0) and scaled by the learning rate.

    def update_weights(self, X_batch, y_batch):
        # - Forward propagation is performed to obtain the predicted output for the current batch.
        train_output = self.forward_propagation(X_batch)

        # - The 'backward_propagation' method is then called to calculate the deltas with respect
        self.backward_propagation(X_batch, y_batch, train_output)


    def train_batch(self, X_train, y_train, X_test, y_test, batch_size):
        # Set the acceptable error for early stopping
        acceptable_error = 0.01  

        # Initialize the previous test MSE to positive infinity
        prev_test_mse = float('inf')

        # Iterate through epochs
        for epoch in range(self.epochs):
            # Iterate through batches in the training data
            for i in range(0, len(X_train), batch_size):
                # Extract a batch of training features (X_batch) and corresponding targets (y_batch)

                X_batch = X_train[i:i + batch_size]  # Select a subset of training features starting from index i
                y_batch = y_train[i:i + batch_size]  # Select the corresponding subset of training targets
                                # Update weights and biases using the current batch
                self.update_weights(X_batch, y_batch)

            # Perform forward propagation on the test set
            test_output = self.forward_propagation(X_test)

            # Calculate the mean squared error (MSE) on the test set
            test_mse = self.calculateMse(y_test, test_output)

            # Print the current epoch and test MSE
            print(f'Epoch {epoch+1}, Test MSE: {test_mse}')

            # Check for early stopping conditions
            if test_mse < acceptable_error:
                print("Stopping training, error is less than acceptable value...")
                break
            elif epoch > 0 and test_mse > prev_test_mse:
                print("Early stopping...")
                break

            # Update the previous test MSE for the next iteration
            prev_test_mse = test_mse

    def predict(self, X):
        # Perform forward propagation to make predictions on input data
        return self.forward_propagation(X)


# Determine the number of features in the input data (number of columns)
input_size = features_train.shape[1]

# Set the size of the hidden layer in the neural network
hidden_size = 16  # Increasing hidden_size may allow the model to capture more complex patterns

# Set the size of the output layer in the neural network (1 for regression)
output_size = 1  # Adjust output_size based on the task, e.g., predicting multiple values

# Set the number of training epochs (iterations over the entire dataset)
epochs = 1000  # More epochs allow the model to see the training data more times, potentially improving performance

# Set the learning rate for the neural network optimization
learning_rate = 0.001  # Higher learning_rate can make the model learn faster but might lead to overshooting(finding the right balance between learning quickly and not missing it)

# Create an instance of the NeuralNetwork class with specified parameters
model = NeuralNetwork(input_size, hidden_size, output_size, epochs, learning_rate)  # It initializes the neural network with random weights and biases and sets hyperparameters for training.

# Train the neural network using batch training on the training data and validate on the test data
model.train_batch(
    features_train,                # Training features (input data)
    targets_train.reshape(-1, 1),  # Training targets (actual output), reshaped to a column vector
    features_test,                 # Test features for validation
    targets_test.reshape(-1, 1),   # Test targets for validation, reshaped to a column vector
    batch_size=16                  # Size of each batch used during training
)

# Make predictions on the test set
predictions = model.predict(features_test)

# Evaluate the model
test_mae = mean_absolute_error(targets_test, predictions)
test_r2 = model.calculateMse(targets_test, predictions)

# Print the evaluation results
print(f"Mean Absolute Error on Test Set: {test_mae}")
print(f"Mean Squared Error on Test Set: {test_r2}")

def get_user_input():
    # Prompt the user to enter the amount of cement and convert it to a float
    cement = float(input("Enter cement: "))

    # Prompt the user to enter the amount of water and convert it to a float
    water = float(input("Enter water: "))

    # Prompt the user to enter the amount of superplasticizer and convert it to a float
    superplasticizer = float(input("Enter superplasticizer: "))

    # Prompt the user to enter the age of the concrete and convert it to a float
    age = float(input("Enter age: "))

    # Create a NumPy array with the user input values
    user_data = np.array([cement, water, superplasticizer, age])

    # Normalize the user input features using the mean and standard deviation from the training set
    user_data_normalized = normalize_features(user_data, mean_train, std_train)

    # Reshape the normalized user data to match the input shape expected by the neural network
    user_data_reshaped = user_data_normalized.reshape(1, -1)

    # Return the reshaped and normalized user input
    return user_data_reshaped


# Get user input for prediction
user_data = get_user_input()

# Make predictions on user input
predicted_strength = model.predict(user_data)

# Denormalize the predicted strength
predicted_strength_actual = (predicted_strength * std_target) + mean_target

# Print the predicted strengths
print(f"The predicted cement strength is: {predicted_strength}")
print(f"The predicted actual cement strength is: {predicted_strength_actual}")