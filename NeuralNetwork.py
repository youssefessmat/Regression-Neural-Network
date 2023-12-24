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
        self.hiddenBias = np.zeros((1, self.hidden_size))
        self.outputBias = np.zeros((1, self.output_size))


    def sigmoid(self, x):
        # Calculate the sigmoid activation function for a given input 'x'
        return 1 / (1 + np.exp(-x))  # transforming 'x' to a value between 0 and 1.
                                                       
    def sigmoid_derivative(self, x):
        # Calculate the derivative of the sigmoid activation function for a given input 'x'
        return self.sigmoid(x) * (1 - self.sigmoid(x)) # - The product of the sigmoid function and its complementary part gives the derivative of the sigmoid.

    def setParameters(self, epochs, learning_rate):
        # Set the number of training epochs and learning rate for the neural network
        self.epochs = epochs                   #the number of training epochs,the number of times the entire training dataset is passed through the network.
        self.learning_rate = learning_rate     # the step size used during gradient descent to update the model parameters.                                              
                                                        

    def forward_propagation(self, X):
        # Calculate the input to the hidden layer by performing a dot product of input features and weights,
        # then adding the bias for the hidden layer
        self.hiddenLayerInput = np.dot(X, self.weights_input_to_hidden) + self.hiddenBias

        # Apply the sigmoid activation function to the hidden layer input to obtain the hidden layer output
        self.hiddenLayerOutput = self.sigmoid(self.hiddenLayerInput)

        # Calculate the input to the output layer by performing a dot product of hidden layer output and weights,
        # then adding the bias for the output layer
        self.output_layer_input = np.dot(self.hiddenLayerOutput, self.weights_hidden_to_output) + self.outputBias

        # Return the calculated input to the output layer
        return self.output_layer_input

    
    def backward_propagation(self, feature, target, output):
        prediction_error = target - output      #represents the difference between the predicted output and the actual target.

        # Calculate the delta (sensitivity to change) at the output layer using the prediction error
        outputLayerchange = self.calculateOutputDelta(prediction_error, output)  # how much a change in the output layer's input contributes to the change in the loss.

        # Calculate the error at the hidden layer by propagating the output layer delta backward
        hidden_layer_error = self.calculateHiddenError(outputLayerchange)  # the output layer delta through the weights connecting the hidden and output layers.

        # Calculate the delta at the hidden layer using the hidden layer error
        hiddenLayerChange = self.calculateHiddenDelta(hidden_layer_error)  # how much a change in the hidden layer's input contributes to the change in the loss.

        # Update the weights and biases based on the calculated deltas and the input feature
        self.update_weight_bias(feature, outputLayerchange, hiddenLayerChange)    #   through the 'update_weight_bias' method, which contributes to the model's learning process.

    def calculateOutputDelta(self, error, output):
        # Calculate the delta at the output layer using the provided error and the derivative of the sigmoid function
        return error * self.sigmoid_derivative(output)
    
    def calculateHiddenError(self, output_delta):
        # Calculate the error at the hidden layer by propagating the output delta backward
        return np.dot(output_delta, np.transpose(self.weights_hidden_to_output))

    def calculateHiddenDelta(self, hidden_error):
        # Calculate the delta at the hidden layer using the provided hidden layer error
        return hidden_error * self.sigmoid_derivative(self.hiddenLayerOutput)

    def calculateMse(self, y_true, y_pred):
        # Calculate the Mean Squared Error (MSE) between true and predicted values
        return np.mean(np.square(y_true - y_pred))
    
    def update_weight_bias(self, feature, output_layer_delta, hidden_layer_delta):
        # Update weights and biases connecting the hidden layer to the output layer
        self.weights_hidden_to_output += np.dot(np.transpose(self.hiddenLayerOutput), output_layer_delta) * self.learning_rate # - The weights connecting the hidden layer to the output layer are updated based on the product
                                                                                                                                #   of the transpose of the hidden layer's output and the output layer delta, scaled by the learning rate.

        # Update biases at the output layer
        self.outputBias += np.sum(output_layer_delta, axis=0, keepdims=True) * self.learning_rate  #The biases at the output layer are updated by summing the output layer delta along the samples

        # Update weights and biases connecting the input layer to the hidden layer
        self.weights_input_to_hidden += np.dot(feature.T, hidden_layer_delta) * self.learning_rate  #   updated based on the product
                                                                                                    #   of the transpose of the input feature and the hidden layer delta, scaled by the learning rate.

        # Update biases at the hidden layer
        self.hiddenBias += np.sum(hidden_layer_delta, axis=0, keepdims=True) * self.learning_rate  # updated by summing the hidden layer delta along the samples
                                                                                                    #   axis (axis=0) and scaled by the learning rate.

    def update_weights(self, X_batch, y_batch):
        # - Forward propagation is performed to obtain the predicted output for the current batch.
        train_output = self.forward_propagation(X_batch)

        # - The 'backward_propagation' method is then called to calculate the deltas with respect
        self.backward_propagation(X_batch, y_batch, train_output)


    def train_batch(self, F_train, T_train, X_test, y_test, batchSize):
        # Set the acceptable error for early stopping
        acceptable_error = 0.01  

        # Initialize the previous test MSE to positive infinity
        prev_test_mse = float('inf')

        # Iterate through epochs
        for epoch in range(self.epochs):
            # Iterate through batches in the training data
            for i in range(0, len(F_train), batchSize):
                # Extract a batch of training features (X_batch) and corresponding targets (y_batch)

                xBatch = F_train[i:i + batchSize]  # Select a subset of training features starting from index i
                yBatch = T_train[i:i + batchSize]  # Select the corresponding subset of training targets
                                # Update weights and biases using the current batch
                self.update_weights(xBatch, yBatch)

            # Perform forward propagation on the test set
            test_output = self.forward_propagation(X_test)

            # Calculate the mean squared error (MSE) on the test set
            test_mse = self.calculateMse(y_test, test_output)

            # Print the current epoch and test MSE
            print(f'Epoch {epoch+1}, Test MSE: {test_mse}')

            # Check for early stopping conditions
            if test_mse < acceptable_error:
                print("error less than the acceptance error")
                break
            elif epoch > 0 and test_mse > prev_test_mse:
                print("Early stopping...the MSE is increasing")
                break

            # Update the previous test MSE for the next iteration
            prev_test_mse = test_mse

    def predict(self, X):
        # Perform forward propagation to make predictions on input data
        return self.forward_propagation(X)


# Determine the number of features in the input data (number of columns)
input_size = features_train.shape[1]

hidden_size = 16  # Increasing hidden_sizeallow the model to capture more complex patterns

output_size = 1  

epochs = 1000  # More epochs allow the model to see the training data more times, improving performance

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
    batchSize=16                  # Size of each batch used during training
)

# Make predictions on the test set
predictions = model.predict(features_test)

# Evaluate the model
MAEtest = mean_absolute_error(targets_test, predictions)
MSEtest = model.calculateMse(targets_test, predictions)
R2test = r2_score(targets_test, predictions)

# Print the evaluation results
print(f"Mean Absolute Error on Test Set: {MAEtest}")
print(f"Mean Squared Error on Test Set: {MSEtest}")
print(f"R-squared on Test Set: {R2test}")

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

# Print the predicted strengths
print(f"The predicted cement strength is: {predicted_strength}")
print(f"The predicted actual cement strength is: {predicted_strength_actual}")