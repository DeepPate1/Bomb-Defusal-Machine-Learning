# Import necessary libraries
import random
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import concurrent.futures

# Function to generate a wiring diagram with random color assignments
def generate_wiring_diagram():
    colors = ['Red', 'Blue', 'Yellow', 'Green']  # Define available colors
    matrix = [['' for _ in range(20)] for _ in range(20)]  # Initialize a 20x20 matrix
    color_order = []  # List to keep track of color order

    # Assign a random color to a randomly selected row
    row = random.randint(0, 19)
    color = random.choice(colors)
    color_order.append(color)
    for c in range(20):
        matrix[row][c] = color

    # Assign a different color to a randomly selected column
    col = random.randint(0, 19)
    second_color = random.choice([c for c in colors if c != color])
    color_order.append(second_color)
    for r in range(20):
        matrix[r][col] = second_color

    # Assign a third color to another row different from the first
    second_row = random.choice([r for r in range(20) if r != row])
    third_color = random.choice([c for c in colors if c != color and c != second_color])
    color_order.append(third_color)
    for c in range(20):
        matrix[second_row][c] = third_color

    # Assign a fourth color to another column different from the first
    second_col = random.choice([c for c in range(20) if c != col])
    fourth_color = next(c for c in colors if c not in [color, second_color, third_color])
    color_order.append(fourth_color)
    for r in range(20):
        matrix[r][second_col] = fourth_color

    return matrix

# Function to print the wiring diagram with color coding
def print_diagram(wiring_diagram):
    # Define color codes for printing
    color_codes = {
        'Red': '\033[91m',
        'Green': '\033[92m',
        'Yellow': '\033[93m',
        'Blue': '\033[94m',
        '': '',  # Safeguard for empty strings
        'End': '\033[0m'
    }

    # Iterate over the matrix and print each cell with the corresponding color
    for row in wiring_diagram:
        for color in row:
            print(f"{color_codes.get(color, '')}██{color_codes['End']}", end="")
        print()

# Function to check if a wiring diagram is dangerous
def is_dangerous(matrix):
    # Check for dangerous combinations (Red next to Yellow and vice versa) in rows
    for row in matrix:
        for i in range(len(row) - 1):
            if (row[i] == 'Red' and row[i + 1] == 'Yellow') or (row[i] == 'Yellow' and row[i + 1] == 'Red'):
                return True

    # Check for dangerous combinations (Red next to Yellow and vice versa) in columns
    for col in zip(*matrix):
        for i in range(len(col) - 1):
            if (col[i] == 'Red' and col[i + 1] == 'Yellow') or (col[i] == 'Yellow' and col[i + 1] == 'Red'):
                return True
    
    return False


# Function to extract various features from the matrix for machine learning
def extract_features(matrix):
    colors = ['Red', 'Blue', 'Yellow', 'Green']
    features = []

    # Feature 1: Color density in each row and column
    for color in colors:
        row_density = [row.count(color) for row in matrix]
        col_density = [col.count(color) for col in zip(*matrix)]
        features.extend(row_density + col_density)

    # Feature 2: Adjacent color pairs in rows and columns
    for color1 in colors:
        for color2 in colors:
            if color1 != color2:
                row_pairs = sum(color1 == matrix[i][j] and color2 == matrix[i][j+1] 
                                for i in range(20) for j in range(19))
                col_pairs = sum(color1 == matrix[i][j] and color2 == matrix[i+1][j] 
                                for i in range(19) for j in range(20))
                features.append(row_pairs + col_pairs)

    # Feature 3: Majority color in each row/column
    for row in matrix:
        row_majority = max(set(row).intersection(colors), key=row.count) if set(row).intersection(colors) else 'None'
        features.append(colors.index(row_majority) if row_majority in colors else -1)
    
    for col in zip(*matrix):
        col_majority = max(set(col).intersection(colors), key=col.count) if set(col).intersection(colors) else 'None'
        features.append(colors.index(col_majority) if col_majority in colors else -1)

    # Feature 4: Color proximity
    for color1 in colors:
        for color2 in colors:
            if color1 != color2:
                distance = [abs(row.index(color1) - row.index(color2)) for i, row in enumerate(matrix) if color1 in row and color2 in row]
                avg_distance = sum(distance) / len(distance) if distance else -1
                features.append(avg_distance)

    return features

# Function to encode color names to integers
def encode_diagram(diagram):
    color_to_int = {'Red': 0, 'Blue': 1, 'Yellow': 2, 'Green': 3}
    encoded_diagram = [color_to_int[color] for color in diagram]
    print(f"{diagram} -> {encoded_diagram}")
    return encoded_diagram

# Function to create a dataset for model training
def create_dataset(num_diagrams):
    dataset = []
    for _ in range(num_diagrams):
        diagram = generate_wiring_diagram()
        features = extract_features(diagram)
        label = 1 if is_dangerous(diagram) else 0  # 1 for 'Dangerous', 0 for 'Safe'
        dataset.append((features, label))
    return dataset

###### STEP 2) MODEL (LOGISTIC REGRESSION)
# Function for min-max scaling of features
def min_max_scale(features):
    min_val = np.min(features, axis=0)
    max_val = np.max(features, axis=0)
    return (features - min_val) / (max_val - min_val)

# Function for z-score normalization of features
def z_score_normalize(features):
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    return (features - mean) / std

# Sigmoid function used in logistic regression
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Function to train a logistic regression model
def train_logistic_regression(X, y, learning_rate=0.01, epochs=100):
    m, n = X.shape
    X = np.hstack((np.ones((m, 1)), X))  # Add intercept term
    weights = np.zeros(n + 1)
    losses = []  # List to store loss values during training

    for epoch in range(epochs):
        z = np.dot(X, weights)
        predictions = sigmoid(z)
        errors = predictions - y
        gradient = np.dot(X.T, errors) / m
        weights -= learning_rate * gradient

        # Calculate and record the loss
        loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        losses.append(loss)

    return weights, losses

# Function to make predictions using the logistic regression model
def predict(X, weights):
    X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add intercept term
    predictions = sigmoid(np.dot(X, weights))
    return [1 if i > 0.5 else 0 for i in predictions]


### STEP 3) EVAL
# Function to split the dataset into training and testing sets
def split_dataset(features, labels, test_size=0.2):
    total_samples = len(features)
    test_samples = int(total_samples * test_size)
    indices = np.random.permutation(total_samples)

    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]

    return features[train_indices], features[test_indices], labels[train_indices], labels[test_indices]

# Function to calculate accuracy of the model
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def train_and_evaluate(run_id, num_diagrams, max_epochs, increment_epochs):
    # Generate a set of wiring diagrams and their respective features and labels
    diagrams = [generate_wiring_diagram() for _ in range(num_diagrams)]
    features = np.array([extract_features(diagram) for diagram in diagrams])
    labels = np.array([1 if is_dangerous(diagram) else 0 for diagram in diagrams])

    # Splitting the dataset into training (80%) and testing (20%) sets
    split_index = int(len(features) * 0.8)
    X_train, X_test = features[:split_index], features[split_index:]
    y_train, y_test = labels[:split_index], labels[split_index:]

    # Scale features using min-max scaling for training and testing sets
    X_train = min_max_scale(X_train)
    X_test = min_max_scale(X_test)

    # Train the logistic regression model using the training set
    weights = train_logistic_regression(X_train, y_train, learning_rate=0.05, epochs=max_epochs)

    # Make predictions on the testing set
    predictions = predict(X_test, weights)

    # Calculate and print the accuracy of the model
    acc = accuracy(y_test, predictions)
    print(f"Run {run_id + 1}/{num_runs} - Epochs: {max_epochs}, Accuracy: {acc}")
    return max_epochs, acc

def run_training(run_id, num_diagrams, max_epochs, increment_epochs):
    print(f"Starting Run {run_id + 1}")
    run_results = []

    for epochs in range(100, max_epochs + 1, increment_epochs):
        print(f"({run_id}) Training with Epochs: {epochs}")
        diagrams = [generate_wiring_diagram() for _ in range(num_diagrams)]
        features = np.array([extract_features(diagram) for diagram in diagrams])
        labels = np.array([1 if is_dangerous(diagram) else 0 for diagram in diagrams])

        # Splitting the dataset into training and testing sets
        split_index = int(len(features) * 0.8)
        X_train, X_test = features[:split_index], features[split_index:]
        y_train, y_test = labels[:split_index], labels[split_index:]

        # Feature scaling
        X_train = min_max_scale(X_train)
        X_test = min_max_scale(X_test)

        # Train the logistic regression model and record losses
        weights, losses = train_logistic_regression(X_train, y_train, learning_rate=0.05, epochs=epochs)

        # Make predictions on the test set
        predictions = predict(X_test, weights)

        # Evaluate the model's accuracy
        acc = accuracy(y_test, predictions)
        print(f"({run_id}) Accuracy after {epochs} epochs: {acc}")
        run_results.append((epochs, acc, losses))

    return run_results

def main():
    num_diagrams = [500, 1000, 2500, 5000]  # Number of diagrams
    max_epochs = 1000  # Maximum number of epochs
    increment_epochs = 100  # Increment in epochs
    num_runs = 5  # Number of runs to average results

    for num in num_diagrams:
        all_epochs = []
        all_accuracies = []
        all_losses = []

        max_epochs_in_runs = 0  # To keep track of the maximum number of epochs in runs

        for run_id in range(num_runs):
            print(f"Starting Run {run_id + 1}")
            run_results = run_training(run_id, num, max_epochs, increment_epochs)
            epochs, accuracies, losses = zip(*run_results)
            all_epochs.append(epochs)
            all_accuracies.append(accuracies)

            # Pad losses with NaN values to ensure consistent shape
            max_epochs_in_run = max(epochs)
            max_epochs_in_runs = max(max_epochs_in_runs, max_epochs_in_run)
            padded_losses = list(losses)
            padded_losses.extend([np.nan] * (max_epochs_in_runs - max_epochs_in_run))
            all_losses.append(padded_losses)

        # Calculate the mean and standard deviation of accuracies and losses across runs
        mean_epochs = np.mean(all_epochs, axis=0)
        mean_accuracies = np.mean(all_accuracies, axis=0)

        # Create an array for mean_losses by taking the nanmean across all_losses
        mean_losses = []
        for i in range(max_epochs_in_runs):
            values = [losses[i] for losses in all_losses if i < len(losses)]
            mean_loss = np.nanmean(values)
            mean_losses.append(mean_loss)

        std_accuracies = np.std(all_accuracies, axis=0)

        # Plot the results and save as PNG
        plt.figure(figsize=(12, 6))
        plt.title(f"Average Accuracy vs. Epochs (Num Diagrams: {num})")

        # Ensure both mean_epochs and mean_losses have the same length
        min_length = min(len(mean_epochs), len(mean_losses))
        mean_epochs = mean_epochs[:min_length]
        mean_losses = mean_losses[:min_length]

        plt.errorbar(mean_epochs, mean_accuracies[:min_length], yerr=std_accuracies[:min_length], marker='o', linestyle='-', label='Accuracy')
        plt.xlabel("Epochs")
        plt.ylabel("Average Accuracy")
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.twinx()
        plt.plot(mean_epochs, mean_losses, marker='s', linestyle='--', color='red', label='Loss')
        plt.ylabel("Average Loss")
        plt.legend(loc='upper left')
        plt.savefig(f"accuracy_loss_num_diagrams_{num}.png")
        plt.close()

if __name__ == "__main__":
    main()
