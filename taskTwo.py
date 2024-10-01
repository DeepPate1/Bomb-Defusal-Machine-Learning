import random
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import concurrent.futures

# Function to generate a wiring diagram
def generate_wiring_diagram():
    colors = ['Red', 'Blue', 'Yellow', 'Green']
    matrix = [['' for _ in range(20)] for _ in range(20)]
    color_order = []

    # Assigning first color to a random row
    row = random.randint(0, 19)
    color = random.choice(colors)
    color_order.append(color)
    for c in range(20):
        matrix[row][c] = color

    # Assigning second color to a random column
    col = random.randint(0, 19)
    second_color = random.choice([c for c in colors if c != color])
    color_order.append(second_color)
    for r in range(20):
        matrix[r][col] = second_color

    # Assigning third color to another row
    second_row = random.choice([r for r in range(20) if r != row])
    third_color = random.choice([c for c in colors if c != color and c != second_color])
    color_order.append(third_color)
    for c in range(20):
        matrix[second_row][c] = third_color

    # Assigning fourth color to another column
    second_col = random.choice([c for c in range(20) if c != col])
    fourth_color = next(c for c in colors if c not in [color, second_color, third_color])
    color_order.append(fourth_color)
    for r in range(20):
        matrix[r][second_col] = fourth_color

    return matrix, color_order

# Function to print the wiring diagram
def print_diagram(wiring_diagram):
    color_codes = {
        'Red': '\033[91m',
        'Green': '\033[92m',
        'Yellow': '\033[93m',
        'Blue': '\033[94m',
        '': '',
        'End': '\033[0m'
    }

    for row in wiring_diagram:
        for color in row:
            print(f"{color_codes.get(color, '')}██{color_codes['End']}", end="")
        print()

# Function to check if a wiring diagram is dangerous
def is_dangerous(matrix):
    for row in matrix:
        for i in range(len(row) - 1):
            if (row[i] == 'Red' and row[i + 1] == 'Yellow') or (row[i] == 'Yellow' and row[i + 1] == 'Red'):
                return True
    
    for col in zip(*matrix):
        for i in range(len(col) - 1):
            if (col[i] == 'Red' and col[i + 1] == 'Yellow') or (col[i] == 'Yellow' and col[i + 1] == 'Red'):
                return True
    
    return False

# Function to determine what to cut from a dangerous diagram
def what_to_cut_fr(dangerous, color_order):
    cut = color_order[2] if dangerous else "None"
    return cut

# Function to check if color order in a wiring diagram is dangerous
def is_dangerous_fr(color_order):
    if 'Red' in color_order and 'Yellow' in color_order:
        return color_order.index('Red') < color_order.index('Yellow')

    return False

# Function to extract features from the wiring diagram
def extract_features(matrix, color_order):
    colors = ['Red', 'Blue', 'Yellow', 'Green']
    features = []

    # Feature extraction for color density and adjacency
    for color in colors:
        row_density = [row.count(color) for row in matrix]
        col_density = [col.count(color) for col in zip(*matrix)]
        features.extend(row_density + col_density)

    # Feature extraction for adjacent color pairs
    for color1 in colors:
        for color2 in colors:
            if color1 != color2:
                row_pairs = sum(color1 == matrix[i][j] and color2 == matrix[i][j+1] 
                                for i in range(20) for j in range(19))
                col_pairs = sum(color1 == matrix[i][j] and color2 == matrix[i+1][j] 
                                for i in range(19) for j in range(20))
                features.append(row_pairs + col_pairs)

    # Feature extraction for majority color
    for row in matrix:
        if set(row).intersection(colors):
            row_majority = max(set(row).intersection(colors), key=row.count)
        else:
            row_majority = 'None'
        features.append(colors.index(row_majority) if row_majority in colors else -1)

    for col in zip(*matrix):
        if set(col).intersection(colors):
            col_majority = max(set(col).intersection(colors), key=col.count)
        else:
            col_majority = 'None'
        features.append(colors.index(col_majority) if col_majority in colors else -1)

    # Adding feature for color order
    color_to_int = {'Red': 0, 'Blue': 1, 'Yellow': 2, 'Green': 3}
    order_feature = [color_to_int[color] for color in color_order[:4]]
    features.extend(order_feature)

    return features

# Function to encode the diagram for analysis
def encode_diagram(diagram):
    color_to_int = {'Red': 0, 'Blue': 1, 'Yellow': 2, 'Green': 3}
    encoded_diagram = [color_to_int[color] for color in diagram]
    print(f"{diagram} -> {encoded_diagram}")
    return encoded_diagram

# Logistic regression model functions
def min_max_scale(features):
    min_val = np.min(features, axis=0)
    max_val = np.max(features, axis=0)
    return (features - min_val) / (max_val - min_val)

def z_score_normalize(features):
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    return (features - mean) / std

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, learning_rate=0.5, epochs=100):
    m, n = X.shape
    num_classes = len(np.unique(y))
    X = np.hstack((np.ones((m, 1)), X))
    weights = np.zeros((n + 1, num_classes))

    for class_idx in range(num_classes):
        y_class = np.where(y == class_idx, 1, 0)
        for _ in range(epochs):
            z = np.dot(X, weights[:, class_idx])
            predictions = sigmoid(z)
            errors = predictions - y_class
            gradient = np.dot(X.T, errors) / m
            weights[:, class_idx] -= learning_rate * gradient

    return weights

def predict(X, weights):
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    z = np.dot(X, weights)
    return np.argmax(z, axis=1)

# Functions for dataset evaluation
def split_dataset(features, labels, test_size=0.2):
    total_samples = len(features)
    test_samples = int(total_samples * test_size)
    indices = np.random.permutation(total_samples)

    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]

    return features[train_indices], features[test_indices], labels[train_indices], labels[test_indices]

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def get_color_order(diagram):
    color_order = []
    for row in diagram:
        for color in row:
            if color not in color_order:
                color_order.append(color)
    return color_order

# Function to run the training process
def run_training(run_id, num_diagrams, max_epochs, increment_epochs):
    # Indicating the start of a training run
    print(f"Starting Run {run_id + 1}")
    run_results = []

    # Looping through a range of epochs
    for epochs in range(100, max_epochs + 1, increment_epochs):
        # Logging the current number of epochs
        print(f"({run_id}) Training with Epochs: {epochs}")

        # Generating wiring diagrams and selecting only the dangerous ones
        generated_data = [generate_wiring_diagram() for _ in range(num_diagrams)]
        dangerous_diagrams = [(diagram, color_order) for diagram, color_order in generated_data if is_dangerous_fr(color_order)]

        # Skip the run if no dangerous diagrams are generated
        if not dangerous_diagrams:
            print("No dangerous diagrams generated. Skipping this run.")
            continue

        # Extracting features from the dangerous diagrams
        features = np.array([extract_features(diagram, color_order) for diagram, color_order in dangerous_diagrams])
        # Mapping colors to integers for labeling
        color_to_int = {'Red': 0, 'Blue': 1, 'Yellow': 2, 'Green': 3}
        # Creating labels for training
        labels = np.array([color_to_int[what_to_cut_fr(True, color_order)] for _, color_order in dangerous_diagrams])

        # Splitting data into training and testing sets
        split_index = int(len(features) * 0.8)
        X_train, X_test = features[:split_index], features[split_index:]
        y_train, y_test = labels[:split_index], labels[split_index:]

        # Scaling features using min-max scaling
        X_train = min_max_scale(X_train)
        X_test = min_max_scale(X_test)

        # Training the logistic regression model
        weights = train_logistic_regression(X_train, y_train, learning_rate=0.05, epochs=epochs)

        # Making predictions with the trained model
        predictions = predict(X_test, weights)
        for i, prediction in enumerate(predictions):
            int_to_color = {0: 'Red', 1: 'Blue', 2: 'Yellow', 3: 'Green'}
            actual_color_to_cut = int_to_color[y_test[i]]
            predicted_color_to_cut = int_to_color[prediction]
            print(f"({run_id}) Diagram {i + 1}: Predicted Wire Color to Cut = {predicted_color_to_cut}, Actual Wire Color to Cut = {actual_color_to_cut}")

        # Calculating and logging the accuracy
        acc = accuracy(y_test, predictions)
        print(f"({run_id}) Accuracy after {epochs} epochs: {acc}")
        run_results.append((epochs, acc))

    return run_results

# Main function to orchestrate the training and evaluation
def main():
    # Setting parameters for training
    num_diagrams = 5000  # Number of diagrams to generate
    max_epochs = 1000  # Maximum epochs for training
    increment_epochs = 100  # Epoch increments for training
    num_runs = 5  # Total number of training runs

    results = []

    # Using thread pool executor to run training in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_training, run_id, num_diagrams, max_epochs, increment_epochs) for run_id in range(num_runs)]

        # Collecting results from each future as they complete
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    # Calculating average accuracy over all runs
    averaged_results = np.mean(results, axis=0)

    # Extracting epochs and accuracies for plotting
    epochs, accuracies = zip(*averaged_results)

    # Plotting average accuracy against epochs
    plt.figure(figsize=(12, 6))
    plt.title("Average Accuracy vs. Epochs")
    plt.plot(epochs, accuracies, marker='o', linestyle='-')
    plt.xlabel("Epochs")
    plt.ylabel("Average Accuracy")
    plt.grid(True)
    plt.show()

# Checking if the script is the main program and running main
if __name__ == "__main__":
    main()
