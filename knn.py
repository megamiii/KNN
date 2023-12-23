import numpy as np
import matplotlib.pyplot as plt


def load_knn_data():
    test_inputs = np.genfromtxt('knn-dataset/test_inputs.csv', delimiter=','),
    test_labels = np.genfromtxt('knn-dataset/test_labels.csv', delimiter=','),
    train_inputs = np.genfromtxt('knn-dataset/train_inputs.csv', delimiter=','),
    train_labels = np.genfromtxt('knn-dataset/train_labels.csv', delimiter=','),
    return train_inputs, train_labels, test_inputs, test_labels


'''
This function implements the KNN classifier to predict the label of a data point. 
Measure distances with the Euclidean norm (L2 norm).  
When there is a tie between two (or more) labels, break the tie by choosing any label.

Inputs:
    **x**: input data point for which we want to predict the label (numpy array of M features)
    **inputs**: matrix of data points in which neighbors will be found (numpy array of N data points x M features)
    **labels**: vector of labels associated with the data points  (numpy array of N labels)
    **k_neighbors**: # of nearest neighbors that will be used
Outputs:
    **predicted_label**: predicted label (integer)
'''   
def predict_knn(x, inputs, labels, k_neighbors):
    predicted_label = 0
    ########
    # TO DO:
    # Extract numpy arrays from tuples if necessary
    inputs = inputs[0] if isinstance(inputs, tuple) else inputs
    labels = labels[0] if isinstance(labels, tuple) else labels

    # Measure distances with the Euclidean norm (L2 norm)
    distances = np.sqrt(((inputs - x)**2).sum(axis=1))
    
    # Get the indices of the k_neighbors smallest distances
    knn_indices = distances.argsort()[:k_neighbors]
    
    # Get the labels of the nearest neighbors
    knn_labels = labels[knn_indices]
    
    # Get the most frequent label among the nearest neighbors
    (values, counts) = np.unique(knn_labels, return_counts=True)
    predicted_label = values[np.argmax((counts))]

    ########
    return predicted_label


'''
This function evaluates the accuracy of the KNN classifier on a dataset. 
The dataset to be evaluated consists of (inputs, labels). 
The dataset used to find nearest neighbors consists of (train_inputs, train_labels).

Inputs:
   **inputs**: matrix of input data points to be evaluated (numpy array of N data points x M features)
   **labels**: vector of target labels for the inputs (numpy array of N labels)
   **train_inputs**: matrix of input data points in which neighbors will be found (numpy array of N' data points x M features)
   **train_labels**: vector of labels for the training inputs (numpy array of N' labels)
   **k_neighbors**: # of nearest neighbors to be used (integer)
Outputs:
   **accuracy**: percentage of correctly labeled data points (float)
'''
def eval_knn(inputs, labels, train_inputs, train_labels, k_neighbors):
    accuracy = 0
    ########
    # TO DO:
    # Extract numpy arrays from tuples if necessary
    inputs = inputs[0] if isinstance(inputs, tuple) else inputs
    labels = labels[0] if isinstance(labels, tuple) else labels
    train_inputs = train_inputs[0] if isinstance(train_inputs, tuple) else train_inputs
    train_labels = train_labels[0] if isinstance(train_labels, tuple) else train_labels

    # Initialize a counter for the number of correct predictions
    correct_predictions = 0
    
    # Loop over each data point and its corresponding true label in the inputs and labels
    for x, true_label in zip(inputs, labels):
        # Reshape x to be a single data point
        x_reshaped = x.reshape(1, -1)

        # Predict the label
        predicted_label = predict_knn(x_reshaped, train_inputs, train_labels, k_neighbors)
        
        # If the predicted label matches the true label, increment the counter
        if predicted_label == true_label:
            correct_predictions += 1
            
    # Calculate accuracy as the ratio of correct predictions to the total number of predictions
    accuracy = correct_predictions / len(labels)

    ########
    return accuracy


'''
This function performs k-fold cross validation to determine the best number of neighbors for KNN.
        
Inputs:
    **k_folds**: # of folds in cross-validation (integer)
    **hyperparameters**: list of hyperparameters where each hyperparameter is a different # of neighbors (list of integers)
    **inputs**: matrix of data points to be used when searching for neighbors (numpy array of N data points by M features)
    **labels**: vector of labels associated with the inputs (numpy array of N labels)
Outputs:
    **best_hyperparam**: best # of neighbors for KNN (integer)
    **best_accuracy**: accuracy achieved with best_hyperparam (float)
    **accuracies**: vector of accuracies for the corresponding hyperparameters (numpy array of floats)
'''
def cross_validation_knn(k_folds, hyperparameters, inputs, labels):
    best_hyperparam = 0
    best_accuracy = 0
    accuracies = np.zeros(len(hyperparameters))
    ########
    # TO DO:
    # Extract numpy arrays from tuples if necessary
    inputs = inputs[0] if isinstance(inputs, tuple) else inputs
    labels = labels[0] if isinstance(labels, tuple) else labels

    # Calculate the size of each fold
    fold_size = len(inputs) // k_folds
    
    # Iterate over each hyperparameter (k_neighbors)
    for k in hyperparameters:
        fold_accuracies = []
        
        # Perform k-fold cross-validation
        for i in range(k_folds):
            # Determine the start and end indices of the validation set
            val_start = i * fold_size
            val_end = (i + 1) * fold_size
            
            # Split the dataset into validation and training sets
            val_inputs = inputs[val_start:val_end]
            val_labels = labels[val_start:val_end]
            train_inputs = np.vstack([inputs[:val_start], inputs[val_end:]])
            train_labels = np.concatenate([labels[:val_start], labels[val_end:]])
            
            # Evaluate the accuracy on the validation set
            fold_accuracy = eval_knn(val_inputs, val_labels, train_inputs, train_labels, k)
            fold_accuracies.append(fold_accuracy)
        
        # Calculate the average accuracy across all folds for the current hyperparameter
        avg_accuracy = np.mean(fold_accuracies)
        accuracies[hyperparameters.index(k)] = avg_accuracy
        
        # Update best_hyperparam and best_accuracy if current hyperparameter outperforms previous ones
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_hyperparam = k

    ########
    return best_hyperparam, best_accuracy, accuracies


'''
This function plots the KNN accuracies for different # of neighbors (hyperparameters) based on cross validation

Inputs:
    **accuracies**: vector of accuracies for the corresponding hyperparameters (numpy array of floats)
    **hyperparams**: list of hyperparameters where each hyperparameter is a different # of neighbors (list of integers)
'''
def plot_knn_accuracies(accuracies, hyperparams):
    plt.plot(hyperparams, accuracies)
    plt.ylabel('accuracy')
    plt.xlabel('k neighbors')
    plt.show()


def main():
    # load data
    train_inputs, train_labels, test_inputs, test_labels = load_knn_data()
    print(train_inputs)
    
    # number of neighbors to be evaluated by cross validation
    hyperparams = range(1,31)
    k_folds = 10

    # use k-fold cross validation to find the best # of neighbors for KNN
    best_k_neighbors, best_accuracy, accuracies = cross_validation_knn(k_folds, hyperparams, train_inputs, train_labels)

    # plot results
    plot_knn_accuracies(accuracies, hyperparams)
    print('best # of neighbors k: ' + str(best_k_neighbors))
    print('best cross validation accuracy: ' + str(best_accuracy))

    # evaluate with best # of neighbors
    accuracy = eval_knn(test_inputs, test_labels, train_inputs, train_labels, best_k_neighbors)
    print('test accuracy: '+ str(accuracy))


if __name__ == "__main__":
    main()