# Kateryna Osadchuk
# hw 4

import numpy as np
from sklearn.neural_network import MLPClassifier
import pandas as pd


def main():
    filename1 = "train.csv"
    filename2 = "test.csv"
    # make matrix of training data from csv file
    train_data = np.genfromtxt(filename1, delimiter=",", dtype='float32', autostrip=True, skip_header=1)
    # first column is ID number, delete it or else it skews results
    train_data = np.delete(train_data, 0, 1)
    # make matrix of testing data from csv file
    test_data = np.genfromtxt(filename2, delimiter=",", dtype='float32', autostrip=True, skip_header=1)
    # first column is ID number, delete it or else it skews results
    test_data = np.delete(test_data, 0, 1)

    # save number of features
    num_features = train_data.shape[1]


    # split training data into features and labels
    labels = train_data[:, num_features - 1]
    train_data = np.delete(train_data, num_features - 1, 1)

    # center and scale the features of training and testing data
    # by subtracting the mean and dividing by the standard deviation
    # use mean and stdev of training data on both training and testing data
    means = np.mean(train_data, axis=0)
    dev = np.std(train_data, axis=0)

    train_data = np.subtract(train_data, means)
    train_data = np.divide(train_data, dev)

    test_data = np.subtract(test_data, means)
    test_data = np.divide(test_data, dev)


    # create a neural network that uses a relu activation function and has 3 hidden layers with (30, 20, 20) nodes
    model = MLPClassifier(activation='relu', batch_size=64, hidden_layer_sizes=(30, 20, 20), max_iter=500,
                         nesterovs_momentum=True,  random_state=0, shuffle=True, solver='adam',
                          tol=0.000001, validation_fraction=0.15,
                         verbose=True)

    # train the neural network using the training data and corresponding labels
    model.fit(train_data, labels)

    # use model to predict probabilities of each example
    # note that predict_proba method returns a matrix with 2 columns: column 0 is probability of the example
    # being labelled 0 and column 1 is probability of example being labelled 1
    probability_predictions = model.predict_proba(test_data)

    # we only want probability that an example is a sinkhole, so extract all of column 1 into a new vector
    probability_isSinkhole = np.zeros([test_data.shape[0], 1])
    for i in range(len(test_data)):
        probability_isSinkhole[i] = probability_predictions[i][1]

    # save vector of probabilities that an example is a sinkhole into a CSV file
    df = pd.DataFrame(probability_isSinkhole, columns=['IsSinkhole'])
    df.to_csv('results.csv')


if __name__ == "__main__":
    main()