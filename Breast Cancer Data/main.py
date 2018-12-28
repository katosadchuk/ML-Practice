import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import svm

# classes are labeled as 2 or 4 in data, change to 0/1 labels
def encodeClasses(labels):
    newLabels = np.zeros(labels.shape)

    index = 0
    for x in labels:
        if x==2:
            newLabels[index] = 0
        if x==4:
            newLabels[index] = 1
        index+=1

    return newLabels

def main():
    data = np.genfromtxt("breast-cancer-wisconsin.data.txt", delimiter=',', skip_header=1, dtype='int32')
    # first column is ID number, delete it
    data = np.delete(data, 0, axis=1)

    # split into labels and features - change label encoding to 0/1
    labels = data[:, -1]
    data = np.delete(data, -1, axis=1)
    betterLabels = encodeClasses(labels)


    # split into training/testing data
    train_data, test_data, train_labels, test_labels = train_test_split(data, betterLabels, test_size=0.20, random_state=42)

    # training data is unbalanced so oversample
    ros = RandomOverSampler(random_state=0)
    train_data, train_labels = ros.fit_resample(train_data, train_labels)


    # standardize and center data
    means = np.average(train_data, axis=0)
    stdev = np.std(train_data, axis=0)
    train_data = np.subtract(train_data, means)
    train_data = np.divide(train_data, stdev)
    test_data = np.subtract(test_data, means)
    test_data = np.divide(test_data, stdev)

    # train adaboost
    model1 = AdaBoostClassifier(n_estimators=150)
    model1.fit(train_data, train_labels)
    predictions1 = model1.predict(test_data)

    # train randomforest
    model2 = RandomForestClassifier(n_estimators=100)
    model2.fit(train_data, train_labels)
    predictions2 = model2.predict(test_data)

    # train svm
    model3 = svm.SVC(gamma='scale')
    model3.fit(train_data, train_labels)
    predictions3 = model3.predict(test_data)


    # vote on assignment
    finalPred = np.zeros(predictions3.shape)
    for x in range(test_data.shape[0]):
        sum = predictions1[x] + predictions2[x] + predictions3[x]
        if sum > 1:
            finalPred[x] = 1

    # print final accuracy
    finalScore = accuracy_score(test_labels, finalPred)
    print(finalScore)




if __name__ == "__main__":
    main()
