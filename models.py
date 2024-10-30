from enum import Enum


class FeatureType(str, Enum):
    tf_idf = "tf_idf"
    word2vec = "word2vec"
    bag_of_word = "bag_of_word"


class ModelType(str, Enum):
    knn = "KNN"
    bayes = "Bayes"
    decision_tree = "Decision Tree"
    random_forest = "Random Forest"
    logistic_regression = "Logistic Regression"
    svm_linear = "SVM Linear"
    svm_non_linear = "SVM Non-linear"
