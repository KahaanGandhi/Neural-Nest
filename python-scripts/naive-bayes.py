import pandas as pd
import numpy as np

class NaiveBayesClassifier:
    def __init__(self, alpha=1):                                    # Initialize an untrained NB classifier
        self.alpha = alpha                                          # Laplacian smoothing as default
        self.log_priors = None
        self.class_conditional_probs = None
        self.features = None

    def train(self, data, target_class):
        """
        Calculates priors 
        # Calculating class frequencies and log priors"""
        training_instances = len(data)                              # Count the number of instances in the training data
        class_frequencies = {}                                      # Calculate the frequency of each class in target column (helpful for priors)
        for i in range(training_instances):
            class_label = data.iloc[i][target_class]
            if class_label in class_frequencies:
                class_frequencies[class_label] += 1
            else:
                class_frequencies[class_label] = 1
        self.log_priors = {}                                        # Calculate log of prior probabilities for each class
        for class_label, freq in class_frequencies.items():
            theta = (freq + self.alpha) / (training_instances + self.alpha * len(class_frequencies))
            self.log_priors[class_label] = np.log(theta)

        self.class_conditional_probs = {}                           # Initialize structure to store class-conditional probabilities
        for class_value in class_frequencies.keys():                # Iterate over every feature for every class, skipping target class
            self.class_conditional_probs[class_value] = {}
            for feature in data.columns:                            # Count instances of this feature for the current class
                if feature != target_class:
                    total_count = 0
                    for feature_value in data[feature].unique():
                        total_count += len(data[(data[target_class] == class_value) & (data[feature] == feature_value)])

                    V_j = len(data[feature].unique())                                           # Count how many unique values (options) for this feature
                    for feature_value in data[feature].unique():                                # Now, iterating over each unique value of the feature
                        N_k_vj = len(data[(data[target_class] == class_value) & (data[feature] == feature_value)])
                        theta_k_j_v = (N_k_vj + self.alpha) / (total_count + self.alpha * V_j)
                        if feature not in self.class_conditional_probs[class_value]:            # Log-tranform probability and store it in dictionary
                            self.class_conditional_probs[class_value][feature] = {}
                        self.class_conditional_probs[class_value][feature][feature_value] = np.log(theta_k_j_v)

        self.features = [feature for feature in data.columns if feature != target_class]        # List of columns that will be used to train

    def predict(self, new_instance):
        class_probabilities = {}                                        # Initialize structure to store probability of new instance being each class
        for class_label in self.log_priors.keys():                      # Use each prior as initial guess...
            log_prob = self.log_priors[class_label]
            for feature in self.features:                               # add the conditional probability of each feature given the current class...
                feature_value = new_instance.get(feature)
                log_prob += self.class_conditional_probs[class_label][feature][feature_value]
            class_probabilities[class_label] = log_prob

        return max(class_probabilities, key=class_probabilities.get)    # and return class which yields the highest probability
    
classifier = NaiveBayesClassifier()

train_data = pd.read_csv("./datasets/1994_census_cleaned_train.csv")    # Training NB classifier on 1994 Adult Census Income
target_class = "sex"
classifier.train(train_data, target_class)

test_data = pd.read_csv("./datasets/1994_census_cleaned_test.csv")      # Testing it on a different unique batch 

new_instance = {}
correct_labels, incorrect_labels = 0, 0
for i in range(len(test_data)):
    for j in range(len(test_data.columns)):
        if test_data.columns[j] != target_class:
            new_instance[test_data.columns[j]] = test_data.iloc[i][j]
        else:
            correct_value = test_data[target_class][i]
    if classifier.predict(new_instance) == correct_value:
        correct_labels += 1
    else:
        incorrect_labels += 1

print("Accuracy rate:", round((correct_labels / (correct_labels + incorrect_labels)), 2), "%")