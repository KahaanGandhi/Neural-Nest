import pandas as pd
import numpy as np
import string
import re

original_data = pd.read_csv("/Users/kahaan/Desktop/Torch/datasets/spam.csv", encoding='ISO-8859-1')

filtered_data = original_data.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])  
renamed_data = filtered_data.rename(columns={"v1": "Label", "v2": "Text"})
shuffled_data = renamed_data.sample(frac=1)                          # Clean and shuffle dataset
split_index = int(len(shuffled_data) * 0.8)                          # Split up into testing and training batches                                               
training_data = shuffled_data[:split_index]
testing_data = shuffled_data[split_index:]

#print(renamed_data["Label"].value_counts(normalize=True))           # Confirm that both batches reflect original
#print(training_data["Label"].value_counts(normalize=True))
#print(testing_data["Label"].value_counts(normalize=True))

cleaned_training_data = training_data.copy()                        # Further clean training set
cleaned_training_data['Text'] = training_data['Text'].str.replace('[^a-zA-Z ]', '', regex=True).str.lower()
cleaned_training_data['Text'] = cleaned_training_data['Text'].str.split()

N_ham = shuffled_data["Label"].value_counts()[0]                    # Calculate class priors
N_spam = shuffled_data["Label"].value_counts()[1]
alpha = 1                                                           # Laplacian smoothing
number_of_classes = 2
n = len(shuffled_data)
P_ham = (N_ham + 1) / (n + number_of_classes)
P_spam = (N_spam + 1) / (n + number_of_classes)
log_priors = {'ham': np.log(P_ham),                                 # Store class priors in log-space
              'spam': np.log(P_spam)}

vocabulary = set()                                                  # Count unique words in training data
for text in cleaned_training_data['Text']:
    for word in text:
        vocabulary.add(word)
vocabulary = list(vocabulary)

word_frequency = {}                                                 # Use vocab to create word-counting structure
for key in {'ham', 'spam'}:
    if not key in word_frequency:
        word_frequency[key] = {}
        for word in vocabulary:                                     
            word_frequency[key][word] = 0

spam_messages = cleaned_training_data[cleaned_training_data['Label'] == 'spam']
ham_messages = cleaned_training_data[cleaned_training_data['Label'] == 'ham']

words_in_spam, words_in_ham = 0, 0                                  # Count frequency of words for each class
for text in spam_messages['Text']:
    for word in text:
        word_frequency['spam'][word] += 1
        words_in_spam += 1

for text in ham_messages['Text']:
    for word in text:
        word_frequency['ham'][word] += 1
        words_in_ham += 1
        

word_log_prob = {}                                                  # Create a structure to calculate conditional probability
alpha = 1
N_vocab = len(vocabulary)
for key in word_frequency.keys():                                   # Calculate conditional probalities of word given class
    if not key in word_log_prob:
        word_log_prob[key] = {}
    for word in vocabulary:
        if key == "spam":
            N_words = words_in_spam
        else:
            N_words = words_in_ham
        N_wi_label = word_frequency[key][word]
        prob = (N_wi_label + alpha) / (N_words + alpha * N_vocab)
        word_log_prob[key][word] = np.log(prob)
        
def classify(message:str):                                          # Classify a new message
    message = re.sub('[^a-zA-Z]', ' ', message).lower()             # Clean new message
    words = message.split()
    
    class_probabilities = {}
    for class_label in log_priors.keys():                           # For each potential class...
        log_prob = log_priors[class_label]                          # begin with class prior
        for word in words:                                          # For each word...
            if word in vocabulary:                                  # if it was in training vocab...             
                log_prob += word_log_prob[class_label][word]        # add conditional probability
            else:                                                   # Otherwise, calculate it now
                if key == "spam":
                    N_words = words_in_spam
                else:
                    N_words = words_in_ham
                prob = (1 + alpha) / (N_words + alpha * len(vocabulary))
                log_prob += prob
        class_probabilities[class_label] = log_prob                 # Store probabilities, to be argmax'd
    most_likely_class = max(class_probabilities, key=class_probabilities.get)
    return most_likely_class

correct_labels = 0                                                  # Test classifier on training data
for i in range(len(testing_data)):
    text = testing_data.iloc[i][1]
    if classify(text) == testing_data.iloc[i][0]:
        correct_labels += 1
print("Accuracy rate:",round(correct_labels/(len(testing_data)),4) * 100, "%")