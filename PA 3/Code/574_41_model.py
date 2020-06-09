import numpy as np
from sklearn import svm
from Preprocessing import preprocess
from Postprocessing import *
from utils import *

metrics = ["race", "sex", "age", 'c_charge_degree', 'priors_count', 'c_charge_desc']
training_data, training_labels, test_data, test_labels, categories, mappings = preprocess(metrics)

np.random.seed(42)
SVR = svm.LinearSVR(C=1.0/float(len(test_data)), max_iter=5000)
SVR.fit(training_data, training_labels)

training_class_predictions = SVR.predict(training_data)
training_predictions = []
test_class_predictions = SVR.predict(test_data)
test_predictions = []

for i in range(len(training_labels)):
    training_predictions.append(training_class_predictions[i])

for i in range(len(test_labels)):
    test_predictions.append(test_class_predictions[i])

training_race_cases = get_cases_by_metric(training_data, categories, "race", mappings, training_predictions, training_labels)
test_race_cases = get_cases_by_metric(test_data, categories, "race", mappings, test_predictions, test_labels)

epsilon_value = 0.01
training_race_cases, thresholds_training = enforce_equal_opportunity(training_race_cases, epsilon_value)
test_race_cases, thresholds_testing = enforce_equal_opportunity(test_race_cases, epsilon_value)

for group in test_race_cases.keys():
    training_race_cases[group] = apply_threshold(training_race_cases[group], thresholds_training[group])

for group in test_race_cases.keys():
    test_race_cases[group] = apply_threshold(test_race_cases[group], thresholds_testing[group])

# ADD MORE PRINT LINES HERE - THIS ALONE ISN'T ENOUGH
# YOU NEED ACCURACY AND COST FOR TRAINING AND TEST DATA
# PLUS WHATEVER RELEVANT METRICS ARE USED IN YOUR POSTPROCESSING METHOD, TO ENSURE EPSILON WAS ENFORCED
print("Accuracy on training data")
print(get_total_accuracy(training_race_cases))
print()

print("Cost on training data")
print('${:,.0f}'.format(apply_financials(training_race_cases)))
print()

print("F1 Score on training data")
f1_score = []
for group in training_race_cases.keys():
            f1_score += training_race_cases[group]
print(calculate_Fscore(f1_score))
print()

print("Metrics for training data")
for group in training_race_cases.keys():
            TPR = get_true_positive_rate(training_race_cases[group])
            print("TPR for " + group + ": " + str(TPR))

tpr = []
for group in training_race_cases.keys():
    tpr += training_race_cases[group]

t = get_true_positive_rate(tpr)
print("TPR for all training data", t)
print()

for group in training_race_cases.keys():
    print("Threshold for " + group + ": " + str(thresholds_training[group]))
print()

print("Accuracy on test data")
print(get_total_accuracy(test_race_cases))
print()

print("Cost on test data")
print('${:,.0f}'.format(apply_financials(test_race_cases)))
print()

print("F1 Score on test data")
f1_score = []
for group in test_race_cases.keys():
            f1_score += test_race_cases[group]
print(calculate_Fscore(f1_score))
print()

print("Metrics for test data")
for group in test_race_cases.keys():
            TPR = get_true_positive_rate(test_race_cases[group])
            print("TPR for " + group + ": " + str(TPR))

tpr = []
for group in test_race_cases.keys():
            tpr += test_race_cases[group]
t = get_true_positive_rate(tpr)
print("TPR for all test data", t)
print()

for group in test_race_cases.keys():
    print("Threshold for " + group + ": " + str(thresholds_testing[group]))