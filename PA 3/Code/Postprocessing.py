from utils import *
import pandas as pd

#######################################################################################################################
# YOU MUST FILL OUT YOUR SECONDARY OPTIMIZATION METRIC (either accuracy or cost)!
# The metric chosen must be the same for all 5 methods.
#
# Chosen Secondary Optimization Metric: # accuracy
#######################################################################################################################
""" Determines the thresholds such that each group has equal predictive positive rates within
    a tolerance value epsilon. For the Naive Bayes Classifier and SVM you should be able to find
    a nontrivial solution with epsilon=0.02.
    Chooses the best solution of those that satisfy this constraint based on chosen
    secondary optimization criteria.
"""
def enforce_demographic_parity(categorical_results, epsilon):

    thresholds = {}
    demographic_parity_data = {}

    # Must complete this function!
    # return demographic_parity_data, thresholds
    thresholded_data = {}
    constraints = {}
    for key,value in categorical_results.items():
        keys_dict = {}
        thsr = 0.01
        while thsr <= 1.00:
            thresholded_data[key] = apply_threshold(value, thsr)
            total_pos_pred = get_num_predicted_positives(thresholded_data[key])
            keys_dict[thsr] = total_pos_pred/len(value)
            thsr += 0.01
            thsr = round(thsr, 2)
        constraints[key] = keys_dict

    final_list = get_matching_threshold_values(constraints, epsilon)
    thresholds, demographic_parity_data = secondary_optimization_accuracy(final_list, categorical_results)

    return demographic_parity_data, thresholds

#######################################################################################################################
""" Determine thresholds such that all groups have equal TPR within some tolerance value epsilon,
    and chooses best solution according to chosen secondary optimization criteria. For the Naive
    Bayes Classifier and SVM you should be able to find a non-trivial solution with epsilon=0.01
"""
def enforce_equal_opportunity(categorical_results, epsilon):

    thresholds = {}
    equal_opportunity_data = {}

    # Must complete this function!
    # return equal_opportunity_data, thresholds
    thresholded_data = {}
    constraints_tpr = {}
    for key,value in categorical_results.items():
        keys_dict_tpr = {}
        thsr = 0.01
        while thsr <= 1.00:
            thresholded_data[key] = apply_threshold(value, thsr)
            tpr = get_true_positive_rate(thresholded_data[key])
            keys_dict_tpr[thsr] = tpr
            thsr += 0.01
            thsr = round(thsr, 2)
        constraints_tpr[key] = keys_dict_tpr
    final_list_tpr = get_matching_threshold_values(constraints_tpr, epsilon)
    thresholds, equal_opportunity_data = secondary_optimization_accuracy(final_list_tpr, categorical_results)

    return equal_opportunity_data, thresholds

#######################################################################################################################

"""Determines which thresholds to use to achieve the maximum profit or maximum accuracy with the given data
"""

def enforce_maximum_profit(categorical_results):
    mp_data = {}
    thresholds = {}

    # Must complete this function!
    # return mp_data, thresholds
    max_accuracy = {}
    thresholded_data = {}
    for key, value in categorical_results.items():
        thsr = 0.01
        accuracies = {}
        while thsr <= 1.00:
            thresholded_data[key] = apply_threshold(value, thsr)
            total_num_cases = 0
            total_correct = 0
            for prediction, label in thresholded_data[key]:
                total_num_cases += 1.0
                if prediction == label:
                    total_correct += 1.0
            accuracies[thsr] = total_correct / total_num_cases
            thsr += 0.01
            thsr = round(thsr, 2)
        thresholds[key] = max(accuracies,key=accuracies.get)
        mp_data[key] = apply_threshold(value, thresholds[key])

    return mp_data, thresholds

#######################################################################################################################
""" Determine thresholds such that all groups have the same PPV, and return the best solution
    according to chosen secondary optimization criteria
"""

def enforce_predictive_parity(categorical_results, epsilon):
    predictive_parity_data = {}
    thresholds = {}

    # Must complete this function!
    # return predictive_parity_data, thresholds
    thresholded_data = {}
    constraints_ppv = {}
    for key,value in categorical_results.items():
        keys_dict_ppv = {}
        thsr = 0.01
        while thsr <= 1.00:
            thresholded_data[key] = apply_threshold(value, thsr)
            pos_pred_val = get_positive_predictive_value(thresholded_data[key])
            keys_dict_ppv[thsr] = pos_pred_val
            thsr += 0.01
            thsr = round(thsr, 2)
        constraints_ppv[key] = keys_dict_ppv
    final_list_ppv = get_matching_threshold_values(constraints_ppv, epsilon)
    thresholds, predictive_parity_data = secondary_optimization_accuracy(final_list_ppv, categorical_results)

    return predictive_parity_data, thresholds

###################################################################################################################
""" Apply a single threshold to all groups, and return the best solution according to
    chosen secondary optimization criteria
"""

def enforce_single_threshold(categorical_results):
    single_threshold_data = {}
    thresholds = {}

    # Must complete this function!
    # return single_threshold_data, thresholds
    thresholded_data = {}
    accuracies = {}
    thsr = 0.01
    while thsr <= 1.00:
        for key,value in categorical_results.items():
            thresholded_data[key] = apply_threshold(value, thsr)
        accuracies[thsr] = get_total_accuracy(thresholded_data)
        thsr += 0.01
        thsr = round(thsr, 2)
    final_threshold = max(accuracies,key=accuracies.get)

    # Enforcing the found most accurate threshold
    for key,value in categorical_results.items():
            single_threshold_data[key] = apply_threshold(value, final_threshold)
            thresholds[key] = final_threshold

    return single_threshold_data, thresholds

#####################################################################################################################
"""Performs secondary optimization w.r.t accuracy

@:param list_dicts:     List of dictionaries containing ethinicity and threshold key value pair

@:param data: categorical_results

@:returns optimized thresholds, optim_thresholded data
"""

def secondary_optimization_accuracy(list_of_dicts, data):
    final_data = {}
    accuracies = []
    i = 0
    for dictionary in list_of_dicts:
        thresholdedData = {}
        for key, value in data.items():
            thresholdedData[key] = apply_threshold(value, dictionary[key])
        i += 1
        tot_acc = get_total_accuracy(thresholdedData)
        accuracies.append(tot_acc)

    if len(accuracies) == len(list_of_dicts):
        maxIndex = accuracies.index(max(accuracies))
        final_thresholds = list_of_dicts[maxIndex]
        for key,value in data.items():
            final_data[key] = apply_threshold(value, final_thresholds[key])

        return final_thresholds, final_data
    else:
        return None,None

#######################################################################################################################
"""Retrieves the matching threshold values between different ethnicities

@:param constraints:     List of dictionaries containing ethinicity and threshold key value pair

@:param epsilon:

@:returns final_list    List of all possible combinations
"""

def get_matching_threshold_values(constraints, epsilon):

    final_list = []
    constraint_keys = list(constraints.keys())
    constraint_values = list(constraints.values())
    change_scale = 0.02

    for threshold1, value1 in constraint_values[0].items():
        for threshold2, value2 in constraint_values[1].items():
                if compare_probs(value1, value2, epsilon):
                    for threshold3, value3 in constraint_values[2].items():
                        if compare_probs(value1, value3, epsilon) and compare_probs(value2, value3, epsilon):
                            for threshold4, value4 in constraint_values[3].items():
                                if compare_probs(value1, value4, epsilon) and compare_probs(value2, value4, epsilon) and compare_probs(value3, value4, epsilon):
                                    combination = {}
                                    combination[constraint_keys[0]] = threshold1 - (threshold1%change_scale)
                                    combination[constraint_keys[1]] = threshold2 - (threshold2%change_scale)
                                    combination[constraint_keys[2]] = threshold3 - (threshold3%change_scale)
                                    combination[constraint_keys[3]] = threshold4 - (threshold4%change_scale)
                                    final_list.append(combination)
    df_unique = pd.DataFrame(final_list).drop_duplicates()
    final_list = df_unique.T.to_dict().values()

    return list(final_list)

