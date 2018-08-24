#!/usr/bin/python

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
import os
from model_runner import CLFRunner
from metrics import CLFMetrics

def print_and_save_log(method, metrics, test_type, log_file):
    header = "{}_{}".format(test_type, method)
    print(header)
    log_file.write(header + '\n')

    line = 'Instance Accuracy: {}'.format(metrics.instance_accuracy())
    print(line)
    log_file.write(line + '\n')

    line = 'Recommendation Accuracy: {}'.format(
        metrics.recommendation_accuracy())
    print(line)
    log_file.write(line + '\n')

    line = 'Subset Accuracy: {}'.format(metrics.exact_match_accuracy())
    print(line)
    log_file.write(line + '\n')

    line = 'Label Accuracy: {}'.format(metrics.label_accuracy())
    print(line)
    log_file.write(line + '\n')

    line = 'Label FScores:'
    print(line)
    log_file.write(line + '\n')
    fscores = metrics.label_fscore()
    print(fscores)
    log_file.write(str(fscores) + '\n')

    avg_fscore = np.mean(list(fscores.values()))
    line = 'Label Average FScore: {}'.format(avg_fscore)
    print(line)
    log_file.write(line + '\n')

    line = 'Subset FScores:'
    print(line)
    log_file.write(line + '\n')
    fscores = metrics.label_fscore(subset=True)
    print(fscores)
    log_file.write(str(fscores) + '\n')

    avg_fscore = np.mean(list(fscores.values()))
    line = 'Subset Average FScore: {}'.format(avg_fscore)
    print(line)
    log_file.write(line + '\n')

    print('-' * 50)
    log_file.write('-' * 50 + '\n')


def main():
    results_dir = 'results'
    try:
        os.mkdir(results_dir)
    except FileExistsError as e:
        pass
    log_filepath = '{}/results.log'.format(results_dir)
    log_file = open(log_filepath, 'w')
    model = RandomForestClassifier(n_estimators=100, max_depth=10,
                                   random_state=0)

    splitter = LeaveOneOut()
    multi_label_handlers = {'Multi-labels': MultiLabelBinarizer(),
                            'Powerset': LabelEncoder()}
    runner = CLFRunner(model, splitter)

    dataset_filepath = 'dataset/qap_dataset.dat'
    x_data, y_data = load_svmlight_file(dataset_filepath, multilabel=True)

    for method in multi_label_handlers:
        encoder = multi_label_handlers[method]
        y_data_trans = encoder.fit_transform(y_data)
        instance_order_filepath = 'dataset/instances_order.txt'
        with open(instance_order_filepath, 'r') as inst_file:
            instances_order = [line.rstrip('\n') for line in
                               inst_file.readlines()]
        runner.run(x_data, y_data_trans, instances_order)
        metrics = CLFMetrics(*runner.get_results(), encoder)
        test_type = 'full'
        print_and_save_log(method, metrics, test_type, log_file)

        predicts_filename = '{}/{}_instances_predictions.dat'.format(
            results_dir,method)
        with open(predicts_filename, 'wb') as predicts_file:
            pickle.dump(metrics.instances_predictions(), predicts_file)

        metrics.remove_multi_labeled()
        test_type = 'reduced'
        print_and_save_log(method, metrics, test_type, log_file)
    log_file.close()

if __name__ == "__main__":
    main()
