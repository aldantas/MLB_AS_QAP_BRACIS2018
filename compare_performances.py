import glob
import os
import pickle
import numpy as np
import scipy.stats as stat

def load_predicts(predict_file):
    with open(predict_file, 'rb') as predicts_file:
        predicts_dict = pickle.load(predicts_file)
    return predicts_dict

def load_alg_performances():
    performances_file = 'algorithms_performances/performances.dat'
    with open(performances_file, 'rb') as performances_file:
        performances_dict = pickle.load(performances_file)
    return performances_dict

def load_best_known():
    best_known_dict = {}
    for instance_path in glob.glob(os.path.join('qap_instances', '*.sln')):
        instance_name = instance_path.split('/')[-1]
        instance_name = instance_name.split('.')[0]
        with open(instance_path) as inst_best_file:
            best_cost = inst_best_file.readline().split()[-1]
            best_known_dict[instance_name] = best_cost
    return best_known_dict

def eval_alg_performances(performances_dict, best_known_dict, condition):
    algs_perf_eval_dict = {
        'breakout_local_search': {'eq': 0, 'dists': []},
        'max_min_ant_system_bi': {'eq': 0, 'dists': []},
        'robust_tabu_search': {'eq': 0, 'dists': []}
    }
    for instance in best_known_dict:
        best_known = best_known_dict[instance]
        for algorithm in performances_dict[instance]:
            alg_perform = performances_dict[instance][algorithm]
            if condition == 'all':
                # consider the average cost
                achieved_cost = alg_perform[0]
            else:
                # consider only the best cost
                achieved_cost = alg_perform[-1]
            if int(achieved_cost) == int(best_known):
                algs_perf_eval_dict[algorithm]['eq'] += 1
            else:
                diff = int(achieved_cost) - int(best_known)
                distance = (diff * 100) / int(best_known)
                algs_perf_eval_dict[algorithm]['dists'].append(distance)
    return algs_perf_eval_dict

def print_performances(perf_eval_dict):
    for key in perf_eval_dict:
        print(key)
        print('Equals: {}'.format(perf_eval_dict[key]['eq']))
        avg_dist = np.mean(perf_eval_dict[key]['dists'])
        std_dist = np.std(perf_eval_dict[key]['dists'])
        print('Average Distance: {} ({})'.format(avg_dist, std_dist))

def eval_clf_performances(predicts_dict, performances_dict, best_known_dict,
                          condition):
    algs_dict = {0: 'breakout_local_search', 1: 'max_min_ant_system_bi',
                 2: 'robust_tabu_search'}
    clf_perf_eval_dict = {
        'clf_best_case': {'eq': 0, 'dists': [], 'samples': {}},
        'clf_worst_case': {'eq': 0, 'dists': [], 'samples': {}}
    }
    for instance in predicts_dict:
        selected_algs = []
        for i, pred_class in enumerate(predicts_dict[instance]['pred']):
            if pred_class:
                alg_name = algs_dict[i]
                avg_cost = performances_dict[instance][alg_name]
                selected_algs.append((algs_dict[i], avg_cost))
        for case in clf_perf_eval_dict:
            if case == 'clf_best_case':
                selected_algs.sort(key=lambda tup: tup[1])
            else:
                selected_algs.sort(key=lambda tup: tup[1], reverse=True)

            predicted_alg = selected_algs[0][0]
            best_known = best_known_dict[instance]
            predicted_perf = performances_dict[instance][predicted_alg]
            if condition == 'all':
                # consider the average cost
                achieved_cost = predicted_perf[0]
            else:
                # consider only the best cost
                achieved_cost = predicted_perf[-1]
            if int(achieved_cost) == int(best_known):
                clf_perf_eval_dict[case]['eq'] += 1
            else:
                diff = int(achieved_cost) - int(best_known)
                distance = (diff * 100) / int(best_known)
                clf_perf_eval_dict[case]['dists'].append(distance)
                samples = performances_dict[instance][predicted_alg][1]
                clf_perf_eval_dict[case]['samples'][instance] = samples
    return clf_perf_eval_dict

def compare_samples(clf_perf_eval_dict, performances_dict, alg_key):
    for case in clf_perf_eval_dict:
        inst_samples_dict = clf_perf_eval_dict[case]['samples']
        better_count = 0
        worst_count = 0
        equivalent_count = 0
        for instance in inst_samples_dict:
            clf_sample = [int(cost) for cost in inst_samples_dict[instance]]
            alg_sample = [int(cost) for cost in
                          performances_dict[instance][alg_key][1]]
            p_value = stat.kruskal(clf_sample, alg_sample)[1]
            if np.mean(clf_sample) <= np.mean(alg_sample):
                if p_value < 0.05:
                    better_count += 1
                else:
                    equivalent_count += 1
            else:
                if p_value < 0.05:
                    worst_count += 1
                else:
                    equivalent_count += 1
        print(case, 'x', alg_key)
        print('Sample size:', len(inst_samples_dict))
        print('Better:', better_count)
        print('Worst:', worst_count)
        print('Equivalent:', equivalent_count)


conditions = ['best', 'all']
methods = ['Multi-labels', 'Powerset']
for cond in conditions:
    print('#' * 40)
    print(cond)
    print('#' * 40)
    performances_dict = load_alg_performances()
    best_known_dict = load_best_known()
    algs_perf_eval_dict = eval_alg_performances(performances_dict,
                                                best_known_dict, cond)
    print_performances(algs_perf_eval_dict)
    for method in methods:
        print('-' * 40)
        print(method)
        print('-' * 40)
        pred_file = "results/{}_instances_predictions.dat".format(method)
        predicts_dict = load_predicts(pred_file)
        clf_perf_eval_dict = eval_clf_performances(predicts_dict,
                                                   performances_dict,
                                                   best_known_dict, cond)
        print_performances(clf_perf_eval_dict)
        print('*' * 20)
        print('Stats')
        print('*' * 20)
        compare_samples(clf_perf_eval_dict, performances_dict,
                        'max_min_ant_system_bi')
