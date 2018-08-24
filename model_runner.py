from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import multiprocessing as mp
import numpy as np

class CLFRunner(object):
    def __init__(self, clf, splitter):
        self.clf = clf
        self.splitter = splitter
        self.true_list = []
        self.pred_list = []
        self.prob_list = []
        self.inst_list = []
        self.instances_pred_dict = {}
        self.max_processes = 20
        self._is_multilabel = False

    def fold_validation(self, x_train, x_test, y_train, y_test, test_insts=[]):
        self.clf.fit(x_train, y_train)
        preds = self.clf.predict(x_test)
        probs = self.clf.predict_proba(x_test)
        if self._is_multilabel:
            # if no label is predicted, assign the sample to the class with the
            # highest probability
            for sample_idx, (pred_y, true_y) in enumerate(zip(preds, y_test)):
                if len(set(np.where(pred_y)[0])) == 0:
                    pred_y = [0] * len(probs)
                    higher_prob = float('-inf')
                    for j, class_prob in enumerate(probs):
                        if class_prob[sample_idx][0] > higher_prob:
                            higher_prob = class_prob[sample_idx][0]
                            class_idx = j
                    pred_y[class_idx] = 1
                    preds[sample_idx] = np.array(pred_y)
        return y_test, preds, probs, test_insts

    def _check_labels(self, true_list):
        try:
            true_list[0][0]
            self._is_multilabel = True
        except IndexError:
            self._is_multilabel = False

    def update_results(self, pool_results):
        true_list, pred_list, prob_list, inst_list = pool_results
        self.true_list.extend(true_list)
        self.pred_list.extend(pred_list)
        self.prob_list.extend(prob_list)
        self.inst_list.extend(inst_list)
        for instance, true, pred in zip(inst_list, true_list, pred_list):
            try:
                true = [int(l) for l in true]
                pred = [int(l) for l in pred]
            except TypeError:
                pass
            self.instances_pred_dict[instance] = {'true': true, 'pred': pred}

    def clear_values(self):
        self.true_list = []
        self.pred_list = []
        self.prob_list = []

    def get_results(self):
        return self.true_list, self.pred_list, self.inst_list

    def run(self, x_data, y_data, instances_list=None):
        pool = mp.Pool(processes=self.max_processes)
        self._check_labels(y_data)
        self.clear_values()
        for train_idxs, test_idxs in self.splitter.split(x_data):
            x_train, x_test = x_data[train_idxs], x_data[test_idxs]
            y_train, y_test = y_data[train_idxs], y_data[test_idxs]
            if instances_list:
                test_instances = np.array(instances_list)[test_idxs]
            else:
                test_instances = []
            pool.apply_async(self.fold_validation, args=(
                x_train, x_test, y_train, y_test, test_instances),
                callback=self.update_results)
        pool.close()
        pool.join()

    def run_single_thread(self, x_data, y_data, instances_list=None):
        self._check_labels(y_data)
        self.clear_values()
        for train_idxs, test_idxs in self.splitter.split(x_data):
            x_train, x_test = x_data[train_idxs], x_data[test_idxs]
            y_train, y_test = y_data[train_idxs], y_data[test_idxs]
            if instances_list:
                test_instances = np.array(instances_list)[test_idxs]
            else:
                test_instances = []
            fold_results = self.fold_validation(
                x_train, x_test, y_train, y_test, test_instances)
            self.update_results(fold_results)

    def print_instances_predicts(self, encoder=None):
        for instance in self.instances_pred_dict:
            labels_dict = self.instances_pred_dict[instance]
            true = labels_dict['true']
            pred = labels_dict['pred']
            if not self._is_binarized and encoder:
                true, pred = self._to_binarized([true], [pred], encoder)
                true, pred = true[0], pred[0]
            print('{}: {} - {}'.format(instance, true, pred))

    def _binarized_labels(self):
        if self._is_binarized:
            return self.true_list, self.pred_list
        elif self.encoder:
            true_list = encoder.inverse_transform(self.true_list)
            true_list = MultiLabelBinarizer().fit_transform(true_list)
            pred_list = encoder.inverse_transform(self.pred_list)
            pred_list = MultiLabelBinarizer().fit_transform(pred_list)
            return np.array(true_list, dtype=int), np.array(pred_list, dtype=int)
