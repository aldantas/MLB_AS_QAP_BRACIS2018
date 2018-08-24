from skmultilearn.problem_transform import LabelPowerset
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from scipy.sparse import csr_matrix
from operator import eq
import sklearn.metrics as skm
import multiprocessing as mp
import numpy as np

class CLFMetrics(object):
    def __init__(self, true_list, pred_list, inst_list, encoder=None):
        self.true_list = np.array(true_list, dtype=int)
        self.pred_list = np.array(pred_list, dtype=int)
        self.inst_list = inst_list
        self.encoder = encoder
        self._check_labels()

    def _check_labels(self):
        try:
            self.true_list[0][0]
            self._is_binarized = True
        except IndexError:
            self._is_binarized = False

    def remove_multi_labeled(self, only_full=True):
        bin_true_list, _ =  self._binarized_labels()
        new_true_list, new_pred_list, new_inst_list = [], [], []
        for bin_true, true, pred, inst in zip(bin_true_list, self.true_list,
                                              self.pred_list, self.inst_list):
            if sum(bin_true) == 1 or (sum(bin_true) < len(bin_true) and
                                      only_full):
                new_true_list.append(true)
                new_pred_list.append(pred)
                new_inst_list.append(inst)
        self.true_list = np.array(new_true_list)
        self.pred_list = np.array(new_pred_list)
        self.inst_list = new_inst_list

    # set as corretly classified if there is no false positive in the
    # intance prediction
    def recommendation_accuracy(self):
        if not self._is_binarized and not self.encoder:
            return self.exact_match_accuracy()
        true_list, pred_list = self._binarized_labels()
        success = 0
        for true, pred in zip(true_list, pred_list):
            if -1 not in true - pred:
                success += 1
        return success / len(self.true_list)

    def _count_multi_label(self, labels_list):
        class_count_dict = {}
        for label_set in labels_list:
            for i, l in enumerate(label_set):
                if l == 1:
                    try:
                        class_count_dict[i] += 1
                    except KeyError:
                        class_count_dict[i] = 1
        return {v: k for k, v in class_count_dict.items()}

    def _count_subset(self, labels_list):
        class_count_dict = {}
        for label in labels_list:
            try:
                class_count_dict[label] += 1
            except KeyError:
                class_count_dict[label] = 1
        return {v: k for k, v in class_count_dict.items()}

    def _binarized_labels(self):
        if self._is_binarized:
            return self.true_list, self.pred_list
        elif self.encoder:
            true_list = self.encoder.inverse_transform(self.true_list)
            true_list = MultiLabelBinarizer().fit_transform(true_list)
            pred_list = self.encoder.inverse_transform(self.pred_list)
            pred_list = MultiLabelBinarizer().fit_transform(pred_list)
            return np.array(true_list, dtype=int), np.array(pred_list, dtype=int)

    def label_fscore(self, subset=False):
        if subset:
            true_list, pred_list = self.true_list, self.pred_list
            if self._is_binarized:
                # transform multilabel to multiclass for subset measurement
                lp = LabelPowerset()
                transformed = lp.transform(np.concatenate((true_list,
                                                           pred_list)))
                true_list, pred_list = np.split(transformed, 2)
        else:
            true_list, pred_list = self._binarized_labels()
        prec, rec, fscore, count = skm.precision_recall_fscore_support(
            true_list, pred_list)
        fscores_dict = {}
        for c, f in zip(count, fscore):
            # label = class_by_count[c]
            # for when remove_multi_labeled is used
            if c == 0:
                continue
            fscores_dict[c] = f
        return fscores_dict

    def exact_match_accuracy(self):
        return skm.accuracy_score(self.true_list, self.pred_list)

    def instance_accuracy(self):
        true_list, pred_list = self._binarized_labels()
        return skm.jaccard_similarity_score(true_list, pred_list)

    def hamming_loss(self):
        true_list, pred_list = self._binarized_labels()
        return skm.hamming_loss(true_list, pred_list)

    def label_accuracy(self):
        true_list, pred_list = self._binarized_labels()
        return skm.accuracy_score(true_list.ravel(), pred_list.ravel())

    def labels_precision_recall_fscore_support(self, average=None):
        return skm.precision_recall_fscore_support(
            self.true_list, self.pred_list, average=average)

    def instances_predictions(self):
        true_list, pred_list = self._binarized_labels()
        instances_pred_dict = {}
        for true, pred, inst in zip(true_list, pred_list, self.inst_list):
            instances_pred_dict[inst] = {'true': true, 'pred': pred}
        return instances_pred_dict

    def print_instances_predicts(self):
        true_list, pred_list = self._binarized_labels()
        for true, pred, inst in zip(true_list, pred_list, self.inst_list):
            print('{}: {} - {}'.format(inst, true, pred))
