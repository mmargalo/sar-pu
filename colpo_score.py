import numpy as np
import sklearn

def evaluate_all(y, y_pred, e_pred):
    #e = e[y == 1]
    #s = s[y == 1]
    #e_pred = e_pred[y == 1]
    f_results = evaluate_classification(y, y_pred)
    #s_results = evaluate_classification(s, e_pred)
    #e_results = evaluate_propensity_scores(e, e_pred)

    return {
        **{"f_" + k: v for k, v in f_results.items()},
    #    **{"s_" + k: v for k, v in s_results.items()},
    #    **{"e_" + k: v for k, v in e_results.items()}
    }

def evaluate_classification(real, pred):
    tp, fp, tn, fn = tpfptnfn(real, (pred > 0.5).astype(float))
    results = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    results["f1"] = f1_score_tpfptnfn(tp, fp, tn, fn)
    results["accuarcy"] = accuracy_tpfntnfn(tp, fp, tn, fn)
    results["precision"] = precision_tpfptnfn(tp, fp, tn, fn)
    results["recall"] = recall_tpfptnfn(tp, fp, tn, fn)
    results["prp"] = prp_tpfptnfn(tp, fp, tn, fn)
    results["rec2"] = rec2_tpfptnfn(tp, fp, tn, fn)

    tp, fp, tn, fn = tpfptnfn(real, pred)
    results = {**results, **{"p_tp": tp, "p_fp": fp, "p_tn": tn, "p_fn": fn}}

    results["p_f1"] = f1_score_tpfptnfn(tp, fp, tn, fn)
    results["p_accuarcy"] = accuracy_tpfntnfn(tp, fp, tn, fn)
    results["p_precision"] = precision_tpfptnfn(tp, fp, tn, fn)
    results["p_recall"] = recall_tpfptnfn(tp, fp, tn, fn)
    results["p_prp"] = prp_tpfptnfn(tp, fp, tn, fn)
    results["p_rec2"] = rec2_tpfptnfn(tp, fp, tn, fn)
    results["mse"] = sklearn.metrics.mean_squared_error(real, pred)
    results["mae"] = sklearn.metrics.mean_absolute_error(real, pred)
    results["roc_auc"] = sklearn.metrics.roc_auc_score(real, pred)
    results["average_precision"] = sklearn.metrics.average_precision_score(real, pred)
    results["log_loss"] = sklearn.metrics.log_loss(real, pred)

    results["prior"] = real.mean()
    results["pred_prior"] = pred.mean()
    results["prior_err"] = results["pred_prior"] - results["prior"]
    results["prior_abs_err"] = abs(results["prior_err"])
    results["prior_square_err"] = results["prior_err"]**2

    return results

def evaluate_propensity_scores(real, pred):
    tp, fp, tn, fn = tpfptnfn(real, pred)
    results = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    results["f1"] = f1_score_tpfptnfn(tp, fp, tn, fn)
    results["accuarcy"] = accuracy_tpfntnfn(tp, fp, tn, fn)
    results["precision"] = precision_tpfptnfn(tp, fp, tn, fn)
    results["recall"] = recall_tpfptnfn(tp, fp, tn, fn)
    results["prp"] = prp_tpfptnfn(tp, fp, tn, fn)
    results["rec2"] = rec2_tpfptnfn(tp, fp, tn, fn)

    results["mse"] = sklearn.metrics.mean_squared_error(real, pred)
    results["mae"] = sklearn.metrics.mean_absolute_error(real, pred)

    results["prior"] = real.mean()
    results["pred_prior"] = pred.mean()
    results["prior_err"] = results["pred_prior"] - results["prior"]
    results["prior_abs_err"] = abs(results["prior_err"])
    results["prior_square_err"] = results["prior_err"]**2

    return results

def accuracy(true, pred):
    return np.average(1 - abs(pred - true))


def tp(true, pred):
    return (pred * true).mean()
    #return sum(pred[true==1])


def fp(true, pred):
    return (pred * (1 - true)).mean()
    #return sum(pred[true==0])


def tn(true, pred):
    return ((1 - pred) * (1 - true)).mean()
    #return sum((1-pred)[true==0])


def fn(true, pred):
    return ((1 - pred) * true).mean()
    #return sum((1-pred)[true==1])


def tpfptnfn(true, pred):
    return tp(true, pred), fp(true, pred), tn(true, pred), fn(true, pred)


def accuracy_tpfntnfn(tp, fp, tn, fn):
    return (tp + tn) / (tp + fp + tn + fn)


def precision_tpfptnfn(tp, fp, tn, fn):
    if tp + fp == 0.0:
        return 0.0
    return tp / (tp + fp)


def recall_tpfptnfn(tp, fp, tn, fn):
    if tp + fn == 0.0:
        return 0.0
    return tp / (tp + fn)


def f1_score_tpfptnfn(tp, fp, tn, fn):
    prec = precision_tpfptnfn(tp, fp, tn, fn)
    rec = recall_tpfptnfn(tp, fp, tn, fn)
    if (prec + rec) == 0.0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def prp_tpfptnfn(tp, fp, tn, fn):
    return (tp + fp) / (tp + fp + tn + fn)


def rec2_tpfptnfn(tp, fp, tn, fn):
    rec = recall_tpfptnfn(tp, fp, tn, fn)
    prp = prp_tpfptnfn(tp, fp, tn, fn)
    if prp == 0.0:
        return float('inf')
    return rec * rec / prp


def expected_loglikelihood(class_probabilities, propensity_scores, labels):
    prob_labeled = class_probabilities * propensity_scores
    prob_unlabeled_pos = class_probabilities * (1 - propensity_scores)
    prob_unlabeled_neg = 1 - class_probabilities
    prob_pos_given_unl = prob_unlabeled_pos / (
        prob_unlabeled_pos + prob_unlabeled_neg)
    prob_neg_given_unl = 1 - prob_pos_given_unl
    prob_unlabeled_pos[prob_unlabeled_pos ==
                       0] = 0.00000001  #prevent problems of taking log
    prob_unlabeled_neg[prob_unlabeled_neg ==
                       0] = 0.00000001  #prevent problems of taking log
    return (labels * np.log(prob_labeled) + (1 - labels) *
            (prob_pos_given_unl * np.log(prob_unlabeled_pos) +
             prob_neg_given_unl * np.log(prob_unlabeled_neg))).mean()


def label_frequency(class_probabilities, propensity_scores):
    total_pos = class_probabilities.sum()
    total_labeled = (class_probabilities * propensity_scores).sum()
    return total_labeled / total_pos
