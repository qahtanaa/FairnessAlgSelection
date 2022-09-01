import numpy as np
import pandas as pd
import random
from random import sample
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics import utils
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score

from BlackBoxAuditing.repairers.GeneralRepairer import Repairer
from FairAST.biasmitigationalgorithms import *
from aif360.algorithms.preprocessing import *
from aif360.algorithms.preprocessing.optim_preproc_helpers import distortion_functions, opt_tools
from aif360.algorithms.inprocessing import *
from aif360.algorithms.postprocessing import *
from fairlearn.preprocessing import *
from fairlearn.postprocessing import *
from fairlearn.reductions import *
import tensorflow
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

import tensorflow
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

import tensorflow
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

class Algorithms():
  """
  Class that stores the requirements of all the algorithms, in combination with the rules and the compatibility of fairness measures.
  Also contains all the functions to run and evaluate the bias mitigation algorithms, and a baseline Logistic Regression classifier.

  ATTRIBUTES:
  measure : (str) specifying desired fairness measure to optimize for
  compatible : (dict) specifying which fairness measures are compatible with each other.
  requirements : (dict) containing all the algorithms' requirements for the data. 
  rules : (dict) specifying the rules behind the requirements.

  METHODS:
  get_compatibility() : Return a dictionary specifying which fairness measures are compatible with each other.
  get_requirements() : Return a dictionary containing all the algorithms' requirements for the data. 
  get_rules() : Return a dictionary specifying the rules behind the requirements.

  evaluate(test_data, pred, priv_group, unpriv_group) : Return the performance and fairness results based on the original data and the predictions.
  run_LogReg(X_train, y_train, X_test, test_data, weights=None) : run a Logistic Regression classifier
  run_baseline(data, X_train, X_test, y_train, y_test, train_data, test_data): run a baseline algorithm without bias mitigation
  run_{DIR/Rew/LFR/OP/CR/AD/GFC/MFC/PR/EGR/GSR/FC/CEOP/EOP/ROC/TO}(data, X_train, X_test, y_train, y_test, train_data, test_data) : run the bias mitigation algorithm
  run_algorithms(list_of_algorithms, data, X_train, X_test, y_train, y_test, train_data, test_data) : Run all algorithms specified in the list_of_algorithms.
  """

  def __init__(self, measure, strict=False):
    """
    measure : (str) specifying fairness measure to optimize for. Implemented for: 'SP', 'EO', 'TPR', 'FPR', 'C', 'TNR', 'FNR', 'PPV', 'FDR'.
    strict : (bool) specifies whether the allowed fairness measures should strictly the same as the measure (True), or compatible with the measure (False).
    """
    self.measure = measure
    self.strict = strict
    self.measure_names = self.get_measure_names()
    self.compatible = self.get_compatibility()
    self.requirements = self.get_requirements()
    self.rules = self.get_rules()

  def get_compatibility(self):
    """
    Return a dictionary specifying which fairness measures the algorithm is also allowed to optimize for, that do not prohibit the desired measure to improve.
    """
    if self.strict == True:
      #The algorithm should specifically optimize for the specified fairness measure.
      strict = {
        'SP' : ['SP'],
        'EO' : ['EO'],
        'TPR': ['TPR'],
        'FPR': ['FPR'],
        'C'  : ['C'],

        'TNR': ['TNR'],
        'FNR': ['FNR'],
        'PPV': ['PPV'],
        'FDR': ['FDR'],
        'Calibration': ['Calibration']
      }
      return strict
    else:
      #The algorithm should optimize for a fairness measure that is compatible with the specified fairness measure.
      compatible = {
      'SP':   ['SP', 'EO', 'TPR', 'FPR', 'C', 'TNR', 'FNR', 'PPV', 'FDR', 'Calibration'], 
      'EO':   ['EO','SP', 'TPR', 'FPR', 'C', 'TNR', 'FNR'],
      'TPR':  ['TPR', 'SP', 'EO', 'FPR', 'C', 'TNR', 'FNR', 'PPV', 'FDR'],
      'FPR':  ['FPR', 'SP', 'EO', 'TPR', 'C', 'TNR', 'FNR', 'PPV', 'FDR'],
      'C':    ['C', 'SP', 'EO', 'TPR', 'FPR', 'TNR', 'FNR', 'PPV', 'FDR'],
      'TNR':  ['TNR', 'SP', 'EO', 'TPR', 'FPR', 'C', 'FNR', 'PPV', 'FDR'],
      'FNR':  ['FNR', 'SP', 'EO', 'TPR', 'FPR', 'C', 'TNR', 'PPV', 'FDR',],
      'PPV':  ['PPV', 'SP', 'TPR', 'FPR', 'C', 'TNR', 'FNR', 'FDR'],
      'FDR':  ['FDR', 'SP', 'TPR', 'FPR', 'C', 'TNR', 'FNR', 'PPV'],
      'Calibration':['Calibration', 'SP', 'C', 'PPV', 'FDR']
      }
      return compatible

  def get_requirements(self):
    """
    Return a dictionary containing all the algorithms' requirements for the data.
    """
    req = {
        'DIR':  {'technique':'PRE', 'label':["binary"], 'measure': [value for key,value in self.compatible.items() if key in ['SP']], 'max_nr_prot':1, 'feat_type':['numerical', 'mix']},
        'LFR':  {'technique':'PRE', 'label':["binary"], 'measure': [value for key,value in self.compatible.items() if key in ['SP']], 'max_nr_prot':1, 'max_nr_priv':1, 'max_nr_unpriv':1, 'nr_feat_values':'small', 'balance':'balanced'},
        'OP':   {'technique':'PRE', 'label':["binary"], 'measure': [value for key,value in self.compatible.items() if key in ['SP']], 'nr_feat_values':'small', 'feat_type':['categorical']},
        'Rew':  {'technique':'PRE', 'label':["binary"], 'measure': [value for key,value in self.compatible.items() if key in ['SP']]},
        'CR':   {'technique':'PRE', 'label':["binary"], 'measure': [value for key,value in self.compatible.items() if key in ['SP']]},
        'AD':   {'technique':'IN',  'label':["binary"], 'measure': [value for key,value in self.compatible.items() if key in ['EO']], 'max_nr_prot':1, 'max_nr_priv':1, 'max_nr_unpriv':1},
        'GFC':  {'technique':'IN',  'label':["binary"], 'measure': [value for key,value in self.compatible.items() if key in ['FPR', 'FNR']],'protected_type':['single-multi', 'multiple-binary', 'multiple-multi']},
        'MFC':  {'technique':'IN',  'label':["binary"], 'measure': [value for key,value in self.compatible.items() if key in ['SP','FDR']], 'max_nr_prot':1, 'protected_type':['single-binary']},
        'PR':   {'technique':'IN',  'label':["binary"], 'measure': [value for key,value in self.compatible.items() if key in ['SP']], 'max_nr_prot':1},
        'EGR':  {'technique':'IN',  'label':["binary", "regression"], 'measure': [value for key,value in self.compatible.items() if key in ['SP', 'EO', 'TPR', 'FPR', 'TNR', 'FNR']]},
        'GSR':  {'technique':'IN',  'label':["binary", "regression"], 'measure': [value for key,value in self.compatible.items() if key in ['SP', 'EO', 'TPR', 'FPR', 'TNR', 'FNR']]},
        'FC':   {'technique':'IN',  'label':["binary"], 'measure': [value for key,value in self.compatible.items() if key in ['SP']]},
        'CEOP': {'technique':'POST','label':["binary"], 'measure': [value for key,value in self.compatible.items() if key in ['Calibration']]},
        'EOP':  {'technique':'POST','label':["binary"], 'measure': [value for key,value in self.compatible.items() if key in ['EO']]},
        'ROC':  {'technique':'POST','label':["binary"], 'measure': [value for key,value in self.compatible.items() if key in ['SP', 'EO', 'TPR']]},
        'TO':   {'technique':'POST','label':["binary"], 'measure': [value for key,value in self.compatible.items() if key in ['EO', 'SP','TPR', 'FPR', 'TNR', 'FNR']]}
        }
    return req

  def get_rules(self):
    """
    Return a dictionary specifying the rules behind the requirements.
    """
    import itertools
    rules = {
      'label': lambda p, x: True if p in x else False, #the label should be in the allowed types of classification labels
      'balance' : lambda p, x: p == x, #the label balance should be balanced
      'max_nr_prot': lambda p, x: p <= x, #the number of protected attributes should be equal to or less than the max
      'nr_feat_values': lambda p, x: p == x, #the number of features/values can be either 'large' or 'small'
      'feat_type': lambda p, x: True if p in x else False, # the type of data should be one of the options for the algorithm
      'protected_type': lambda p, x: True if p in x else False, #the type of protected features should be one of the options for the algorithm
      'max_nr_priv': lambda p, x: p <= x, #the number of privileged groups should be equal to or less than the max
      'max_nr_unpriv': lambda p, x: p <= x, #the number of unprivileged groups should be equal to or less than the max
      'measure' : lambda p, x : True if p in list(itertools.chain(*x)) else False, #True if the algorithm optimizes for the specified (or a compatible) measure
      'technique' : lambda p, x: True if x in p else False #True if the algorithm falls into one of the desired technique categories
    }
    return rules

  def get_measure_names(self):

    measure_names = ['b_acc_p', 'b_acc_up', 'precision', 'recall', 'f1', 'TPR', 'FPR', 'EO', 'SP', 'C']
    if self.measure not in measure_names:
      measure_names = measure_names.append(self.measure)
    return measure_names

  """
  RUNNING AND EVALUATING THE ALGORITHMS
  """

  def evaluate(self, test_data, pred, priv_group, unpriv_group):
    """
    Return the performance and fairness results based on the original data and the predictions.

    test_data : (BinaryLabelDataset) ground truth labels
    pred : (BinaryLabelDataset) predicted labels
    priv_group : (list(dict)) representation of the privileged group
    unpriv_group : (list(dict)) representation of the unprivileged group
    """
    #Evaluation of the model
    cm = ClassificationMetric(test_data, pred, unprivileged_groups=unpriv_group, privileged_groups=priv_group)
    dm = BinaryLabelDatasetMetric(pred, unprivileged_groups=unpriv_group, privileged_groups=priv_group)

    priv_cond = utils.compute_boolean_conditioning_vector(
                            test_data.protected_attributes,
                            test_data.protected_attribute_names,
                            condition=priv_group)
    unpriv_cond = utils.compute_boolean_conditioning_vector(
                            test_data.protected_attributes,
                            test_data.protected_attribute_names,
                            condition=unpriv_group)
    privs_orig = test_data.labels[priv_cond]
    unprivs_orig = test_data.labels[unpriv_cond]
    privs_pred = pred.labels[priv_cond]
    unprivs_pred = pred.labels[unpriv_cond]
    
    measure_scores = {
        'b_acc_p':   balanced_accuracy_score(privs_orig, privs_pred),
        'b_acc_up':  balanced_accuracy_score(unprivs_orig, unprivs_pred),
        'precision': precision_score(test_data.labels, pred.labels),
        'recall':    recall_score(test_data.labels, pred.labels),
        'f1' :       f1_score(test_data.labels, pred.labels),

        'SP' : dm.disparate_impact(),
        'EO' : cm.average_abs_odds_difference(),
        'TPR': abs(cm.true_positive_rate_difference()),
        'FPR': abs(cm.false_positive_rate_difference()),
        'C'  : dm.consistency()[0],
        'TNR': abs(cm.true_negative_rate(privileged=True)-cm.true_negative_rate(privileged=False)),
        'FNR': cm.false_negative_rate_difference(),
        'PPV': abs(cm.positive_predictive_value(privileged=True)-cm.positive_predictive_value(privileged=False)),
        'FDR': cm.false_discovery_rate_difference()
    }

    results = []
    for elem in self.measure_names: 
      score = measure_scores[elem]
      results.append(score)
    
    return results

  def run_LogReg(self, X_train, y_train, X_test, test_data, weights=None):
    """
    Run a Logistic Regression classifier, return the predictions.
    """
    clf = LogisticRegression(solver='liblinear')
    clf.fit(X_train, y_train.ravel(), sample_weight=weights)
    pred = test_data.copy(deepcopy = True)
    pred.labels = clf.predict(X_test).reshape(-1,1)

    return pred

  def run_baseline(self, data, X_train, X_test, y_train, y_test, train_data, test_data):
    """
    Return the results of a baseline algorithm (Logistic Regression, without bias mitigation).
    """
    pred = self.run_LogReg(X_train, y_train, X_test, test_data)
    results = self.evaluate(test_data, pred, data.priv, data.unpriv)

    return results

  def run_DIR(self, data, X_train, X_test, y_train, y_test, train_data, test_data):
    """
    Run AIF360's Disparate Impact Remover algorithm, return the results.
    """
    #Remove categorical data
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    categorical = [c for c in data.categorical if c not in [data.label_name, 'stratify', data.protected[0]]]
    #Convert to BinaryLabelDataset
    train_data = BinaryLabelDataset(favorable_label=1.0, unfavorable_label=0.0, df=train.drop(categorical, axis=1), label_names=[data.label_name], protected_attribute_names=data.protected)
    test_data = BinaryLabelDataset(favorable_label=1.0, unfavorable_label=0.0, df=test.drop(categorical, axis=1), label_names=[data.label_name], protected_attribute_names=data.protected)
    
    #Disparate Impact Remover
    dir = DisparateImpactRemover(repair_level=1)
    dir_train_data = dir.fit_transform(train_data)
    dir_test_data = dir.fit_transform(test_data)

    #Remove the sensitive feature from the data, add categorical data back  
    train_transf = dir_train_data.convert_to_dataframe()[0]
    test_transf = dir_test_data.convert_to_dataframe()[0]
    train_transf = train_transf.drop(data.protected, axis=1)
    test_transf = test_transf.drop(data.protected, axis=1)
    train_transf = train_transf.drop([data.label_name], axis=1)
    test_transf = test_transf.drop([data.label_name], axis=1)
    cat_train = train[categorical]
    dir_train_features = pd.concat([train_transf.reset_index(drop=True),cat_train.reset_index(drop=True)], axis=1)
    cat_test = test[categorical]
    dir_test_features = pd.concat([test_transf.reset_index(drop=True),cat_test.reset_index(drop=True)], axis=1)

    pred = self.run_LogReg(dir_train_features, dir_train_data.labels, dir_test_features, test_data)

    results = self.evaluate(test_data, pred, data.priv, data.unpriv)
    
    return results

  def run_LFR(self, data, X_train, X_test, y_train, y_test, train_data, test_data):
    """
    Run AIF360's Learning Fair Representations algorithm, return the results.
    """
    #Learning Fair Representations
    lfr = LFR(unprivileged_groups=data.unpriv, privileged_groups=data.priv, k=10)
    lfr.fit(train_data)
    lfr_train_data = lfr.transform(train_data)
    lfr_test_data = lfr.transform(test_data)

    pred = self.run_LogReg(lfr_train_data.features, lfr_train_data.labels, lfr_test_data.features, test_data)

    results = self.evaluate(test_data, pred, data.priv, data.unpriv)

    return results

  def run_OP(self, data, X_train, X_test, y_train, y_test, train_data, test_data):
    """
    Run AIF360's Optimized Preprocessing algorithm, return the results.
    """
    from aif360.algorithms.preprocessing.optim_preproc_helpers import distortion_functions, opt_tools
    if set(X_train.columns) == set(['credit_history', 'savings', 'employment', 'sex', 'age']):
      distortion = distortion_functions.get_distortion_german
    elif set(X_train.columns) == set(['age_cat', 'c_charge_degree', 'priors_count', 'sex', 'race']):
      distortion = distortion_functions.get_distortion_compas
    elif set(X_train.columns) == set(['Age (decade)', 'Education Years', 'sex', 'race']):
      distortion = distortion_functions.get_distortion_adult
    else:
      def get_distortion_mydata(vold, vnew):
        """
        Note: Users can use this as templates to create other distortion functions.
        Args:
          vold (dict) : {attr:value} with old values
          vnew (dict) : dictionary of the form {attr:value} with new values
        Returns:
          d (value) : distortion value
        """
        distort = {}

        # Create distortion functions

        # Check if distortion functions have been created:
        assert bool(distort)

        # Calculate total cost
        total_cost = 0.0
        for k in vold:
          if k in vnew:
            total_cost += distort[k].loc[vnew[k], vold[k]]

        return total_cost

      distortion = get_distortion_mydata
      try:
        distortion({1:1},{2:1})
      except:
        distortion = None
        print("""For OP: Please complete the distortion function called 'get_distortion_mydata'. 
        See examples: https://github.com/Trusted-AI/AIF360/blob/master/aif360/algorithms/preprocessing/optim_preproc_helpers/distortion_functions.py""")

    if distortion == None:
      return [0] * len(self.measure_names)
    else:
      optim_options = {
        "distortion_fun": distortion,
        "epsilon": 0.05,
        "clist": [0.99, 1.99, 2.99],
        "dlist": [.1, 0.05, 0]
      }

      #Optimized Preprocessing
      opp = OptimPreproc(opt_tools.OptTools, optim_options)  
      opp.fit(train_data)
      opp_train_data = opp.transform(train_data)
      opp_test_data = opp.transform(test_data)

      pred = self.run_LogReg(opp_train_data.features, opp_train_data.labels, opp_test_data.features, test_data)
    
      results = self.evaluate(test_data, pred, data.priv, data.unpriv)
      return results

  def run_Rew(self, data, X_train, X_test, y_train, y_test, train_data, test_data):
    """
    Run AIF360's Reweigher algorithm, return the results.
    """
    #Reweigh the data
    rw = Reweighing(unprivileged_groups=data.unpriv, privileged_groups=data.priv)
    rw.fit(train_data)
    rw_train_data = rw.transform(train_data)

    pred = self.run_LogReg(rw_train_data.features, rw_train_data.labels, test_data.features, test_data, rw_train_data.instance_weights)

    results = self.evaluate(test_data, pred, data.priv, data.unpriv)
    
    return results

  def run_CR(self, data, X_train, X_test, y_train, y_test, train_data, test_data):
    """
    Run Fairlearn's Disparate Impact Remover algorithm, return the results.
    """
    #Correlation Remover
    cr = CorrelationRemover(sensitive_feature_ids=data.protected, alpha=1)
    cr.fit(X_train)
    cr_train_data_features = cr.transform(X_train)
    cr_test_data_features = cr.transform(X_test)

    pred = self.run_LogReg(cr_train_data_features, train_data.labels, cr_test_data_features, test_data)

    results = self.evaluate(test_data, pred, data.priv, data.unpriv)

    return results

  def run_AD(self, data, X_train, X_test, y_train, y_test, train_data, test_data):
    """
    Run AIF360's Adversarial Debiasing algorithm, return the results.
    """
    #Adversarial debiasing
    sess=tf.Session()
    debiased = AdversarialDebiasing(privileged_groups = data.priv,
                          unprivileged_groups = data.unpriv,
                          scope_name='debiased',
                          debias=True, sess=sess)
    debiased.fit(train_data)
    pred = debiased.predict(test_data)
    sess.close()
    tf.reset_default_graph()
    sess = tf.Session()

    results = self.evaluate(test_data, pred, data.priv, data.unpriv)

    return results

  def run_GFC(self, data, X_train, X_test, y_train, y_test, train_data, test_data):
    """
    Run AIF360's Gerry Fair Classification algorithm, return the results.
    """
    measures = {
        'FPR' : 'FP', 
        'FNR': 'FN'
        }
    if self.measure not in measures.keys():
      for m in measures.keys():
        if self.measure in self.compatible[m]:
          measure = measures[m]
          break
    else:
      measure=measures[self.measure]

    split_point = len(X_train)
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis =1)
    all_data = pd.concat([train, test], axis=0)
    sensitive_attrs = data.protected

    categorical = [x for x in data.categorical if x not in [data.label_name, 'stratify']]

    prot_att_names = []
    for att in sensitive_attrs:
      if att in categorical:
        for value in all_data[att].unique():
          prot_att_names.append(str(att)+'_'+str(value))

    all_data = pd.get_dummies(all_data, columns=categorical)

    gerry_train = all_data[:split_point]
    gerry_test = all_data[split_point:]

    gerry_train_data = BinaryLabelDataset(favorable_label=1.0, unfavorable_label=0.0, df=gerry_train, label_names=[data.label_name], protected_attribute_names=prot_att_names)
    gerry_test_data = BinaryLabelDataset(favorable_label=1.0, unfavorable_label=0.0, df=gerry_test, label_names=[data.label_name], protected_attribute_names=prot_att_names)

    GFC = GerryFairClassifier(fairness_def=measure)
    GFC.fit(gerry_train_data, early_termination=True)
    gerry_pred = GFC.predict(gerry_test_data, threshold=0.5)

    pred = test_data.copy(deepcopy = True)
    pred.labels = gerry_pred.labels

    results = self.evaluate(test_data, pred, data.priv, data.unpriv)
    
    return results

  def run_MFC(self, data, X_train, X_test, y_train, y_test, train_data, test_data):
    """
    Run AIF360's Meta Fair Classification algorithm, return the results.
    """
    measures = {
        'SP' : 'sr', 
        'FDR': 'fdr'
        }
    if self.measure not in measures.keys():
      for m in measures.keys():
        if self.measure in self.compatible[m]:
          measure = measures[m]
          break
    else:
      measure=measures[self.measure]

    
    train_data.privileged_protected_attributes = [data.priv[0][data.protected[0]]]
    test_data.privileged_protected_attributes = [data.priv[0][data.protected[0]]]
    mfc = MetaFairClassifier(tau=0.8, type=measure)
    mfc.fit(train_data)
    pred = mfc.predict(test_data)

    results = self.evaluate(test_data, pred, data.priv, data.unpriv)

    return results    

  def run_PR(self, data, X_train, X_test, y_train, y_test, train_data, test_data):
    """
    Run AIF360's Prejudice Remover algorithm, return the results.
    """
    train_data.privileged_protected_attributes=[{data.protected[0] : 1}]
    train_data.unprivileged_protected_attributes=[{data.protected[0] : 0}]
    test_data.privileged_protected_attributes=[{data.protected[0] : 1}]
    test_data.unprivileged_protected_attributes=[{data.protected[0] : 0}]

    pr = PrejudiceRemover(eta=1.0, sensitive_attr=data.protected[0], class_attr=data.label_name)
    pr.fit(train_data)
    pred = pr.predict(test_data)

    results = self.evaluate(test_data, pred, data.priv, data.unpriv)

    return results

  def run_EGR(self, data, X_train, X_test, y_train, y_test, train_data, test_data):
    """
    Run Fairlearn's Exponentiaded Gradient Reduction algorithm, return the results.
    """
    measures = {
        'EO' :EqualizedOdds, 
        'SP' :DemographicParity, 
        'TPR':TruePositiveRateParity, 
        'FPR':FalsePositiveRateParity
        }
    if self.measure not in measures.keys():
      for m in measures.keys():
        if self.measure in self.compatible[m]:
          measure = measures[m]
          break
    else:
      measure=measures[self.measure]

    #ExponentiatedGradientReduction
    estimator = LogisticRegression(solver='liblinear')
    egr = ExponentiatedGradient(estimator=estimator, constraints=measure())
    egr.fit(X=train_data.features, y=train_data.labels, sensitive_features=train_data.protected_attributes)
    pred = test_data.copy(deepcopy=True)
    pred.labels = egr.predict(test_data.features).reshape(-1,1)

    results = self.evaluate(test_data, pred, data.priv, data.unpriv)

    return results

  def run_GSR(self, data, X_train, X_test, y_train, y_test, train_data, test_data):
    """
    Run Fairlearn's Grid Search Reduction algorithm, return the results.
    """
    measures = {
        'EO' :EqualizedOdds, 
        'SP' :DemographicParity, 
        'TPR':TruePositiveRateParity, 
        'FPR':FalsePositiveRateParity 
        }
    if self.measure not in measures.keys():
      for m in measures.keys():
        if self.measure in self.compatible[m]:
          measure = measures[m]
          break
    else:
      measure=measures[self.measure]

    #GridSearchReduction
    estimator = LogisticRegression(solver='liblinear')
    gs = GridSearch(estimator=estimator, constraints=measure())
    gs.fit(X=X_train.drop(data.protected, axis=1), y=train_data.labels, sensitive_features=train_data.protected_attributes)
    pred = test_data.copy(deepcopy=True)
    pred.labels = gs.predict(X_test.drop(data.protected, axis=1)).reshape(-1,1)
    
    results = self.evaluate(test_data, pred, data.priv, data.unpriv)

    return results

  def run_FC(self, data, X_train, X_test, y_train, y_test, train_data, test_data):
    """
    Run the Fairness Constraints algorithm, return the results.
    """
    split_point = len(X_train)
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis =1)
    all_data = pd.concat([train, test], axis=0)
    sensitive_attrs = data.protected

    x_control = {}
    for p in sensitive_attrs:
      x_control[p] = list(all_data[p])

    y = []
    for class_label in all_data[data.label_name]:
      if class_label in [0, 0., 0.0]:
        class_label = -1
      elif class_label in [1, 1., 1.0]:
        class_label = 1
      else:
        raise Exception("Invalid class label value")
    
      y.append(class_label)

    features = all_data.drop([data.label_name], axis=1)
    features = features.drop(data.protected, axis=1)
    categorical = []
    for col in data.categorical:
      if col not in data.protected:
        if col not in [data.label_name, 'stratify']:
          categorical.append(col)
    features = pd.get_dummies(features, columns=categorical)
    X = []
    for feat in features.columns:
      col = features[feat].tolist()
      X.append(col)

    X = np.array(X, dtype=float).T
    y = np.array(y, dtype =int)
    for k, v in x_control.items(): x_control[k] = np.array(v, dtype=int)

    x_train = X[:split_point]
    x_test = X[split_point:]
    y_train = y[:split_point]
    y_test = y[split_point:]
    x_control_train = {}
    x_control_test = {}
    for k in x_control.keys():
      x_control_train[k] = x_control[k][:split_point]
      x_control_test[k] = x_control[k][split_point:]
    
    def train_test_classifier():
	    w = train_model(x_train, y_train, x_control_train, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma)
	    distances_boundary_test = (np.dot(x_test, w)).tolist()
	    all_class_labels_assigned_test = np.sign(distances_boundary_test)	
	    return all_class_labels_assigned_test

    apply_fairness_constraints = 1
    apply_accuracy_constraint = 0
    sep_constraint = 0
    loss_function = _logistic_loss
    sensitive_attrs = data.protected
    sensitive_attrs_to_cov_thresh = {}
    for p in data.protected:
      sensitive_attrs_to_cov_thresh[p] = [0] * data.df[p].nunique()
    gamma = None

    pred = test_data.copy(deepcopy = True)
    labels = train_test_classifier()
    for l in range(len(labels)):
      if labels[l] in  [-1.0, -1., -1]:
        labels[l] = 0.
      elif labels[l] in [1.0, 1., 1]:
        labels[l] = 1.
    pred.labels = labels.reshape(-1,1)

    results = self.evaluate(test_data, pred, data.priv, data.unpriv)

    return results


  def run_CEOP(self, data, X_train, X_test, y_train, y_test, train_data, test_data):
    """
    Run AIF360's Calibrated EqOdds Postprocessing algorithm, return the results.
    """
    #Initial classifier
    clf = LogisticRegression(solver='liblinear')
    clf.fit(train_data.features, train_data.labels.ravel())

    # positive class index
    pos_ind = np.where(clf.classes_ == train_data.favorable_label)[0][0]
    # get the prediction scores
    train_pred = train_data.copy(deepcopy = True)
    train_pred.scores = clf.predict_proba(train_data.features)[:,pos_ind].reshape(-1,1)
    test_pred = test_data.copy(deepcopy=True)
    test_pred.scores = clf.predict_proba(test_data.features)[:,pos_ind].reshape(-1,1)

    #CalibratedEqOdds
    ceop = CalibratedEqOddsPostprocessing(unprivileged_groups=data.unpriv, privileged_groups=data.priv, cost_constraint='weighted')
    ceop.fit(train_data, train_pred)
    pred = ceop.predict(test_pred)

    results = self.evaluate(test_data, pred, data.priv, data.unpriv)

    return results

  def run_EOP(self, data, X_train, X_test, y_train, y_test, train_data, test_data):
    """
    Run AIF360's EqOdds Postprocessing algorithm, return the results.
    """
    #Initial classifier
    clf = LogisticRegression(solver='liblinear')
    clf.fit(train_data.features, train_data.labels.ravel())
    train_pred = train_data.copy(deepcopy = True)
    train_pred.labels = clf.predict(train_data.features).reshape(train_pred.labels.shape)
    test_pred = test_data.copy(deepcopy = True)
    test_pred.labels = clf.predict(test_data.features).reshape(test_pred.labels.shape)

    #EqOddsPostprocessing
    eop = EqOddsPostprocessing(data.unpriv, data.priv)
    eop.fit(train_data, train_pred)
    pred = test_data.copy(deepcopy=True)
    pred = eop.predict(test_pred)

    results = self.evaluate(test_data, pred, data.priv, data.unpriv)

    return results

  def run_ROC(self, data, X_train, X_test, y_train, y_test, train_data, test_data):
    """
    Run AIF360's Reject Option Classification algorithm, return the results.
    """
    measures = {
        'EO' :"Average odds difference", 
        'SP' :"Statistical parity difference", 
        'TPR':"Equal opportunity difference", 
        }
    if self.measure not in measures.keys():
      for m in measures.keys():
        if self.measure in self.compatible[m]:
          measure = measures[m]
          break
    else:
      measure=measures[self.measure]

    #Training a classifier
    clf = LogisticRegression(solver='liblinear')
    clf.fit(train_data.features, train_data.labels.ravel())
    # positive class index
    pos_ind = np.where(clf.classes_ == train_data.favorable_label)[0][0]
    # get the prediction scores
    train_pred = train_data.copy(deepcopy = True)
    train_pred.scores = clf.predict_proba(train_data.features)[:,pos_ind].reshape(-1,1)
    test_pred = test_data.copy(deepcopy=True)
    test_pred.scores = clf.predict_proba(test_data.features)[:,pos_ind].reshape(-1,1)

    #Using the predictions from the initial model to fit a post processing classifier
    roc = RejectOptionClassification(unprivileged_groups=data.unpriv, 
                                 privileged_groups=data.priv, 
                                 low_class_thresh=0.01, high_class_thresh=0.99,
                                  num_class_thresh=100, num_ROC_margin=50,
                                  metric_name=measure,
                                  metric_ub=0.05, metric_lb=-0.05)
    roc.fit(train_data, train_pred)
    pred = roc.predict(test_pred)
    
    results = self.evaluate(test_data, pred, data.priv, data.unpriv)

    return results

  def run_TO(self, data, X_train, X_test, y_train, y_test, train_data, test_data):
    """
    Run Fairlearn's Threshold Optimizer algorithm, return the results.
    """
    measures = {
        'EO' :"equalized_odds", 
        'SP' :"demographic_parity", 
        'TPR':"true_positive_rate_parity", 
        'FPR':"false_positive_rate_parity", 
        'TNR':"true_negative_rate_parity", 
        'FNR':"false_negative_rate_parity"
        }
    if self.measure not in measures.keys():
      for m in measures.keys():
        if self.measure in self.compatible[m]:
          measure = measures[m]
          break
    else:
      measure=measures[self.measure]

    #Initial classifier
    clf = LogisticRegression(solver='liblinear')
    clf.fit(train_data.features, train_data.labels.ravel())

    #ThresholdOptimizer
    to = ThresholdOptimizer(estimator=clf, constraints=measure, prefit=True, predict_method='auto')
    to.fit(X=train_data.features, y=train_data.labels.ravel(), sensitive_features=train_data.protected_attributes)
    pred = test_data.copy(deepcopy=True)
    pred.labels = to.predict(X=test_data.features, sensitive_features=test_data.protected_attributes, random_state=1).reshape(-1,1)
    
    results = self.evaluate(test_data, pred, data.priv, data.unpriv)

    return results


  def run_algorithms(self, list_of_algorithms, data, X_train, X_test, y_train, y_test, train_data, test_data):
    """
    Run all algorithms specified in the list_of_algorithms.

    list_of_algorithms : (list) containing the names of the algorithms that need to be run
    data : (DataPreparation object)
    X_train : (DataFrame) containing all train features
    X_test : (DataFrame) containing all test features
    y_train : (DataFrame) containing all train labels
    y_test : (DataFrame) containing all test labels
    train_data : (BinaryLabelDataset) the features and labels in the train set
    test_data : (BinaryLabelDataset) the features and labels in the test set
    """
    run = {
        'baseline' : self.run_baseline,
        'DIR' : self.run_DIR,
        'Rew' : self.run_Rew,
        'LFR' : self.run_LFR,
        'OP' : self.run_OP,
        'CR' : self.run_CR,
        'AD' : self.run_AD,
        'GFC' : self.run_GFC,
        'MFC' : self.run_MFC,
        'PR' : self.run_PR,
        'EGR' : self.run_EGR,
        'GSR' : self.run_GSR,
        'FC' : self.run_FC,
        'CEOP' : self.run_CEOP,
        'EOP' : self.run_EOP,
        'ROC' : self.run_ROC,
        'TO' : self.run_TO
    }
    
    all_results = []
    for a in list_of_algorithms:
      alg_results = run[a](data, X_train, X_test, y_train, y_test, train_data, test_data)
      all_results = all_results + [alg_results]
    return all_results
