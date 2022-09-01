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

from FairAST.datapreparation import DataPreparation
from FairAST.profile import Profile
from FairAST.algorithms import Algorithms

class Recommendation():
  """
  Recommend the best bias mitigation algorithm to use, in order to optimize the fairness with respect to a desired fairness measure. 
  The recommendation is based on the properties of the data and the requirements of the algorithms.
  In case there are multiple suitable algorithms, the possible algorithms are tested on a sample of the data.
  The best performing algorithms are tested on the complete dataset.

  METHODS
    check_requirements() : Check for which algorithms the dataset fits the requirements, and return a list of possible algorithms.
    split_data() : Split the data in a train and test set.
    test_baseline() : Test a baseline algorithm (Logistic Regression) and return the results.
    test_sample() : Test the possible algorithms on a sample of the data. 
    test_final() : Test the best algorithm(s) on the complete dataset.
    test() : Test the suitable algorithms.
    recommendation() : Check the requirements and test the algorithms.
    recommend() : Perform all steps in the recommendation tool: prepare the data, create a profile of the data, check the requirements and test the algorithms. 
  """

  def __init__(self, data=None, profile=None, algorithms=None, sample_runs=3, final_top=3):
    """
    Construct all necessary attributes for the recommendation.

    PARAMETERS
    data : (DataPreparation object) (optional) containing the data 
    profile : (dict) (optional) storing the profile of the data
    algorithms : (Algorithms object) (optional) storing the algorithms and their requirements
    sample_runs : (int) specifying the number of sample runs before recommending the best performing algorithm
    final_top : (int) specifying the numer of best performing algorithms to finally test on the complete dataset
    """
    self.data = data
    self.profile = profile
    self.algorithms = algorithms
    self.sample_runs = sample_runs
    self.final_top = final_top

  def check_requirements(self):
    """
    Check for which algorithms the dataset fits the requirements,
    and return a list of possible algorithms.
    """
    self.possibilities = []
    for alg, a in self.algorithms.requirements.items():
      checks = []
      for prof, p in self.profile.items():
        if prof in a:
          if self.algorithms.rules[prof](p, a[prof]):
            checks.append(True)
          else:
            checks.append(False)
            break
      if all(checks):
        self.possibilities.append(alg)
    return self.possibilities

  def split_data(self, data, train_size, test_size, n):
    """
    Split the data in a train and test set. Return the train/test features and labels, and the train and test set in a BinaryLabelDataset format.

      data : (DataPreparation object) stores the data in the correct format 
      train_size : (int/float) if int: absolute number of instances in train set, if float: proportion of the data in the train data
      test_size : (int/float) if int: absolute number of instances in test set, if float: proportion of the data in the test data
      n : (int) specifying the random state
    """
    #Prepare train and test data
    strat = data.df['stratify']
    features = data.df.drop([data.label_name, 'stratify'], axis=1)
    labels = data.df[data.label_name]
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size,  train_size=train_size, shuffle=True, random_state=n, stratify=strat)
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    #Convert to BinaryLabelDataset
    train_data = BinaryLabelDataset(favorable_label=1.0, unfavorable_label=0.0, df=train, label_names=[data.label_name], protected_attribute_names=data.protected)
    test_data = BinaryLabelDataset(favorable_label=1.0, unfavorable_label=0.0, df=test, label_names=[data.label_name], protected_attribute_names=data.protected)

    return X_train, X_test, y_train, y_test, train_data, test_data   

  def test_baseline(self):
    """
    Test a baseline algorithm (Logistic Regression) and return the results.
    """
    X_train, X_test, y_train, y_test, train_data, test_data = self.split_data(self.data, train_size=0.7, test_size=0.3, n=1)
    baseline = self.algorithms.run_algorithms(['baseline'], self.data, X_train, X_test, y_train, y_test, train_data, test_data)
    baseline_results = pd.DataFrame(baseline, columns =self.algorithms.measure_names, index = ['LogReg'])
    return baseline_results

  def test_sample(self, possibilities, n=1):
    """
    Test the possible algorithms on a sample of the data. 
    A sample consists of 5000 instances (3500 train, 1500 test), or the whole dataset if there are less than 5000 instances.
    Return top_list, a dataframe showing the ranking of the algorithms' performances with respect to the chosen fairness measure.

      possibilities : (list(str)) list of algorithms that need to be tested.
    """

    if len(self.data.df) > 5000: 
      X_train, X_test, y_train, y_test, train_data, test_data = self.split_data(self.data, train_size=3500, test_size=1500, n=n)
    else:
      X_train, X_test, y_train, y_test, train_data, test_data = self.split_data(self.data, train_size=0.7, test_size=0.3, n=n)

    sample_results = self.algorithms.run_algorithms(possibilities, self.data, X_train, X_test, y_train, y_test, train_data, test_data)
    all_results = pd.DataFrame(sample_results, columns =self.algorithms.measure_names, index = possibilities)
    if self.profile['measure'] in ['SP', 'C']:
      #sort results by which value in SP or C is closest to 1
      sorted_top_list = sorted(all_results.index.values, key=lambda k: abs(all_results[self.profile['measure']][k] - 1))
      top_list = all_results.reindex(sorted_top_list)
    else:
      top_list = all_results.sort_values(by = self.profile['measure'], ascending=True)
      
    return top_list

  def test_final(self, algorithms):
    """
    Test the best algorithm(s) on the complete dataset.
    Return results, a dataframe showing the performance and fairness measures.

      algorithms: (list(str)) list containing the algorithms that needs to be tested.
    """
    X_train, X_test, y_train, y_test, train_data, test_data = self.split_data(self.data, train_size=0.7, test_size=0.3, n=1)
    final_results = self.algorithms.run_algorithms(algorithms, self.data, X_train, X_test, y_train, y_test, train_data, test_data)
    all_results = pd.DataFrame(final_results, columns =self.algorithms.measure_names, index = algorithms)
    if self.profile['measure'] in ['SP', 'C']:
      sorted_top_list = sorted(all_results.index.values, key=lambda k: abs(all_results[self.profile['measure']][k] - 1))
      top_list = all_results.reindex(sorted_top_list)
    else:
      top_list = all_results.sort_values(by = self.profile['measure'], ascending=True)
    return top_list

  def test(self):
    """
    Test the suitable algorithms. If there are multiple suitable algorithms, first test these on a sample of the data. Return the test results of the best performing algorithm. 
    """    
    #When no suitable algorithm has been found:
    if len(self.possibilities) == 0:
      results = []
      print("Sorry, there are no suitable algorithms found.")
    
    #When only one or two suitable algorithms have been found:
    elif len(self.possibilities) <= 2:
      print("There is/are {} suitable algorithm found: {}".format(len(self.possibilities), self.possibilities))
      results = self.test_final(self.possibilities, 1)

    #When multiple suitable algorithms have been found:
    else:
      print("There are {} suitable algorithms found: {}".format(len(self.possibilities), self.possibilities))
      print("Running these algorithms on a sample of the data gives the following results:")
      scores = dict.fromkeys(self.possibilities, 0)
      for n in range(self.sample_runs):
        top_list = self.test_sample(self.possibilities, n)
        display(top_list)
        top = top_list.index.values
        for i in range(len(top)):
          scores[top[i]] += len(top)-i
      #scores = {k:v for k,v in scores.items() if v != 0}
      print("scores", scores.items())
      scores_values = list(set(scores.values()))
      print("values", scores_values)
      best = [key for key, value in scores.items() if value in sorted(scores_values, reverse=True)[:self.final_top]]
      print("best", best)
      results = self.test_final(best)
    return results

  def recommendation(self):
    """
    Check the requirements and test the algorithms.
    """
    self.check_requirements()
    results = self.test()
    return results
  
  def recommend(self, df, protected, label, priv, unpriv, fav, unfav, measure, categorical=[], technique=['PRE', 'IN', 'POST'], strict=False):
    """
    Perform all steps in the recommendation tool: prepare the data, create a profile of the data, check the requirements and test the algorithms. 

    df : (pandas DataFrame) containing the data
    protected : (list(str)) specifying the column names of all protected features
    label : (str) specifying the label column
    priv : (list(dicts)) representation of the privileged groups
    unpriv : (list(dicts)) representation of the unprivileged groups
    fav : (str/int/..) value representing the favorable label
    unfav : (str/int/..) value representing the unfavorable label
    measure : (str) specifying desired fairness measure to optimize for
              'EO' = equalized odds, 
              'SP' = statistical parity/demographical parity, 
              'TPR' = equal True Positive Rate/equality of opportunity
    categorical : (list(str)) (optional) specifying column names of categorical features
    technique : (list(str)) (optional) specifying the allowed type(s) of bias mitigation technique. Options are:
                'PRE' = preprocessing techniques
                'IN' = in-processing techniques
                'POST' = postprocessing techniques
    strict : (bool) 'False' allows to select algorithms that optimize for compatible measures, 
                    'True' only considers algorithms that specifically optimize for the desired fairness measure
    """
    self.data = DataPreparation(df, protected, label, priv, unpriv, fav, unfav, categorical).prepare()
    self.profile = Profile(self.data, measure, technique).create_profile()
    self.algorithms = Algorithms(measure, strict)
    print("1. DATA PREPARATION")
    print("The resulting data after preprocessing:")
    display(self.data.df)
    print()
    print("2. PROFILING")
    print("The profile of the data:")
    print(self.profile)
    print()
    print("3. BASELINE")
    print("Testing a Logistic Regression model without bias mitigation gives the following results:")
    baseline = self.test_baseline()
    display(baseline)
    print()
    print("4. FIND BIAS MITIGATION ALGORITHM")
    results = self.recommendation()
    print()
    print("5. RECOMMENDATION")
    print("Running the best algorithm(s) on the data gives the following results:")
    display(results)
    return results
