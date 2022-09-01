# FairAST (Fair Algorithm Selection Tool)
Algorithmic Fairness: which algorithm suits my purpose?
Master Thesis Project Utrecht University

## About the project

Many bias mitigation algorithms have been proposed that aim to reduce the impact of 
biases in the data on the decisions made by a Machine Learning algorithm. 
In this project, we investigated how to decide which bias mitigation algorithm
can best be applied on a given dataset. 

For this purpose, we analyzed multiple bias mitigation algorithms, 
and determined several characteristics of datasets that will help to determine 
whether a bias mitigation algorithm will be able to perform well.

FairAST is a tool that recommends the best performing bias mitigation algorithms, 
based on the dataset and the fairness measure the user wants to improve. It does so
by creating a profile of the data, and checking this against the requirements of the algorithms.
The suitable algorithms will be tested on a sample of the data, to see which algorithms perform best.

## How to use FairAST

### Dependencies
FairAST uses [AIF360](https://pypi.org/project/aif360/), [Fairlearn](https://fairlearn.org/), [BlackBoxAuditing](https://pypi.org/project/BlackBoxAuditing/), [scikit-learn](https://scikit-learn.org/stable/install.html), [numpy](https://numpy.org/install/) and [pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html).


### User input
FairAST takes in a (cleaned) dataset and a couple of properties the user needs to specify.

The data should be a dataframe, consisting only of the features and the label that should be used for the classification task.
Any rows containing missing values will be removed by tool. The implementation only works for binary classification tasks.

The user should specify:
- the column name of the label
- which label values correspond to the favorable and the unfavorable label.
- which features should be considered the 'protected attributes'.
- the privileged and the unprivileged groups.
- which fairness measure they want to improve (see Implemented Fairness Measures).

Optionally, the user can specify:
- which features in the data should be considerd as categorical features. The tool will already handle all binary features and features consisting strings as categorical features.
- the techniques to consider, which are either pre-, in-, or post-processing techniques.
- whether to only select algorithms that specifically optimize for the desired fairness measure, by setting 'strict' to True (default=False).

### Making a recommendation
In order to perform all steps of FairAST and make a recommendation, the user only needs to create a Recommendation object, and run the 'recommend' method.
An example:
``` 
rec = Recommendation(sample_runs=3,final_top=3)
rec.recommend(compas_data, ['race-sex'], 'two_year_recid', [{'race-sex': 1}], [{'race-sex': 0}], 'did_not_recid', 'did_recid', 'EO')
```

## How to add a bias mitigation algorithm

New bias mitigation algorithms can be added to the Algorithms class. 

1. Make sure the tool can run the algorithm by importing the package or code of the algorithm.
2. Decide on (the abbreviation of) the name of the algorithm (NAME), and use this name consistently. Make sure it is different from the other implemented algorithms.
3. Create a function in the Algorithms class that can be used to train and test NAME, according to the following format:
```
def run_NAME(self, data, X_train, X_test, y_train, y_test, train_data, test_data):
  """
  Run the NAME algorithm, return the results.
  """
  #TO DO: Make sure the data is in the correct format for the algorithm
  #TO DO: Set any parameters of the algorithm

  #TO DO: Perform all necessary steps to train the algorithm

  pred = #TO DO: make predictions after bias mitigation

  results = self.evaluate(test_data, pred, data.priv, data.unpriv)

  return results
```

4. Add NAME to the dictionary inside the function 'run_algorithms', by adding:
```
'NAME' : self.run_NAME
```
5. Add the requirements of NAME to the dictionary in the function 'get_requirements'.
In case NAME has any new requirements, make sure to add a rule in the function 'get_rules' on how to satisfy this requirement. 
Also make sure to add a function to the Profile, that finds this new characteristic of the data and adds it to the profile.

## Implemented Bias Mitigation Algorithms

From AIF360:
- [Disparate Impact Remover](https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.preprocessing.DisparateImpactRemover.html#aif360.algorithms.preprocessing.DisparateImpactRemover)
- [Learning Fair Representations](https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.preprocessing.LFR.html)
- [Optimized Preprocessing](https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.preprocessing.OptimPreproc.html) 
- [Reweighing](https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.preprocessing.Reweighing.html) 
- [Adversarial Debiasing](https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.inprocessing.AdversarialDebiasing.html) 
- [Gerry Fair Classification](https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.inprocessing.GerryFairClassifier.html) 
- [Meta Fair Classification](https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.inprocessing.MetaFairClassifier.html) 
- [Prejudice Remover](https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.inprocessing.PrejudiceRemover.html)
- [Calibrated Equalized Odds Postprocessing](https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.postprocessing.CalibratedEqOddsPostprocessing.html) 
- [Equalized Odds Postprocessing](https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.postprocessing.EqOddsPostprocessing.html)
- [Reject Option Classification](https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.postprocessing.RejectOptionClassification.html)

From Fairlearn:
- [Correlation Remover](https://fairlearn.org/v0.7.0/api_reference/fairlearn.preprocessing.html)
- [Exponentiated Gradient Reduction](https://fairlearn.org/v0.7.0/api_reference/fairlearn.reductions.html)
- [Grid Search Reduction](https://fairlearn.org/v0.7.0/api_reference/fairlearn.reductions.html)
- [Threshold Optimizer](https://fairlearn.org/v0.7.0/api_reference/fairlearn.postprocessing.html)

Other:
- [Fairness Constraints](https://arxiv.org/abs/1507.05259)

## Implemented Fairness Measures

- Statistical Parity 
- Equalized Odds
- True Positive Rate difference
- False Positive Rate difference
- True Negative Rate difference
- False Negative Rate difference
- Positve Predictive Value difference
- False Discovery Rate difference
- Consistency
