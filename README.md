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


### Required user input
FairAST takes in a (cleaned) dataset and a couple of properties the user needs to specify.

The data should be a dataframe, consisting only of the features and the label that should be used for the classification task.
Any rows containing missing values will be removed by tool.

The implementation only works for binary classification tasks.
The user should specify the column name of the label, and which label values correspond to the favorable and the unfavorable label.

The user needs to determine which features should be considered the 'protected attributes'.
The user needs to determine the privileged and the unprivileged groups.

The user needs to specify which fairness measure they want to improve (see Implemented Fairness Measures).

Optionally, the user can specify which features in the data should be considerd as categorical features. 
The tool will already handle all binary features and features consisting strings as categorical features.

Optionally, the user can only consider a subset of algorithms based on their technique, which are either pre-, in-, or post-processing techniques.

Optionally, the user can consider only algorithms that specifically optimize for the desired fairness measure, by setting 'strict' to True.

### Making a recommendation
In order to perform all steps of FairAST and make a recommendation, the user only needs to create a Recommendation object, and run the 'recommend' method.
An example:
'''
rec = Recommendation(sample_runs=3,final_top=3)
rec.recommend(compas_data, ['race-sex'], 'two_year_recid', [{'race-sex': 1}], [{'race-sex': 0}], 'did_not_recid', 'did_recid', 'EO')
'''

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
- Flse Positive Rate difference
- True Negative Rate difference
- False Negative Rate difference
- Positve Predictive Value difference
- False Discovery Rate difference
- Consistency
