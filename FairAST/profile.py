import numpy as np
import pandas as pd

class Profile():
  """
  Create a profile of the dataset, storing important information.

  ATTRIBUTES
  data : (DataPreparation object) stores the data and its properties
  measure : (str) specifying fairness measure to optimize for. Implemented for: 
    'SP' = statistical parity/demographic parity, calculated by the disparate impact ratio, 
    'EO' = equalized odds, equal TPR and FPR, calculated by the absolute average odds difference, 
    'TPR' = equality of opportunity, equal TPR, calculated by the true positive rate difference,
    'FPR' = equal FPR, calculated by the false positive rate difference,
    'C' = consistency
    'TNR' = equal TNR, calculated by the true negative rate difference,
    'FNR' = equal FNR, calculated by the false negative rate difference,
    'PPV' = equal PPV, calculated by the positive predictive value difference,
    'FDR' = equal FDR, calculated by the false discovery rate difference.
  technique : (list(str)) (optional), allowed types of bias mitigation techniques, default: list of all possible techniques ['PRE', 'IN', 'POST']

  METHODS
  label() : Store the type of label in the data.
  balance() : Store the balance of favorable over unfavorable labels
  max_nr_protected() : Store the number of protected attributes in the data.
  feat_type() : Store the type of features in the data.
  max_nr_priv() : Store the number of privileged groups in the data.
  max_nr_unpriv() : Store the number of unprivileged groups in the data.
  nr_feat_values() : Store the size of features/feature value combinations (large or small).
  create_profile() : Return the profile, a dictionary representing the profile of the data.
  """

  def __init__(self, data, measure, technique=['PRE', 'IN', 'POST']):
    """
    Initialize all necessary attributes. 

    PARAMETERS
    data : (DataPreparation object) stores the data and its properties
    measure : (str) specifying desired fairness measure to optimize for
    technique : (list(str)) (optional), specifies the allowed types of bias mitigation techniques to use (pre-, in-, or post-processing)
                        default: list of all possible techniques ['PRE', 'IN', 'POST']
    """
    self.data = data
    self.measure = measure
    self.technique = technique

  def label(self):
    """
    Store the kind of label in the data.
    """
    if self.data.df[self.data.label_name].nunique() == 2:
      label = 'binary'
    elif self.data.df[self.data.label_name].nunique() > 2:
      label = 'multi'
    else:
      label = 'single'
    self.label = label

  def balance(self):
    """
    Calculate the balance of favorable and unfavorable labels.
    """
    n_fav = self.data.df[self.data.label_name].value_counts()[1]
    n_unfav = self.data.df[self.data.label_name].value_counts()[0]
    ratio = n_fav/n_unfav
    if ratio <= 0.5 or ratio >= 2:
      self.balance = 'imbalanced'
    else:
      self.balance = 'balanced'
  
  def feat_type(self):
    """
    Store the kind of features in the data.
    """
    if len(self.data.categorical) == 0 and len(self.data.numerical) > 0:
      self.feat_type = 'numerical'
    elif len(self.data.numerical) == 0 and len(self.data.categorical) > 0:
      self.feat_type = 'categorical'
    else:
      self.feat_type = 'mix'

  def max_nr_prot(self):
    """
    Store the number of protected attributes in the data.
    """
    self.max_nr_prot = len(self.data.protected)

  def max_nr_priv(self):
    """
    Store the number of privileged groups in the data.
    """
    self.max_nr_priv = len(self.data.priv)

  def max_nr_unpriv(self):
    """
    Store the number of unprivileged groups in the data.
    """
    self.max_nr_unpriv = len(self.data.unpriv)

  def protected_type(self):
    """
    The type of protected attribute(s),
    either single-binary, single-multi, multiple-binary or multiple-multi
    """
    if len(self.data.protected) == 1:
      number = 'single'
    else:
      number = 'multiple'
    if all([self.data.df[i].nunique() == 2 for i in self.data.protected]):
      values = 'binary'
    else:
      values = 'multi'
    self.protected_type = number+'-'+values

  def nr_feat_values(self):
    """
    Store the size of features/feature value combinations (large or small).
      'small' if it is possible to create a dataframe based on the cartesian product of the features and feature values.
      'large' if it is not possible to take the cartesian product, because it will be too big.
    """    
    product = 1
    for i in self.data.df.nunique():
      product = product * i
    if product < 10000000:
      try:
        features = list(set(self.data.df.columns) - set(['stratify']))
        df = pd.DataFrame(index=range(product),columns=range(len(features)))
        values = [self.data.df[feature].unique().tolist() for feature in features]
        DXY_index = pd.MultiIndex.from_product(values,names=features)
      except:
        self.nr_feat_values = 'large'
      else:
        self.nr_feat_values = 'small'
    else:
      self.nr_feat_values = 'large'

  def create_profile(self):
    """
    Return the profile, a dictionary representing the profile of the data.
    """
    self.label()
    self.balance()
    self.feat_type()
    self.protected_type()
    self.max_nr_prot()
    self.max_nr_priv()
    self.max_nr_unpriv()
    self.nr_feat_values()

    profile = vars(self)
    profile.pop("data")
    return profile
