import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from aif360.metrics import utils

class DataPreparation():
  """
  Prepare the data for profiling and running the bias mitigation algorithms.

  ATTRIBUTES
    df : (pandas DataFrame) containing the data
    protected : (list(str)) specifying the column names of all protected features
    label : (str) specifying the label column
    priv : (list(dicts)) representation of the privileged groups
    unpriv : (list(dicts)) representation of the unprivileged groups
    fav : (str/int/..) value representing the favorable label
    unfav : (str/int/..) value representing the unfavorable label
    categorical : (list(str)) (optional) specifying column names of categorical features

  METHODS
    check_missing_values() : Check for missing values in the data. All rows containing missing values will be removed.
    binarize_labels() : Change label to binary representation where unfavorable label = 0, favorable label = 1.
    stratification() : Create groups based on the (un)privileged groups and the label, that can be used for stratification of the data.
    dtypes() : Store the data type (categorical or numerical) of each feature.
    normalization() : Perform normalization on the numerical features.
    make_numerical() : Turn all categorical features into numerical values.
    prepare() : Perform all preprocessing steps.
  """

  def __init__(self, df, protected, label, priv, unpriv, fav, unfav, categorical=[]):
    """
    Construct all necessary attributes for the data preparation.

    df : (pandas DataFrame) containing the data
    protected : (list(str)) specifying the column names of all protected features
    label_name : (str) specifying the label column
    priv : (list(dicts)) representation of the privileged groups
    unpriv : (list(dicts)) representation of the unprivileged groups
    fav : (str/int/..) value representing the favorable label
    unfav : (str/int/..) value representing the unfavorable label
    categorical : (list(str)) (optional) specifying column names of categorical features
    """
    self.df = df
    self.protected = protected
    self.label_name = label
    self.priv = priv
    self.unpriv = unpriv
    self.fav = fav
    self.unfav = unfav
    self.categorical = categorical

  def check_missing_values(self):
    """
    Check for missing values in the data. 
    All rows containing missing values will be removed.
    """
    missing = self.df.isna().sum().sum()
    if missing > 0:
      print("Missing values detected. All rows with missing values will be removed.")
      self.df = self.df.dropna() 
    else:
      pass

  def binarize_label(self):
    """
    Change label to binary representation where unfavorable label = 0, favorable label = 1.
    """
    #Check that label is binary
    assert self.df.nunique()[self.label_name] == 2, "The label should take exactly two different values."

    #Change unfavorable label to 0 and favorable label to 1
    self.df[self.label_name].replace([self.unfav, self.fav], [0, 1], inplace=True)

  
  def stratification(self):
    """
    Create groups based on the (un)privileged groups and the label, that can be used for stratification of the data.
    """
    #Find the privileged and unprivileged instances in the data
    priv_cond = utils.compute_boolean_conditioning_vector(
                            self.df[self.protected].to_numpy(),
                            self.protected,
                            condition=self.priv)
    unpriv_cond = utils.compute_boolean_conditioning_vector(
                            self.df[self.protected].to_numpy(),
                            self.protected,
                            condition=self.unpriv)
    #Create the column 'stratify', representing the different groups in the data
    #Groups are based on being privileged/unprivileged and the label (favorable/unfavorable)
    self.df['stratify'] = 0
    self.df['stratify'] = np.where(priv_cond, 1, self.df['stratify'])
    self.df['stratify'] = np.where(unpriv_cond, 2, self.df['stratify'])
    self.df['stratify'] = self.df['stratify'].astype(str) + '-' + self.df[self.label_name].astype(str)

  def dtypes(self):
    """
    Store the data type (categorical or numerical) of each feature.
    """
    self.numerical = []
    for n in self.df.columns:
      if n in self.categorical:
        pass
      elif self.df[n].nunique() == 2:
        self.categorical.append(n)
      elif self.df[n].dtypes in ['object', 'str', 'category']:
        self.categorical.append(n)
      else:
        self.numerical.append(n)

  def normalization(self):
    """
    Perform normalization on the numerical features.
    """
    norm = self.df.drop(self.categorical, axis=1)
    names = norm.columns
    scaler = MinMaxScaler()
    norm = scaler.fit_transform(norm)
    norm = pd.DataFrame(norm, columns=names)
    self.df = pd.concat((norm, self.df.drop(names, axis=1)), axis=1)

  def make_numerical(self):
    """
    Turn all categorical features into numerical values.
    """
    to_change = []
    for n in self.categorical:
      if self.df[n].dtypes != 'int64':
        to_change.append(n)

    encoders = {}
    for col in to_change:
      enc = OrdinalEncoder(dtype='int64')
      enc.fit(self.df.loc[:, [col]])
      self.df[col] = enc.transform(self.df.loc[:, [col]])
      encoders[col] = enc

    #Adjust the values in the priv/unpriv representations
    priv_df = pd.DataFrame(self.priv)
    unpriv_df = pd.DataFrame(self.unpriv)
    dicts = [priv_df, unpriv_df]
    for frame in dicts:
      for col in frame.columns:
        if col in encoders.keys():
          frame[col] = encoders[col].transform(frame.loc[:,[col]])
    self.priv = priv_df.to_dict('records')
    self.unpriv = unpriv_df.to_dict('records')
    
  def prepare(self):  
    """
    Perform all preprocessing steps.
    """
    self.check_missing_values()
    self.binarize_label()
    self.stratification()
    self.dtypes() 
    self.normalization()
    self.make_numerical()
    return self
