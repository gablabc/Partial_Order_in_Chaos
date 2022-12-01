""" 
Utility functions for features including mappers, scalers, and embbeders.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
import torch
from torch import nn


# Mappers take feature values and assing them a high level representation
# e.g. numerical -> (low, medium, high), categorical 3 -> "Married" etc.
# They are primarily used when visualising local explanations, where the 
# feature values used by the model may not be easily interpretable.


# Boolean feature
class bool_value_mapper(object):
    """ Organise feature values as 1->true or 0->false """

    def __init__(self):
        self.values = ["False", "True"]

    # map 0->False  1->True
    def __call__(self, x):
        return self.values[round(x)]


# Ordinal/Nominal encoding of categorical features
class cat_value_mapper(object):
    """ Organise categorical features  int_value->'string_value' """

    def __init__(self, categories_in_order):
        self.cats = categories_in_order

    # x takes values 0, 1, 2 ,3  return the category
    def __call__(self, x):
        return self.cats[round(x)]


# Numerical features x in [xmin, xmax]
class numerical_value_mapper(object):
    """ Organise feature values in quantiles  value->{low, medium, high}"""

    def __init__(self, num_feature_values):
        self.quantiles = np.quantile(num_feature_values, [0, 0.2, 0.4, 0.6, 0.8, 1])
        self.quantifiers = ["very small", "small", "medium", "large", "very large"]

    # map feature value to very small, small, medium, large, very large
    def __call__(self, x):
        return self.quantifiers[
            np.where((x <= self.quantiles[1:]) & (x >= self.quantiles[0:-1]))[0][0]
        ]


# Numerical features but with lots of zeros x in {0} U [xmin, xmax]
class sparse_numerical_value_mapper(object):
    """ Organise feature values in quantiles but treat 0-values differently
    """

    def __init__(self, num_feature_values):
        idx = np.where(num_feature_values != 0)[0]
        self.quantiles = np.quantile(
            num_feature_values[idx], [0, 0.2, 0.4, 0.6, 0.8, 1]
        )
        self.quantifiers = ["very small", "small", "medium", "large", "very large"]

    # map feature value to very small, small, medium, large, very large
    def __call__(self, x):
        if x == 0:
            return int(x)
        else:
            return self.quantifiers[
                np.where((x <= self.quantiles[1:]) & (x >= self.quantiles[0:-1]))[0][0]
            ]



class Features(object):
    """ Abstraction of the concept of a feature. Useful when doing
    feature importance plots """

    def __init__(self, X, feature_names, feature_types):
        self.names = feature_names
        self.types = []
        # Nominal categorical features that will need to be encoded
        self.nominal = []
        # map feature values to interpretable text
        self.maps = []
        for i, feature_type in enumerate(feature_types):
            # If its a list then the feature is categorical
            if type(feature_type) == list:
                self.types.append(feature_type[0]) # ordinal or nominal
                self.maps.append(cat_value_mapper(feature_type[1:]))
                if feature_type[0] == "nominal":
                    self.nominal.append(i)
            else:   
                self.types.append(feature_type)
                if feature_type == "num":
                    self.maps.append(numerical_value_mapper(X[:, i]))
                    
                elif feature_type == "sparse_num":
                    self.maps.append(sparse_numerical_value_mapper(X[:, i]))
                    
                elif feature_type == "bool":
                    self.maps.append(bool_value_mapper())
                    
                elif feature_type == "num_int":
                    self.maps.append(lambda x: round(x))
                elif feature_type == "percent":
                    self.maps.append(lambda x: f"{100*x:.0f}%")
                else:
                    raise ValueError("Wrong feature type")
                    
        # non_nominal refer to numerical+ordinal features i.e. all features that
        # are naturally represented with numbers
        self.non_nominal = list( set(range(len(feature_types))) - 
                                 set(self.nominal) )
                
    def map_values(self, x):
        """ Map values of x into interpretable text """
        return [f"{self.names[i]}={self.maps[i](x[i])}" for i in range(len(x))]

    def __len__(self):
        return len(self.names)



# Scalers allow to modify numerical variables so that their range is closer to
# 1 and their mean is near the origin. This is important for models such as
# linear regression and MLPs. These scalers always put numerical columns before
# nominal columns.

class PytorchScaler(nn.Module):

    def __init__(self, n_features, features_idx=None):
        super().__init__()
        self.n_features = n_features
        if features_idx == None:
            self.features_idx = list(range(self.n_features))
        else:
            self.features_idx = features_idx
        # Init parameters
        self.scale = nn.Parameter(torch.zeros((1, n_features)), requires_grad=False)
        self.loc = nn.Parameter(torch.zeros((1, n_features)), requires_grad=False)

    def to(self, device):
        if device == "Default":
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        super().to(self.device)
        return self

    def transform(self, X):
        if self.n_features == len(self.features_idx):
            return (X - self.loc) / self.scale
        else:
            X_tmp = X
            X_tmp[:, self.features_idx] =\
                    (X_tmp[:, self.features_idx] - self.loc) / self.scale
            return X_tmp

    def inv_transform(self, X_p):
        if self.n_features == len(self.features_idx):
            return self.scale * X_p + self.loc
        else:
            X_tmp = X_p
            X_tmp[:, self.features_idx] =\
                    self.scale * X_tmp[:, self.features_idx] + self.loc
            return X_tmp


class PytorchStandardScaler(PytorchScaler):

    def fit(self, X):
        self.loc = nn.Parameter(X[:, self.features_idx].mean(0, keepdims=True), requires_grad=False)
        self.scale = nn.Parameter(X[:, self.features_idx].std(0, keepdims=True), requires_grad=False)
        return self


class PytorchMinMaxScaler(PytorchScaler):

    def fit(self, X):
        self.loc = nn.Parameter(X[:, self.features_idx].min(0, keepdims=True)[0], requires_grad=False)
        self.scale = nn.Parameter(X[:, self.features_idx].max(0, keepdims=True)[0] - self.loc, requires_grad=False)
        return self



class PytorchOHE(object):
    
    def __init__(self):
        pass

    def fit(self, X, features):
        input_dim = len(features)
        self.n_nominal = len(features.nominal)
        assert self.n_nominal > 0

        # OHE embeddings
        self.embeddings = []
        for i, nominal_feature in enumerate(features.nominal):
            n_cats = len(features.maps[nominal_feature].cats)
            input_dim += n_cats - 1
            self.embeddings.append(torch.eye(n_cats))
        return input_dim

    def transform(self, X):
        # Split
        X_num = torch.as_tensor(X[:, :-self.n_nominal], dtype=torch.float32)
        X_cat = torch.as_tensor(X[:, -self.n_nominal:], dtype=torch.long)
        embedded_features = []
        # Embed categorial features
        for i, embedding in enumerate(self.embeddings):
            embedded_features.append(embedding[X_cat[:, i], :])
        return torch.cat([X_num] + embedded_features, dim=1)



class Scaler(object):
    """ Object used to scale numerical/ordinal features and regression target """

    def __init__(self, scaler_type, task, features):
        """Initialize the feature scaler and target scalers.

        Args:
            scaler_type (string): Scaler_type name (e.g. MinMax, Standard).
            task (string): Type of task (e.g. regression or classsification).
            features (Features object)

        """
        self.non_nominal = features.non_nominal
        self.nominal = features.nominal
        if task == "regression":
            if scaler_type == "MinMax":
                scaler_x = MinMaxScaler()
                self.scaler_y = MinMaxScaler()
                          
            elif scaler_type == "Standard":
                scaler_x = StandardScaler()
                self.scaler_y = StandardScaler()
        else:
            if scaler_type == "MinMax":
                scaler_x = MinMaxScaler()
                self.scaler_y = None
                          
            elif scaler_type == "Standard":
                scaler_x = StandardScaler()
                self.scaler_y = None
            
        self.scaler_x = ColumnTransformer([('num_scaler', scaler_x, 
                                             features.non_nominal)], 
                                             remainder='passthrough') 


    def fit(self, X, y):
        """ Fit the scalers

        Args:
            X (np.array/DataFrame/torch.Tensor): (N, d) Array of all X values.
            y (np.array/DataFrame/torch.Tensor): (N, 1) Array of all y values.

        """
        self.scaler_x.fit(X)
        if self.scaler_y:
            self.scaler_y.fit(y.reshape((-1, 1)))
             
            
    def transform(self, X, y):
        """ Transform the unprocessed data which is still stored as 
            Numpy arrays or DataFrames

        Args:
            X (np.array/DataFrame/torch.Tensor): (N, d) Array of all X values.
            y (np.array/DataFrame/torch.Tensor): (N, 1) Array of all y values.

        Returns:
            (np.array/DataFrame/torch.Tensor): Transformed X and y arrays.
        """
        x_s = self.scaler_x.transform(X)
        if self.scaler_y:
            y_s = self.scaler_y.transform(y.reshape((-1, 1)))
            if len(y.shape)==1:
                y_s= y_s.ravel() 
        else:
             y_s = y
        if type(X) == torch.Tensor:
            x_s = torch.tensor(x_s, dtype=X.dtype)
        if type(y) == torch.Tensor:
            y_s = torch.tensor(y_s, dtype=y.dtype)
        return x_s, y_s
    
    
    def fit_transform(self, X, y):
        """ Fit and transform the unprocessed data which is still stored as 
            Numpy arrays or DataFrames

        Args:
            X (np.array/DataFrame/torch.Tensor): (N, d) Array of all X values.
            y (np.array/DataFrame/torch.Tensor): (N, 1) Array of all y values.

        Returns:
            (np.array/DataFrame/torch.Tensor): Fit and transformed X and y arrays.
        """
        self.fit(X, y)
        return self.transform(X, y)
    
    
    def inverse_transform_x(self, X):
        """ Apply the inverse transform to the X

        Args:
            X (np.array/DataFrame/torch.Tensor): (N, d) Array of all X values.

        Returns:
            (np.array/DataFrame/torch.Tensor): Un-Transformed X array.
        """
        X_orig = self.scaler_x.transformers_[0][1].inverse_transform(X[:, 
                                                                self.non_nominal])
        X_orig = np.hstack((X_orig, X[:, self.nominal]))
        if type(X) == torch.Tensor:
            X_orig = torch.tensor(X_orig, dtype=X.dtype)
        
        return X_orig
    
    
    def inverse_transform_y(self, y):
        """ Apply the inverse transform to the target and go back to the original

        Args:
            y (np.array/DataFrame/torch.Tensor): (N, M) Array of all y values for M models.

        Returns:
            (np.array/DataFrame/torch.Tensor): Un-Transformed y array.
        """
        if self.scaler_y:
            y_orig = self.scaler_y.inverse_transform(y.reshape((-1, 1)))
            y_orig = y_orig.reshape(y.shape)
        else:
            y_orig = y
        if type(y) == torch.Tensor:
            y_orig = torch.tensor(y_orig, dtype=y.dtype)
        return y_orig
    
        
    def inverse_transform(self, X, y):
        """ Apply the inverse transform to the data to go back to the original

        Args:
            X (np.array/DataFrame/torch.Tensor): (N, d) Array of all X values.
            y (np.array/DataFrame/torch.Tensor): (N, 1) Array of all y values.

        Returns:
            (np.array/DataFrame/torch.Tensor): Un-Transformed X and y arrays.
        """
        return self.inverse_transform_x(X), self.inverse_transform_y(y)
    
    
    def invscale_target(self, normalised_array):
        """ Scale back a given quantity so it has the same units as the
        unormalised target. For instance, if Shapley values are computed
        on a MLP whose training dataset was normalised, then the Shapley
        values must be scaled by a multiplicative factor to have the original
        units ($, number of rented bikes, etc.)

        Args:
            normalised_array(np.array/torch.Tensor): Array whose scale
                must be changed to match the original target scale.

        Returns:
            Tensor: Scaled array with the same shape as the original.
        """
        if type(normalised_array) in [float, np.float32, np.float64, np.ndarray]:
            if type(self.scaler_y) == MinMaxScaler:
                return normalised_array / float(self.scaler_y.scale_[0])
            elif type(self.scaler_y) == StandardScaler:
                return normalised_array * float(self.scaler_y.scale_[0])
        elif type(normalised_array) == torch.Tensor:
            if type(self.scaler_y) == MinMaxScaler:
                return normalised_array.cpu() / self.scaler_y.scale_[0]
            elif type(self.scaler_y) == StandardScaler:
                return normalised_array.cpu() * self.scaler_y.scale_[0]
            
            

# Embedders allow to map categorical features to R^p. These include the well
# known one-hot encodding although entity embedding is also possible.

def embed(X, embeddings):
    """
    Map nominal features to a multi-dimensinal
    embedding.

    Args:
        X (np.array or torch.Tensor): (batch_size, p) Input whose last 
                columns are nominal features encoded with integers.
                
        embeddings (list): List containing the (n_categories, embbed_dim)
                arrays representing the embeddings of each categorial variable.

    Returns:
        torch.Tensor: (batch_size, augmented_features) Tensor of embbeded features.
    """
    n_nominal = len(embeddings)
    assert n_nominal > 0
    X_num = torch.as_tensor(X[:, :-n_nominal], dtype=torch.float32)
    X_cat = torch.as_tensor(X[:, -n_nominal:], dtype=torch.long)
    embedded_features = []
    # embed categorial features
    for i, embedding in enumerate(embeddings):
        embedded_features.append(embedding[X_cat[:, i], :])
    return torch.cat([X_num] + embedded_features, dim=1)
