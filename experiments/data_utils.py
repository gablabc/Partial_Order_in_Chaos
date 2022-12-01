""" 
Utility functions for features and datasets
"""
import numpy as np
import pandas as pd
# Sklearn
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle
import os, sys

# Add path to uxai
sys.path.append(os.path.join("../"))
from uxai.features import Features


data_dir = "../../datasets/"

def get_data_bike():
    df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "datasets", "Bike-Sharing/hour.csv")
    )
    df.drop(columns=["dteday", "casual", "registered", "instant"], inplace=True)

    # Remove correlated features
    df.drop(columns=["atemp", "season"], inplace=True)

    # Rescale temp to Celcius
    df["temp"] = 41 * df["temp"]

    # Month count starts at 0
    df["mnth"] -= 1

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42)

    # Scale all features
    feature_names = list(df.columns[:-1])

    X = df.to_numpy()[:, :-1]
    y = df.to_numpy()[:, [-1]]

    # Generate Features object
    feature_types = [
        ["ordinal", "2011", "2012"],
        ["ordinal",
            "January", "February", "March", "April", "May", "June", "July",
            "August", "September", "October","November", "December",
        ],
        "num_int",
        "bool",
        ["ordinal",
         "Sunday", "Monday", "Thuesday", "Wednesday", "Thursday",
         "Friday", "Saturday"],
        "bool",
        "num_int",
        "num",
        "num",
        "num",
    ]

    features = Features(X, feature_names, feature_types)

    return X, y, features



def get_data_houses():
    # https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data?select=train.csv
    df = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__), "datasets", "kaggle_houses", "train.csv"
        )
    )

    # dropping categorical features
    df.drop(
        labels=[
            "Id", "MSSubClass", "MSZoning", "Street", "Alley", "LotShape",
            "LandContour", "Utilities", "LotConfig", "LandSlope", "Neighborhood",
            "Condition1", "Condition2", "BldgType", "HouseStyle", "MSZoning",
            "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType",
            "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond",
            "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC",
            "Electrical", "KitchenQual", "Functional", "FireplaceQu", "GarageType",
            "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence",
            "MiscFeature", "SaleType", "SaleCondition", "CentralAir","PavedDrive"
        ],
        axis=1,
        inplace=True,
    )

    # shuffle the data
    df = df.sample(frac=1, random_state=42)

    # Replace missing values by the mean
    imp = SimpleImputer(missing_values=np.nan, strategy="mean")

    df["MasVnrArea"] = imp.fit_transform(df[["MasVnrArea"]])
    df["GarageYrBlt"] = imp.fit_transform(df[["GarageYrBlt"]])

    # Dropping LotFrontage because it is missing 259/1460 values which is a lot (GarageYrBlt: 81 and MasVnrArea: 8 is reasonable)
    df.drop(labels=["LotFrontage"], axis=1, inplace=True)

    # High correlation
    df.drop(labels=["GarageCars"], axis=1, inplace=True)

    # Dropping GarageYrBlt because it is highly correlated (0.84791) with YearBuilt.
    df.drop(columns=["GarageYrBlt"], inplace=True)


    # Add the ratio of completion of the basement as a feature
    assert (df["TotalBsmtSF"] == df["BsmtUnfSF"] + df["BsmtFinSF1"] + df["BsmtFinSF2"]).all()
    has_basement = df["TotalBsmtSF"]>0
    df.insert(0, "BsmtPercFin", (has_basement).astype(int))
    df.loc[has_basement, "BsmtPercFin"] = 1 - df.loc[has_basement, "BsmtUnfSF"]/df.loc[has_basement, "TotalBsmtSF"]
    # Drop other features that involve the basement
    df.drop(columns=["TotalBsmtSF"], inplace=True)
    df.drop(columns=["BsmtUnfSF"], inplace=True)
    df.drop(columns=["BsmtFinSF1"], inplace=True)
    df.drop(columns=["BsmtFinSF2"], inplace=True)

    # High correlation >0.6 with BsmtPercFin
    df.drop(columns=["BsmtFullBath"], inplace=True)

    # GrLivArea: Above grade (ground) living area square feet and TotRmsAbvGrd: Total rooms above
    # grade (does not include bathrooms) are highly correlated (0.827874)
    df.drop(columns=["TotRmsAbvGrd"], inplace=True)

    # All features under 0.1: BsmtFinSF2, LowQualFinSF, BsmtHalfBath, 3SsnPorch, PoolArea, MiscVal, MoSold, YrSold
    # Almost no values == (almost all values have a value of 0)

    # BsmtHalfBath has a really low correlation (2nd lowest) with the target (-0.0121889), almost no values
    df.drop(columns=["BsmtHalfBath"], inplace=True)

    # MoSold low correlation with target -0.0298991, I
    # removed year because it was the lowest correlation of all features with the target
    df.drop(columns=["MoSold"], inplace=True)

    # LowQualFinSF: Low quality finished square feet (all floors), almost no values
    df.drop(columns=["LowQualFinSF"], inplace=True)

    # MiscVal almost no values
    df.drop(columns=["MiscVal"], inplace=True)

    # Pool area almost no values
    df.drop(columns=["PoolArea"], inplace=True)

    # 3SsnPorch almost no values
    df.drop(columns=["3SsnPorch"], inplace=True)

    # Low Model Reliance
    df.drop(columns=["FullBath"], inplace=True)
    df.drop(columns=["HalfBath"], inplace=True)

    # Too similar to first and second floor areas
    df.drop(columns=["GrLivArea"], inplace=True)

    # Solve the weird issue with YearBuild and YearRemodAdd
    bool_idx = df["YearRemodAdd"]==1950
    df.loc[bool_idx, "YearRemodAdd"] = df.loc[bool_idx, "YearBuilt"]
    df.drop(columns=["YearBuilt"], inplace=True)

    # In the end remove any feature that represent time
    df.drop(columns=["YearRemodAdd", "YrSold"], inplace=True)

    # Remove outliers
    df = df[df["SalePrice"] < 500000]
    df = df[df["SalePrice"] > 50000]

    # Rename a feature
    feature_names = list(df.columns[:-1])
    #feature_names[4] = "YearRenovation"

    X = df.to_numpy()[:, :-1]
    y = df.to_numpy()[:, [-1]]

    # Generate Features object
    feature_types = [
        "percent",
        "sparse_num",
        "num_int",
        "num_int",
        "sparse_num",
        "sparse_num",
        "sparse_num",
        "num_int",
        "num_int",
        "num_int",
        "sparse_num",
        "sparse_num",
        "sparse_num",
        "sparse_num",
        "sparse_num",
    ]

    features = Features(X, feature_names, feature_types)

    return X, y, features



def get_data_adults():

    # load train
    raw_data_1 = np.genfromtxt(os.path.join(os.path.dirname(__file__), 'datasets', 
                                            'Adult-Income','adult.data'), 
                                                     delimiter=', ', dtype=str)
    # load test
    raw_data_2 = np.genfromtxt(os.path.join(os.path.dirname(__file__), 'datasets', 
                                            'Adult-Income','adult.test'),
                                      delimiter=', ', dtype=str, skip_header=1)

    feature_names = ['age', 'workclass', 'fnlwgt', 'education',
                     'educational-num', 'marital-status', 'occupation', 
                     'relationship', 'race', 'gender', 'capital-gain', 
                     'capital-loss', 'hours-per-week', 'native-country', 'income']

    # Shuffle train/test
    df = pd.DataFrame(np.vstack((raw_data_1, raw_data_2)), columns=feature_names)


    # For more details on how the below transformations 
    df = df.astype({"age": np.int64, "educational-num": np.int64, 
                    "hours-per-week": np.int64, "capital-gain": np.int64, 
                    "capital-loss": np.int64 })

    # Reduce number of categories
    df = df.replace({'workclass': {'Without-pay': 'Other/Unknown', 
                                   'Never-worked': 'Other/Unknown'}})
    df = df.replace({'workclass': {'?': 'Other/Unknown'}})
    df = df.replace({'workclass': {'Federal-gov': 'Government', 
                                   'State-gov': 'Government', 'Local-gov':'Government'}})
    df = df.replace({'workclass': {'Self-emp-not-inc': 'Self-Employed', 
                                   'Self-emp-inc': 'Self-Employed'}})

    df = df.replace({'occupation': {'Adm-clerical': 'White-Collar', 
                                    'Craft-repair': 'Blue-Collar',
                                    'Exec-managerial':'White-Collar',
                                    'Farming-fishing':'Blue-Collar',
                                    'Handlers-cleaners':'Blue-Collar',
                                    'Machine-op-inspct':'Blue-Collar',
                                    'Other-service':'Service',
                                    'Priv-house-serv':'Service',
                                    'Prof-specialty':'Professional',
                                    'Protective-serv':'Service',
                                    'Tech-support':'Service',
                                    'Transport-moving':'Blue-Collar',
                                    'Unknown':'Other/Unknown',
                                    'Armed-Forces':'Other/Unknown',
                                    '?':'Other/Unknown'}})

    df = df.replace({'marital-status': {'Married-civ-spouse': 'Married', 
                                        'Married-AF-spouse': 'Married', 
                                        'Married-spouse-absent':'Married',
                                        'Never-married':'Single'}})

    df = df.replace({'income': {'<=50K': 0, '<=50K.': 0,  '>50K': 1, '>50K.': 1}})

    df = df.replace({'education': {'Assoc-voc': 'Assoc', 'Assoc-acdm': 'Assoc',
                                   '11th':'School', '10th':'School', 
                                   '7th-8th':'School', '9th':'School',
                                   '12th':'School', '5th-6th':'School', 
                                   '1st-4th':'School', 'Preschool':'School'}})

    # Put numeric+ordinal before nominal and remove fnlwgt-country
    df = df[['age', 'educational-num', 'capital-gain', 'capital-loss',
             'hours-per-week', 'gender', 'workclass','education', 'marital-status', 
             'occupation', 'relationship', 'race', 'income']]


    # df = df.rename(columns={'educational-num': 'educational_num',
    #                         'marital-status': 'marital_status', 
    #                         'hours-per-week': 'hours_per_week', 
    #                         'capital-gain': 'capital_gain', 
    #                         'capital-loss': 'capital_loss'})

    df = shuffle(df, random_state=42)
    feature_names = df.columns[:-1]
    
    # Make a column transformer for ordinal encoder
    encoder = ColumnTransformer(transformers=
                      [('identity', FunctionTransformer(), df.columns[:5]),
                       ('encoder', OrdinalEncoder(), df.columns[5:-1])
                      ])
    X = encoder.fit_transform(df.iloc[:, :-1])
    y = df["income"].to_numpy().reshape((-1, 1))
    
    # Generate Features object
    feature_types = ["num", "num", "sparse_num", "sparse_num", "num", ["ordinal", "Female", "Male"]]+\
        [(["nominal"] + list(l)) for l in encoder.transformers_[1][1].categories_[1:]]
    
    features = Features(X, feature_names, feature_types)
    
    return X, y, features
    

# Mappings for the different datasets to customize the code

DATASET_MAPPING = {
    "bike": get_data_bike,
    "kaggle_houses" : get_data_houses,
    "adult_income" : get_data_adults
}

THRESHOLDS_MAPPING = {
    "kaggle_houses": 0.95*28090,   # Random Forest 28090  (22500 also possible)
    "bike": 0.95*46.3,             # Random Forest 46.3
    "adult_income": 100-0.95*15.11 # Logistic Regression 15.11 error -> 84.89 accuracy
}


TASK_MAPPING = {
    "kaggle_houses": "regression",
    "bike": "regression",
    "adult_income": "classification"
}