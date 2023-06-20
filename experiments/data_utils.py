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


def get_data_compas():
    # Process data
    
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 
                    "datasets", "COMPAS", "compas-scores-two-years.csv"))
    # Same preprocessing as done by ProPublica but we also only keep Caucasians and Blacks
    keep = (df["days_b_screening_arrest"] <= 30) &\
        (df["days_b_screening_arrest"] >= -30) &\
        (df["score_text"] != "nan") &\
        ((df["race"] == "Caucasian") | (df["race"] == "African-American")) 
    df = df[keep]

    # Binarize some features
    df.loc[:, 'sex_Male'] = (df['sex'] == 'Male').astype(int)
    df.loc[:, 'race_Black'] = (df['race'] == "African-American").astype(int)
    df.loc[:, 'c_charge_degree_F'] = (df['c_charge_degree'] == 'F').astype(int)

    # Features to keep
    features = ['sex_Male', 'race_Black', 'c_charge_degree_F',
                'priors_count', 'age', 'juv_fel_count', 'juv_misd_count']
    names = list(df['name'])
    X = df[features]
    # Rename some columns
    X = X.rename({"sex_Male" : "Sex", "race_Black" : "Race", "c_charge_degree_F" : "Charge", 
              "priors_count" : "Priors", "age" : "Age", "juv_fel_count" : "Juv_felonies", 
              "juv_fel_count" : "Juv_misds"})
    X = X.to_numpy().astype(np.float64)
    # New Features to keep
    features = ['Sex', 'Race', 'Charge', 'Priors', 'Age', 'JuvFelonies', 'JuvMisds']

    # Target
    y = df["decile_score"].to_numpy().astype(np.float64)

    # Generate Features object
    feature_types = [
        ["nominal", "Female", "Male"],
        ["nominal", "White", "Black"],
        ["nominal", "Misd", "Felony"],
        "num_int",
        "num_int",
        "num_int",
        "num_int"
    ]

    features = Features(X, features, feature_types)

    return X, y, features, names



def get_data_houses(remove_correlations=False, submission=False):
    # https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data?select=train.csv
    
    if submission:
        df = pd.read_csv(
            os.path.join(
                os.path.dirname(__file__), "datasets", "kaggle_houses", "test.csv"
            )
        )
    else:
        df = pd.read_csv(
            os.path.join(
                os.path.dirname(__file__), "datasets", "kaggle_houses", "train.csv"
            )
        )
    Id = df["Id"]

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

    #### Missing Data ####
    # Replace missing values by the median
    columns_with_nan = df.columns[np.where(df.isna().any())[0]]
    if columns_with_nan is not None:
        imp = SimpleImputer(missing_values=np.nan, strategy="mean")
        df[columns_with_nan] = imp.fit_transform(df[columns_with_nan])

    #### Features which are mostly zero ####
    # Dropping LotFrontage because it is missing 259/1460 values 
    # which is a lot (GarageYrBlt: 81 and MasVnrArea: 8 is reasonable)
    df.drop(labels=["LotFrontage"], axis=1, inplace=True)
    # 1408 houses have MiscVal=0
    df.drop(columns=["MiscVal"], inplace=True)
    # 1453 houses have Pool=0
    df.drop(columns=["PoolArea"], inplace=True)
    # 1436 houses have 3SsnPorch=0
    df.drop(columns=["3SsnPorch"], inplace=True)
    # 1420 houses have LowQualFinSF=0
    df.drop(columns=["LowQualFinSF"], inplace=True)

    ### Ignore HalfBathrooms ####
    df.drop(columns=["BsmtHalfBath", "HalfBath"], inplace=True)

    #### Ignore time-related features ####
    # We mainly care about the PHYSICAL properties of the houses
    df.drop(columns=["YearRemodAdd", "YrSold", "YearBuilt"], inplace=True)
    df.drop(columns=["GarageYrBlt", "MoSold"], inplace=True)

    #### Multiple Features regarding the Basement are multi-colinear ####
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

    #### Almost Perfect multi-colinearity GrLivArea=1st + 2nd floors ####
    assert np.isclose(df["GrLivArea"], df["1stFlrSF"] + df["2ndFlrSF"]).mean() > 0.95
    df.drop(columns=["GrLivArea"], inplace=True)
    
    # Remove correlated/redundant features
    if remove_correlations:
        #### High Spearman Correlation ####
        # High correlation of 0.85 with GarageArea
        df.drop(labels=["GarageCars"], axis=1, inplace=True)

        # High correlation >0.6 with BsmtPercFin
        df.drop(columns=["BsmtFullBath"], inplace=True)

        # High correlation >0.6 with BedroomAbvGrd
        df.drop(columns=["TotRmsAbvGr"], inplace=True)

        # High correlation ~0.6 with OverallQual
        df.drop(columns=["FullBath"], inplace=True)

    # # Solve the weird issue with YearBuild and YearRemodAdd
    # bool_idx = df["YearRemodAdd"]==1950
    # df.loc[bool_idx, "YearRemodAdd"] = df.loc[bool_idx, "YearBuilt"]
    # df.drop(columns=["YearBuilt"], inplace=True)

    # Process the target (careful here)
    # df = df[df["SalePrice"] < 500000]
    # df = df[df["SalePrice"] > 50000]
    # df['SalePrice'] = np.log1p(df['SalePrice'])

    # Determine the ordering of the features
    if remove_correlations:
        feature_names = \
            ['LotArea', 'OverallQual', 'OverallCond', 'MasVnrArea',
            'BsmtPercFin', '1stFlrSF', '2ndFlrSF',
            'BedroomAbvGr',  'KitchenAbvGr', 'Fireplaces', 'GarageArea',
            'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch']

        feature_types = [
            "sparse_num",
            "num_int",
            "num_int",
            "sparse_num",
            "percent",
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

    else:
        feature_names = \
            ['LotArea', 'OverallQual', 'OverallCond', 'MasVnrArea',
            'BsmtPercFin', '1stFlrSF', '2ndFlrSF', 'FullBath', 
            'BsmtFullBath', 'BedroomAbvGr', 'TotRmsAbvGr', 'KitchenAbvGr', 
            'Fireplaces', 'GarageArea', 'GarageCars',
            'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch']

        feature_types = [
            "sparse_num",
            "num_int",
            "num_int",
            "sparse_num",
            "percent",
            "sparse_num",
            "sparse_num",
            "num_int",
            "num_int",
            "num_int",
            "num_int",
            "num_int",
            "num_int",
            "sparse_num",
            "num_int",
            "sparse_num",
            "sparse_num",
            "sparse_num",
            "sparse_num",
        ]
    
    if submission:
        df = df[feature_names]
        X = df.to_numpy()
        y = None
    else:
        df = df[feature_names+['SalePrice']]
        X = df.to_numpy()[:, :-1]
        y = np.log(df.to_numpy()[:, [-1]])
    
    features = Features(X, feature_names, feature_types)
    return X, y, features, Id



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
    "kaggle_houses" : get_data_houses,
    "adult_income" : get_data_adults,
    "compas" : get_data_compas
}


TASK_MAPPING = {
    "kaggle_houses": "regression",
    "adult_income": "classification",
    "compas" : "regression"
}
