from pathlib import Path
import pandas as pd
import tarfile 
import urllib.request

import matplotlib.pyplot as plt

import numpy

from zlib import crc32


############# PULLING DATA FROM THE INTERNET ####################
def load_housing_data():
    tarball_path= Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets", filter="data")
    return pd.read_csv(Path("datasets/housing/housing.csv"))


########### SIMPLE WAY TO CREATE TRAIN AND TEST SETS ###############
def shuffle_and_split_data(data, test_ratio, rng):
    shuffled_indices = rng.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)

    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[test_indices], data.iloc[train_indices]


def is_id_in_test_set(identifier, test_ratio):
    return crc32(numpy.int64(identifier)) < test_ratio * 2**32


def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))

    return data.loc[~in_test_set], data.loc[in_test_set]


housing_full = load_housing_data()
# print(housing_full.head())
# print(housing_full.info())
# print(housing_full["ocean_proximity"].value_counts())

# housing_full.hist(bins=50, figsize=(12, 8))
# plt.show()

test_set, train_set = shuffle_and_split_data(housing_full, 0.2, numpy.random.default_rng())

housing_with_id = housing_full.reset_index()
# print(housing_with_id)

train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")

housing_with_id["id"] = (housing_with_id["longitude"] * 1000 + housing_with_id["latitude"])
# print(housing_with_id)

test_set, train_set = split_data_with_id_hash(housing_with_id, 0.2, "id")
# print(test_set)


from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing_full, test_size=0.2, random_state=42)
# print(test_set)

############ ADDING INCOME CATEGORY COLUMN ###################
housing_full["income_cat"] = pd.cut(housing_full["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., numpy.inf], labels=[1, 2, 3, 4, 5])
cat_counts = housing_full["income_cat"].value_counts().sort_index()
cat_counts.plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
# plt.show()


############ STRATIFIYING THE DATA SETS ##################
from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []
for train_index, test_index in splitter.split(housing_full, housing_full["income_cat"]):
    strat_train_set_n = housing_full.iloc[train_index]
    strat_test_set_n = housing_full.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])

strat_train_set, strat_test_set = train_test_split(housing_full, test_size=0.2, stratify=housing_full["income_cat"], random_state=42)
# print(strat_test_set)

########## DROPPING THE INCOME CATEGORY ##################
for set_ in (strat_test_set, strat_train_set):
    set_.drop("income_cat", axis=1, inplace=True) 

#####################################################
########## VISUALIZING GEOGRAPHICAL DATA ############
#####################################################
# housing_full.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
housing_full.plot(kind="scatter", x="longitude", y="latitude", grid=True, s=housing_full["population"]/100, label="population", c="median_house_value", cmap="jet", colorbar=True, sharex=False, figsize=(10, 7))
# plt.show()


######## STANDARD CORRELATION COEFFICIENT (PEARSON'S R) ########
corr_matrix = housing_full.corr(numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))


############ MOST CORRELATION ATTRIBUTE ############
housing_full.plot(kind="scatter", x="median_income", y="median_house_value", grid=True, alpha=0.1)
# plt.show()


###################################################################
############### PREPARING DATA FOR ML ALGORITHMS ##################
###################################################################
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

######## CLEANING THE DATA OF AN ATTRIBUTE #########
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median") # specifiying that we want to replace missing values with the median of that attribute
housing_num = housing.select_dtypes(include=[numpy.number]) # copy of the data with only numberical attributes (exclude ocean proximity stuff)
# imputer.fit(housing_num)
X = imputer.fit_transform(housing_num) # apply trained imputer to transform the set by replacing missing values with median values
print(imputer.statistics_) # where the computed median is stored

# There's more powerful imputers instead of SimpleImputer like KNNInputer - replaces each missing value with the mean of k nearest neighbours
# Also, IterativeImputer - trains a regression model per feature to predict the missing values based on all the other available features

######## RECOVERING FROM NUMPY ARRAY TO PANDAS DATA FRAME ########
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

######## HANDLING TEXT AND CATEGORICAL ATTRIBUTES ########
from sklearn.preprocessing import OrdinalEncoder

housing_cat = housing["ocean_proximity"]
ordinal_encoder = OrdinalEncoder()
housing_cat_econded = ordinal_encoder.fit_transform(housing_cat)

####### FIXING AN ISSUE WITH ENCODED VALUES THROUGH OneHotEncoder ########
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat) # outputs a sparse matrix


######## FEATURE SCALING AND TRANSFORMATION ########
# ML algorithms are bad when scaling is very different feature scaling helps with that. There are two ways fix this:

#1. Min-max scaling (Normalization) - Scaling all the values to be from -1 to 1 (subtracts the min value from all values, divides the results by the difference between min and max)
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)

#2. Standardization - Subtracts the mean value, divides the result by standard deviation
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)
