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

