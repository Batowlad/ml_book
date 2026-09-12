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
print(f"Standard Correlation Coefficient:\n{corr_matrix["median_house_value"].sort_values(ascending=False)}")


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
print(f"Imputer statistics: {imputer.statistics_}") # where the computed median is stored

# There's more powerful imputers instead of SimpleImputer like KNNInputer - replaces each missing value with the mean of k nearest neighbours
# Also, IterativeImputer - trains a regression model per feature to predict the missing values based on all the other available features

######## RECOVERING FROM NUMPY ARRAY TO PANDAS DATA FRAME ########
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

######## HANDLING TEXT AND CATEGORICAL ATTRIBUTES ########
from sklearn.preprocessing import OrdinalEncoder

housing_cat = housing[["ocean_proximity"]]
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

############ RBF ##############
# (used when a feature has multimodal distributation (mode - peak)) You add a feature for each of the modes
# representing the similarity between the housing median age and that particular mode.
# gaussian rbf - output value decays exponentially as the input value moves away from the fixed point.
#                The equation for rbf of similarity x and 35 would be: exp(-y(x-35)^2); y - gamma
from sklearn.metrics.pairwise import rbf_kernel

age_simil = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)

######### TARGET VALUE TRANSFORMATION #########
from sklearn.linear_model import LinearRegression

target_scaler = StandardScaler()
scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())

model = LinearRegression()
model.fit(housing[["median_income"]], scaled_labels)
new_data = housing[["median_income"]].iloc[:5] #pretending this is new data

scaled_predictions = model.predict(new_data)
predictions = target_scaler.inverse_transform(scaled_predictions)

#A better way to do this is to use TransformedTargetRegressor - it automatically uses transformer to scale the labels and train the regression model
from sklearn.compose import TransformedTargetRegressor

model = TransformedTargetRegressor(regressor=LinearRegression(), transformer=StandardScaler())
model.fit(housing[["median_income"]], housing_labels)
predictions = model.predict(new_data)

# print(housing_labels)

########## CUSTOM TRANSFORMERS ##########
from sklearn.preprocessing import FunctionTransformer

log_transformer = FunctionTransformer(numpy.log)
log_pop = log_transformer.transform(housing[["population"]])

#Another example
sf_coords = 37.7749, -122.41
sf_transformer = FunctionTransformer(rbf_kernel, kw_args=dict(Y=[sf_coords], gamma=0.1))

sf_simil = sf_transformer.transform(housing[["latitude", "longitude"]])


######### FULLY BUILDING A CUSTOM TRANSOFRMER ##########
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True): #with_mean - determines whether to center the data by subtracting the mean
        self.with_mean = self.with_mean

    def fit(self, X, y=None): #                 y is required even without using, X is data
        X = check_array(X) #                    checks that X is an array with finite float values
        self.mean_ = X.mean(axis=0) #           axis=0 - goes down (by column); axis=1 - goes right (by row)
        self.scale_ = X.std(axis=0) #           std - calculates standard deviation
        self.n_features_in_ = X.shape[1] #      every estimator stores this in fit()
        return self #                           ALWAYS RETURN self

    def transform(self, X):
        check_is_fitted(self) #looks for learned attributes (with trailing)
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1]
        if self.with_mean:
            X = X - self.mean_

        return X / self.scale_


# A custom transformer that uses other estimators
from sklearn.cluster import KMeans

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)

        return self #always return self in fit

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

######## TRANSFORMATION PIPELINE ##########
from sklearn.pipeline import Pipeline

num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("standardize", StandardScaler())
])
#NOTE: Your last arg determines what method it will call if you pass a transformer it will sequentually apply the transformer method. 
#      If the last estimator were a predictor calling it would sequentuallhy apply all the transformations and pass result to predictor's predict()

# You can also do it without naming transformers just by passing their classes
from sklearn.pipeline import make_pipeline

num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

#NOTE: If you call the fit() method on a pipeline it calls fit_transform()

housing_num_prepared = num_pipeline.fit_transform(housing_num)
df_housing_num_prepared = pd.DataFrame(housing_num_prepared, columns=num_pipeline.get_feature_names_out(), index=housing_num.index) #If you want the output to be a nice DataFrame

#NOTE: Pipelines also support indexing - if you call pipeline[1] it will return the second estimator.
#      Or you can also call pipeline["simple_imputer"] and it wil return a SimpleImputer() estimator in the pipeline.



########## COLUMN TRANSFORMER ############
from sklearn.compose import ColumnTransformer

num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedroos", "population", "households", "median_income"]
cat_attribs = ["ocean_proximity"]

cat_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
    ])

# An automatic way to select all the features of a given type
from sklearn.compose import make_column_selector, make_column_transformer

preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=numpy.number)), 
    (cat_pipeline, make_column_selector(dtype_include=object))
    )

housing_prepared = preprocessing.fit_transform(housing)


############### A FULL PIPELINE BEFORE TRAINING (FINALLY) ################
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"] #feature names out

def ratio_pipeline():
    return make_pipeline(SimpleImputer(strategy="median"),
                         FunctionTransformer(column_ratio, feature_names_out=ratio_name),
                         StandardScaler())

log_pipeline = make_pipeline(SimpleImputer(strategy="median"), FunctionTransformer(numpy.log, feature_names_out="one-to-one"), StandardScaler())

cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

preprocessing = ColumnTransformer([
    ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]), 
    ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]), 
    ("people_per_house", ratio_pipeline(), ["population", "households"]), 
    ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population", "households", "median_income"]), 
    ("geo", cluster_simil, ["latitude", "longitude"]), 
    ("cat", cat_pipeline, make_column_selector(dtype_include=object))], 
    remainder=default_num_pipeline)

###################################
######### ACTUAL ML ALGO ##########
###################################
from sklearn.linear_model import LinearRegression

lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)

housing_predictions = lin_reg.predict(housing)
print(housing_predictions)

######### MEASURING MODEL'S RMSE ##########
from sklearn.metrics import root_mean_squared_error
lin_rmse = root_mean_squared_error(housing_labels, housing_predictions) 
print(lin_rmse)

######### USING A STRONGER MODEL ##########
from sklearn.tree import DecisionTreeRegressor

tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)

housing_predictions = tree_reg.predict(housing)
print(housing_predictions)

tree_rmse = root_mean_squared_error(housing_labels, housing_predictions)
print(tree_rmse) # outputs: 0 - badly overfits the data, we need a better evaluator

########### A BETTER EVALUATOR ############
# We are going to be using Cross-validation

