import pandas as pd
import numpy as np

def read_data(train, test):
    """Load train and test CSVs into DataFrames."""
    data_train = pd.read_csv(train)
    data_test = pd.read_csv(test)
    return data_train, data_test

def nulls_nota_previa(train, val):
    """
    Impute missing nota_previa values.
    
    The first semester (sem_prev=4045, i.e. 2022-1) has no prior grade by definition,
    so we impute using the median of sem_next (4046, i.e. 2022-2) per school in train.
    Any remaining NaNs (including schools not seen in train) are filled with the 
    global median of nota_previa from train.
    """
    sem_prev = 4045
    sem_next = 4046

    # Compute per-school medians from sem_next in train
    medians = train[train["semestre"] == sem_next].groupby("escuela")["nota_previa"].median()
    
    # Global fallback: median of all non-null nota_previa in train
    global_median = train["nota_previa"].median()

    for data in [train, val]:
        for escuela in data["escuela"].unique():
            value = medians.get(escuela)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                value = global_median

            mask = (
                (data["escuela"] == escuela) &
                (data["semestre"] == sem_prev) &
                (data["nota_previa"].isna())
            )
            data.loc[mask, "nota_previa"] = value

        # Final safety net: fill any remaining NaNs (other semesters, edge cases)
        data["nota_previa"] = data["nota_previa"].fillna(global_median)

    return train, val


def adjust_nulls(train, val, data_test=None):
    """
    Impute missing values for numeric features using per-school medians from train.
    Falls back to global train median for schools not present in train (e.g. group split).
    """
    features_nulls = ["horas_estudio", "horas_sueno", "participacion", "nivel_socioeconomico"]

    datasets = [train, val]
    if data_test is not None:
        datasets.append(data_test)

    for feature in features_nulls:
        # Compute per-school medians ONLY from train
        medians = train.groupby("escuela")[feature].median()
        # Global fallback for schools not in train
        global_median = train[feature].median()

        for data in datasets:
            for escuela in data["escuela"].unique():
                value = medians.get(escuela)
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    value = global_median

                mask = (data["escuela"] == escuela) & (data[feature].isna())
                data.loc[mask, feature] = value

            # Final safety net
            data[feature] = data[feature].fillna(global_median)

    if data_test is not None:
        return train, val, data_test
    else:
        return train, val


def one_hot_encoder_school(data1, data2, data3):
    """One-hot encode the 'escuela' column and drop the 'H' dummy."""
    datasets = []
    for data in [data1, data2, data3]:
        data = pd.get_dummies(data, columns=["escuela"])
        # The omitted category is implicitly represented when all one-hot encoded variables are equal to zero, acting as the baseline in the model.
        data = data.drop(columns=["escuela_H"])
        datasets.append(data)
    return datasets[0], datasets[1], datasets[2]

def one_hot_encoder2(train, val):
    # for excercise 2, we need to one-hot encode the school variable, but we want to keep all the dummies to be able to analyze the importance of each school in the model.
    train = pd.get_dummies(train, columns=["escuela"])
    val = pd.get_dummies(val, columns=["escuela"])
    
    val = val.reindex(columns=train.columns, fill_value=0)

    return train, val

def semestre_to_int(s):
    year, sem = s.split("-")
    return int(year) * 2 + int(sem)

def normalize_train(data, features):
    """Z-score normalize the given features and return the per-feature (mean, std) statistics."""
    statistics = {}
    for feature in features:
        mean = data[feature].mean()
        std = data[feature].std()
        data[feature] = (data[feature] - mean) / std
        statistics[feature] = (mean, std)
    return data, statistics

def normalize_val_test(data, features, statistics):
    """Apply train-set statistics to normalize test/validation features."""
    data = data.copy()
    for feature in features:
        mean, std = statistics[feature]
        data[feature] = (data[feature] - mean) / std
    return data