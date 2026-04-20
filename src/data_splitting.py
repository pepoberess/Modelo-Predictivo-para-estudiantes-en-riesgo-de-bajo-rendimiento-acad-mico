import numpy as np

def train_val_split_by_school(data, test_size=0.2, random_state=42):
    
    np.random.seed(random_state)
    
    train_data = []
    val_data = []
    
    for escuela in data["escuela"].unique():
        
        values_school = data[data["escuela"] == escuela].index.values
        np.random.shuffle(values_school)
        
        split = int(len(values_school) * (1 - test_size))
        
        train_data.extend(values_school[:split])
        val_data.extend(values_school[split:])
    
    train = data.loc[train_data].reset_index(drop=True)
    val = data.loc[val_data].reset_index(drop=True)
    
    return train, val

def random_split(data, test_size=0.2, random_state=42):
    
    np.random.seed(random_state)
    
    train_idx = []
    val_idx = []
    
    for label in data["target_b"].unique():
        
        idx = data[data["target_b"] == label].index.values
        np.random.shuffle(idx)
        
        split = int(len(idx) * (1 - test_size))
        
        train_idx.extend(idx[:split])
        val_idx.extend(idx[split:])
    
    train = data.loc[train_idx].reset_index(drop=True)
    val = data.loc[val_idx].reset_index(drop=True)
    
    return train, val

def group_split_by_school(data, num_schools_val=2, random_state=42):
    
    np.random.seed(random_state)
    
    schools = data["escuela"].unique()
    np.random.shuffle(schools)
    
    val_idxs = schools[:num_schools_val]
    train_idxs = schools[num_schools_val:]
    
    train = data[data["escuela"].isin(train_idxs)].reset_index(drop=True)
    val = data[data["escuela"].isin(val_idxs)].reset_index(drop=True)
    
    return train, val

def Kfold_group(data):
    schools = data["escuela"].unique()
    splits_group = [(data[data["escuela"] != s].index.values, data[data["escuela"] == s].index.values)for s in schools]
    
    return splits_group

def temporal_split(data, random_state=42):
    np.random.seed(random_state)

    train = data[data["semestre"] < 4050].reset_index(drop=True)
    val = data[data["semestre"] >= 4050].reset_index(drop=True)

    return train, val

def cross_validate(data, lambdas, features, preprocess_fn, model_class, f1_fn, k=5, group=False, random_state=42):
    """
    For each lambda, runs k-fold CV (random or group by school) and returns the F1 score.
    All predictions are collected across folds before computing F1 (no averaging).
 
    Parameters
    ----------
    data           : full training DataFrame
    lambdas        : list of lambda values to evaluate
    features       : list of feature column names (passed to preprocess_fn)
    preprocess_fn  : function(train, val, features) -> (train, val)
    model_class    : LogisticRegression class
    f1_fn          : f1_score function
    k              : number of folds (ignored if group=True, uses one school per fold)
    group          : if True, uses GroupKFold by school; if False, uses random KFold
    random_state   : random seed
 
    Returns
    -------
    f1_scores : list of F1 scores, one per lambda
    best_lam   : lambda value with highest F1
    splits     : list of (train_idx, val_idx) tuples for the best lambda
    """
    np.random.seed(random_state)
 
    # Build fold index splits
    if group:
        schools = data["escuela"].unique()
        splits = [
            (data[data["escuela"] != s].index.values, data[data["escuela"] == s].index.values)
            for s in schools
        ]
    else:
        folds = [[] for _ in range(k)]
        for label in data["target_b"].unique():
            idx = data[data["target_b"] == label].index.values.copy()
            np.random.shuffle(idx)
            for i, chunk in enumerate(np.array_split(idx, k)):
                folds[i].extend(chunk)
        splits = [
            (sum([folds[j] for j in range(k) if j != i], []), folds[i])
            for i in range(k)
        ]
 
    f1_scores = []
 
    for lam in lambdas:
        all_y_true = []
        all_y_pred = []
 
        for train_idx, val_idx in splits:
            train_f = data.loc[train_idx].reset_index(drop=True)
            val_f   = data.loc[val_idx].reset_index(drop=True)
 
            train_f, val_f = preprocess_fn(train_f, val_f, features)
 
            X_train = train_f.drop(columns=["rendimiento", "target_b"]).values.astype(float)
            y_train = train_f["target_b"].values
            X_val   = val_f.drop(columns=["rendimiento", "target_b"]).values.astype(float)
            y_val   = val_f["target_b"].values
 
            model = model_class(X_train, y_train, L2=lam)
            model.fit(L2=True)
 
            y_pred = model.prediction(X_val)
            all_y_true.extend(y_val)
            all_y_pred.extend(y_pred)
 
        f1 = f1_fn(np.array(all_y_true), np.array(all_y_pred))
        f1_scores.append(f1)
        print(f"  λ={lam:.4f} -> F1={f1:.4f}")

    best_lam = lambdas[int(np.argmax(f1_scores))]

    return f1_scores, best_lam, splits

