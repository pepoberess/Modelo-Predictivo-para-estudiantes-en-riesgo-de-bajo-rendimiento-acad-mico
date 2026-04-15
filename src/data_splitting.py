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

def temporal_split(data, random_state=42):
    np.random.seed(random_state)

    train = data[data["semestre"] < 4050].reset_index(drop=True)
    val = data[data["semestre"] >= 4050].reset_index(drop=True)

    return train, val