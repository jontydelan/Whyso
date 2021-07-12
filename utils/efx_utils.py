from collections import defaultdict
from typing import Dict, List
from sklearn.model_selection import train_test_split
import pickle, os


LOC = r"C:\Users\M1049231\Dev\Equifax\artifacts"

def make_split(X,y,FINAL_FEAT_v2):
    X = X[FINAL_FEAT_v2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)


def _save(df, prefix:str, loc = LOC):
    with open(os.path.join(loc, prefix + ".pkl"), 'wb') as f:
        pickle.dump(df , f)


def _load(prefix:str, loc = LOC):
    """make sure file name has the necessary extensions eg:'.pkl' """
    
    with open(os.path.join(loc, prefix), 'rb') as f:
        _ = pickle.load(f)
    return _

def _load_all(files:List = None, loct = LOC)->Dict:
    if files:
        files = [file+".pkl" for file in files]
        print("other files : ",os.listdir(loct))
    else:
        files = [file for file in os.listdir(loct) if ".pkl" in file]

    print("files being loaded are : ", files)

    res = dict()

    for file in files:
        res[file.replace(".pkl","")] = _load(file, loc = loct)
    
    return res
