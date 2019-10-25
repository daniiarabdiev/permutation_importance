import numpy as np 
from sklearn.metrics import roc_auc_score, accuracy_score


def score(model, X, y):
    y_pred = model.predict(X)
    return accuracy_score(y, y_pred)

def score_auc(model, X, y):
    y_pred_prob = [i[1] for i in model.predict_proba(X)]
    return roc_auc_score(y, y_pred_prob)

def shuffled_scores(X_orig, y, model, score_func):
    scores = []
    for col in X_orig.columns:
        X = X_orig.copy()
        X[col] = np.random.permutation(X[col])
        score = score_func(model, X, y)
        scores.append(score)
    return np.array(scores)

def permutation_feature_importance(X, y, model, score_func, n_iter):
    base_score = score_func(model, X, y)
    score_decreases = []
    for i in range(n_iter):
        scores = shuffled_scores(X, y, model, score_func)
        score_decreases.append(-scores + base_score)
    return base_score, score_decreases

def get_permutation_importance(X, y, model, score_func):
    '''
    Call this function to get permutation feature importances.
    X - pandas dataframe with all the features used to train the model
    y - pandas dataframe or series object 
    model - Model object used for prediction. Any model objects with predict or .predict_proba methods are acceptable. 
    score_func - function to score the model. AUC and ACCURACY are available above
    '''
    
    base_score, score_decreases = permutation_feature_importance(X, y, model, score_func, 5)
    permutation_feature_importances = np.mean(score_decreases, axis=0)

    print(permutation_feature_importances)

    return permutation_feature_importances

# get_permutation_importance(df_features, y, score_func)
