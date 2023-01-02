import pandas as pd
import numpy as np
from joblib import dump, load
# from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score, confusion_matrix, precision_score, auc, recall_score


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds=5):
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    fold_size = int(dataset.shape[0]/n_folds)
    folds = list()
    start = 0
    end = start+fold_size
    while(end<dataset.shape[0]):
        folds.append(dataset[start:end])
        start = end+1
        end = start+fold_size
    folds.append(dataset[start:dataset.shape[0]])
    return folds

def cross_validation_score(dataset, n_folds, pipeline, param_grid):
    folds = cross_validation_split(dataset, n_folds)
    auc_scores = []
    pipelines = []
    for param_value in param_grid['C']:
        pipeline.set_params(logistic_clf__C = param_value)
        cum_score = 0
        for n in range(len(folds)):
            test_df = folds[n]
            train_df = pd.concat(folds[0:n] + folds[n+1:], axis=0)
            pipeline.fit(train_df, train_df.event_note)
            cum_score += score(test_df, pipeline)
        print(" C:{}  avg auc score:{}".format(pipeline.get_params()['logistic_clf'].C, cum_score/len(folds)))
        auc_scores.append(cum_score/len(folds))
        pipelines.append(pipeline)
    return auc_scores, pipelines


def model_diag(df, predictions, classes): # classes = encoder.classes_   |   pipeline.classes_
    df = df.reset_index(drop=True)
    ranks = pd.DataFrame(predictions, columns=classes)

    ranks['rank']= ranks.loc[:,['Add to picklist', 'Updated the on hands quantity for the item']].sum(axis=1)

    # 'Location updated for the item', 'New price print sign has been printed'
    df['act_bool']= df.event_note.isin(['Add to picklist', 'Updated the on hands quantity for the item'])*1
    df['rank'] = ranks['rank']
    #results_df = df.loc[:,['run_date','club_nbr','OLD_NBR','rank','event_note']]

    auc_score=roc_auc_score(df['act_bool'], df['rank'])

    print("AUC under ROC Curve:\n", auc_score)

    return df
