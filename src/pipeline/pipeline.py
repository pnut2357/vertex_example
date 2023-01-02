# NEED TO CHANGE TO [future]pipeline.py
import sys
import os
from pathlib import Path

# Imports for vertex pipeline
from google.cloud import aiplatform
import google_cloud_pipeline_components
from google_cloud_pipeline_components import aiplatform as gcc_aip
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
from kfp.v2 import compiler
from kfp.v2.dsl import (
    Artifact,
    component,
    Condition,
    pipeline,
    Input,
    Output,
    Model,
    Dataset,
    InputPath,
    OutputPath,
)
import kfp.components as comp
import kfp.dsl as dsl
from typing import NamedTuple

# import c_utils
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime

sys.path.append(str(Path(".").absolute().parent))
sys.path.append(str(Path(".").absolute().parent) + "/utils")
sys.path.append(str(Path(".").absolute().parent.parent))
sys.path.append(str(Path(".").absolute().parent.parent.parent))

import pipeline_utils
import argparse

try:
    args = pipeline_utils.get_args()
except:
    parser = argparse.ArgumentParser()
    parser.add_argument("--COMMIT_ID", required=True, type=str)
    parser.add_argument("--BRANCH", required=True, type=str)
    parser.add_argument("--is_prod", required=False, type=bool)
    sys.args = [
        "--COMMIT_ID", "1234",
        "--BRANCH", "dev",
        "--is_prod", False,
    ]
    args = parser.parse_args(sys.args)


PARAMS = pipeline_utils.yaml_import('settings.yml')

# Env flag for indentifying what env is used. valid values are: "dev" "stage" "prod"
BRANCH_ID = args.BRANCH
is_prod = args.is_prod

if BRANCH_ID == "stage" and is_prod == True:
    BRANCH_ID = "prod"
    
ENV = BRANCH_ID

PARAMS = pipeline_utils.yaml_import("settings.yml")

# Model Parameters

# MODE "test" --STAGE1_FLAG "train" --ENSEMBLE_FLAG "train" --RF_CLF_MODEL_PATH "" --LOGISTIC_CLF_MODEL_PATH "" --STAGE1_NN_MODEL_PATH "" --GNB_MODEL_PATH "" --STG1_FEATURE_SELECTOR_MODEL_PATH "" --NOSALES_MODEL_PATH ""
MODE = PARAMS["envs"][ENV]["MODE"] # "test" 
STAGE1_FLAG = PARAMS["envs"][ENV]["STAGE1_FLAG"] # "train" 
ENSEMBLE_FLAG = PARAMS["envs"][ENV]["ENSEMBLE_FLAG"] # "train" 
RF_CLF_MODEL_PATH = PARAMS["envs"][ENV]["RF_CLF_MODEL_PATH"] # "" 
LOGISTIC_CLF_MODEL_PATH = PARAMS["envs"][ENV]["LOGISTIC_CLF_MODEL_PATH"] # "" 
STAGE1_NN_MODEL_PATH = PARAMS["envs"][ENV]["STAGE1_NN_MODEL_PATH"] # "" 
GNB_MODEL_PATH = PARAMS["envs"][ENV]["GNB_MODEL_PATH"] # "" 
STG1_FEATURE_SELECTOR_MODEL_PATH = PARAMS["envs"][ENV]["STG1_FEATURE_SELECTOR_MODEL_PATH"] # ""
NOSALES_MODEL_PATH = PARAMS["envs"][ENV]["NOSALES_MODEL_PATH"] # ""


print("MODE:", MODE,"\nSTAGE1_FLAG:", STAGE1_FLAG,"\nENSEMBLE_FLAG:",ENSEMBLE_FLAG,"\nRF_CLF_MODEL_PATH:",RF_CLF_MODEL_PATH, "\nLOGISTIC_CLF_MODEL_PATH:",LOGISTIC_CLF_MODEL_PATH,"\nSTAGE1_NN_MODEL_PATH:", STAGE1_NN_MODEL_PATH, "\nGNB_MODEL_PATH:",
GNB_MODEL_PATH,"\nSTG1_FEATURE_SELECTOR_MODEL_PATH:", STG1_FEATURE_SELECTOR_MODEL_PATH,"\nNOSALES_MODEL_PATH:", NOSALES_MODEL_PATH)

NETWORK = PARAMS['envs'][ENV]['VPC_NETWORK']
# GCP Project id, service account, region, and docker images. 
PROJECT_ID = PARAMS['envs'][ENV]['PROJECT_ID']
REGION = PARAMS['envs'][ENV]['REGION']
BASE_IMAGE = PARAMS['envs'][ENV]['BASE_IMAGE']
# Training Pipeline.
PIPELINE_ROOT = PARAMS['envs'][ENV]['PIPELINE_ROOT']
PIPELINE_NAME = PARAMS['envs'][ENV]['PIPELINE_NAME']
PIPELINE_JSON = PARAMS['envs'][ENV]['PIPELINE_JSON']
TMP_PIPELINE_JSON = os.path.join("/tmp", PIPELINE_JSON)


TRAINING_TABLE_NAME = PARAMS['envs'][ENV]['TRAINING_TABLE_NAME']
TRAINING_DATA_BQ_QUERY = f'select * from {TRAINING_TABLE_NAME}' #f'select * from {TRAINING_TABLE_NAME}'  
# MLFlow image location and version, MLFlow name, and Model registry name as OYI No Sales model
MLFLOW_IMAGE = PARAMS['envs'][ENV]['MLFLOW_IMAGE']
MLFLOW_EXP_NAME = PARAMS['envs'][ENV]['MLFLOW_EXP_NAME']
MODEL_REGISTRY_NAME = PARAMS['envs'][ENV]['MODEL_REGISTRY_NAME']

#############
# TRANSITION_IMAGE = PARAMS['envs'][ENV]['TRANSITION_IMAGE']
# MODEL_PREFIX = PARAMS['envs'][ENV]['MODEL_PREFIX']
TRANSITION_IMAGE = "gcr.io/wmt-mlp-p-oyi-ds-or-oyi-dsns/version-transition-dev:latest"
#############

SERVICE_ACCOUNT = PARAMS['envs'][ENV]['SERVICE_ACCOUNT']

TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
 
# Matches on non-word, non-regular-punctuation characters.
MATCHER = r"""[^a-zA-Z0-9'"!@#$%\^&*()\[\]{}:;<>?,.-=_+ ]+""" 

CLUB_THRESH_PATH = PARAMS['envs'][ENV]['CLUB_THRESH_PATH']
LATEST_NOSALES_MODEL_PATH = PARAMS['envs'][ENV]['LATEST_NOSALES_MODEL_PATH']
LATEST_PIPELINE_PATH = PARAMS['envs'][ENV]['LATEST_PIPELINE_PATH']
RUN_PIPELINE = PARAMS['envs'][ENV]['RUN_PIPELINE']
print(f"ENV: {ENV}, \nPROJECT_ID: {PROJECT_ID}, \nBASE_IMAGE: {BASE_IMAGE}, \nMLFLOW_IMAGE: {MLFLOW_IMAGE}") # \nTRANSITION_IMAGE: {TRANSITION_IMAGE}")
print(f"\nPIPELINE_NAME: {PIPELINE_NAME}, \nPIPELINE_JSON: {PIPELINE_JSON}") #, \nMODEL_PREFIX: {MODEL_PREFIX}")
print(f"\nNETWORK: {NETWORK}")


# # Training Pipeline.
# RUN_PIPELINE = PARAMS['envs'][ENV]['RUN_PIPELINE']
# PIPELINE_ROOT = PARAMS['envs'][ENV]['PIPELINE_ROOT']
# PIPELINE_NAME = PARAMS['envs'][ENV]['PIPELINE_NAME']
# PIPELINE_JSON = PIPELINE_NAME + ".json" # PARAMS['envs'][ENV]['PIPELINE_JSON']
# TMP_PIPELINE_JSON = os.path.join("/tmp", PIPELINE_JSON)


@component(base_image=BASE_IMAGE)
def data_preprocessing(
    training_data_bq_query_input: str,
    matcher: str,
    project_id: str,
    env: str,
    pipeline_root: str,
    training_data_output: Output[Dataset]):
    
    import pandas as pd
    from datetime import timedelta
    import utils
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)
    data = client.query(training_data_bq_query_input).to_dataframe()
    nosales_data = data[
      (data.report_type!='C') &
      (data.display_ind == "Display") &
      (data.oh_qty>=0)]
    nosales_data["item_desc"] = nosales_data["item_desc"].str.replace(matcher, "", regex=True)
    nosales_data["run_date"] = pd.to_datetime(nosales_data["run_date"])
    max_date = nosales_data["run_date"].max()
    cutoff_date = (max_date - timedelta(days=182)).strftime('%Y-%m-%d')
    nosales_data = nosales_data[nosales_data.run_date > cutoff_date]
    
    nosales_data.replace("No Action Taken, already OFS", "No Action Taken, already out for sale", inplace=True)
    nosales_data.replace("Updated the NOSALES type with scrubber event", "No Action Taken, already out for sale", inplace=True)
    nosales_data.sort_values(by = ["run_date","club_nbr","item_nbr","event_ts"],inplace = True)
    nosales_data.drop_duplicates(["old_nbr","club_nbr","run_date"], keep="first",inplace = True)
    
    nosales_ext = utils.calculate_all_level_tpr(df=nosales_data, env=env, pipeline_root=pipeline_root, path='', save=True)
    nosales_ext.fillna(0, inplace=True)
    nosales_ext.to_csv(training_data_output.path, index=False)


@component(base_image=BASE_IMAGE)
def train_test_split(
    nosales_ext_input: Input[Dataset],
    nosales_train_ext_output: Output[Dataset],
    nosales_test_ext_output: Output[Dataset],
    nosales_train_usampled_output: Output[Dataset]

):
    import pandas as pd
    from datetime import timedelta
    
    nosales_ext = pd.read_csv(nosales_ext_input.path)
    nosales_ext['run_date'] = pd.to_datetime(nosales_ext['run_date'])
    split_date = (nosales_ext.run_date.max() - timedelta(days=50)).strftime('%Y-%m-%d')
    nosales_train_ext = nosales_ext[pd.to_datetime(nosales_ext.run_date) < split_date].copy() 
    nosales_test_ext  = nosales_ext[pd.to_datetime(nosales_ext.run_date) >= split_date].copy() 

    x=nosales_train_ext.shape[0]
    y=nosales_test_ext.shape[0]
    print(f"split_date is {split_date}.")
    print("Train/Test ratio:", x*100/(x+y))
    seed = 2019
    frac = 11
    grouped = nosales_train_ext[nosales_train_ext.event_note == "No Action Taken, already out for sale"].groupby('club_nbr')
    u1 = grouped.apply(lambda x: x.sample(n=int(x.shape[0]/frac),  random_state=seed)).reset_index(drop=True)

    u2 = nosales_train_ext[nosales_train_ext.event_note != "No Action Taken, already out for sale"]

    nosales_train_usampled = pd.concat([u1, u2])
    nosales_train_usampled = nosales_train_usampled.sample(frac=1)
    print(nosales_train_usampled.shape)
    nosales_train_usampled.event_note.value_counts()
    
    nosales_train_ext.to_csv(nosales_train_ext_output.path, index=False)
    nosales_test_ext.to_csv(nosales_test_ext_output.path, index=False)
    nosales_train_usampled.to_csv(nosales_train_usampled_output.path, index=False)


@component(base_image=BASE_IMAGE)
def train_eval_model(
    nosales_ext_input: Input[Dataset],
    nosales_train_ext_input: Input[Dataset],
    nosales_test_ext_input: Input[Dataset],
    nosales_train_usampled_input: Input[Dataset],
    mode: str,
    stage1_flag: str,
    ensemble_flag: str,
    rf_clf_model_path_input: str,
    logistic_clf_model_path_input: str,
    stage1_nn_model_path_input: str,
    gnb_model_path_input: str,
    stg1_feature_selector_model_path_input: str,
    nosales_model_path_input: str,
    latest_nosales_model_path_input: str,
    project_id: str,
    region: str,
    timestamp: str,
    rf_clf_model_output: Output[Model],
    logistic_clf_model_output: Output[Model],
    stage1_nn_model_output: Output[Model],
    gnb_model_output: Output[Model],
    stg1_feature_selector_model_output: Output[Model],
    nosales_model_output: Output[Model],
    nosales_test_ext_output: Output[Dataset]
) -> float:
    import os 
    import pandas as pd
    from sklearn.pipeline import Pipeline, make_pipeline
    import utils
    import diagnosis_utils
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from keras.wrappers.scikit_learn import KerasClassifier
    from sklearn.cluster import KMeans
    import pickle
    from google.cloud import storage, aiplatform
    
    nosales_ext = pd.read_csv(nosales_ext_input.path)
    nosales_train_ext = pd.read_csv(nosales_train_ext_input.path)
    nosales_test_ext = pd.read_csv(nosales_test_ext_input.path)
    nosales_train_usampled = pd.read_csv(nosales_train_usampled_input.path)
    
    nosales_ext['run_date'] = pd.to_datetime(nosales_ext['run_date'])
    nosales_train_ext['run_date'] = pd.to_datetime(nosales_train_ext['run_date'])
    nosales_test_ext['run_date'] = pd.to_datetime(nosales_test_ext['run_date'])
    nosales_train_usampled['run_date'] = pd.to_datetime(nosales_train_usampled['run_date'])
    
    tpr_features = [col for col in nosales_train_ext.columns if '_tpr' in col]  # len(tpr_features) : 45

    numerical_features= ['gap_days','exp_scn_in_nosale_period', 'unit_retail','oh_qty','avg_sales_interval']
    numerical_features.extend(tpr_features)
    categorical_features = ['club_nbr','state','cat']

    all_features = numerical_features + categorical_features
    target = ['event_note']

    top_features = list(['oh_qty_log',  'club_nbr_cat_update_loc_tpr_log',  'club_nbr_cat_new_price_sign_tpr_log',  'club_nbr_update_loc_tpr_log',
    'club_nbr_new_price_sign_tpr_log',  'club_nbr_cat_add_to_picklist_tpr_log',  'item_nbr_update_ohq_tpr_log',
    'item_nbr_add_to_picklist_tpr_log',  'club_nbr_add_to_picklist_tpr_log',  'avg_sales_interval_log', 
    'club_nbr_cat_no_action_taken_tpr_log',  'club_nbr_no_action_taken_tpr_log',  'item_nbr_no_action_taken_tpr_log',
    'cat_add_to_picklist_tpr_log',  'unit_retail_log',  'exp_scn_in_nosale_period_log',  'club_nbr_cat_update_ohq_tpr_log', 
    'cat_update_ohq_tpr_log',  'club_nbr_update_ohq_tpr_log',  'state_cat_add_to_picklist_tpr_log',  'reg_cat_update_ohq_tpr_log',
    'state_cat_new_price_sign_tpr_log',  'mkt_cat_new_price_sign_tpr_log',  'mkt_cat_update_ohq_tpr_log', 
    'reg_cat_add_to_picklist_tpr_log',  'state_cat_update_ohq_tpr_log',  'cat_new_price_sign_tpr_log', 
    'mkt_cat_update_loc_tpr_log',  'mkt_update_loc_tpr_log',  'mkt_new_price_sign_tpr_log', 
    'mkt_cat_add_to_picklist_tpr_log',  'mkt_no_action_taken_tpr_log',  'reg_no_action_taken_tpr_log', 
    'cat_no_action_taken_tpr_log',  'mkt_cat_no_action_taken_tpr_log',  'state_cat_update_loc_tpr_log', 
    'gap_days_log',  'reg_new_price_sign_tpr_log',  'mkt_update_ohq_tpr_log',  'state_cat_no_action_taken_tpr_log'])

    if mode == 'test':
        verbose_flag = True
    else:
        verbose_flag = False


    feature_flags = {'kmeans_clustering': False}

    class_weights = dict(nosales_train_usampled.event_note.value_counts()[0]/nosales_train_usampled.event_note.value_counts()[:])


    # pipeline: location-feat
    location_features_tf= Pipeline([
        ('select_loc', utils.DataFrameSelector(['sales_floor_location']))
    ])

    # pipeline: time-feat
    time_features_tf= Pipeline([
        ('select_rundate', utils.DataFrameSelector(['run_date'])),
        ('time_featurize', utils.TimeExtractor())
    ])


    # pipeline: other-catg-feat
    add_cat_tf= Pipeline([
        ('select_other_cat', utils.DataFrameSelector(['club_nbr','cat','state']))
    ])  


    # pipeline: K-means clustering
    kmeans_tf = make_pipeline(
        utils.DataFrameSelector(numerical_features),
        utils.MinMaxScalerTransformer(),
        utils.ModelTransformer(KMeans(2))
    )

    ######################################## Assembling 'Catg' n 'Numeric' Features  #####################################

    # list(catg pipelines)
    list_of_pipelines_for_catg_feat = [
        ('loc_features',location_features_tf),
        ('time_features',time_features_tf),
        ('other_cat_features', add_cat_tf)
    ]
    if feature_flags['kmeans_clustering']:
        list_of_pipelines_for_catg_feat.append(('clusters', kmeans_tf))


    # pipeline: encoding the catg features.
    cat_tf = Pipeline([
        ('combine_cats', utils.ColumnMerge(transformer_list=list_of_pipelines_for_catg_feat)),
        ('cat_featurize', utils.CategoryFeaturizer())
    ])


    # pipeline: numeric_features + log-transformation   
    num_features_tf= Pipeline([
        ('select_num', utils.DataFrameSelector(numerical_features)),
        ('log', utils.LogFeaturizer()),
        ('select_top_features', utils.DataFrameSelector(top_features))
    ])

    stage2_init_feature_num = 20
    num_features_tf2= Pipeline([
        ('select_num', utils.DataFrameSelector(numerical_features)),
        ('log', utils.LogFeaturizer()),
        ('select_top_features', utils.DataFrameSelector(top_features[:stage2_init_feature_num]))
    ])


    # all_feat => catg_feat + numerical_feat
    add_all_tf= utils.ColumnMerge([
        ('num_features',num_features_tf),
        ('cat_features',cat_tf)
    ])

    ############################################################## Final pipelines ##################################################################

    # Lone classifier-pipelines and pre-processors

    #1
    rf_clf = RandomForestClassifier(n_jobs=-1, criterion='gini',n_estimators=50, max_depth=7,max_features='sqrt',
                                    class_weight = class_weights )

    #2
    logistic_clf = LogisticRegression(n_jobs=-1, multi_class='multinomial', solver='lbfgs', max_iter=1000, penalty='l2', class_weight=class_weights)
    

    #3
    gnb = utils.CustomizedGaussianNB()

    #4
    stage1_nn = utils.Stage1_NeuralNetwork(num_classes=5, batch_size=128, epochs=25, verbose=verbose_flag)


    stage1_classifiers = {'rf_clf':rf_clf, 'logistic_clf':logistic_clf, 'stage1_nn':stage1_nn, 'gnb':gnb}

    stage2_nn_input_dimen = stage2_init_feature_num + len(stage1_classifiers)*5
    stage2_estimator = KerasClassifier(build_fn=utils.stage2_nn, input_dimen=stage2_nn_input_dimen, epochs=5, batch_size=128, verbose=verbose_flag)
    
    ##set flags when in mode: 'test'#####
    # True: if you want to save stage1 models during test. Will automatically set to False when in prod
    s1_save_flag = True

    # Stage 1 models
    ####################################
    # Force flag to be 'train' during prod
    if mode == "prod":
        s1_save_flag = False
        stage1_flag = "train"

    stg1_feature_selector = num_features_tf



    if stage1_flag == "train": 
        print("Training and saving models...")
        X_train = stg1_feature_selector.fit_transform(nosales_train_usampled, nosales_train_usampled.event_note)
        X_train = X_train.astype("float128")
        y_train = nosales_train_usampled.event_note
        if s1_save_flag:
            with open(stg1_feature_selector_model_output.path, "wb") as file:  
                pickle.dump(stg1_feature_selector, file)
    
        X_test= stg1_feature_selector.transform(nosales_test_ext)
        stage1_model_output_paths = {"rf_clf":rf_clf_model_output.path, "logistic_clf":logistic_clf_model_output.path,
                               "stage1_nn":stage1_nn_model_output.path, "gnb":gnb_model_output.path}
        for clf in stage1_classifiers:
            print(clf)

            model = stage1_classifiers[clf]
            # filename = clf + ".model"
            model.fit(X_train, y_train)

            print("\n")
            if s1_save_flag:
                save_path = stage1_model_output_paths[clf]
                with open(save_path, "wb") as file:  
                    pickle.dump(model, file)

    else:
        print("Loading models...")
        
        with open(rf_clf_model_path_input, "rb") as handler:
            rf_clf = pickle.load(handler)
       
        with open(logistic_clf_model_path_input, "rb") as handler:
            logistic_clf = pickle.load(handler)
        
        with open(stage1_nn_model_path_input, "rb") as handler:
            stage1_nn = pickle.load(handler)
        
        with open(gnb_model_path_input, "rb") as handler:
            gnb = pickle.load(handler)
       
        stage1_classifiers = {"rf_clf":rf_clf, "logistic_clf":logistic_clf, "stage1_nn":stage1_nn, "gnb":gnb}
        
        with open(stg1_feature_selector_model_path_input, "rb") as handler:
            stg1_feature_selector = pickle.load(handler)
        X_test= stg1_feature_selector.transform(nosales_test_ext)
        for clf in stage1_classifiers:
            print(clf)
            model = stage1_classifiers[clf]
            # To-Do: append the values in array; Check with Deepesh
            nosales_test_ext, current_auc_score_stage1 = diagnosis_utils.model_diag(nosales_test_ext, model.predict_proba(X_test), model.classes_)
            print("\n")
        
        rf_clf_model_output.path = rf_clf_model_path_input
        logistic_clf_model_output.path = logistic_clf_model_path_input
        stage1_nn_model_output.path = stage1_nn_model_path_input
        gnb_model_output.path = gnb_model_path_input
        stg1_feature_selector_model_output.path = stg1_feature_selector_model_path_input
        


    # ensemble model
    #################################################################### 
    if mode == "test":
        train_x = nosales_train_ext
        train_y = nosales_train_ext.event_note

    if mode == "prod":
        ensemble_flag = "train"
        train_x = nosales_ext
        train_y = nosales_ext.event_note


    print(mode, ensemble_flag, train_x.shape[0])  

    stg2_feture_selector = num_features_tf2

    if ensemble_flag == "train": 
        print("Training and saving ensemble...")
        stack_pipeline = Pipeline([
            ("ensemble_classifier", utils.EnsembleClassifier(stg1_feature_selector, list(stage1_classifiers.values()),
                                                     stg2_feture_selector, stage2_estimator)) ])
        stack_pipeline.fit(train_x, train_y)
        with open(nosales_model_output.path, "wb") as file:  
            pickle.dump(stack_pipeline, file)
        
        with open("latest_nosales_model_output", "wb") as file:  
            pickle.dump(stack_pipeline, file) 
        blob = storage.blob.Blob.from_string(latest_nosales_model_path_input, client=storage.Client())
        blob.upload_from_filename("latest_nosales_model_output")
        print("Saved the final model")
        
        if mode == "test":
            nosales_test_ext, current_auc_score_stack = diagnosis_utils.model_diag(nosales_test_ext, stack_pipeline.predict_proba(nosales_test_ext), stack_pipeline.classes_)
        

    else:
        print("Loading ensemble...")
        with open(nosales_model_path_input, "rb") as handler:
            stack_pipeline = pickle.load(handler)
        nosales_test_ext, current_auc_score_stack = diagnosis_utils.model_diag(nosales_test_ext, stack_pipeline.predict_proba(nosales_test_ext), stack_pipeline.classes_)
        
        nosales_model_output.path = nosales_model_path_input
        with open("latest_nosales_model_output", "wb") as file:  
            pickle.dump(stack_pipeline, file) 
        blob = storage.blob.Blob.from_string(latest_nosales_model_path_input, client=storage.Client())
        blob.upload_from_filename("latest_nosales_model_output")
       
        
    nosales_test_ext.to_csv(nosales_test_ext_output.path, index = False)
    
    # add the metric in blob storage
    blob = storage.Client().bucket(latest_nosales_model_path_input.split('/')[2]).blob("stack_auc") 
    with blob.open("w") as f:
        f.write(str(current_auc_score_stack))
    
    return current_auc_score_stack


@component(base_image=BASE_IMAGE)
def update_thresholds(
        nosales_test_ext_input: Input[Dataset],
        club_thresh_path_input: str,
        nosales_model_input: Input[Model],
        club_threshold_output: Output[Dataset]
    ):
    
    import utils
    import pandas as pd
    import pickle
    import os
    from google.cloud import storage
    from tempfile import TemporaryFile
    
    nosales_test_ext = pd.read_csv(nosales_test_ext_input.path)
    nosales_test_ext['run_date'] = pd.to_datetime(nosales_test_ext['run_date'])
    
    # if not os.path.exists(nosales_model_input.path):
    #     raise FileNotFoundError("feature file not found: {0}".format(nosales_model_input.path))
    with open(nosales_model_input.path, "rb") as handler:
        stack_pipeline = pickle.load(handler)
    
    nosales_thresh = utils.gen_thresholds(df = nosales_test_ext,  predictions = stack_pipeline.predict_proba(X=nosales_test_ext), classes = stack_pipeline.classes_)
    df_nosales_thresh = pd.DataFrame(nosales_thresh.items(), columns = ['club_nbr','nosales_club_thresh']) # nosales_club_thresh
    
    club_threshold_file_path = os.path.join(club_thresh_path_input, "club_thresh_chain.csv")
    df_cancelled_thresh = pd.read_csv(club_threshold_file_path).drop(columns = 'nosales_club_thresh')
    all_thresh = df_cancelled_thresh.merge(df_nosales_thresh, how = 'left', on = 'club_nbr')
    club_threshold_output.path = club_threshold_file_path
    all_thresh.to_csv(club_threshold_file_path, index = False)
    
    # club_threshold_file_path = os.path.join(club_thresh_path_input, "club_thresh_chain.csv")
    # df_cancelled_thresh = pd.read_csv(club_threshold_file_path).drop(columns = 'nosales_club_thresh')
    # all_thresh = df_cancelled_thresh.merge(df_nosales_thresh, how = 'left', on = 'club_nbr')
    # all_thresh.to_csv(club_threshold_output.path, index = False)
    # # df_output.to_csv(arg_path.path, index=False)

@dsl.pipeline(pipeline_root=PIPELINE_ROOT, name=PIPELINE_NAME)
def pipeline():
    data = data_preprocessing(training_data_bq_query_input=TRAINING_DATA_BQ_QUERY,
                              matcher=MATCHER,
                              project_id=PROJECT_ID, 
                              env=ENV, 
                              pipeline_root=PIPELINE_ROOT)
    
    train_test_data = train_test_split(nosales_ext_input=data.outputs['training_data_output'])
    
    train_eval_data = train_eval_model(nosales_ext_input=data.outputs['training_data_output'],
                                       nosales_train_ext_input=train_test_data.outputs['nosales_train_ext_output'],
                                       nosales_test_ext_input=train_test_data.outputs['nosales_test_ext_output'],
                                       nosales_train_usampled_input=train_test_data.outputs['nosales_train_usampled_output'],
                                       mode=MODE,
                                       stage1_flag=STAGE1_FLAG,
                                       ensemble_flag=ENSEMBLE_FLAG,
                                       rf_clf_model_path_input=RF_CLF_MODEL_PATH,
                                       logistic_clf_model_path_input=LOGISTIC_CLF_MODEL_PATH,
                                       stage1_nn_model_path_input=STAGE1_NN_MODEL_PATH,
                                       gnb_model_path_input=GNB_MODEL_PATH,
                                       stg1_feature_selector_model_path_input=STG1_FEATURE_SELECTOR_MODEL_PATH,
                                       nosales_model_path_input=NOSALES_MODEL_PATH,
                                       latest_nosales_model_path_input=LATEST_NOSALES_MODEL_PATH,
                                       project_id=PROJECT_ID,
                                       region=REGION,
                                       timestamp=TIMESTAMP)
    
    updated_thresholds = update_thresholds(nosales_test_ext_input=train_eval_data.outputs['nosales_test_ext_output'],  
                                           club_thresh_path_input=CLUB_THRESH_PATH,
                                           nosales_model_input=train_eval_data.outputs['nosales_model_output'])
    
    
    with Condition(train_eval_data.outputs["Output"] >= 0.7, name='model-upload-condition'):
        element_model_registry = CustomTrainingJobOp(
            project=PROJECT_ID,
            location=REGION,
            service_account=SERVICE_ACCOUNT,
            network="projects/12856960411/global/networks/vpcnet-private-svc-access-usc1",
            display_name="mlflow-model-registry",
            worker_pool_specs=[{
                "replica_count": 1,
                "machine_spec": {
                    "machine_type": "n1-standard-4",
                    "accelerator_count": 0,
                },
                # The below dictionary specifies:
                #   1. The URI of the custom image to run this CustomTrainingJobOp against
                #      - this image is built from ../../custom_image_builds/model_registry_image_build.ipynb
                #   2. The command to run against that image
                #   3. The arguments to supply to that custom image 
                "container_spec": {
                    "image_uri": TRANSITION_IMAGE,
                    #TRANSITION_IMAGE,
                    "command": [
                        "python3", "nosales_model_registry.py"
                    ],
                    "args": [
                        "--GCS_MODEL_PATH", LATEST_NOSALES_MODEL_PATH,
                        "--MLFLOW_EXP_NAME", MLFLOW_EXP_NAME,
                        "--MODEL_REGISTRY_NAME", MODEL_REGISTRY_NAME,
                    ],
                },
            }],

        ).set_display_name("element-mlflow-model-registry")
        element_model_registry.after(train_eval_data)
        
        version_transition = CustomTrainingJobOp(
            project=PROJECT_ID,
            location=REGION,
            service_account=SERVICE_ACCOUNT,
            network=NETWORK,
            display_name="mlflow-version-transition",
            worker_pool_specs=[{
                "replica_count": 1,
                "machine_spec": {
                    "machine_type": "n1-standard-4",
                    "accelerator_count": 0,
                },
                "container_spec": {
                    "image_uri": TRANSITION_IMAGE,
                    "command": [
                        "python3", "version_transition.py"
                    ],
                    "args": [
                        "--MODEL_REGISTRY_NAME", MODEL_REGISTRY_NAME,
                    ],
                },
            }],
        ).set_display_name("element-mlflow-version-transition")
        version_transition.after(element_model_registry)


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=pipeline, 
        package_path=TMP_PIPELINE_JSON)
    
    ##########################################

    # pipeline_job = aiplatform.PipelineJob(
    #     display_name=f"{PIPELINE_NAME}-{TIMESTAMP}",
    #     template_path=TMP_PIPELINE_JSON,
    #     pipeline_root=PIPELINE_ROOT,
    #     parameter_values={},
    #     enable_caching=False,
    # )

    # pipeline_utils.store_pipeline(
    #     storage_path=LATEST_PIPELINE_PATH, 
    #     filename=TMP_PIPELINE_JSON
    # )

    # pipeline_job.submit(service_account=SERVICE_ACCOUNT,network=NETWORK)

# if __name__ == "__main__":
#     if str(RUN_PIPELINE).lower() == "true":
#         pipeline_job.submit(service_account=SERVICE_ACCOUNT)
# else:
#     pass
