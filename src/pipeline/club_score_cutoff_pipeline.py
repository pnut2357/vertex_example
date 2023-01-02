#!/usr/bin/env python
# coding: utf-8
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

import datetime

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
        "--BRANCH", "stage",
        "--is_prod", False,
    ]
    args = parser.parse_args(sys.args)
    
BRANCH_ID = args.BRANCH
is_prod = args.is_prod

if BRANCH_ID == "stage" and is_prod == True:
    BRANCH_ID = "prod"
    
ENV = BRANCH_ID

from_date = ( datetime.datetime.now()  - datetime.timedelta(days=7*8) ).strftime("%Y-%m-%d")
to_date   = ( datetime.datetime.now()  - datetime.timedelta(days=1) ).strftime("%Y-%m-%d")
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

PARAMS = pipeline_utils.yaml_import('settings.yml')

# ENV = PARAMS['env_flag']
NETWORK = PARAMS['envs'][ENV]['VPC_NETWORK']

BASE_IMAGE = PARAMS['envs'][ENV]['BASE_IMAGE']
SERVICE_ACCOUNT = PARAMS['envs'][ENV]['SERVICE_ACCOUNT']
CLUB_THRESH_PATH = PARAMS['envs'][ENV]['CLUB_THRESH_PATH']
PROJECT_ID = PARAMS['envs'][ENV]['PROJECT_ID']
RUN_PIPELINE = PARAMS['envs'][ENV]['RUN_PIPELINE']
PIPELINE_ROOT = PARAMS['envs'][ENV]['CLUB_THRESH_PIPELINE_ROOT']
PIPELINE_NAME = PARAMS['envs'][ENV]['CLUB_THRESH_PIPELINE_NAME']
PIPELINE_JSON = PARAMS['envs'][ENV]['CLUB_THRESH_PIPELINE_JSON']
TMP_PIPELINE_JSON = os.path.join("/tmp", PIPELINE_JSON)
LATEST_PIPELINE_PATH = PARAMS['envs'][ENV]['CLUB_THRESH_LATEST_PIPELINE_PATH']


# print(f"ENV: {ENV}, \nPROJECT_ID: {PROJECT_ID}, \nBASE_IMAGE: {BASE_IMAGE}, \nPIPELINE_NAME: {PIPELINE_NAME}, \nPIPELINE_JSON: {PIPELINE_JSON}")

    
@component(base_image=BASE_IMAGE)
def get_logger(from_date: str,
               to_date: str,
               project_id: str,
               df_subset_output: Output[Dataset]
):
    from google.cloud import bigquery
    import pandas as pd

    client = bigquery.Client(project=project_id)
    sql = """(select * from oyi.rm_report_logger where event_ts>= '{from_date}' and event_ts<= '{to_date}' AND
TRIM(LOWER(event_user)) LIKE '%.%' AND TRIM(LOWER(event_txt)) = 'root_cause')""".format(from_date=from_date,to_date=to_date)
    df = client.query(sql).to_dataframe()
    action_thresh = 5
    df.event_ts= pd.to_datetime(df.event_ts)
    df['central_ts']= df.event_ts.dt.tz_convert('US/Central')
    df_subset= df[(df.event_txt=='root_cause') & (df.event_user.str.match('\w+\.\w+'))].copy()
    df_subset['central_dt']=df_subset.central_ts.dt.date
    df_subset= df_subset.sort_values(['central_dt','club_nbr','item_nbr','central_ts'], ascending=False)
    #df_subset= df_subset[~df_subset.duplicated(['central_dt','club_nbr','item_nbr'],keep= 'first')]
    df_subset= df_subset[~df_subset.duplicated(['ds_uuid','club_nbr','item_nbr'],keep= 'first')]
    df_subset= df_subset.sort_values(['club_nbr','event_user','central_ts'],ascending=True)
    gp= df_subset.groupby(['club_nbr','event_user'])
    df_subset['ts_shifted']=gp.central_ts.transform(lambda x:x.shift(1))
    #df_subset.ts_shifted= df_subset.ts_shifted.dt.tz_localize('GMT').dt.tz_convert('US/Central')
    df_subset.ts_shifted= df_subset.ts_shifted.dt.tz_convert('US/Central')
    df_subset['ts_diff']=  df_subset.central_ts- df_subset.ts_shifted
    df_subset['spurious']= ~(df_subset.ts_diff.isna()) & (df_subset.ts_diff.dt.seconds <= action_thresh)
    df_subset= df_subset[~df_subset.spurious].copy()

    df_subset.to_csv(df_subset_output.path, index=False)

    
@component(base_image=BASE_IMAGE)
def get_inv(from_date: str,
            to_date: str,
            project_id: str,
            invdash_output: Output[Dataset]
):
    from google.cloud import bigquery
    import pandas as pd
    
    from_rundate= (pd.to_datetime(from_date)- pd.Timedelta('1 days')).date().strftime('%Y-%m-%d')
    to_rundate= (pd.to_datetime(to_date)- pd.Timedelta('1 days')).date().strftime('%Y-%m-%d')
    
    client = bigquery.Client(project=project_id)
    sql = f"""select club_nbr,item_nbr,old_nbr,run_date,raw_score,special_item, report_type, uuid from oyi_prod.inventory_dashboard_history 
        where run_date>= '{from_rundate}' and run_date <= '{to_rundate}'
        and display_ind='Display'
        """.format(from_rundate=from_rundate,to_rundate=to_rundate)
    invdash = client.query(sql).to_dataframe()
    print(invdash.columns)
    invdash.run_date = pd.to_datetime(invdash.run_date)
    invdash['actual_date']= invdash.run_date+ pd.Timedelta('1 day')
    invdash.actual_date= invdash.actual_date.dt.date
    invdash= invdash[~invdash.duplicated(['actual_date','club_nbr','old_nbr'],keep= 'first')].copy()
    invdash.actual_date = invdash.actual_date.astype(str)
    #Removing special items which are always added to the list
    invdash = invdash[~(invdash.special_item==1)]
    
    invdash.to_csv(invdash_output.path, index=False)


@component(base_image=BASE_IMAGE)
def dataprep(logger_input: Input[Dataset],
             inv_input: Input[Dataset],
             match_nosales_output: Output[Dataset],
             match_cancelled_output: Output[Dataset]
):
    import pandas as pd
    
    logger = pd.read_csv(logger_input.path)
    inv = pd.read_csv(inv_input.path)
  
    match= pd.merge(left= logger,
                    right= inv, 
                    left_on = ['ds_uuid','club_nbr','item_nbr'],
                    right_on= ['uuid','club_nbr','old_nbr'],
                    how= 'inner', indicator=True, validate='one_to_one')
    match['run_date'] = match['run_date'].astype(str)
    match['action']= ~(match.event_note.isin(['No Action Taken, already out for sale','No Action Taken, already OFS']))
    match_nosales= match[~ (match.report_type=='C')]
    match_cancelled= match[(match.report_type=='C')]
  
    match_nosales.to_csv(match_nosales_output.path, index=False)
    match_cancelled.to_csv(match_cancelled_output.path, index=False)


@component(base_image=BASE_IMAGE)
def get_raw_score_thresholds(train_input: Input[Dataset],
                             club_thresh_output: Output[Dataset]
):    
    import numpy as np
    import pandas as pd
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
    
    club_thresh = {}
    club_prec = {}
    club_recall = {}
    
    mins, maxs= {},{}
    
    train = pd.read_csv(train_input.path)
    for club in train.club_nbr.unique():
        train_club = train[train.club_nbr==club]
        thresholds = np.sort(list(set(np.round(train_club.raw_score.unique(), 4))))

        f1_arr = []
        prec_arr = []
        recall_arr= []
        for th in thresholds:
            y_pred = list(train_club.raw_score >= th)
            y_true = list(train_club.action == True)
            f1 = f1_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1_arr.append(f1)
            prec_arr.append(prec)
            recall_arr.append(recall)
        
        club_thresh[club] = thresholds[np.argmax(f1_arr)]
        # get the precision and recall associated with the max F1 score
        club_prec[club] = np.round(prec_arr[np.argmax(f1_arr)], 4)
        club_recall[club] = np.round(recall_arr[np.argmax(f1_arr)], 4)
   
    df_club_prec = pd.DataFrame(club_prec.items(),columns = ['club_nbr','precision'])
    df_club_recall = pd.DataFrame(club_recall.items(),columns = ['club_nbr','recall'])
    df_club_thresh = pd.DataFrame(club_thresh.items(),columns = ['club_nbr','club_thresh'])
    df_thresholds = df_club_thresh.merge(df_club_prec, how = 'left', on = 'club_nbr')\
                                      .merge(df_club_recall, how = 'left', on = 'club_nbr')
    df_thresholds.to_csv(club_thresh_output.path, index = False)


@component(base_image=BASE_IMAGE)
def combine_results(nosales_club_thresholds_input: Input[Dataset],
                    cancelled_club_thresholds_input: Input[Dataset],
                    club_thresh_chain_path_input: str,
                    regularized_club_thresh_chain_output: Output[Dataset],
                    unregularized_club_thresh_chain_output: Output[Dataset] 
):
    import pandas as pd
    
    nosales_club_thresholds = pd.read_csv(nosales_club_thresholds_input.path)
    nosales_club_thresholds = nosales_club_thresholds.rename(columns = {'club_thresh': 'nosales_club_thresh',
                                                                        'precision': 'nosales_precision',
                                                                        'recall': 'nosales_recall'})
    cancelled_club_thresholds = pd.read_csv(cancelled_club_thresholds_input.path)
    cancelled_club_thresholds = cancelled_club_thresholds.rename(columns = {'club_thresh': 'cancelled_club_thresh',
                                                                        'precision': 'cancelled_precision',
                                                                        'recall': 'cancelled_recall'})
   
    # merge the DFs
    df_thresholds = nosales_club_thresholds.merge(cancelled_club_thresholds, how = 'left', on = 'club_nbr')
    # Regularize the chosen values by averaging the results with the group mean.
    df_thresholds['nosales_club_thresh'] = ((df_thresholds['nosales_club_thresh'] + df_thresholds['nosales_club_thresh'].mean()) / 2).round(4)
    df_thresholds['cancelled_club_thresh'] = ((df_thresholds['cancelled_club_thresh'] + df_thresholds['cancelled_club_thresh'].mean()) / 2).round(4)


    current_time = pd.datetime.now()
    df_thresholds['update_ts'] = current_time
    df_thresholds.to_csv(regularized_club_thresh_chain_output.path, index=False)
    unregularized_club_thresh_chain_output.path = f'{club_thresh_chain_path_input}/club_thresh_chain.csv'
    
    df_thresholds_unregularized = nosales_club_thresholds[['club_nbr', 'nosales_club_thresh']].merge(cancelled_club_thresholds[['club_nbr', 'cancelled_club_thresh']],
                                                                      how = 'left', on = 'club_nbr')
    df_thresholds_unregularized.to_csv(f'{club_thresh_chain_path_input}/club_thresh_chain.csv', index=False)

    
@dsl.pipeline(pipeline_root=PIPELINE_ROOT, name=PIPELINE_NAME)
def pipeline():
    logger = get_logger(from_date=from_date,
                           to_date=to_date,
                           project_id = PROJECT_ID)
    
    invdash = get_inv(from_date=from_date,
                      to_date=to_date,
                      project_id = PROJECT_ID)
    
    match_data = dataprep(logger_input=logger.outputs['df_subset_output'],
                       inv_input=invdash.outputs['invdash_output'])
    
    nosales_result = get_raw_score_thresholds(train_input=match_data.outputs['match_nosales_output'])
    cancelled_result = get_raw_score_thresholds(train_input=match_data.outputs['match_cancelled_output'])
    
    club_thresh_chain = combine_results(nosales_club_thresholds_input=nosales_result.outputs['club_thresh_output'],
                                        cancelled_club_thresholds_input=cancelled_result.outputs['club_thresh_output'],
                                        club_thresh_chain_path_input=CLUB_THRESH_PATH)


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=pipeline, 
        package_path=TMP_PIPELINE_JSON)

    pipeline_job = aiplatform.PipelineJob(
        display_name=f"{PIPELINE_NAME}-{TIMESTAMP}",
        template_path=TMP_PIPELINE_JSON,
        pipeline_root=PIPELINE_ROOT,
        parameter_values={},
        enable_caching=False,
    )
    
    pipeline_utils.store_pipeline(
        storage_path=LATEST_PIPELINE_PATH, 
        filename=TMP_PIPELINE_JSON
    )

    pipeline_job.submit(service_account=SERVICE_ACCOUNT,network=NETWORK)


# if __name__ == "__main__":
#     if str(RUN_PIPELINE).lower() == "true":
#         pipeline_job.submit(service_account=SERVICE_ACCOUNT, network=NETWORK)
# else:
#     pass
