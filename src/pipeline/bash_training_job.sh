#!/bin/bash
set -x
curl \
-X POST \
-H "Authorization: Bearer $(gcloud auth print-access-token)" \
-H "Content-Type: application/json" \
https://us-central1-aiplatform.googleapis.com/v1beta1/projects/335163835346/locations/us-central1/customJobs \
-d '{
    "displayName": "'"cc_test_$(date +%Y%m%d%H%M)"'",
    "jobSpec": {
        "workerPoolSpecs": [
            {
                "machineSpec": {
                    "machineType": "n1-standard-4"
                },
                "replicaCount": "1",
                "diskSpec": {
                    "bootDiskType": "pd-ssd",
                    "bootDiskSizeGb": 100
                },
                "containerSpec": {
                    "imageUri": "gcr.io/wmt-mlp-p-oyi-ds-or-oyi-dsns/prework_test:0.0.4",
                    "command": [
                        "sh",
                        "-c",
                        "python /app/run_test.py"
                    ]
                }
            }
        ],
        "service_account": "",
        "network": "projects/12856960411/global/networks/vpcnet-private-svc-access-usc1",
        "baseOutputDirectory": {},
    }
}'