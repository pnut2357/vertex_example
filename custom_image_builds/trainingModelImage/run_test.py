import pandas as pd


training_data_path = "gs://oyi-ds-vertex-pipeline-bucket-nonprod/335163835346/oyi-nosales-model-pipeline-dev-20221029205839/data-preprocessing_-8891711501161725952/training_data_output"
training_data = pd.read_csv(training_data_path)

nosales_train_ext_path = "gs://oyi-ds-vertex-pipeline-bucket-nonprod/335163835346/oyi-nosales-model-pipeline-dev-20221029205839/train-test-split_-4280025482734338048/nosales_train_ext_output"
nosales_train_ext = pd.read_csv(nosales_train_ext_path)

nosales_test_ext_path = "gs://oyi-ds-vertex-pipeline-bucket-nonprod/335163835346/oyi-nosales-model-pipeline-dev-20221029205839/train-test-split_-4280025482734338048/nosales_test_ext_output"
nosales_test_ext = pd.read_csv(nosales_test_ext_path)

nosales_train_usampled_path = "gs://oyi-ds-vertex-pipeline-bucket-nonprod/335163835346/oyi-nosales-model-pipeline-dev-20221029205839/train-test-split_-4280025482734338048/nosales_train_usampled_output"
nosales_train_usampled = pd.read_csv(nosales_train_usampled_path)

print("success")