#Below is the command, to run the training model 
#
fnet train --json /ihome/zsiddiqu/singhp/prefbz16_1.json --gpu_ids 0
#
#Below is the command, to run the predict function , the test images path must be present in N_test.csv file 
fnet predict --path_model_dir "/ihome/zsiddiqu/singhp/model/model.p" --dataset fnet.data.MultiChTiffDataset --dataset_kwargs '{"path_csv": "/ihome/zsiddiqu/singhp/N_test.csv"}' --gpu_ids 0
#





