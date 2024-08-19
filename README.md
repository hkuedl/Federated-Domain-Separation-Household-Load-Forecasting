# FederatedDomainSeparation
Codes for "Federated Domain Separation for Distributed Forecasting of Non-IID Household Loads"


# Federated Domain Separation for Distributed Forecasting of Non-IID Household Loads
> We propose an advanced FL-based household load forecasting framework combined with federated domain separation. It can comprehensively acquire useful knowledge from all households while excluding potentially contaminating parts, thus giving more accurate forecasts even in the presence of non-IID load data.

Codes for the paper "Federated Domain Separation for Distributed Forecasting of Non-IID Household Loads". https://ieeexplore.ieee.org/abstract/document/10440505

Authors: Nan Lu; Shu Liu; Qingsong Wen; Qiming Chen; Liang Sun; Yi Wang


## Requirements
>[!NOTE]
>Please ensure that at least the following packages are installed

Python version: 3.8.10
Pytorch version: 1.13.1+cu116
numpy version: 1.24.2
pandas version: 1.5.3
matplotlib version: 3.7.1


## Experiments
Please refer to the file "Experiments".

If you would like to see the definition of the client, please refer to
```
"client.py".
```

If you would like to see how the central server works, please refer to
```
"server.py".
```

If you would like to see the details about the model construction, please refer to
```
"Forecasting_models.py".
```

### Data
The data file is too large to upload. I will be uploading the data to the cloud storage as soon as possible. 

### Reproduction
>[!IMPORTANT]
>Make sure first that your codes can be run on your PC/servers with the **same** environments mentioned above.


To reproduce the experiments in the paper, please run
```
"read_forecasting_results" and "read_results".
```

These two files will read the trained model parameters and present the corresponding experimental results in this paper.

And if you would like to retrain the model, please set the hyperparameters in
```
"experimental_parameters.py".
```
then run 
```
"main.py".
```
