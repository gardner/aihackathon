# Forecast

## Data

```shell
mkdir -p data && cd data && \
wget https://www.emi.ea.govt.nz/Wholesale/Datasets/Volumes/Reconciliation/2016/ReconciledInjectionAndOfftake_201604_20170624_100756.csv.gz && \
gzip -d *csv.gz
```

Code to train


## Setup

### macOS (M1/M2)

    conda install pytorch torchvision torchaudio -c pytorch-nightly
    pip install pytorch-forecasting

Update existing:

    conda update pytorch torchvision torchaudio -c pytorch-nightly




Reading:

https://huggingface.co/blog/time-series-transformers



## Methods

### Pytorch
[TemporalFusionTransformer](https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer.html#pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer)


### Gluon / DeepAR

[TimeSeriesPredictor](https://aws.amazon.com/blogs/machine-learning/easy-and-accurate-forecasting-with-autogluon-timeseries/)

https://www.sciencedirect.com/science/article/pii/S1877050922018907?via%3Dihub

Trans-T2V