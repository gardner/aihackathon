#!/usr/bin/env python3

# TemporalFusionTransformer

from multiprocessing import freeze_support
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.tuner import Tuner
from pytorch_forecasting import QuantileLoss, TimeSeriesDataSet, TemporalFusionTransformer
import pandas as pd
from datetime import datetime, timedelta
import os
import numpy as np

import torch
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium')
    print('YAY CUDA')
else:
    print ("CUDA device not found.")

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def half_hour_intervals(start, end):
    delta = end - start
    # Calculate the total number of seconds between the two datetimes
    # and then divide by the number of seconds in half an hour (1800 seconds).
    return int(delta.total_seconds() // 1800)

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    # start_mem = df.memory_usage().sum() / 1024**2
    # print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        # print(str(col_type))

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int' or str(col_type)[:4] == 'uint':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            elif not 'datetime' in str(col_type):
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    # print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    # print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

def set_time_idx(df):
    # df = df.where(df["TradingDate"] == '2016-04-03').sort_values("TradingPeriod").dropna()
    # 2023‐08‐08T19:05:07−07:00
    df["NZDT"] = pd.to_datetime(df["TradingDate"] + 'T' + df["TradingPeriodStartTime"])
    df["NZDT"] = df["NZDT"].dt.tz_localize('Pacific/Auckland', ambiguous=False)
    df["UTCDT"] = df["NZDT"].dt.tz_convert('UTC')
    df.drop(["NZDT"], inplace=True, axis=1)

    ## TODO This needs to be fixed
    ## we should set time_idx to be the number of intervals since the start of the dataset

    start_date = df["UTCDT"].min()
    # Set the time_idx to be the trading period since the start of the dataset
    df["time_idx"] = df["UTCDT"].apply(lambda x: half_hour_intervals(start_date, x))
    # df["time_idx"] = (df["UTCDT"].dt.dayofyear * 48) - 48
    # df["time_idx"] += df["TradingPeriod"].astype(int)
    # df["time_idx"] -= df["time_idx"].min()
    df.drop_duplicates(subset=["PointOfConnection", "TradingDate", "TradingPeriod", "TradingPeriodStartTime"], keep="first", inplace=True)
    df = df.sort_values(["time_idx"]).dropna()

    return df

def main():
    df = pd.read_csv('data/ReconciledInjectionAndOfftake_201604_20170624_100756.csv')
    # df.drop(["Network", "Island", "TradingDate" ], inplace=True, axis=1)

    df = reduce_mem_usage(set_time_idx(df))

    print(df.head(5))
    ## TODO Calculate holidays and off-peak times

    ### Actually train the model

    # # define dataset
    max_encoder_length = 192
    max_prediction_length = 48
    training_cutoff = df["time_idx"].max() - max_prediction_length

    print("Training cutoff: ", training_cutoff)
    training = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="KilowattHours",
        group_ids=[ "PointOfConnection" ],
        min_encoder_length=0,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["Network", "Island", "Participant", "FlowDirection"],
        time_varying_known_reals=["TradingPeriod"],
        time_varying_unknown_reals=["KilowattHours"],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    # create validation and training dataset
    validation = TimeSeriesDataSet.from_dataset(training, df, min_prediction_idx=training.index.time.max() + 1, stop_randomization=True)

    print("Training dataset: ", len(training))
    batch_size = max_encoder_length * 10
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=4)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=4)

    # # define trainer with early stopping
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="auto",
        gradient_clip_val=0.1,
        log_every_n_steps=10,
        limit_train_batches=30000,
        callbacks=[lr_logger, early_stop_callback],
    )

    # # create the model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=32,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=16,
        output_size=7,
        loss=QuantileLoss(),
        log_interval=2,
        reduce_on_plateau_patience=4
    )
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

    # find optimal learning rate (set limit_train_batches to 1.0 and log_interval = -1)
    res = Tuner(trainer).lr_find(
        tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, early_stop_threshold=1000.0, max_lr=0.3,
    )

    print(f"suggested learning rate: {res.suggestion()}")
    fig = res.plot(show=True, suggest=True)
    fig.show()

    print("Training model")
    # fit the model
    trainer.fit(
        tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
    )

    print("Finished training")
    predictions = tft.predict(val_dataloader)
    print(predictions)


if __name__ == '__main__':
    print("Starting...")
    main()