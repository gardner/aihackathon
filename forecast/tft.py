#!/usr/bin/env python3

# TemporalFusionTransformer

# import torch
# if torch.backends.mps.is_available():
#     mps_device = torch.device("mps")
#     x = torch.ones(1, device=mps_device)
#     print(x)
# else:
#     print ("MPS device not found.")
#     exit()

# import lightning.pytorch as pl
# from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
# from lightning.pytorch.tuner import Tuner
# from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
import pandas as pd
from datetime import datetime, timezone

import pytz

df = pd.read_csv('data/ReconciledInjectionAndOfftake_201604_20170624_100756.csv')

# Ignore DST for now
# dt = datetime.strptime('2015-01-03', '%Y/%m/%d')
# print(dt)

pytz_nz = pytz.timezone('Pacific/Auckland')


# df = df.where(df["TradingDate"] == '2016-04-03').sort_values("TradingPeriod").dropna()
# 2023‐08‐08T19:05:07−07:00
df = df.where(df["PointOfConnection"] == 'OHB2201').sort_values(["TradingDate", "TradingPeriod"]).dropna()
df["NZDT"] = pd.to_datetime(df["TradingDate"] + 'T' + df["TradingPeriodStartTime"])
df["NZDT"] = df["NZDT"].dt.tz_localize('Pacific/Auckland', ambiguous=False)
df["UTCDT"] = df["NZDT"].dt.tz_convert('UTC')
df.drop(["NZDT"], inplace=True, axis=1)

## TODO This needs to be fixed
## we should set time_idx to be the number of intervals since the start of the dataset

# Set the time_idx to be the trading period since the start of the dataset
df["time_idx"] = (df["UTCDT"].dt.dayofyear * 48) - 48
df["time_idx"] += df["TradingPeriod"].astype(int)
df["time_idx"] -= df["time_idx"].min()

# print(df)

## Testing
df = df.sort_values(["TradingDate", "TradingPeriod"]).dropna()
df.drop_duplicates(subset=["PointOfConnection", "TradingDate", "TradingPeriod", "TradingPeriodStartTime"], keep="first", inplace=True)
# df.drop(["Network", "Participant", "Island", "KilowattHours", "PointOfConnection", "FlowDirection"], inplace=True, axis=1)

df.to_csv('data/ohb2201.csv', index=False)
# print(df.head(400).to_string(index=False))

exit()


# print(df.sort_values("time_idx").where(df["TradingPeriod"] == 2).dropna())

# print(df.sort_values("time_idx").head(50))
# # PointOfConnection Network Island Participant TradingDate  TradingPeriod TradingPeriodStartTime FlowDirection  KilowattHours




# # define dataset
# max_encoder_length = 36
# max_prediction_length = 6
# training_cutoff = "2022-01-01"  # day for cutoff

# training = TimeSeriesDataSet(
#     data[lambda x: x.date < training_cutoff],
#     time_idx= ...,
#     target= ...,
#     # weight="weight",
#     group_ids=[ ... ],
#     max_encoder_length=max_encoder_length,
#     max_prediction_length=max_prediction_length,
#     static_categoricals=[ ... ],
#     static_reals=[ ... ],
#     time_varying_known_categoricals=[ ... ],
#     time_varying_known_reals=[ ... ],
#     time_varying_unknown_categoricals=[ ... ],
#     time_varying_unknown_reals=[ ... ],
# )

# # create validation and training dataset
# validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=training.index.time.max() + 1, stop_randomization=True)
# batch_size = 128
# train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=2)
# val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=2)

# # define trainer with early stopping
# early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min")
# lr_logger = LearningRateMonitor()
# trainer = pl.Trainer(
#     max_epochs=100,
#     accelerator="auto",
#     gradient_clip_val=0.1,
#     limit_train_batches=30,
#     callbacks=[lr_logger, early_stop_callback],
# )

# # create the model
# tft = TemporalFusionTransformer.from_dataset(
#     training,
#     learning_rate=0.03,
#     hidden_size=32,
#     attention_head_size=1,
#     dropout=0.1,
#     hidden_continuous_size=16,
#     output_size=7,
#     loss=QuantileLoss(),
#     log_interval=2,
#     reduce_on_plateau_patience=4
# )
# print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# # find optimal learning rate (set limit_train_batches to 1.0 and log_interval = -1)
# res = Tuner(trainer).lr_find(
#     tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, early_stop_threshold=1000.0, max_lr=0.3,
# )

# print(f"suggested learning rate: {res.suggestion()}")
# fig = res.plot(show=True, suggest=True)
# fig.show()

# # fit the model
# trainer.fit(
#     tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
# )