#!/bin/bash


for dataset in "Embedded_generation" "Grid_export" "Grid_export_to_Tiwai" "Grid_import" "HVDC_Flows" "Reactive_power" "Unit_level_generation_IR"; do
  ./dload.sh $dataset
done