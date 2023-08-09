#!/bin/bash

for year in {2005..2023}; do
    curl "https://www.emi.ea.govt.nz/Wholesale/Datasets/Volumes/Reconciliation/$year" | \
    egrep -i --color "href=\".*?ReconciledInjectionAndOfftake.*.csv.gz\"" | \
    awk -F'="/' ' { print $2 }' | \
    awk -F'"' '{ print "https://www.emi.ea.govt.nz/" $1 }'| \
    sort | uniq > "$year.txt"
    mkdir -p "data/recon/$year"
    aria2c -x4 -c -i "$year.txt" -d "data/recon/$year"
done

