#!/bin/bash

curl "https://www.emi.ea.govt.nz/Wholesale/Datasets/Metered_data/$1" | \
egrep -i --color "href=\".*?_$1.csv\"" | \
awk -F'="/' ' { print $2 }' | \
awk -F'"' '{ print "https://www.emi.ea.govt.nz/" $1 }'| \
sort | uniq > "$1.txt"

mkdir -p "data/$1"
aria2c -x4 -c -i "$1.txt" -d "data/$1"

