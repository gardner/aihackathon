# PointOfConnection,Network,Island,Participant,TradingDate,TradingPeriod,TradingPeriodStartTime,FlowDirection,KilowattHours
# NMA0331,TPCO,SI,MERI,2016-04-14,2,00:30,Injection,27075.0
# OHB2201,MERI,SI,MERI,2016-04-14,14,06:30,Injection,47691.0
# OKN0111,LINE,NI,MERI,2016-04-14,28,13:30,Offtake,96.0
# OKN0111,LINE,NI,MERI,2016-04-14,30,14:30,Injection,1.0
# OWH0111,HAWK,NI,MERI,2016-04-14,2,00:30,Offtake,68.0
# PAO1101,POCO,NI,MERI,2016-04-14,5,02:00,Offtake,2709.0
# PNI0331,CKHK,NI,MERI,2016-04-14,15,07:00,Offtake,525.0
# ROS0221,VECT,NI,MERI,2016-04-14,47,23:00,Offtake,2577.0
# ROT0111,HAWK,NI,MERI,2016-04-14,17,08:00,Offtake,396.0

from glob import glob
import math
import multiprocessing
import os
import pandas as pd
from tqdm import tqdm
import json


point_of_connection = set()
network = set()
island = { 'NI', 'SI' }
participant = set()
flow_direction = { 'Offtake', 'Injection' }

def build_map(filename):
    df = pd.read_csv(filename, compression='gzip', index_col=None, header=0)
    return {
        "PointOfConnection": pd.unique(df["PointOfConnection"]),
        "Network": pd.unique(df["Network"]),
        "Participant": pd.unique(df["Participant"]),
    }

if __name__ == "__main__":
    all_files = glob(os.path.join('data' , '**', "*.csv.gz"), recursive=True)

    # for filename in tqdm.tqdm(all_files):
    #     build_map(filename)

    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    print("mapping ...")
    results = tqdm(pool.imap(build_map, all_files), total=len(all_files))
    # tqdm(pool.imap(crunch, range(40)), total=40)
    for result in results:
        point_of_connection.update(result["PointOfConnection"])
        network.update(result["Network"])
        participant.update(result["Participant"])

    print("Waiting for pool to finish...")
    pool.close()
    pool.join()
    print("Done")

    dictionary = {}
    for idx, connection in enumerate(point_of_connection):
        dictionary[connection] = idx

    # Save dictionary to file as json
    with open('maps/point_of_connection.json', 'w') as fp:
        json.dump(dictionary, fp)

    print("point_of_connection")

    dictionary = {}
    for idx, network in enumerate(network):
        dictionary[network] = idx

    # Save dictionary to file as json
    with open('maps/network.json', 'w') as fp:
        json.dump(dictionary, fp)

    print("network")

    dictionary = {}
    for idx, participant in enumerate(participant):
        dictionary[participant] = idx

    # Save dictionary to file as json
    with open('maps/participant.json', 'w') as fp:
        json.dump(dictionary, fp)

    print("participant")

    # print("network", network)
    # print("participant", participant)
