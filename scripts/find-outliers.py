#!/usr/bin/env python3
# vim: set ts=8 sts=4 et sw=4 tw=99:
#
# Prints outliers in database
#

import numpy as np
import math
import pandas as pd
import nltk
from nltk.cluster.kmeans import KMeansClusterer

# TODO federation not in openpowerlifting.csv

# 0         1      2    3   4       5       6   7          8              9           10       11       12       13         14        15      16         17     18       19          20          21          22           23            24           25     26    27    28
# MeetID,LifterID,Name,Sex,Event,Equipment,Age,Division,BodyweightKg,WeightClassKg,Squat1Kg,Squat2Kg,Squat3Kg,Squat4Kg,BestSquatKg,Bench1Kg,Bench2Kg,Bench3Kg,Bench4Kg,BestBenchKg,Deadlift1Kg,Deadlift2Kg,Deadlift3Kg,Deadlift4Kg,BestDeadliftKg,TotalKg,Place,Wilks,McCulloch

indexes = {}

# TODO normalizations
# TODO remove null data

age_normalize = 1
bw_normalize = 1
squat_normalize = 1
bench_normalize = 1
deadlift_normalize = 1

NUM_CLUSTERS = 2
REPEATS = 1

def get_data(path):
    db = pd.read_csv(path)

    for i in range(len(db.columns)):
        indexes[db.columns[i]] = i
    print(indexes)

    # TODO normalize

    return db.values

def euclid_component(a, b, normalization):
    if(np.isnan(a)):
        return 0
    if(np.isnan(b)):
        return 0
    return (float(a) - float(b)) ** 2 / (normalization ** 2)

def find_outlier(cluster):
    pass

# u and v are vectors
def distance_func(u, v):

    if u[indexes["Sex"]] is v[indexes["Sex"]]:
        sex = 1
    else:
        sex = 0

    if u[indexes["Event"]] is v[indexes["Event"]]:
        event = 1
    else:
        event = 0

    if u[indexes["Equipment"]] is v[indexes["Equipment"]]:
        equip = 1
    else:
        equip = 0

    if u[indexes["Division"]] is v[indexes["Division"]]:
        division = 1
    else:
        division = 0

    distance = math.sqrt(
        # categorical variables
        sex + event + equip + division +
        # continuous variables
        euclid_component(u[indexes["Age"]], v[indexes["Age"]], age_normalize) +
        euclid_component(u[indexes["BodyweightKg"]], v[indexes["BodyweightKg"]], bw_normalize) +
        euclid_component(u[indexes["BestSquatKg"]], v[indexes["BestSquatKg"]], squat_normalize) +
        euclid_component(u[indexes["BestBenchKg"]], v[indexes["BestBenchKg"]], bench_normalize) +
        euclid_component(u[indexes["BestDeadliftKg"]], v[indexes["BestDeadliftKg"]], deadlift_normalize)
    )

    return distance

if __name__ == '__main__':
    import sys

    kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=distance_func, repeats=REPEATS)
    data = get_data(sys.argv[1])
    clusters = kclusterer.cluster(data, assign_clusters=True)

    print(clusters)