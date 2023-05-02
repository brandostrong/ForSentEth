from datetime import datetime, timedelta
import pandas as pd

print(timedelta)


def readFile(filename):
    arr = []
    with open(filename) as f:
        for line in f:
            curr = line.strip().split(",")
            split = [datetime.fromtimestamp(
                int(curr[0])), float(curr[1]), float(curr[2])]
            arr.append(split)
            # print(datetime.datetime.fromtimestamp(int(line.split(",")[0])))

    return (arr)


def discreteDates(arr):
    # first datetime: datetime.datetime(2015, 8, 7, 7, 3, 25)
    # first block is (2015, 8, 7, 7, 0, 0)
    # look at starting time, and find nearest last hour
    start = datetime(2015, 8, 7, 7, 0, 0)
    delta = timedelta(hours=1)
    blockend = datetime(2015, 8, 7, 8, 0, 0)
    for i, x in enumerate(arr):
        if i % 100000 == 0:
            print(i, "\n")
            print((i / len(arr))*100, "%")
        placed = False
        currtime = x[0]
        while not placed:
            if currtime >= start and currtime < blockend:
                arr[i].append(start)
                placed = True
            else:
                start = start + timedelta(hours=1)
                blockend = blockend + timedelta(hours=1)
    return arr


def discretizeToCsv(filename):
    arr = readFile(filename)
    discretedates = discreteDates(arr)
    df = pd.DataFrame(discretedates)
    del discretedates
    df.columns = ["time", "price", "quantity", "discretehour"]
    # df["price"] = df["price"].astype(float)
    # df["quantity"] = df["quantity"].astype(float)
    df['weight'] = df.groupby("discretehour").apply(
        lambda x: x["price"] * x["quantity"]).values
    df['meanprice'] = df.groupby("discretehour").transform(
        'sum')["weight"] / df.groupby("discretehour").transform('sum')["quantity"]

    discrete = pd.DataFrame()

    discrete["meanprice"] = df.groupby("discretehour").apply(
        'sum')["weight"] / df.groupby("discretehour").apply('sum')["quantity"]
    discrete["volume"] = df.groupby("discretehour")["quantity"].sum()
    discrete["low"] = df.groupby("discretehour")["price"].min()
    discrete["high"] = df.groupby("discretehour")["price"].max()
    discrete["open"] = df.groupby("discretehour")["price"].first()
    discrete["close"] = df.groupby("discretehour")["price"].last()
    discrete["transactions"] = df.groupby(
        "discretehour")["discretehour"].count()

    df.to_csv("fullcont.csv")
    del df
    discrete.to_csv("discrete.csv")



print(1)
