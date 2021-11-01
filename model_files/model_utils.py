import pandas as pd
import numpy as np
from datetime import datetime

def construct_weather_data(response, station, cols) -> pd.DataFrame:
    timestamps = sorted(response.data.keys())
    d = {}
    d["time"] = timestamps
    
    for col in cols:
        print(col)
        values = []
        for t in timestamps:
            values.append(response.data[t][station][col]["value"])
        
        d[col] = values

    return pd.DataFrame.from_dict(d)

        
# placeholder for data preprocessing function
def preprocess(weather):
    

    for col in ["Air temperature", "Wind speed", "Wind direction", "Wind gust", "Humidity", "Dew point"]:
        weather[col] = weather[col].interpolate().bfill()
    
    # print(weather[weather.isnull().any(axis=1)])

    weather = parse_timestamps(weather)
    print(weather.head())
    # print(weather.columns)
    weather = weather.rename(columns={
        "Air temperature": "Ilman lämpötila (degC)",
        "Wind speed": "Tuulen nopeus (m/s)",
        "Wind direction": "Tuulen suunta (deg)",
        "Wind gust": "Puuskanopeus (m/s)",
        "Humidity": "Suhteellinen kosteus (%)",
        "Dew point": "Kastepistelämpötila (degC)"
    })

    # here model was trained on finnish data

    return weather

def parse_timestamps(df): 
    # print(df["time"].iloc[0].timetuple())
    df = df.assign(year=[x.timetuple().tm_year for x in df['time']])
    df = df.assign(yday=[x.timetuple().tm_yday for x in df['time']])
    df = df.assign(day=[x.timetuple().tm_mday for x in df['time']])
    df = df.assign(month=[x.timetuple().tm_mon for x in df['time']])
    df = df.assign(wday=[x.timetuple().tm_wday for x in df['time']])
    df = df.assign(hour=[x.timetuple().tm_hour for x in df['time']])

    # placeholder values (not done yet)
    df["week"] = np.repeat(42, len(df))
    # df['week'] = df.apply(lambda row: row.time.isocalendar().week, axis=1)
    df["id"] = np.repeat(113, len(df))
    df["yhour"] = np.repeat(1, len(df))

    print(df.head())

    return df

def predict(config, model):
    if isinstance(config, dict):
        config = pd.DataFrame(config)

    return model.predict(config)


def count_drops(station):
    minbikes = min(station['predicted'])

    counts = 0
    was_min = False

    for i, row in station.iterrows():
        bikes = row['predicted'] 

        # drop starts
        if not was_min and bikes == minbikes:
            counts += 1
            was_min = True

        # drop ends
        elif was_min and bikes != minbikes:
            was_min = False

    return counts, int(minbikes)


def count_changes(station):
    counts = 0
    previous = -1

    for i, row in station.iterrows():
        bikes = row['predicted'] 

        # print(previous, bikes)
        if bikes != previous:
            # found change 
            counts += 1
            previous = bikes

        # print(counts)
    return counts
