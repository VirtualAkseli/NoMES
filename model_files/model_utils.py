import pandas as pd
import numpy as np
import datetime
import os

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
    

    weather = parse_timestamps(weather)
    print(weather.head())

    # here model was trained on finnish data
    weather = weather.rename(columns={
        "Air temperature": "Ilman lämpötila (degC)",
        "Wind speed": "Tuulen nopeus (m/s)",
        "Wind direction": "Tuulen suunta (deg)",
        "Wind gust": "Puuskanopeus (m/s)",
        "Humidity": "Suhteellinen kosteus (%)",
        "Dew point": "Kastepistelämpötila (degC)"
    })

    return weather

def parse_timestamps(df): 
    # print(df["time"].iloc[0].timetuple())
    df = df.assign(year=[x.timetuple().tm_year for x in df['time']])
    df = df.assign(yday=[x.timetuple().tm_yday for x in df['time']])
    df = df.assign(day=[x.timetuple().tm_mday for x in df['time']])
    df = df.assign(month=[x.timetuple().tm_mon for x in df['time']])
    df = df.assign(wday=[x.timetuple().tm_wday for x in df['time']])
    df = df.assign(hour=[x.timetuple().tm_hour for x in df['time']])
    df['week'] = df.apply(lambda row: row.time.isocalendar()[1], axis=1)

    
    # placeholder values (not done yet)
    df["id"] = np.repeat(1, len(df))
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

def get_start_end_times(time):
    now = datetime.datetime.utcnow()

    start_time = now
    end_time = start_time + datetime.timedelta(hours=time)
    start_time = start_time.isoformat(timespec="seconds") + "Z"
    end_time = end_time.isoformat(timespec="seconds") + "Z"

    return start_time, end_time


def get_station_names():
    file_names = []
    for file in os.listdir("./model_files/models/"):
        file = file.replace(".pkl", "").replace("model_", "")
        file_names.append(file)

    return file_names