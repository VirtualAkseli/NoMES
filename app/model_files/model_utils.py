import pandas as pd


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
    # interpolate for NaN values
    # timestamps to correct format
    return weather


def predict(config, model):
    if isinstance(config, dict):
        config = pd.DataFrame(config)
    # else:
    #     df = config

    return model.predict(config)