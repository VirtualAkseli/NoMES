from flask import Flask
from fmiopendata.wfs import download_stored_query
import datetime
from .model_files import model_utils
import pickle

app = Flask('app')


@app.route('/', methods=['GET'])
def test():
    return 'Pinging Model Application!'


@app.route('/fmi', methods=['GET'])
def get_weather_fmi():
    now = datetime.datetime.utcnow()

    start_time = now - datetime.timedelta(hours=24)
    start_time = start_time.isoformat(timespec="seconds") + "Z"

    forecast = download_stored_query("fmi::forecast::hirlam::surface::obsstations::multipointcoverage", args=[f"starttime={start_time}"])

    station = "Kaisaniemi"
    cols = ["Air temperature", "Wind speed", "Wind direction", "Wind gust", "Humidity", "Dew point", "Precipitation amount 1 hour"]
    weather = model_utils.construct_weather_data(forecast, station, cols)
    print(weather.head())
    print(weather.tail())

    with open('./model_files/model.pkl', 'rb') as f:
        model = pickle.load(f)
        f.close()




    # preprocess ei vielä tee mitään
    X = model_utils.preprocess(weather)
    prediction = model_utils.predict(X, model)

    # ei näytä vielä mitään
    return prediction




if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
