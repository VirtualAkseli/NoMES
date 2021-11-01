from flask import Flask, render_template, request
from fmiopendata.wfs import download_stored_query
import datetime
from model_files import model_utils
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os

import model_files

app = Flask('app')
# app.config['EXPLAIN_TEMPLATE_LOADING'] = True


@app.route('/', methods=['GET'])
def test():
    return 'Pinging Model Application!'


@app.route('/predict', methods=['GET'])
def form():

    file_names = []
    for file in os.listdir("./model_files/models/"):
        file = file.replace(".pkl", "").replace("model_", "")
        file_names.append(file)

    return render_template("form.html", file_names=file_names)


# @app.route("/result", methods=["POST"])
# def result():
#     return render_template("result.html", name=request.form["name"])


@app.route('/result', methods=['GET', 'POST'])
def get_weather_fmi():
    station_name = request.form["station"] 
    print(station_name)
    now = datetime.datetime.utcnow()

    start_time = now - datetime.timedelta(hours=24)
    start_time = start_time.isoformat(timespec="seconds") + "Z"

    forecast = download_stored_query("fmi::forecast::hirlam::surface::obsstations::multipointcoverage", args=[f"starttime={start_time}"])

    station = "Kaisaniemi"
    cols = ["Air temperature", "Wind speed", "Wind direction", "Wind gust", "Humidity", "Dew point"]
    weather = model_utils.construct_weather_data(forecast, station, cols)
    print(weather.head())

    with open(f'model_files/models/model_{station_name}.pkl', 'rb') as f:
    
        print("loading model...")
        model = pickle.load(f)
        print("loaded model with features ", model.feature_names)

        f.close()

    X = model_utils.preprocess(weather)
    prediction = model_utils.predict(X, model)

    result = pd.DataFrame({"predicted": prediction, "time": weather["time"]})

    result.plot(x="time", figsize=(15,5))
    plt.savefig(f"static/{station_name}.png")

    print(result)

    counts = model_utils.count_changes(result)
    drops, min_bikes = model_utils.count_drops(result)

    print(counts, drops, min_bikes)

    # return dict(result.predicted)
    return render_template("result.html", img = f"static/{station_name}.png", name=station_name, changes=counts, drops_to_min = drops, min_bikes = min_bikes)



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
