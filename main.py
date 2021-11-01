from flask import Flask, render_template, request
from fmiopendata.wfs import download_stored_query
import datetime
from model_files import model_utils
import pickle
import pandas as pd
import matplotlib.pyplot as plt

app = Flask('app')
# app.config['EXPLAIN_TEMPLATE_LOADING'] = True


@app.route('/', methods=['GET'])
def test():
    return 'Pinging Model Application!'


@app.route('/predict', methods=['GET'])
def form():
    return render_template("form.html")


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

    with open(f'model_files/models/model_unioninkatu.pkl', 'rb') as f:
    
        print("loading model...")
        model = pickle.load(f)
        print("loaded model with features ", model.feature_names)

        f.close()

    X = model_utils.preprocess(weather)
    prediction = model_utils.predict(X, model)

    result = pd.DataFrame({"predicted": prediction, "time": weather["time"]})
    result.plot(x="time")
    plt.savefig(f"static/{station_name}.png")

    print(result)
    # return dict(result.predicted)
    return render_template("result.html", img = f"static/{station_name}.png", name=station_name)



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
