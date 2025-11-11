from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import requests
from datetime import datetime, timedelta

app = Flask(__name__)

# Load trained RandomForest model
model = joblib.load("rf_model.pkl")

# Your OpenWeatherMap API Key
apiKey = "8b9939a6b97f3144527f5441a517aaa9"

def calculate_module_temp(ambient, irradiation):
    """Approximate module temperature based on ambient and irradiation."""
    return ambient + ((45 - 20) / 800) * irradiation

def get_weather(city=None, lat=None, lon=None):
    """Fetch weather forecast using OpenWeatherMap API."""
    if city:
        url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={apiKey}&units=metric"
    elif lat and lon:
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={apiKey}&units=metric"
    else:
        return None

    res = requests.get(url)
    if res.status_code != 200:
        return None

    data = res.json()
    today = datetime.utcnow().date()
    tomorrow = today + timedelta(days=1)

    today_forecast = next((item for item in data['list']
                           if item['dt_txt'].startswith(str(today)) and "12:00:00" in item['dt_txt']),
                          data['list'][0])
    tomorrow_forecast = next((item for item in data['list']
                              if item['dt_txt'].startswith(str(tomorrow)) and "12:00:00" in item['dt_txt']),
                             data['list'][0])

    return {
        "today_temp": today_forecast['main']['temp'],
        "tomorrow_temp": tomorrow_forecast['main']['temp'],
        "cloud": tomorrow_forecast['clouds']['all']
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_prediction", methods=["POST"])
def get_prediction():
    """Predict solar power generation, compare with actual, and apply penalty."""
    try:
        data = request.get_json()
        rows = int(data.get("rows", 1))
        cols = int(data.get("cols", 1))
        panel_size = rows * cols

        ambient = float(data.get("ambient"))
        irradiation = float(data.get("irradiation"))

        # Calculate module temperature
        module_temp = calculate_module_temp(ambient, irradiation)

        # Predict power using trained RandomForest model
        X = np.array([[ambient, module_temp, irradiation]])
        predicted_per_panel = model.predict(X)[0]  # kWh per 1 panel
        total_predicted = predicted_per_panel * panel_size

        # Estimate actual value (simplified assumption)
        # You can adjust the 0.9 factor to represent efficiency
        actual_value = (irradiation / 1000) * panel_size * 0.9  

        # Penalty based on deviation
        penalty_rate = 0.1  # 10% penalty on deviation
        penalty = abs(total_predicted - actual_value) * penalty_rate

        # Determine prediction status
        if total_predicted > actual_value:
            status = "Over Prediction"
        elif total_predicted < actual_value:
            status = "Under Prediction"
        else:
            status = "Accurate"

        return jsonify({
            "ambient_temp": round(ambient, 2),
            "module_temp": round(module_temp, 2),
            "irradiation": round(irradiation / 1000, 2),  # W/m² → kW/m²
            "panel_size": panel_size,
            "predicted": round(float(total_predicted), 2),
            "actual": round(float(actual_value), 2),
            "penalty": round(float(penalty), 2),
            "status": status
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/get_weather", methods=["POST"])
def get_weather_route():
    """Get weather data and estimate irradiation."""
    try:
        data = request.get_json()
        city = data.get("city")
        lat = data.get("lat")
        lon = data.get("lon")

        weather = get_weather(city=city, lat=lat, lon=lon)
        if not weather:
            return jsonify({"error": "Could not fetch weather data"}), 400

        # Estimate irradiation based on cloud cover
        irradiation = 1000 * (1 - weather['cloud'] / 100)  # W/m²

        return jsonify({
            "today_temp": round(weather['today_temp'], 1),
            "tomorrow_temp": round(weather['tomorrow_temp'], 1),
            "irradiation": irradiation
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
