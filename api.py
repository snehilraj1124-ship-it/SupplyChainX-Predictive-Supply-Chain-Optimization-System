from flask import Flask, request, jsonify
from src.predict import forecast_next_row

app = Flask(__name__)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "SupplyChainX",
        "version": "1.0"
    }


@app.post("/forecast")
def forecast():
    try:
        payload = request.get_json(force=True)

        if not payload:
            return jsonify({
                "error": "Request body is empty"
            }), 400

        forecast = forecast_next_row(payload)

        return jsonify({
            "status": "success",
            "forecasted_demand": round(forecast, 2)
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


@app.get("/")
def home():
    return {
        "project": "SupplyChainX",
        "description": "Predictive Supply Chain Optimization & Demand Intelligence System",
        "endpoints": {
            "health": "/health",
            "forecast": "/forecast"
        }
    }


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
