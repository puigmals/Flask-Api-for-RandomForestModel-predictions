# -*- coding: utf-8 -*-

from flask import Flask, jsonify, request
from utils.random_forest_functions import randomForestModel, standardScaler

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        inputUserData = request.get_json()
        flowersDataList = []
        for flower in inputUserData:
            flowersDataList += [[*flower.values()]]
        scaler = standardScaler(flowersDataList)
        predictedMatrix = randomForestModel(scaler)
        return jsonify({ "result" : str(predictedMatrix) })
    except:
        return jsonify({ "result" : 0 })

if __name__ == "__main__":
    app.run(debug=True)