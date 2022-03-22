import pickle

with open('random_forest.pkl', 'rb') as fi:
  random_forest_model = pickle.load(fi)
with open('standard_scaler.pkl', 'rb') as si:
    scaler_model = pickle.load(si)

def standardScaler(matrixInput):
    scaledMatrix = scaler_model.transform(matrixInput)
    return scaledMatrix

def randomForestModel(scaledMatrix):
    predictedMatrix = random_forest_model.predict(scaledMatrix)
    return predictedMatrix