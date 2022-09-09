import pickle
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

log_reg = r'C:\Users\Rachel\PycharmProjects\twitter_bot_tutorial\prep\log_reg_eng.dat'
rand_forest = r'C:\Users\Rachel\PycharmProjects\twitter_bot_tutorial\prep\rand_forest_eng.dat'
dataset = r'C:\Users\Rachel\Documents\Twitter Data\english_full.csv'


def run_model(model_path, data_path):
    model = pickle.load(open(model_path, 'rb'))
    data = pd.read_csv(data_path)
    x_test = data.drop(["id", "CAP", "spammer", "label"], axis=1).values
    y_test = data["label"].values
    x_test_scaled = preprocessing.scale(x_test)
    x_pred = model.predict(x_test_scaled)

    accuracy = accuracy_score(y_test, x_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))


run_model(log_reg, dataset)
run_model(rand_forest, dataset)
