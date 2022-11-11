import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay

input_path_en = r'C:\Users\Rachel\Documents\Twitter Data\english_full.csv'
input_path_un = r'C:\Users\Rachel\Documents\Twitter Data\universal_full.csv'
en_path_trimmed = r'C:\Users\Rachel\Documents\Twitter Data\eng_test_trimmed.csv'
un_path_trimmed = r'C:\Users\Rachel\Documents\Twitter Data\univ_test_trimmed.csv'

def orgs_to_bots(input_path):
    df = pd.read_csv(input_path)
    df = df.replace({'label': 2}, {'label': 1})
    return df


def find_zscore(df):
    columns = list(df.columns)
    new_df = pd.DataFrame()
    for col in columns:
        new_col = (df[col] - df[col].mean()) / df[col].std(ddof=0)
        new_df[col] = new_col
    return new_df


def remove_outliers(zscore, input_df, output_path=None):
    human_df = input_df[input_df['label'] == 0]
    bot_df = input_df[input_df['label'] == 1]

    human_df = human_df[np.abs(find_zscore(human_df.loc[:, "astroturf":"self-declared"]) < zscore).all(axis=1)]
    bot_df = bot_df[np.abs(find_zscore(bot_df.loc[:, "astroturf":"self-declared"]) < zscore).all(axis=1)]

    output_df = pd.concat([human_df, bot_df])
    output_df = output_df.loc[:, "astroturf":]

    if output_path is None:
        return output_df
    else:
        output_df.to_csv(output_path)
        return output_df


def logistic_regression(input_df, print_result=False):
    x = input_df.drop(["spammer", "label"], axis=1).values
    if "id" in input_df.columns:
        x = input_df.drop(["id", "CAP", "spammer", "label"], axis=1).values
    y = input_df["label"].values
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.30, random_state=100)

    x_train_scaled = preprocessing.scale(x_train)
    x_test_scaled = preprocessing.scale(x_test)

    model = LogisticRegression()
    model.fit(x_train_scaled, y_train)
    result = model.score(x_test_scaled, y_test)

    pickle.dump(model, open(f'log_reg_eng.dat', 'wb'))

    if print_result:
        print("Accuracy: %.2f%%" % (result * 100.0))
    return result * 100


def best_zscore(zscore, lang, input_path, show=False):
    input_df = orgs_to_bots(input_path)
    zscores = []
    accuracies = []

    while zscore >= 0:
        zscores.append(zscore)
        df = remove_outliers(zscore, input_df)
        result = logistic_regression(df)
        accuracies.append(float("{:.2f}".format(result)))
        zscore -= 0.1

    plt.plot(zscores, accuracies)
    plt.xlabel('zscore')
    plt.ylabel('accuracy (%)')
    plt.suptitle(f"Zscore and Accuracy ({lang})")

    plt.savefig(fr'C:\Users\Rachel\Documents\Twitter Data\zscore_{lang}_test')
    if show:
        plt.show()
