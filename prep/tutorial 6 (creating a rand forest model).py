import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn import model_selection, preprocessing
from xgboost import XGBClassifier

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


def random_forest(input_df, print_result=False):
    x = input_df.drop(["spammer", "label"], axis=1).values
    if "id" in input_df.columns:
        x = input_df.drop(["id", "spammer", "label"], axis=1).values
    y = input_df["label"].values
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.30, random_state=100)

    x_train_scaled = preprocessing.scale(x_train)

    model = XGBClassifier(use_label_encoder=False)
    optimization_dict = {
        'max_depth': [2],
        'n_estimators': [200],
        'learning_rate': [0.1],
    }
    model = model_selection.GridSearchCV(model, optimization_dict,
                                         scoring='accuracy', verbose=1)
    model.fit(x_train_scaled, y_train, eval_metric="error")

    pickle.dump(model, open(f'rand_forest_eng.dat', 'wb'))

    if print_result:
        print(model.best_score_)
        print(model.best_params_)
