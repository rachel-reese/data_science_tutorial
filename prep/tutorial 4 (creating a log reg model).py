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

def confusion_matrix(lang, input_df, show=False):
    x = input_df.drop(["spammer", "label"], axis=1).values
    if "id" in input_df.columns:
        x = input_df.drop(["id", "CAP", "spammer", "label"], axis=1).values
    y = input_df["label"].values

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.30, random_state=100)

    x_train_scaled = preprocessing.scale(x_train)
    x_test_scaled = preprocessing.scale(x_test)

    model = LogisticRegression()
    model.fit(x_train_scaled, y_train)

    title_options = [(f"Confusion Matrix, without normalization\n({lang})", None),
                     (f"Normalization: true\n({lang})", "true"),
                     (f"Normalization: pred\n({lang})", "pred"),
                     (f"Normalization: all\n({lang})", "all")]

    for title, normalize in title_options:
        disp = ConfusionMatrixDisplay.from_estimator(model, x_test_scaled, y_test, display_labels=["bot", "human"],
                                                     cmap=plt.cm.Blues, normalize=normalize)
        disp.ax_.set_title(title)
        plt.savefig(fr'C:\Users\Rachel\Documents\Twitter Data\{normalize}_{lang}_test')

    if show:
        plt.show()


def main(zscore, lang, input_path, output_path, show_matrix=False, print_result=False,):
    df = orgs_to_bots(input_path)
    df_trimmed = remove_outliers(zscore, df, output_path)
    logistic_regression(df_trimmed, print_result)
    confusion_matrix(lang, df_trimmed, show_matrix)


best_zscore(3, "en", input_path_en, True)
best_zscore(3, "un", input_path_un, True)
main(1.7, "en", input_path_en, en_path_trimmed)
main(1.4, "un", input_path_un, un_path_trimmed, True, True)
