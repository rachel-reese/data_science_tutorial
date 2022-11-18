import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay

en_path_trimmed = r'C:\Users\Rachel\Documents\Twitter Data\eng_test_trimmed.csv'
un_path_trimmed = r'C:\Users\Rachel\Documents\Twitter Data\univ_test_trimmed.csv'

en_df_trimmed = pd.read_csv(en_path_trimmed)
un_df_trimmed = pd.read_csv(un_path_trimmed)


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


confusion_matrix(lang, en_df_trimmed, show_matrix)
confusion_matrix(lang, un_df_trimmed, show_matrix)

