def confusion_matrix(lang, input_df, show=False):
    x = input_df.drop(["spammer", "label"], axis=1).values
    if "id" in input_df.columns:
        x = input_df.drop(["id", "spammer", "label"], axis=1).values
    y = input_df["label"].values
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.30, random_state=100)

    x_train_scaled = preprocessing.scale(x_train)
    x_test_scaled = preprocessing.scale(x_test)

    model = XGBClassifier(use_label_encoder=False)
    optimization_dict = {
        'max_depth': [2],
        'n_estimators': [200],
        'learning_rate': [0.1],
    }
    model = model_selection.GridSearchCV(model, optimization_dict,
                                         scoring='accuracy', verbose=1)
    model.fit(x_train_scaled, y_train, eval_metric="error")

    title_options = [(f"Confusion Matrix, without normalization\n({lang})", None),
                     (f"Normalization: true\n({lang})", "true"),
                     (f"Normalization: pred\n({lang})", "pred"),
                     (f"Normalization: all\n({lang})", "all")]

    for title, normalize in title_options:
        disp = ConfusionMatrixDisplay.from_estimator(model, x_test_scaled, y_test, display_labels=["bot", "human"],
                                                     cmap=plt.cm.Blues, normalize=normalize)
        disp.ax_.set_title(title)
        # plt.savefig(fr'C:\Users\Rachel\Documents\Twitter Data\{normalize}_{lang}_test')

    if show:
        plt.show()


dff = orgs_to_bots(input_path_en)
confusion_matrix("eng", remove_outliers(1.7, dff), True)

