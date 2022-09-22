import botometer
import pandas as pd
import matplotlib.pyplot as plt


rapidapi_key = 'adba30d77bmshdcfbc478e5ccc58p18b171jsn5462eff7ccbb'
twitter_app_auth = {
    'consumer_key': 'w8CyF1ZjqyfZISDF1UnR97U4n',
    'consumer_secret': 'MJFsEhID9wnYAJCCfDRbDIme4jyA2tZXRW8InCdVD0teren72h',
}
bom = botometer.Botometer(wait_on_ratelimit=True,
                          rapidapi_key=rapidapi_key,
                          **twitter_app_auth)


def split_csv(input_path, chunksize):
    for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunksize)):
        chunk.to_csv(r'C:\Users\Rachel\Documents\twitter_tutorial_2022\chunk{}.csv'.format(i), index=False)


def create_csv(input_path, output_path_en, output_path_univ):
    chunk = input_path.split('\\')[-1]
    chunk = chunk.replace('chunk', '')
    chunk = chunk.replace('.csv', '')

    training_set = pd.read_csv(input_path)
    ids = training_set["id"]
    columns = ["id", "CAP", "astroturf", "fake follower", "financial", "other", "overall", "self-declared", "spammer",
               "label"]
    eng_df = pd.DataFrame(columns=columns)
    univ_df = pd.DataFrame(columns=columns)
    accounts_processed = 0

    for id, result in bom.check_accounts_in(ids):
        current_row = training_set.loc[training_set["id"] == id]
        account_type = current_row["type"].iloc[0]

        try:
            if account_type.lower() == "human":
                label = 0
            elif account_type.lower() == "bot":
                label = 1
            elif account_type.lower() == "organization":
                label = 2
            else:
                raise Exception("unknown type")

            if result["user"]["majority_lang"] == 'en':
                account_data = [[result["user"]["user_data"]["id_str"]],
                                [result['cap']['english']],
                                [result['display_scores']['english']['astroturf']],
                                [result['display_scores']['english']['fake_follower']],
                                [result['display_scores']['english']['financial']],
                                [result['display_scores']['english']['other']],
                                [result['display_scores']['english']['overall']],
                                [result['display_scores']['english']['self_declared']],
                                [result['display_scores']['english']['spammer']],
                                [label]]
                row = pd.DataFrame(dict(zip(columns, account_data)))
                eng_df = pd.concat([eng_df, row], ignore_index=True)

            else:
                account_data = [[result["user"]["user_data"]["id_str"]],
                                [result['cap']['universal']],
                                [result['display_scores']['universal']['astroturf']],
                                [result['display_scores']['universal']['fake_follower']],
                                [result['display_scores']['universal']['financial']],
                                [result['display_scores']['universal']['other']],
                                [result['display_scores']['universal']['overall']],
                                [result['display_scores']['universal']['self_declared']],
                                [result['display_scores']['universal']['spammer']],
                                [label]]
                row = pd.DataFrame(dict(zip(columns, account_data)))
                univ_df = pd.concat([univ_df, row], ignore_index=True)

            accounts_processed += 1
            print(f'{id} has been processed. ({accounts_processed}/{len(ids)} accounts processed)')

        except Exception as e:
            accounts_processed += 1
            print(f'{id} could not be fetched. ({accounts_processed}/{len(ids)} accounts processed)')

    output_path_en = output_path_en + r'\english{}.csv'.format(chunk)
    output_path_univ = output_path_univ + r'\universal{}.csv'.format(chunk)

    eng_df.to_csv(output_path_en, index=False)
    univ_df.to_csv(output_path_univ, index=False)

    return [eng_df, univ_df]


def merge_files(files):
    if 'english' in files[0].split('/')[-1]:
        lang = 'english'
    else:
        lang = 'universal'
    df = pd.concat(map(pd.read_csv, files), ignore_index=True)
    df.to_csv(r'C:\Users\Rachel\Documents\twitter_tutorial_2022\{}\{} merged.csv'.format(lang, lang), index=False)
    return df


def create_histograms(df, lang, color1=None, color2=None, color3=None):
    human_df = df[df["label"] == 0]
    human_df = human_df.loc[:, "astroturf":"spammer"].astype(float)
    bot_df = df[df["label"] == 1]
    bot_df = bot_df.loc[:, "astroturf":"spammer"].astype(float)
    org_df = df[df["label"] == 2]
    org_df = org_df.loc[:, "astroturf":"spammer"].astype(float)

    human_df.hist(figsize=(15, 12), color=color1)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f'Human ({lang})')
    plt.savefig(r'C:\Users\Rachel\Documents\twitter_tutorial_2022\{}\human_({})'.format(lang, lang))

    bot_df.hist(figsize=(15, 12), color=color2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f'Bot ({lang})')
    plt.savefig(r'C:\Users\Rachel\Documents\twitter_tutorial_2022\{}\bot_({})'.format(lang, lang))

    org_df.hist(figsize=(15, 12), color=color3)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f'Organization ({lang})')
    plt.savefig(r'C:\Users\Rachel\Documents\twitter_tutorial_2022\{}\org_({})'.format(lang, lang))

    plt.show()


input_csv = r'C:\Users\Rachel\Documents\twitter_tutorial_2022\marked_twitter_id2.csv'
chunks_eng = [r'C:\Users\Rachel\Documents\twitter_tutorial_2022\english\english{}.csv'.format(i) for i in range(4)]
chunks_univ = [r'C:\Users\Rachel\Documents\twitter_tutorial_2022\universal\universal{}.csv'.format(i) for i in range(4)]
output_eng = r'C:\Users\Rachel\Documents\twitter_tutorial_2022\english'
output_univ = r'C:\Users\Rachel\Documents\twitter_tutorial_2022\universal'

# split_csv(input_csv, 29)
# create_csv(r'C:\Users\Rachel\Documents\twitter_tutorial_2022\chunk0.csv', output_eng, output_univ)
# create_csv(r'C:\Users\Rachel\Documents\twitter_tutorial_2022\chunk1.csv', output_eng, output_univ)
# create_csv(r'C:\Users\Rachel\Documents\twitter_tutorial_2022\chunk2.csv', output_eng, output_univ)
# create_csv(r'C:\Users\Rachel\Documents\twitter_tutorial_2022\chunk3.csv', output_eng, output_univ)
# eng_merged = merge_files(chunks_eng)
# univ_merged = merge_files(chunks_univ)
# df_list = create_csv(input_csv, output_eng, output_univ)
# create_histograms(df_list[0], "english", "pink", "darkmagenta", "palevioletred")
# create_histograms(df_list[1], "universal", "orchid", "plum", "mediumvioletred")
