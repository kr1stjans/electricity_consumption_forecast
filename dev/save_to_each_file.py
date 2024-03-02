# save to each file
'''
  existing_files = os.listdir('../data/consumers')

    for consumer in set(complete_df.index):

        if consumer + '.csv' in existing_files:
            print("skipping", consumer)
            continue

        df = complete_df.loc[complete_df.index == consumer]

        df = df.set_index(drop=True, keys='dt')

        # fill missing indexes
        df = df.asfreq(pd.offsets.Minute(30))

        # fill missing values
        df = df.fillna(method='pad', axis='index')

        # result must have no nulls or duplicated datetimes
        assert df.isnull().values.any() == False
        assert df.index.duplicated().any() == False

        df.to_csv('../data/consumers/' + consumer + '.csv')

        data_frames.append(df)

    exit(1)
'''