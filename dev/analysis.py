
def plot_nr_of_measurements_per_dt(df):
    measurements_per_dt = df.groupby(df.index).count()
    plt.figure()
    plt.plot(measurements_per_dt.index, measurements_per_dt['id'])
    plt.show()


def plot_summed_values(df):
    sum_per_dt = df.groupby(df.index).sum()
    print(sum_per_dt)
    plt.figure()
    plt.plot(sum_per_dt.index, sum_per_dt['value'])
    plt.show()


def plot_autocorrelations():
    data = DataProcessor.load_data_as_separate_dataframes()

    print('data loaded')
    avg_df = pd.concat(data, axis=1).mean(axis=1)
    avg_df.to_csv('total_average.csv')
    print('data merged')
    acf_values = acf(avg_df['value'].values[:3000], nlags=3000)
    print('values calculated')
    # acf_values = acf(d['value'].values[:350])[1:]
    plt.plot(range(len(acf_values)), acf_values)
    '''
    plt.figure(1)
    for d in data:
        acf_values = acf(d['value'].values[:192], nlags=192)
        # acf_values = acf(d['value'].values[:350])[1:]
        plt.plot(range(len(acf_values)), acf_values)
    '''
    plt.show()


def load_initial_consumers():
    original_data = DataProcessor.load_data_as_separate_dataframes()
    data = pd.concat(original_data, axis=1)
    data = data.fillna(method='ffill', axis='index').fillna(0).values.transpose()

    cluster_size = 25
    kmeans = KMeans(n_clusters=cluster_size).fit(data)
    plt.figure(1)
    for i in range(cluster_size):
        plt.subplot(cluster_size / 5, cluster_size / (cluster_size / 5), i + 1)
        for idx, d in enumerate(original_data):
            if kmeans.labels_[idx] == i:
                plt.plot(range(len(d)), d)
    plt.show()

