import matplotlib.pyplot as plt
from sklearn import linear_model
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.animation as manimation


def get_weekday(weekday):
    if weekday == 0:
        return "Monday"
    elif weekday == 1:
        return "Tuesday"
    elif weekday == 2:
        return "Wednesday"
    elif weekday == 3:
        return "Thursday"
    elif weekday == 4:
        return "Friday"
    elif weekday == 5:
        return "Saturday"
    elif weekday == 6:
        return "Sunday"
    return ""


if __name__ == "__main__":

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
    writer = FFMpegWriter(fps=1, metadata=metadata)

    mm_ids = [
        "889", "311",
        "498"]  # ["1", "2", "14", "26", "30", "45", "49", "68", "79", "82", "94", "149", "176", "222", "267", "323", "369",
    # "421", "483", "505", "556", "600", "633", "762", "913", "992", "1185", "1313", "1388", "1678", "1824", "2092",
    # "2240", "2326", "2801", "3085", "3206", "3497", "4167", "4743", "5835", "6970", "7858", "8892", "9433",
    # "10240", "10809", "11449"]

    for mm_id in mm_ids:

        x_values, y_values, dates = DataManager().get_test_data(mm_id)

        avg_rmse = 0
        avg_variance = 0
        avg_abs_error = 0
        cnt = 0

        predictions = []
        fig = plt.figure()
        with writer.saving(fig, "testing/" + str(mm_id) + "_dummy.mp4", 100):
            for moving_window in range(35327, len(x_values) - 192, 96):
                print('Cross validating @', moving_window, "measurement place", mm_id)

                x_train = x_values[:moving_window]
                y_train = y_values[:moving_window]

                x_test = x_values[moving_window:moving_window + 192]
                y_test = y_values[moving_window:moving_window + 192]

                regr = linear_model.LinearRegression()
                regr.fit(x_train, y_train)
                y_predicted = regr.predict(x_test)

                predictions.extend(y_predicted)

                rmse = mean_squared_error(y_test, y_predicted) ** 0.5
                abs_error = mean_absolute_error(y_test, y_predicted)
                variance = regr.score(x_test, y_test)

                avg_rmse += rmse
                avg_abs_error += abs_error
                avg_variance += variance

                actual, = plt.plot(range(len(y_test)), y_test, color='red', label='actual')
                linear_regression_predicted, = plt.plot(range(len(y_test)), y_predicted, color='blue', label='linear')

                plt.legend([actual, linear_regression_predicted],
                           ['Actual', 'Linear'])
                date = datetime.strptime(dates[moving_window], '%Y-%m-%d %H:%M:%S')

                title = str(datetime.strftime(date, '%d.%m.%Y')) + " (" + get_weekday(
                    date.weekday()) + ")" + "\nRMSE: " + str("%.2f" % rmse) + " ABS: " + str(
                    "%.2f" % abs_error) + " R2: " + str("%.2f" % variance)
                plt.title(title)

                writer.grab_frame()
                plt.clf()
                cnt += 1

        total_rmse = avg_rmse / cnt
        print('avg rmse', total_rmse)
        total_abs_error = avg_abs_error / cnt
        print('avg abs error', total_abs_error)
        total_variance = avg_variance / cnt
        print('avg variance', total_variance)

        # update_query = "INSERT INTO Forecast (mm_id, model_type, rmse, variance, r2) VALUES (" + str(
        #        mm_id) + "," + "'kfold'" + "," + str(total_rmse) + "," + str(total_variance) + "," + str(total_t2) + ")"
        # cur.execute(update_query)
        # conn.commit()
