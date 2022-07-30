import math

from sklearn.metrics import mean_squared_error
from flask import *
import matplotlib.pyplot as plt
import pandas as pd
# from model_prediction import *

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from IPython.display import HTML

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/files'


@app.route('/')
def upload():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        path = './static/files/' + f.filename
        f.save(path)

        def Data_format_conversion(engine_id):

            path = "./static/files/" + request.files['file'].filename
            df = pd.read_csv(path)

            df = df[df['ID'] == engine_id]
            df = df.drop(columns=['ID'])

            ################################## Scalling the DATA
            scaler = MinMaxScaler()
            df = scaler.fit_transform(df)
            print('Shape of df for engine {}: '.format(engine_id), df.shape)

            ################################    Getting into training shape with slidingwindow ###################################

            features = df[:, 0:-1]
            target = df[:, -1]
            if features.shape[0] <= 25:
                return 0


            ts_generator = TimeseriesGenerator(features, target, length=25, sampling_rate=1, batch_size=1)

            ################################ Changing the shape of input to (no of smaples,window_length,features)################################
            X = []
            y = []
            for i in range(len(ts_generator)):

                x_temp, y_temp = ts_generator[i]
                X.append(x_temp.reshape(x_temp.shape[1], x_temp.shape[2], 1))
                y.append(y_temp)

            X = np.array(X)
            y = np.array(y)

            return X, y, scaler, features

        df = pd.read_csv(path)

        a = df['ID'].max()

        # model = joblib.load(open('./model2.pkl', 'rb'))
        from tensorflow.keras.models import load_model

        # load model
        model = load_model('model.h5')
        table = pd.DataFrame()  ##### Data frame to store RUL

        for i in range(1, a):
            engine_id = i

            win_length = 25
            if  Data_format_conversion(engine_id):

                X, y, scaler, features = Data_format_conversion(engine_id)



                prediction = model.predict(X)  # prediction on trained data
                rev_trans = pd.concat([pd.DataFrame(features[win_length:]), pd.DataFrame(prediction)], axis=1)
                rev_trans = scaler.inverse_transform(rev_trans)  # Transforming back to original scale
                rev_trans = pd.DataFrame(rev_trans)
                df_actual = df[df['ID'] == engine_id]

                dict1 = {'Engine No.': [i], 'Actual RUL': [df_actual['RUL'].min()], 'Predicted RUL': [rev_trans[13].min()]}
                dict1 = pd.DataFrame(dict1)

                table = pd.concat([table, dict1], ignore_index=True)

                # print(table)
                #
                #
                # print('prediccted RUL is', rev_trans[13].min())
                # print('actual RUl is', df_actual['RUL'].min())

        html_code = table.to_html(classes='table table-stripped')
        text_file = open('Templates/table.html', "w")
        text_file.write(html_code)
        text_file.close()
    mse = mean_squared_error(table['Predicted RUL'], table['Actual RUL'])
    rmse = math.sqrt(mse)
    plt.plot(table['Engine No.'], table['Actual RUL'])
    plt.plot(table['Engine No.'], table['Predicted RUL'])
    plt.ylabel('RUL')
    plt.xlabel('Engine No.')
    plt.title('RMSE is {} on given test set'.format(rmse))
    plt.legend(['Actual', 'Prediction'], loc='upper right')
    plt.show()
    #plt.savefig('static/RULgraph.png')

    # rmse_dict={'Engine No.': [0], 'Actual RUL': ['RMSE is'], 'Predicted RUL': [rmse]}
    # table=pd.concat([table,rmse_dict])
    # print('rmse is:', rmse)

    return render_template('table.html')


if __name__ == '__main__':
    app.run(debug=True)
