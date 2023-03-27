from flask import Flask, render_template, request
from PandC import *

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        if request.method == 'POST':
            mydict = request.form
            Model = str(mydict['Model'])
            Age = int(mydict['Age'])
            Sex = int(mydict['Sex'])
            RestingBP = int(mydict['RestingBP'])
            Cholesterol = int(mydict['Cholesterol'])
            FastingBS = int(mydict['FastingBS'])
            MaxHR = int(mydict['MaxHR'])
            ExerciseAngina = int(mydict['ExerciseAngina'])
            Oldpeak = float(mydict['Oldpeak'])
            ST_Slope = str(mydict['ST_Slope'])
            RestingECG = str(mydict['RestingECG'])
            ChestPainType = str(mydict['ChestPainType'])

            inputfeatures = [[Age, Sex, RestingBP, Cholesterol, FastingBS, MaxHR, ExerciseAngina, Oldpeak, ST_Slope,
                              RestingECG, ChestPainType]]
            ML_model = [[Model]]
            final_model = './' + Model + '.pkl'

            df = pd.DataFrame(inputfeatures,
                              columns=['Age', 'Sex', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'ExerciseAngina',
                                       'Oldpeak', 'ST_Slope',
                                       'RestingECG', 'ChestPainType'])
            model = HeartDiseasePrediction(final_model)
            model.load_clean_data(df)
            presicted_df = model.predicted_outputs()
            presicted_df.to_csv('Final_prediction.csv')
            result = int(presicted_df['Prediction'])

            # disease_result = {1: 'Have Heart Disease', 0: 'Dont Hve Heart Disease'}
            # final_result=disease_result[result]

            if result == 1:
                return render_template('final.html',disease=True)
            else:
                return render_template('final.html' )
    except:
        string='Please Enter the Values'
        return render_template('error.html',string=string)

    return render_template('index.html')


if __name__ == '__main__':
    app.run()
