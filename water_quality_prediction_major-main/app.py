import pickle
from flask import Flask, render_template, request
import numpy as np
import os

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        # request all the input fields
        ph = float(request.form['ph value'])
        Hardness = float(request.form['Hardness'])
        Solids = float(request.form['Solids'])
        Chloramines = float(request.form['Chloramines'])
        Sulfate = float(request.form['Sulfate'])
        Conductivity = float(request.form['Conductivity'])
        Organic_carbon = float(request.form['Organic carbon'])
        Trihalomethanes = float(request.form['Trihalomethanes'])
        Turbidity = float(request.form['Turbidity'])

        # create numpy array for all the inputs
        val = np.array([ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity])

        # define save model and scaler path
        model_path = os.path.join('models', 'xgboost.sav')
        scaler_path = os.path.join('models', 'scaler.sav')

        # load the model and scaler
        model = pickle.load(open(model_path, 'rb'))
        scc = pickle.load(open(scaler_path, 'rb'))

        # transform the input data using pre fitted standard scaler
        data = scc.transform([val])

        # make a prediction for the given data
        res = model.predict(data)

        # Weight Ratio For WQI Calculation
        
        ph_weight_rel = 0.19047619047619047619047619047619 # 4
        hardness_weight_rel = 0.0952380952380952380952380952381 # 2
        solids_weight_rel = 0.0952380952380952380952380952381 # 2
        chloramines_weight_rel = 0.14285714285714285714285714285714 # 3
        sulfate_weight_rel = 0.0952380952380952380952380952381 # 2
        conductivity_weight_rel = 0.0952380952380952380952380952381 # 2
        organic_carbon_weight_rel = 0.0952380952380952380952380952381 # 2
        trihalomethanes_weight_rel = 0.0952380952380952380952380952381 # 2
        turbidity_weight_rel = 0.0952380952380952380952380952381 # 2

    
        # water quality rating calculation for each parameter
        quality_rating_ph = 100 - 5 * (ph - 7)
        quality_rating_hardness = 100 - 5 * (Hardness - 50) / (450 - 50)
        quality_rating_solids = 100 - 5 * (Solids - 500) / (2000 - 500)
        quality_rating_chloramines = 100 - 5 * (Chloramines - 0.6) / (4.0 - 0.6)
        quality_rating_sulfate = 100 - 5 * (Sulfate - 200) / (400 - 200)
        quality_rating_conductivity = 100 - 5 * (Conductivity - 150) / (1500 - 150)
        quality_rating_organic_carbon = 100 - 5 * (Organic_carbon - 5) / (30 - 5)
        quality_rating_trihalomethanes = 100 - 5 * (Trihalomethanes - 20) / (80 - 20)
        quality_rating_turbidity = 100 - 5 * (Turbidity - 1) / (5 - 1)



        # Sub Index Calculation
        ph_sub_index = quality_rating_ph * ph_weight_rel
        hardness_sub_index = quality_rating_hardness * hardness_weight_rel
        solids_sub_index = quality_rating_solids * solids_weight_rel
        chloramines_sub_index = quality_rating_chloramines * chloramines_weight_rel
        sulfate_sub_index = quality_rating_sulfate * sulfate_weight_rel
        conductivity_sub_index = quality_rating_conductivity * conductivity_weight_rel
        organic_carbon_sub_index = quality_rating_organic_carbon * organic_carbon_weight_rel
        trihalomethanes_sub_index = quality_rating_trihalomethanes * trihalomethanes_weight_rel
        turbidity_sub_index = quality_rating_turbidity * turbidity_weight_rel


        # WQI Calculation
        WQI = ph_sub_index + hardness_sub_index + solids_sub_index + chloramines_sub_index + sulfate_sub_index + conductivity_sub_index + organic_carbon_sub_index + trihalomethanes_sub_index + turbidity_sub_index

        
        if res == 1:
            outcome = 'Drinkable'
        else:
            outcome = 'not Drinkable'
            WQI = WQI + 300
            
        if WQI >= 0 and WQI <= 50:
            wqi_class =  "Excellent"
        elif WQI > 50 and WQI <= 100:
            wqi_class = "Good"
        elif WQI > 100 and WQI <= 200:
            wqi_class = "Fair"
        elif WQI > 200 and WQI <= 300:
            wqi_class= "Poor"
        elif WQI > 300 and WQI <= 400:
            wqi_class = "Very Poor"
        else:
            wqi_class = "Unsatisfactory"


        r_value = f"Water is {outcome}. \n WQI Value: {WQI},  WQI Classification: {wqi_class}"    
        return render_template('index.html', result=r_value)
    return render_template('index.html')

# run application
if __name__ == "__main__":
    app.run(debug=True)
