import joblib
import pandas as pd
from flask import Flask,render_template,request


model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")


app = Flask(__name__)
@app.route("/",methods=["GET","POST"])
def index():
    if request.method == "POST":
        print(request.form)
        longitude = float(request.form["longitude"])
        latitude = float(request.form["latitude"])
        housing_median_age = float(request.form["housing_median_age"])
        households = float(request.form["households"])
        total_rooms = float(request.form["total_rooms"])
        total_bedrooms = float(request.form["total_bedrooms"])
        population = float(request.form["population"])
        median_income = float(request.form["median_income"])
        ocean_proximity = request.form["ocean_proximity"]
        

        # print(f"longitude :- {longitude}")
        # print(f"latitude :- {latitude}")
        # print(f"housing_median_age :- {housing_median_age}")
        # print(f"households :- {households}")
        # print(f"total_rooms :- {total_rooms}")
        # print(f"total_bedrooms :- {total_bedrooms}")
        # print(f"population :- {population}")
        # print(f"median_income :- {median_income}")

        data = pd.DataFrame([[longitude,latitude,housing_median_age,households,
                              total_rooms,total_bedrooms,population,median_income,ocean_proximity]],
                              columns=["longitude","latitude","housing_median_age","households",
                              "total_rooms","total_bedrooms","population","median_income","ocean_proximity"])
        transforemed_data =pipeline.transform(data)
        predictions = model.predict(transforemed_data)
        print(predictions)
        return render_template("output.html",Price=predictions)
    
    return render_template("index.html")

@app.route("/Predict",methods=["GET","POST"])
def predict():
    return render_template("output.html")

app.run(debug=True)