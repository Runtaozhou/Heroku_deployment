import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open("xgb_model.pkl","rb"))

@app.route("/")
def Home():
    return render_template("form.html")

@app.route("/result",methods=['POST'])
def result():
    if request.method == 'POST':
        features = [num for num in request.form.values()]
        final_features = [[float(num) for num in features]]
        prediction = model.predict(final_features)
        types = ""
        if prediction[0] == 0:
            types = "Setosa"
        elif prediction[0] == 1:
            types = "Versicolour"
        else:
            types = "Virginica"

        return render_template('result.html',prediction = "based on your result, the Iris type is {}".format(types))

if __name__ == "__main__":
    app.run(debug = True)