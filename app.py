from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
from sklearn import metrics 
import warnings
import pickle
warnings.filterwarnings('ignore')
from feature import FeatureExtraction

file = open("model.pkl","rb")
gbc = pickle.load(file)
file.close()

app = Flask(__name__)

@app.route("/fraud", methods=["POST"])
def index():
    data = request.get_json()
    url = data['url']

    obj = FeatureExtraction(url)
    x = np.array(obj.getFeaturesList()).reshape(1,30) 

    y_pred =gbc.predict(x)[0]
    pred_values = gbc.predict(x)
    # print(pred_values[:10])
    #1 is safe       
    #-1 is unsafe
    y_pro_phishing = gbc.predict_proba(x)[0,0]
    y_pro_non_phishing = gbc.predict_proba(x)[0,1]
    if(y_pro_phishing*100 >= y_pro_non_phishing*100):
        pred = "It is {0:.2f} %  unsafe to go ".format(y_pro_phishing*100)
    else:
        pred = "It is {0:.2f} %  safe to go ".format(y_pro_non_phishing*100)
    # return jsonify({"phishing_probability": str(y_pro_phishing * 100), "non_phishing_probability": str(y_pro_non_phishing * 100)})
    # return pred
    return jsonify({"predictions": x.tolist()})


# if __name__ == "__main__":
#     app.run(debug=True)