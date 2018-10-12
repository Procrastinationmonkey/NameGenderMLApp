from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap 
import pandas as pd 
import numpy as np 

# ML Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib


app = Flask(__name__)
Bootstrap(app)


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	dataframe= pd.read_csv("data/names.csv")
	
	# Features
	dfX = dataframe.name
	dfY = dataframe.sex
    
    # Vectorization
	c1 = dfX
	cv = CountVectorizer()
	X = cv.fit_transform(c1) 
	
	# ML Model
	naivebayes_model = open("C:\Users\Dell\Desktop\\naivebayesgendermodel.pkl","rb")
	
	clf = joblib.load(naivebayes_model)

	# Name check
	if request.method == 'POST':
		name1 = request.form['checkname']
		d = [name1]
		vect = cv.transform(d).toarray()
		my_prediction = clf.predict(vect)
	return render_template('results.html',prediction = my_prediction,name = name1.upper())


if __name__ == '__main__':
	app.run(debug=True)
