from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
import pickle 

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

ENV = 'dev'

if ENV == 'dev':
    app.debug = True
    app.config['SQLALCHEMY_DATABASE_URI'] = ''
else:
    app.debug = False
    app.config['SQLALCHEMY_DATABASE_URI'] = ''

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class Models(db.Model):
    __tablename__ = 'models'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50))
    model = db.Column(db.PickleType)

# -- populate db with the rf model we made --

# pickle_in = open("heart_rf.pickle","rb")
# rf_in = pickle.load(pickle_in)
# name_in = 'RandomForestHeart'
# data = Models(name=name_in, model=rf_in)
# db.session.add(data)
# db.session.commit()

# -- end --

@app.route('/', methods=['GET'])
def index():
    rf = db.session.query(Models).filter_by(name='RandomForestHeart').first()
    model = rf.model
    body_json = request.json
    if body_json is not None:
        pred = model.predict([[body_json['Age'],body_json['Sex'],body_json['RestingBP'],body_json['Cholesterol'],body_json['FastingBS'],body_json['MaxHR'],body_json['ExerciseAngina'],body_json['Oldpeak'],body_json['ATA'],body_json['NAP'],body_json['TA'],body_json['Normal'],body_json['ST'],body_json['Flat'],body_json['Up']]])
        retjson = {
            'Heart Disease Prediction':pred
        }
    else:
        return jsonify({
            'error':'no body was provided.'
        })
    return jsonify(retjson)

if __name__ == '__main__':
    app.run()
