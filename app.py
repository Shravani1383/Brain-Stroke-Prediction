from flask import Flask, render_template, request
import joblib
import os
import  numpy as np
from numpy.random import randint, choice
import pickle
import math
from sklearn.preprocessing import StandardScaler
num_disease = []

app= Flask(__name__)

@app.route("/")
def index():
    return render_template("home.html")

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# Load the scaler
fp = open("scaler.bin", "rb")
scaler = pickle.load(fp)
fp.close()

# Load LinearSVM
fp = open("scaler.bin", "rb")
model1 = pickle.load(fp)
fp.close()

# Load LogisticRegression
fp = open("LogisticRegression.bin", "rb")
model2 = pickle.load(fp)
fp.close()

@app.route("/result",methods=['POST','GET'])
def result():
    disease = []
    gender=int(request.form['gender'])
    num_disease.append(0)
    age=int(request.form['age'])
    num_disease.append(1)
    hypertension=int(request.form['hypertension'])
    if hypertension == 1:
        disease.append("HYPERTENSION")
        num_disease.append(2)
    heart_disease = int(request.form['heart_disease'])
    if heart_disease == 1:
        disease.append('HEART DISEASE')  
        num_disease.append(3)
        
    ever_married = int(request.form['ever_married'])
    num_disease.append(4)
    work_type = int(request.form['work_type'])
    num_disease.append(5)
    Residence_type = int(request.form['Residence_type'])
    num_disease.append(6)
    avg_glucose_level = float(request.form['avg_glucose_level'])
    if avg_glucose_level > 199:
        disease.append("DIABETES")
        num_disease.append(7)
    
    bmi = float(request.form['bmi'])
    if bmi > 30 and bmi < 35:
        disease.append("OBESITY I")
        num_disease.append(8)
    elif bmi > 35 and bmi < 40:
        disease.append("OBESITY II")
        num_disease.append(8)
    elif bmi > 40:
        disease.append("OBESITY III")
        num_disease.append(8)
    
    
    smoking_status = int(request.form['smoking_status'])
    if smoking_status == 3:
        disease.append("SMOKING")
        num_disease.append(9)
    

    x=np.array([gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,
                avg_glucose_level,bmi,smoking_status]).reshape(1,-1)

    scaler_path=os.path.join('D:/Python37/Projects/Stroke Prediction','models/scaler.pkl')
    scaler=None
    with open(scaler_path,'rb') as scaler_file:
        scaler=pickle.load(scaler_file)

    x=scaler.transform(x)

    model_path=os.path.join('D:/Python37/Projects/Stroke Prediction','models/dt.sav')
    dt=joblib.load(model_path)

    Y_pred=dt.predict(x)

    # for No Stroke Risk
    if Y_pred==0:
        return render_template('nostroke.html')
    else:
        pred=1
        return render_template('stroke.html',prediction_text=pred, dis=",".join(disease))
    
@app.route('/index',methods=['POST','GET'])
def home():
    return render_template('index.html')
    
@app.route('/predict',methods=['POST','GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [[float(x) for x in request.form.values()]]
    final_features = scaler.transform(int_features)
    #output = model.predict(final_features)
    # Now convert linear svm's prediction to probability
    out1 = sigmoid(model1.decision_function(final_features))
    # Get prediction probability from logistic regression
    out2 = model2.predict_proba(final_features)[:,1]

    # Their average is the final output probability
    final_output = np.mean((out1, out2), axis=0)
    #output = prediction.reshape(-1, 1)
    output=(str(round(final_output[0]*100, 2)))
    return render_template('index.html', prediction_text='Severity of brain stroke {}%'.format(output))  

@app.route('/expert',methods=['POST','GET'])
def expert():
    return render_template('expert-review.html', judul="Expert Review")
    
@app.route('/expert_review',methods=['POST','GET'])
def expert_review():
    PHASE,HEALTH,EXERCISE = "","",""
    result=[]
    exercise=[]
    FASE = request.form["phase"]
    if FASE == "PHASE 1":
        result.append("Aerobic exercises like brisk walking, cycling, swimming, or dancing can help improve blood flow and lower blood pressure, which are key factors in preventing stroke.")
        exercise.append(0)
        
    if FASE == "PHASE 2":
        result.append("Resistance training exercises like lifting weights or using resistance bands can help improve muscle strength, which can improve overall health and reduce the risk of stroke.")
        exercise.append(1)
        
    if FASE == "PHASE 3":
        result.append("Yoga- It is a form of exercise that combines physical postures, breathing techniques, and meditation. It can help lower stress levels, improve flexibility and balance, and reduce the risk of stroke.")
        exercise.append(2)
        
    if FASE == "PHASE 4":
        result.append("Tai chi- It is a low-impact exercise that involves slow, gentle movements and breathing techniques. It can improve balance, flexibility, and coordination, and reduce the risk of stroke.")
        exercise.append(3)
    
    if FASE == "PHASE 5":
        result.append("Certain brain exercises like puzzles, reading, and playing games can help improve cognitive function and reduce the risk of stroke..")
        exercise.append(4)
        
    if FASE == "PHASE 6":
        result.append("Swimming is a low-impact exercise that can improve cardiovascular health, reduce stress, and lower the risk of stroke.")
        exercise.append(5)
        
    if FASE == "PHASE 7":
        result.append("Cycling is another low-impact exercise that can improve cardiovascular health, reduce stress, and lower the risk of stroke.")
        exercise.append(6)
        
    if FASE == "PHASE 8":
        result.append("Pilates is a form of exercise that focuses on core strength, flexibility, and balance. It can improve overall health and reduce the risk of stroke.")
        exercise.append(7)
        
    if FASE == "PHASE 9":
        result.append("you have a moderate risk of stroke, you have to exercises that improve balance and coordination, such as tai chi or yoga, can be helpful. These exercises can also help reduce stress levels, which is important in preventing stroke.")
        exercise.append(8)
        
    if FASE == "PHASE 10":
        result.append("you have a high risk of stroke so your exercise routine may need to be modified to accommodate your physical limitations. Walking or swimming may be good options, as they are low-impact exercises that can be done at your own pace. Physical therapy may also be recommended to help improve mobility and strength.")
        exercise.append(9)
        
    PROGRESS =  int(request.form.get("proses", False))
    if PROGRESS > 70 :
        PHASE = FASE.split(' ')[0] +" "+str(int(FASE.split(" ")[1])+1)
    else:
        PHASE = FASE
    HEALTH = choice([str(x)+"%" for x in range(30,90,10)])
    
    pred=result
    
    return render_template("expert-review.html", prediction_text=pred, dis=",".join(result), ph=PHASE, health=HEALTH,)
    
if __name__=="__main__":
    app.run(debug=True,port=7384)