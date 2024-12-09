import os
import pickle

from flask import Flask
from flask import render_template
from flask import request, jsonify, redirect

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def index_page():
    prediction = ""
    if request.method == "POST":
        level = request.form.get("level")
        lang = request.form.get("lang")
        tweets = request.form.get("tweets")
        phd = request.form.get("phd")
        prediction = predict_interviews_well([level, lang, tweets, phd])
    print("prediction:", prediction)
    # goes into templates folder and finds given name
    return render_template("index.html", prediction=prediction) 

# @app.route('/', methods = ['GET', 'POST'])
# def index_page():
#     prediction = ""
#     if request.method == "POST":
#         bmi = request.form.get("BMI")
#         smoking = request.form.get("Smoking")
#         pa = request.form.get("PhysicalActivity")
#         ev1 = request.form.get("LungFunctionFEV1")

#         instance = [bmi, smoking, pa, ev1]
#         prediction = my_random_forest.predict(instance)
#     print("prediction:", prediction)
#     # goes into templates folder and finds given name
#     return render_template("index.html", prediction=prediction) 

# lets add a route for the /predict endpoint
@app.route("/predict")
def predict():
    # lets parse the unseen instance values from the query string
    # they are in the request object
    bmi = request.args.get("BMI") # defaults to None
    smoking = request.args.get("Smoking")
    pa = request.args.get("PhysicalActivity")
    ev1 = request.args.get("LungFunctionFEV1")
    
    instance = [bmi, smoking, pa, ev1]
    # lets make a prediction!
    pred = predict_interviews_well.predict(instance)
    if pred is not None:
        return jsonify({"prediction": pred}), 200
    # something went wrong!!
    return "Error making a prediction", 400

# recursive
def tdidt_classifier(tree, header, instance):
    info_type = tree[0]
    if info_type == "Attribute":
        attribute_index = header.index(tree[1])
        test_value = instance[attribute_index]
        for i in range(2, len(tree)):
            value_list = tree[i]
            if value_list[1] == test_value:
                return tdidt_classifier(value_list[2], header, instance)
    else: # info_type == "Leaf"
        leaf_label = tree[1]
        return leaf_label

def predict_interviews_well(unseen_instance):
    # deserialize to object (unpickle)
    infile = open("tree.p", "rb")
    header, interview_tree = pickle.load(infile)
    infile.close()
    try:
        return tdidt_classifier(interview_tree, header, unseen_instance)
    except:
        return None

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port, debug=False)