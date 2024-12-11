import os
import pickle
import myutils
import classifier
import mypytable

from flask import Flask
from flask import render_template
from flask import request, jsonify, redirect

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def index_page():
    prediction = ""
    if request.method == "POST":
        bmi = request.form.get("BMI")
        smoking = request.form.get("Smoking")
        pa = request.form.get("PhysicalActivity")
        ev1 = request.form.get("LungFunctionFEV1")

        instance = [bmi, smoking, pa, ev1]
        prediction = predict_asthma(instance)
    print("prediction:", prediction)
    # goes into templates folder and finds given name
    return render_template("index.html", prediction=prediction) 

# lets add a route for the /predict endpoint
@app.route("/predict")
def predict():
    # lets parse the unseen instance values from the query string
    # they are in the request object
    bmi = request.args.get("BMI") # defaults to None
    smoking = request.args.get("Smoking")
    pa = request.args.get("PhysicalActivity")
    ev1 = request.args.get("LungFunctionFEV1")
    
    # process to correct data types
    bmi = float(bmi)
    bmi = myutils.discretize_bmi([bmi])[0]
    pa = float(pa)
    pa = myutils.discretize_physicalactivity([pa])[0]
    ev1 = float(ev1)
    ev1 = myutils.discretize_lungfunction([ev1])[0]

    instance = [bmi, smoking, pa, ev1]
    print(instance)
    # lets make a prediction!
    pred = predict_asthma(instance)
    if pred is not None:
        return jsonify({"prediction": pred}), 200
    # something went wrong!!
    return "Error making a prediction", 400

# # classifier
# def tdidt_classifier(tree, header, instance):
#     info_type = tree[0]
    
#     if info_type == "Attribute":
#         attribute_index = header.index(tree[1])
#         test_value = instance[attribute_index]
        
#         for value_list in tree[2:]:
#             if value_list[1] == test_value:
#                 return tdidt_classifier(value_list[2], header, instance)
    
#     else:  # info_type == "Leaf"
#         leaf_label = tree[1]
#         return leaf_label

# # classifier
# def rf_classifier(trees, header, instance):
#     # Collect predictions from all trees
#     all_preds = []
    
#     for tree in trees:
#         # Use the tdidt_classifier to predict from each tree
#         prediction = tdidt_classifier(tree, header, instance)
#         all_preds.append(prediction)
    
#     # Majority vote for each instance
#     votes = Counter(all_preds)
#     majority_vote = votes.most_common(1)[0][0]  # Get the most common prediction
    
#     return majority_vote

def rf_predict(trees, header, instance):

    # Getting asthma data
    table = myutils.load('asthma_disease_data.csv')

    yes = []
    no = []

    for i in range(len(table)):
        if table[i][-1] == "1":
            yes.append(table[i])
        else:
            no.append(table[i])
        
    myutils.randomize_in_place(no)
    table = yes + no[:900]

    myutils.randomize_in_place(table)

    # creating a MyPyTable object and loading from file
    print("\nLoading data from asthma_disease_data.csv file")
    asthma_data = mypytable.MyPyTable()
    asthma_data.load_from_file("asthma_disease_data.csv")
    asthma_data.data = [row[:-1] for row in table]  # Removes last column from every row
    asthma_data.column_names = asthma_data.column_names[:-1]

    # Extract features and target variable as lists
    bmi = list(map(float, asthma_data.get_column('BMI')))  # Convert BMI to float
    bmi = myutils.discretize_bmi(bmi)
    smoking = asthma_data.get_column('Smoking')
    physical_activity = list(map(float, asthma_data.get_column('PhysicalActivity'))) # Encode PhysicalActivity
    physical_activity = myutils.discretize_physicalactivity(physical_activity)
    lung_function = list(map(float, asthma_data.get_column('LungFunctionFEV1')))  # Convert LungFunction to float
    lung_function = myutils.discretize_lungfunction(lung_function)
    y_data = asthma_data.get_column('Diagnosis')  # Diagnosis is assumed to be numeric

    # Combine features into a single list of features
    X_data = list(zip(bmi, smoking, physical_activity, lung_function))  # Now each entry is numeric

    # Fitting My Random Forest and getting tree
    my_random_forest = classifier.MyRandomForestClassifier( N=6, M=1, F=4, seed=0)
    my_random_forest.fit(X_data, y_data)
    # my_random_forest.trees = trees  # trees from fit gotten from pickled
    
    # Reshape instance into a 2D array
    instance = [instance]  # Now instance is a list of lists (2D array)
    y_pred_rf = my_random_forest.predict(instance)
    print(y_pred_rf)
    return y_pred_rf

def predict_asthma(unseen_instance):
    # Deserialize to object (unpickle)
    with open("tree.p", "rb") as infile:
        header, asthma_trees = pickle.load(infile)
        print(header)
        print(asthma_trees)
    
    try:
        # return rf_classifier(asthma_trees, header, unseen_instance)
        return rf_predict(asthma_trees, header, unseen_instance)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

if __name__ == "__main__":
    # Test outside Flask
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port, debug=False)