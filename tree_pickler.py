import pickle # standard library
import myutils
import mypytable
import classifier


# pickling is used for object serialization and deserialization
# pickle: write a binary representation of an object to a file (for use later...)
# un/depickle: read a binary representation of an object from a file (to a python object in this memory)

# imagine you've just trained a MyDecisionTreeClassifier
# imagine you've just trained a MyRandomForestClassifier
# so you pickle the classifier object
# and unpickle later in your web app code

# let's do this with the interview tree from DecisionTreeFun
header = ["level", "lang", "tweets", "phd"]
interview_tree_solution =   ["Attribute", "level", 
                                ["Value", "Junior", 
                                    ["Attribute", "phd", 
                                        ["Value", "yes",
                                            ["Leaf", "False", 2, 5]
                                        ],
                                        ["Value", "no",
                                            ["Leaf", "True", 3, 5]
                                        ]
                                    ]
                                ],
                                ["Value", "Mid",
                                    ["Leaf", "True", 4, 14]
                                ],
                                ["Value", "Senior",
                                    ["Attribute", "tweets",
                                        ["Value", "yes",
                                            ["Leaf", "True", 2, 5]
                                        ],
                                        ["Value", "no",
                                            ["Leaf", "False", 3, 5]
                                        ]
                                    ]
                                ]
                            ]

asthma_header = ["BMI", "Smoking", "PhysicalActivity", "LungFunctionFEV1"]

# Now to generate asthma_trees
# Getting asthma data
table = myutils.load('asthma_disease_data.csv')
headers = table.pop(0)

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
asthma_data.column_names = headers[:-1]

# Extract features and target variable as lists
bmi = list(map(float, asthma_data.get_column('BMI')))  # Convert BMI to float
bmi = myutils.discretize_bmi(bmi)
smoking = [1 if x == 'yes' else 0 for x in asthma_data.get_column('Smoking')]  # Encode Smoking
physical_activity = [1 if x == 'yes' else 0 for x in asthma_data.get_column('PhysicalActivity')]  # Encode PhysicalActivity
physical_activity = myutils.discretize_physicalactivity(physical_activity)
lung_function = list(map(float, asthma_data.get_column('LungFunctionFEV1')))  # Convert LungFunction to float
lung_function = myutils.discretize_lungfunction(lung_function)
y_data = asthma_data.get_column('Diagnosis')  # Diagnosis is assumed to be numeric

# Combine features into a single list of features
X_data = list(zip(bmi, smoking, physical_activity, lung_function))  # Now each entry is numeric

# Fitting My Random Forest and getting tree
my_random_forest = classifier.MyRandomForestClassifier( N=6, M=1, F=4, seed=0)
my_random_forest.fit(X_data, y_data)

asthma_trees = my_random_forest.trees

# let's package the two lists together into one object
packaged_obj = [asthma_header, asthma_trees]
outfile = open("tree.p", "wb")
pickle.dump(packaged_obj, outfile)
outfile.close()