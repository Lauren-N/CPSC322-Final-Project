# pylint: skip-file

"""
Lauren Chin and Lauren Nguyen
CPSC 322, Fall 2024
Final Project
12/11/2024

Description: This file is used to create unit tests for classifiers.
This file imports and uses python packages to help with these unit test
values we are comparing to. This is to implement our test-driven-development.
"""

# copied my test_myclassifiers.py solution from PA4-6 here

import numpy as np
import myutils

from classifier import MyKNeighborsClassifier

from classifier import MyNaiveBayesClassifier

from classifier import MyRandomForestClassifier

# Bramer test sets
# from in-class #1  (4 instances)
X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
y_train_class_example1 = ["bad", "bad", "good", "good"]

# from in-class #2 (8 instances)
# assume normalized
X_train_class_example2 = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]]

y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]

# from Bramer
header_bramer_example = ["Attribute 1", "Attribute 2"]
X_train_bramer_example = [
    [0.8, 6.3],
    [1.4, 8.1],
    [2.1, 7.4],
    [2.6, 14.3],
    [6.8, 12.6],
    [8.8, 9.8],
    [9.2, 11.6],
    [10.8, 9.6],
    [11.8, 9.9],
    [12.4, 6.5],
    [12.8, 1.1],
    [14.0, 19.9],
    [14.2, 18.5],
    [15.6, 17.4],
    [15.8, 12.2],
    [16.6, 6.7],
    [17.4, 4.5],
    [18.2, 6.9],
    [19.0, 3.4],
    [19.6, 11.1]]

y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
           "-", "-", "+", "+", "+", "-", "+"]

def test_kneighbors_classifier_kneighbors():
    # Test case using the in-class 4 instance example
    knn = MyKNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_class_example1, y_train_class_example1)
    
    unseen_instance = [0.33, 1] # Our unseen instance from class
    distances, indices = knn.kneighbors([unseen_instance])
    expected_indices = [[0, 2, 3]]  # Nearest indices: [0,2,3]
    expected_distances = [[0.67, 1.0, 1.053]]  # Closest distances

    assert indices == expected_indices, "Indices do not match expected neighbors"
    assert np.allclose(distances, expected_distances, rtol=1e-4), "Distances do not match expected values"

    # Test case using the in-class 8 instance example
    knn.fit(X_train_class_example2, y_train_class_example2)

    unseen_instance = [2,3]
    distances, indices = knn.kneighbors([unseen_instance])
    expected_indices_2 = [[0, 4, 6]]  # Nearest indices from class
    expected_distances_2 = [[1.4142, 1.4142, 2.0]] # Closest distances

    assert indices == expected_indices_2, "Indices for 8 instance example do not match"
    assert np.allclose(distances, expected_distances_2, rtol=1e-4), "Distances for 8 instance example do not match"

    # Test case using Bramer example
    knn = MyKNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_bramer_example, y_train_bramer_example)

    unseen_instance = [9.1, 11]
    distances, indices = knn.kneighbors([unseen_instance])
    expected_indices_bramer = [[6, 5, 7, 4, 8]]  # Nearest indices based on Appendix E answer key
    expected_distances_bramer = [[0.608, 1.237, 2.202, 2.802, 2.915]]

    assert indices == expected_indices_bramer, "Indices for Bramer example do not match"
    assert np.allclose(distances, expected_distances_bramer, rtol=1e-3), "Distances for Bramer example do not match"

def test_kneighbors_classifier_predict():
    # Test case using the in-class 4 instance example
    knn = MyKNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_class_example1, y_train_class_example1)

    unseen_instance = [0.33, 1]
    predictions = knn.predict([unseen_instance])  # Test point
    expected_predictions = ["good"]  # Expected based on nearest neighbors
    assert list(predictions) == expected_predictions, "Predictions for in-class 4 instance example do not match"

    # Test case using the in-class 8 instance example
    knn.fit(X_train_class_example2, y_train_class_example2)

    unseen_instance = [2,3]
    predictions_2 = knn.predict([unseen_instance])
    expected_predictions_2 = ["yes"]  # Expected based on nearest neighbors (no, yes, yes)
    assert list(predictions_2) == expected_predictions_2, "Predictions for 8 instance example do not match"

    # Test case using Bramer example
    knn = MyKNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_bramer_example, y_train_bramer_example)

    unseen_instance = [9.1, 11]
    predictions_bramer = knn.predict([unseen_instance])
    expected_predictions_bramer = ["+"]  # Expected based on nearest neighbors ("-", "+", "-", "+", "+")
    assert list(predictions_bramer) == expected_predictions_bramer, "Predictions for Bramer example do not match"


# in-class Naive Bayes example (lab task #1)
header_inclass_example = ["att1", "att2"]
X_train_inclass_example = [
    [1, 5], # yes
    [2, 6], # yes
    [1, 5], # no
    [1, 5], # no
    [1, 6], # yes
    [2, 6], # no
    [1, 5], # yes
    [1, 6] # yes
]
y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]

# MA7 (fake) iPhone purchases dataset
header_iphone = ["standing", "job_status", "credit_rating", "buys_iphone"]
X_train_iphone = [
    [1, 3, "fair"],
    [1, 3, "excellent"],
    [2, 3, "fair"],
    [2, 2, "fair"],
    [2, 1, "fair"],
    [2, 1, "excellent"],
    [2, 1, "excellent"],
    [1, 2, "fair"],
    [1, 1, "fair"],
    [2, 2, "fair"],
    [1, 2, "excellent"],
    [2, 2, "excellent"],
    [2, 3, "fair"],
    [2, 2, "excellent"],
    [2, 3, "fair"]
]
y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

# Bramer 3.2 train dataset
header_train = ["day", "season", "wind", "rain", "class"]
X_train_train = [
    ["weekday", "spring", "none", "none"],
    ["weekday", "winter", "none", "slight"],
    ["weekday", "winter", "none", "slight"],
    ["weekday", "winter", "high", "heavy"],
    ["saturday", "summer", "normal", "none"],
    ["weekday", "autumn", "normal", "none"],
    ["holiday", "summer", "high", "slight"],
    ["sunday", "summer", "normal", "none"],
    ["weekday", "winter", "high", "heavy"],
    ["weekday", "summer", "none", "slight"],
    ["saturday", "spring", "high", "heavy"],
    ["weekday", "summer", "high", "slight"],
    ["saturday", "winter", "normal", "none"],
    ["weekday", "summer", "high", "none"],
    ["weekday", "winter", "normal", "heavy"],
    ["saturday", "autumn", "high", "slight"],
    ["weekday", "autumn", "none", "heavy"],
    ["holiday", "spring", "normal", "slight"],
    ["weekday", "spring", "normal", "none"],
    ["weekday", "spring", "normal", "slight"]
]
y_train_train = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                 "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                 "very late", "on time", "on time", "on time", "on time", "on time"]


def test_naive_bayes_classifier_fit():
    # Test 1: In-class Naive Bayes example (8 instances)
    clf = MyNaiveBayesClassifier()
    clf.fit(X_train_inclass_example, y_train_inclass_example)
    
    # Priors:
    assert np.isclose(clf.priors["yes"], 5/8)  # 5/8
    assert np.isclose(clf.priors["no"], 3/8)  # 3/8
    
    # Posteriors setup is [class][feature][value of feature]
    # Conditional probabilities for "yes"
    assert np.isclose(clf.posteriors["yes"][0][1], 0.8) # 4/5
    assert np.isclose(clf.posteriors["yes"][0][2], 0.2) # 1/5 
    assert np.isclose(clf.posteriors["yes"][1][5], 0.4)  # 2/5
    assert np.isclose(clf.posteriors["yes"][1][6], 0.6)  # 3/5
    
    # Conditional probabilities for "no"
    assert np.isclose(clf.posteriors["no"][0][1], 2/3)  # 2/3
    assert np.isclose(clf.posteriors["no"][0][2], 1/3)  # 1/3
    assert np.isclose(clf.posteriors["no"][1][5], 2/3)  # 2/3
    assert np.isclose(clf.posteriors["no"][1][6], 1/3)  # 1/3


    # Test 2: iPhone purchases dataset (15 instances)
    clf.fit(X_train_iphone, y_train_iphone)
    
    # Priors:
    assert np.isclose(clf.priors["yes"], 10/15)  # 10/15
    assert np.isclose(clf.priors["no"], 5/15)  # 5/15
    
    # Posteriors setup is [class][feature][value of feature]
    # Conditional probabilities for "yes" and "no" classes (Posteriors) can be checked similarly
    # Standing
    assert np.isclose(clf.posteriors["yes"][0][1], 0.2)  # 2/10
    assert np.isclose(clf.posteriors["yes"][0][2], 0.8)  # 8/10

    # Job Status
    assert np.isclose(clf.posteriors["yes"][1][1], 0.3)  # 3/10
    assert np.isclose(clf.posteriors["yes"][1][2], 0.4)  # 4/10
    assert np.isclose(clf.posteriors["yes"][1][3], 0.3)  # 3/10

    # Credit Rating
    assert np.isclose(clf.posteriors["yes"][2]["fair"], 0.7)  # 7/10
    assert np.isclose(clf.posteriors["yes"][2]["excellent"], 0.3)  # 3/10

    # Checking the posterior probabilities for the class "no"
    # Standing
    assert np.isclose(clf.posteriors["no"][0][1], 0.6)  # 3/5
    assert np.isclose(clf.posteriors["no"][0][2], 0.4)  # 2/5
    
    # Job Status
    assert np.isclose(clf.posteriors["no"][1][1], 0.2)  # 1/5
    assert np.isclose(clf.posteriors["no"][1][2], 0.4)  # 2/5
    assert np.isclose(clf.posteriors["no"][1][3], 0.4)  # 2/5

    # Credit Rating
    assert np.isclose(clf.posteriors["no"][2]["fair"], 0.4)  # 2/5
    assert np.isclose(clf.posteriors["no"][2]["excellent"], 0.6)  # 3/5


    # Test 3: Bramer 3.2 dataset (20 instances)
    clf.fit(X_train_train, y_train_train)
    
    # Priors:
    assert clf.priors["on time"] == 0.7
    assert clf.priors["late"] == 0.1
    assert clf.priors["very late"] == 0.15
    assert clf.priors["cancelled"] == 0.05

    # Posteriors setup is [class][feature][value of feature]
    # Checking the posterior probabilities for the class "on time"
    # Day
    assert np.isclose(clf.posteriors["on time"][0]["weekday"], 9/14)
    assert np.isclose(clf.posteriors["on time"][0]["saturday"], 2/14)
    assert np.isclose(clf.posteriors["on time"][0]["sunday"], 1/14)
    assert np.isclose(clf.posteriors["on time"][0]["holiday"], 2/14)

    # Season
    assert np.isclose(clf.posteriors["on time"][1]["spring"], 4/14)
    assert np.isclose(clf.posteriors["on time"][1]["summer"], 6/14)
    assert np.isclose(clf.posteriors["on time"][1]["autumn"], 2/14)
    assert np.isclose(clf.posteriors["on time"][1]["winter"], 2/14)

    # Wind
    assert np.isclose(clf.posteriors["on time"][2]["none"], 5/14)
    assert np.isclose(clf.posteriors["on time"][2]["high"], 4/14)
    assert np.isclose(clf.posteriors["on time"][2]["normal"], 5/14)

    # Rain
    assert np.isclose(clf.posteriors["on time"][3]["none"], 5/14)
    assert np.isclose(clf.posteriors["on time"][3]["slight"], 8/14)
    assert np.isclose(clf.posteriors["on time"][3]["heavy"], 1/14)

    # Checking the posterior probabilities for the class "late"
    # Day
    assert np.isclose(clf.posteriors["late"][0]["weekday"], 1/2)
    assert np.isclose(clf.posteriors["late"][0]["saturday"], 1/2)

    # Season
    assert np.isclose(clf.posteriors["late"][1]["winter"], 1)

    # Wind
    assert np.isclose(clf.posteriors["late"][2]["high"], 1/2)
    assert np.isclose(clf.posteriors["late"][2]["normal"], 1/2)

    # Rain
    assert np.isclose(clf.posteriors["late"][3]["none"], 1/2)
    assert np.isclose(clf.posteriors["late"][3]["heavy"], 1/2)

    # Checking the posterior probabilities for the class "very late"
    # Day
    assert np.isclose(clf.posteriors["very late"][0]["weekday"], 1)

    # Season
    assert np.isclose(clf.posteriors["very late"][1]["autumn"], 1/3)
    assert np.isclose(clf.posteriors["very late"][1]["winter"], 2/3)

    # Wind
    assert np.isclose(clf.posteriors["very late"][2]["high"], 1/3)
    assert np.isclose(clf.posteriors["very late"][2]["normal"], 2/3)

    # Rain
    assert np.isclose(clf.posteriors["very late"][3]["none"], 1/3)
    assert np.isclose(clf.posteriors["very late"][3]["heavy"], 2/3)

    # Checking the posterior probabilities for the class "cancelled"
    # Day
    assert np.isclose(clf.posteriors["cancelled"][0]["saturday"], 1)

    # Season
    assert np.isclose(clf.posteriors["cancelled"][1]["spring"], 1)

    # Wind
    assert np.isclose(clf.posteriors["cancelled"][2]["high"], 1)

    # Rain
    assert np.isclose(clf.posteriors["cancelled"][3]["heavy"], 1)

def test_naive_bayes_classifier_predict():
    clf = MyNaiveBayesClassifier()
    
    # Test 1: Predict using the 8 instance training set
    clf.fit(X_train_inclass_example, y_train_inclass_example)
    prediction = clf.predict([[1, 5]])
    assert prediction == ["yes"]

    # Test 2: iPhone purchases dataset (15 instances)
    clf.fit(X_train_iphone, y_train_iphone)
    prediction = clf.predict([[2, 2, "fair"], [1, 1, "excellent"]])
    assert prediction == ["yes", "no"]

    # Test 3: Bramer 3.2 unseen instance
    clf.fit(X_train_train, y_train_train)
    prediction = clf.predict([["weekday", "winter", "high", "heavy"]])
    assert prediction == ["very late"]  # got answer from Bramer 3.2

# interview dataset
header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
X_train_interview = [
    ["Senior", "Java", "no", "no"],
    ["Senior", "Java", "no", "yes"],
    ["Mid", "Python", "no", "no"],
    ["Junior", "Python", "no", "no"],
    ["Junior", "R", "yes", "no"],
    ["Junior", "R", "yes", "yes"],
    ["Mid", "R", "yes", "yes"],
    ["Senior", "Python", "no", "no"],
    ["Senior", "R", "yes", "no"],
    ["Junior", "Python", "yes", "no"],
    ["Senior", "Python", "yes", "yes"],
    ["Mid", "Python", "no", "yes"],
    ["Mid", "Java", "yes", "no"],
    ["Junior", "Python", "no", "yes"]
]
y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

# note: this tree uses the generic "att#" attribute labels because fit() does not and should not accept attribute names
# note: the attribute values are sorted alphabetically
tree_interview = \
        ["Attribute", "att0",
            ["Value", "Junior", 
                ["Attribute", "att3",
                    ["Value", "no", 
                        ["Leaf", "True", 3, 5]
                    ],
                    ["Value", "yes", 
                        ["Leaf", "False", 2, 5]
                    ]
                ]
            ],
            ["Value", "Mid",
                ["Leaf", "True", 4, 14]
            ],
            ["Value", "Senior",
                ["Attribute", "att2",
                    ["Value", "no",
                        ["Leaf", "False", 3, 5]
                    ],
                    ["Value", "yes",
                        ["Leaf", "True", 2, 5]
                    ]
                ]
            ]
        ]

# desk calculation iphone tree
tree_iphone = \
        ["Attribute", "att0",
            ["Value", 1, 
                ["Attribute", "att1",
                    ["Value", 1, 
                        ["Leaf", "yes", 1, 5]
                    ],
                    ["Value", 2,
                        ["Attribute", "att2",
                            ["Value", "excellent", 
                                ["Leaf", "yes", 1, 2]
                            ],
                            ["Value", "fair",
                                ["Leaf", "no", 1, 2]
                            ]
                        ]
                    ],
                    ["Value", 3, 
                        ["Leaf", "no", 2, 5]
                    ]
                ]
            ],
            ["Value", 2,
                ["Attribute", "att2",
                    ["Value", "excellent",
                        ["Attribute", "att1",
                            ["Value", 1,
                                ["Leaf", "no", 1, 2]
                            ],
                            ["Value", 2,
                                ["Leaf", "no", 1, 2]
                            ],
                            ["Value", 3,
                                ["Leaf", "no", 0, 4]
                            ]
                        ]
                    ],
                    ["Value", "fair",
                        ["Leaf", "yes", 6, 10]
                    ]
                ]
            ]
        ]

def test_random_forest_classifier_fit():
    # dataset from decision trees
    header = ["att0", "att1", "att2", "att3"]
    attribute_domains = {"att0": ["Junior", "Mid", "Senior"], 
            "att1": ["Java", "Python", "R"],
            "att2": ["no", "yes"], 
            "att3": ["no", "yes"]}
    X = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]

    y = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

    rf = MyRandomForestClassifier(n_classifiers=3, max_features=2, n_bootstrap=3)

    rf.fit(X, y)

    print(rf.classifiers)

def test_random_forest_classifier_predict():
    pass