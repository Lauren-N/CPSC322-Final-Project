import pickle # a standard library

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

# let's package the two lists together into one object
packaged_obj = [header, interview_tree_solution]
outfile = open("tree.p", "wb")
pickle.dump(packaged_obj, outfile)
outfile.close()