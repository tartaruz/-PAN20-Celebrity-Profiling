
config = {
    "training_path"    : "data/split_data/training/",
    "test_path"        : "data/split_data/test/",
    
    
    

    # Load modus
    "load_celebrity"   : False,
    "load_vectorizer"  : False,

    
    "TFIDF_path": "preprossesing/pickle/tools/tfidf.pkl",
    
    # Retrive from a pickeld file if possible
    "save_mode"        : True,
    "save_path"        : "data/pickled_data/",
    "pickle_suffix"    : ".pkl",

    # Preprossesing 
    "replacement_of_emoji"      : True,
    "replacement_of_link"       : True,
    "removal_of_puntation"      : True,
    "removal_of_stopwords"      : True,
    "normalization_with_stemmer": True,


    # Vectorizer/Embeding
    "vocabulary_size": 4000000
}
