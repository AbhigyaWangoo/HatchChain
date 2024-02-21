from classifier.decisiontree import ExplainableTreeClassifier

def test_invalid_file_construction():
    """ Tests whether an invalid file has been used to load the classifier """
    try:
        ExplainableTreeClassifier([], "", "nonexistant_file")
        assert False
    except NameError:
        assert True
