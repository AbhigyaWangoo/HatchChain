from utils.utils import lbl_to_resumeset, lbl_to_resumeset_multiproc, LABELS
from similarity.cosine import CosineSimilarity
from collections import OrderedDict
from . import base
import json
from typing import Any, List, Tuple, Dict, Set, Union
from abc import ABC
import pandas as pd
import tqdm
import multiprocessing
import pickle
from random import shuffle
import os

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from nltk.tokenize import word_tokenize

from query_engine.src.db import postgres_client, rawtxt_client


import enum
import numpy as np

EMPTY_STRING = ""
DEBUG_HEURISTIC = "This is a sample heuristic for now for debugging purposes"
ROOT = "ROOT"
KEYWORD_HEURISTIC = "Relevant Keywords"
OUTPUT_CSV_DIR = "data/label_keywords/"
DATAMODELS = "data/models/"
SAVED_DTMODEL = "SavedDTModel"
SAVED_DOC2VEC = "data/models/Doc2Vec"


class SavedModelFields(enum.Enum):
    HEURISTIC_LIST = "heuristics"
    DEPTH = "depth"
    HEURISTIC_COUNT = "heuristic_ct"
    CATEGORY = "category"
    HYPERPARAMETER_LIST = "hyperparam_lst"


class Category():
    def __init__(self, name: str) -> None:
        self.name = name


class HyperParameter():
    def __init__(self, name: str) -> None:
        self.name = name


class Node(ABC):
    """
    Very important, left means a failure to pass the heuristic for navigation.
    Right means the success to pass the heuristic.
    """

    def __init__(self) -> None:
        self.left: Node = None
        self.right: Node = None
        self.heuristic: str = ""
        self.hyperparameter_level: HyperParameter = None

    def __init__(self, heuristic: str = None, level: HyperParameter = None) -> None:
        self.left: Node = None
        self.right: Node = None
        self.heuristic: str = heuristic
        self.hyperparameter_level = level


class ExplainableTreeClassifier(base.AbstractClassifier):
    """ 
    A classifier loosely based on the Decision tree classifier. Meant to be 
    more explainable in nature that a traditional decision tree. We construct the 
    tree via training data, and enable a heuristic to seperate following nodes. Leaf nodes
    contain classifications.
    """

    def __init__(self, hyperparams: List[str], category: str, load_file: str = EMPTY_STRING, _heuristic_ct: int = 5, consider_keywords: bool = True) -> None:
        super().__init__(hyperparams)
        self._hyperparam_lst: List[HyperParameter] = []
        for param in hyperparams:
            self._hyperparam_lst.append(HyperParameter(param))
        self._depth = len(hyperparams)
        self._heuristic_ct = _heuristic_ct
        self._category = Category(category)
        self._include_keywords = consider_keywords
        self._dt_doc2vec_model = None
        self._rf_classifiers = None

        if load_file != EMPTY_STRING:
            self.load_model(load_file)
        else:
            # print("Generating heuristic list in classifier.")
            self._heuristic_list = self._generate_heuristic_list(
                consider_keywords)
            # print("Generated!")

        self._root = self._construct_tree(root=None, idx=0)

    def save_model(self, path: str) -> Dict[Any, Any]:
        mode = "w"
        if not os.path.exists(path):
            mode = "a"

        data = {}
        # Data to save
        # 1. Heuristic List
        data[SavedModelFields.HEURISTIC_LIST.value] = self._heuristic_list

        # 2. primitave vars
        data[SavedModelFields.DEPTH.value] = self._depth
        data[SavedModelFields.HEURISTIC_COUNT.value] = self._heuristic_ct
        data[SavedModelFields.CATEGORY.value] = self._category.name
        # data['include_keywords']= self._include_keywords

        # 3. hyperparam list
        list_str = []
        for hyperparam in self._hyperparam_lst:
            list_str.append(hyperparam.name)
        data[SavedModelFields.HYPERPARAMETER_LIST.value] = list_str

        # 4. (AFTERWARDS, integrate the tfidf and deterministic dt)
        # data['rf_classifiers']= self._rf_classifiers
        # data['dt_doc2vec_model']= self._dt_doc2vec_model

        with open(path, mode, encoding="utf8") as fp:
            fp.write(json.dumps(data))

    def load_model(self, path: str) -> Dict[Any, Any] | None:
        if not os.path.exists(path):
            print(f"Path {path} does not exist. Please feed in a correct path.")
            return None

        with open(path, "r", encoding="utf8") as fp:
            data = json.load(fp)

            if len(data) == 0:
                print(f"Data file {path} is empty. Try a different file.")
                return None

            self._heuristic_list = data[SavedModelFields.HEURISTIC_LIST.value]
            # TODO defaulting to false for now to only consider recruiter passed in keywords.
            self._include_keywords = False
            self._depth = data[SavedModelFields.DEPTH.value]
            self._heuristic_ct = data[SavedModelFields.HEURISTIC_COUNT.value]
            self._category = Category(data[SavedModelFields.CATEGORY.value])

            list_str = data[SavedModelFields.HYPERPARAMETER_LIST.value]
            self._hyperparam_lst = []
            for hyperparam in list_str:
                self._hyperparam_lst.append(HyperParameter(hyperparam))

    def get_k_similar(self, job_id: int, resume_id: int, k: int, raw: bool=True) -> OrderedDict[int, str]:
        """ 
        This function gets the top k similar resumes from the db and 
        returns their metadata. It will return either the full raw text, or the 
        json, dependant on the 'raw' flag.

        job_id: int
        resume_id: int
        k: int
        raw: bool
        """
        rv = []

        # 1. Fetch all resume vectors for job id provided. Place into {res_id: vector} dict.
        client = postgres_client.PostgresClient(job_id)
        resumes = client.read_candidates_from_job("", False, True)

        if len(resumes) == 0:
            print("No candidates associated with the job. cant get K similar candidates")
            return rv

        vectors = {}
        res_vector = None
        for resume in resumes:
            res_id=resume[0]

            res = client.read_job_resume(res_id, postgres_client.VECTOR_DATA_FIELD)[0]

            if int(res_id) == resume_id:
                res_vector = res
            else:
                vectors[res_id] = res

        if res_vector == None:
            print("Could not find this resume's vector. Aborting for now.")
            return rv

        # 2. Calc cosine similarity with provided resume id and all others.
        similarity_calculator = CosineSimilarity()
        this_vector = np.array(res_vector)

        # 3. get top k resume ids.
        vectors = similarity_calculator.top_k(this_vector, vectors, k)

        # 4. Use ddb client to retrive from raw text store, or just return jsons
        # retrieved earlier from postgres
        if raw:
            txt_client = rawtxt_client.ResumeDynamoClient(job_id)
            raw_txt_data = txt_client.batch_get_resume(list(vectors.keys()), "", save_to_file=False, return_txt=True)

            # 5. Return list (in order of closest) with raw text
            for vec in vectors:
                vectors[vec] = raw_txt_data[vec]

        return vectors

    def classify(self, input: str, resume_id: int = None, job_id: int = None) -> Tuple[bool, str]:
        win_list, loss_list, reasoning_list = self._traverse_with_input(
            self._root, input, [], [], [])
        # predictions = self._generate_classifications(input=input)

        if resume_id is not None and job_id is not None and abs(len(win_list) - len(loss_list)) <= 1:
            tiebreaker, final_reasoning_list = self.tiebreak(resume_id, job_id, reasoning_list)

            return tiebreaker, ' '.join(reasoning_list)

        return len(win_list) > len(loss_list), ' '.join(reasoning_list)

    def tiebreak(self, resume_id: int, job_id: int, reasoning_list: List[str]) -> Tuple[bool,  List[str]]:
        """
        This is the tiebreaker function. If the |wincount - losscount| <= 1, this function
        will decide whether to push the candidate into or out of the pool depending on
        the k nearest neighbors, and whether a majority of them would be accepted or rejected.
        
        resume_id: id of the resume to fetch
        job_id: id of the job the resume belongs to
        reasoning_list: a list of the provided reasonings.
        """

        # 1. call top k function
        NUM_SIMILAR=5
        similar_resumes = self.get_k_similar(job_id, resume_id, k=NUM_SIMILAR, raw=True)

        # 2. find top k resumes, and find majority acceptance rate
        # 3. Reshape reasoning list depending on a) whether we're accepting or 
        # rejecting based on tiebreaker, and b) the names of the candidates who also were included in the list.

        return True

    def fit(self, dataset: str):
        """ 
        Given a dataset, fits a decision tree classifier and uses the decision path as 
        context when choosing the final classification decision. 
        """

        if not os.path.isfile(SAVED_DOC2VEC):
            corpus = lbl_to_resumeset_multiproc(dataset, set(), disable=False)

        manager = multiprocessing.Manager()

        def __perform_grid_search(rfs: Dict[str, RandomForestClassifier], X, y):
            grid_space = {'max_depth': [3, 5, 10, 15],
                          'n_estimators': [10, 100, 200, 400, 500, 600],
                          'max_features': [2, 3, 4, 5, 6, 7],
                          'min_samples_leaf': [2, 3, 4, 5, 6],
                          'min_samples_split': [2, 3, 4, 5, 6]
                          }

            def __single_grid_search(rf: RandomForestClassifier, category):
                print(f"Performing grid search on {category} classifier")
                grid = GridSearchCV(rf, param_grid=grid_space,
                                    cv=3, scoring='accuracy', n_jobs=2)
                model_grid = grid.fit(X, y)

                with open(f"{category}-gsrch") as fp:
                    print(
                        f'Best hyperparameters for {category} are: {str(model_grid.best_params_)}')
                    fp.write(
                        f'Best hyperparameters for {category} are: {str(model_grid.best_params_)}\n')
                    print(
                        f'Best score for {category} is: {str(model_grid.best_score_)}')
                    fp.write(
                        f'Best score for {category} is: {str(model_grid.best_score_)}\n')

            proccesses = []
            for label in rfs:
                classifier = rfs[label]
                print(f"running grid search on label {label}")
                p = multiprocessing.Process(
                    target=__single_grid_search, args=(classifier, label))
                proccesses.append(p)
                p.start()

            for proc in proccesses:
                proc.join()

        def __build_internal_classifier(vector_size: int = 50, epochs: int = 1, disable: bool = False) -> Tuple[Dict[str, RandomForestClassifier], Doc2Vec]:
            """ 
            This function builds a Doc2Vec model from the dataset, and uses it to build multiple decision tree classifiers,
            one for each category. If either models already exist in the proper directory, they will be loaded from disk.
            """

            if os.path.isfile(SAVED_DOC2VEC):
                model_files = os.listdir(DATAMODELS)

                category_classifiers = {}
                for mfile in model_files:
                    splitted_file = mfile.split("-")

                    if len(splitted_file) > 1 and splitted_file[0] == SAVED_DTMODEL:
                        category_classifiers[splitted_file[-1]
                                             ] = pickle.load(open(F"{DATAMODELS}{mfile}", "rb"))

                return category_classifiers, pickle.load(open(SAVED_DOC2VEC, "rb"))

            def __tag_documents(category: str, documents, tagged_data):
                for doc in tqdm.tqdm(documents, desc=f"Tagging corpus documents for {category}", disable=disable):
                    tokenized_doc = word_tokenize(doc.lower())
                    tags = [category]
                    tagged_data.append(TaggedDocument(
                        words=tokenized_doc, tags=tags))

            # Combine all labels for each document
            tagged_data = manager.list()
            proccesses = []
            for category, documents in corpus.items():
                p = multiprocessing.Process(
                    target=__tag_documents, args=(category, documents, tagged_data))
                proccesses.append(p)
                p.start()

            for proc in proccesses:
                proc.join()

            #  Split the data into training and testing sets
            X_train, X_test = train_test_split(
                tagged_data, test_size=0.2, random_state=42)

            # Train a Doc2Vec model
            model = Doc2Vec(vector_size=vector_size,
                            window=2, min_count=1, workers=4)
            print("Training Doc2Vec model with corpus")
            model.build_vocab(tagged_data)
            print("Built vocab")

            shuffle(tagged_data)
            model.train(list(tagged_data),
                        total_examples=model.corpus_count, epochs=epochs)

            print("Trained Doc2Vec Model")

            # Transform the training data using the trained Doc2Vec model
            X_train_vectors = np.array([model.infer_vector(doc.words) for doc in tqdm.tqdm(
                X_train, desc="Inferring vectors from dataset", disable=disable)])
            print("transformed training data")

            # Train a separate binary classifier for each category
            category_classifiers = {}
            keys = corpus.keys()
            for category in tqdm.tqdm(keys, desc=f"Training {len(keys)} decision trees", disable=disable):
                # Extract binary labels for the current category
                y_train = [1 if category in doc.tags else 0 for doc in X_train]

                # Train a Decision Tree classifier
                rf_classifier = RandomForestClassifier()
                # dt_classifier = DecisionTreeClassifier()
                rf_classifier.fit(X_train_vectors, y_train)

                category_classifiers[category] = rf_classifier

            # Transform the testing data using the trained Doc2Vec model
            X_test_vectors = np.array([model.infer_vector(doc.words) for doc in tqdm.tqdm(
                X_test, desc="Inferring vectors for test set", disable=disable)])

            # Make predictions for each category
            predictions = {}
            for category, classifier in tqdm.tqdm(category_classifiers.items(), desc="Classifying test set", disable=disable):
                predictions[category] = classifier.predict(X_test_vectors)

            # Evaluate the performance for each category
            for category in corpus.keys():
                y_true = [1 if category in doc.tags else 0 for doc in X_test]
                y_pred = predictions[category]
                accuracy = accuracy_score(y_true, y_pred)
                report = classification_report(y_true, y_pred, target_names=[
                                               'Not ' + category, category])

                try:
                    with open("rf_classifier_report.txt", "a") as fp:
                        print(f"Category: {category}")
                        fp.write(f"Category: {category}\n")

                        print(f"Accuracy: {accuracy}")
                        fp.write(f"Accuracy: {accuracy}\n")

                        print(f"Classification Report:\n {report}")
                        fp.write(f"Classification Report:\n {report}\n")
                except Exception as e:
                    print(
                        f"Error saving dt classifier report data to file: {e}")

            for category, classifier in category_classifiers.items():
                pickle.dump(rf_classifier, open(
                    f"{DATAMODELS}{SAVED_DTMODEL}-{category}", 'wb'))
            pickle.dump(model, open(SAVED_DOC2VEC, 'wb'))

            print("performing grid search")
            __perform_grid_search(category_classifiers,
                                  X_train_vectors, y_train)
            print("performed grid search")

            return category_classifiers, model

        self._rf_classifiers, self._dt_doc2vec_model = __build_internal_classifier()

    def _generate_classifications(self, input: str) -> Dict[str, int]:
        if not self._rf_classifiers or not self._dt_doc2vec_model:
            raise ValueError("Models not loaded. Call load_models() first.")

        # Preprocess the input document
        tokenized_doc = word_tokenize(input.lower())
        infer_vector = self._dt_doc2vec_model.infer_vector(tokenized_doc)

        predictions = {}
        for category, classifier in self._rf_classifiers.items():
            # Make prediction for each category
            prediction = classifier.predict([infer_vector])[0]
            predictions[category] = prediction

        return predictions

    def _traverse_with_input(self, cur_node: Node, input: str, win_lst: List[str], loss_lst: List[str], reasoning_lst: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """ Traverses the tree with the input, uses comparison function to generate wins or losses """

        if cur_node is not None:
            go_right, reason = self._navigate(cur_node, input)
            reasoning_lst.append(reason)

            if go_right:
                win_lst.append(cur_node.hyperparameter_level.name)
                self._traverse_with_input(
                    cur_node.right, input, win_lst, loss_lst, reasoning_lst)
            else:
                loss_lst.append(cur_node.hyperparameter_level.name)
                self._traverse_with_input(
                    cur_node.left, input, win_lst, loss_lst, reasoning_lst)

        return win_lst, loss_lst, reasoning_lst

    def _navigate(self, node: Node, input_str: str, max_retry: int = 0) -> Tuple[bool, str]:
        navigation_str = f"""
        You have a candidate and a label. On the bases of the following information
        here: {node.heuristic} decide whether the following candidate: {input_str} fits the category 
        of {self._category.name}. Your output should always be modelled as follows:

        reject | accept:<reasoning for why the candidate should be accepted or rejected>
        """

        res = self._prompt_runpod(navigation_str)
        try:
            reasoning = res.split(":")[-1]

            reject_ct = res.lower().count("reject")
            accept_ct = res.lower().count("accept")

            return accept_ct > reject_ct, reasoning
        except Exception as e:
            error_str = f"""
                Error reading information from navigation. Nav response for 
                node {node.heuristic}: {res}.\n\nError: {e}. Retrying..
            """

            if max_retry > 0:
                print(error_str)
                return self._navigate(node, input_str)

            raise error_str

    def _generate_heuristic_list(self, consider_keywords: bool) -> List[str]:
        heuristics = []

        def __get_keyword_heuristic(term_count: int = 20) -> str:
            keyword_file = f"{OUTPUT_CSV_DIR}{self._category.name}-full.csv"
            df = pd.read_csv(keyword_file)

            trimmed_df = df.head(term_count)
            term_list = trimmed_df.iloc[:, 0].tolist()
            strterms = ','.join(term_list)

            final_heuristic = f"candidates have the following keywords on their resume for {self._category.name}: {strterms}"

            return final_heuristic

        for hyperparam in self._hyperparam_lst:
            heuristic_prompt = f"""
            To be considered capable for {self._category.name}, concerning {hyperparam.name}, generate 
            {self._heuristic_ct} precise qualities some input should have that would make them capable of
            being in this category.
            """

            heuristic = self._prompt_runpod(heuristic_prompt)
            heuristics.append(heuristic)

        if consider_keywords:
            self._hyperparam_lst.append(HyperParameter(KEYWORD_HEURISTIC))
            heuristics.append(__get_keyword_heuristic())
            self._depth += 1

        print(heuristics)

        return heuristics

    def _construct_tree(self, root: Node, idx: int) -> Node:
        hyperparam = self._hyperparam_lst[idx]
        heuristic = self._heuristic_list[idx]

        if root == None:
            root = Node(heuristic=heuristic, level=hyperparam)
        else:
            root.heuristic = heuristic
            root.hyperparameter_level = hyperparam

        if idx < self._depth-1:
            root.left = Node()
            root.right = Node()
            self._construct_tree(root=root.left, idx=idx+1)
            self._construct_tree(root=root.right, idx=idx+1)

        return root

    def print_tree(self):

        def __tree_print(node: Node):
            if node != None:
                __print_node(node)
                __tree_print(node.left)
                __tree_print(node.right)

        def __print_node(node: Node):
            nodestr = f"""
            node hyperparameter: {node.hyperparameter_level.name}
            node heuristic: {node.heuristic}
            """
            print(nodestr)

        __tree_print(self._root)
