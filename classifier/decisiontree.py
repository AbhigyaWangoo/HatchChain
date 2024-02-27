from utils.utils import lbl_to_resumeset_multiproc
from similarity.cosine import CosineSimilarity
from collections import OrderedDict
from . import base
import json
from typing import Any, List, Tuple, Dict
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

from query_engine.src.db import postgres_client


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
NAME = "name"

LIST_MERGING_PROMPT = """
Below, you are given a set of sentences, which aren't bound together very well.
They need some transition phrases (furthermore, additionally, finally, etc.) to bind the 
sentences together. Provide only grammatical improvements, and do not add any supplementary
information. Do not even preface your changes. Just provide the corrected version of each 
sentence.
"""


class ResumeModel:
    def __init__(
        self,
        id: int,
        name: str,
        raw_data: str = None,
        json_data: Dict[Any, Any] = None,
        vector: List[int] = None,
        explainable_classification: bool = None,
    ) -> None:
        self.id = id
        self.name = name
        self.raw_data = raw_data
        self.json_data = json_data
        self.vector = vector
        self.explainable_classification = explainable_classification


class SavedModelFields(enum.Enum):
    HEURISTIC_LIST = "heuristics"
    DEPTH = "depth"
    HEURISTIC_COUNT = "heuristic_ct"
    CATEGORY = "category"
    HYPERPARAMETER_LIST = "hyperparam_lst"


class ClassificationOutput(enum.StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"
    TIE = "tie"


class Category:
    def __init__(self, name: str) -> None:
        self.name = name


class HyperParameter:
    def __init__(self, name: str) -> None:
        self.name = name


class Node(ABC):
    """
    A Node class for the linked list, contains the hyperparameter
    name and the heursitc
    """

    def __init__(self) -> None:
        self.next: Node = None
        self.heuristic: str = ""
        self.hyperparameter_level: HyperParameter = None

    def __init__(self, heuristic: str = None, level: HyperParameter = None) -> None:
        self.next: Node = None
        self.heuristic: str = heuristic
        self.hyperparameter_level = level


class ExplainableTreeClassifier(base.AbstractClassifier):
    """
    A classifier loosely based on the Decision tree classifier. Meant to be
    more explainable in nature that a traditional decision tree. We construct the
    tree via training data, and enable a heuristic to seperate following nodes. Leaf nodes
    contain classifications.
    """

    def __init__(
        self,
        hyperparams: List[str],
        category: str,
        job_description: Dict[Any, Any] = None,
        load_file: str = EMPTY_STRING,
        _heuristic_ct: int = 5,
        consider_keywords: bool = True,
    ) -> None:
        super().__init__(hyperparams)
        self._hyperparam_lst: List[HyperParameter] = []
        for param in hyperparams:
            self._hyperparam_lst.append(HyperParameter(param))
        self._depth = len(hyperparams)
        self._job_description = job_description
        self._heuristic_ct = _heuristic_ct
        self._category = Category(category)
        self._include_keywords = consider_keywords
        self._dt_doc2vec_model = None
        self._rf_classifiers = None

        if load_file != EMPTY_STRING:
            if not self.load_model(load_file):
                raise NameError(
                    f"File {load_file} does not exist to load a file from. Please verify the classifier is receiving the correct file."
                )
        else:
            # print("Generating heuristic list in classifier.")
            self._heuristic_list = self._generate_heuristic_list(consider_keywords)
            # print("Generated!")

        self._root = self._construct_linked_list(root=None, idx=0)

    def prompt_wrapper(self, prompt: str) -> str:
        """
        A failure wrapper around prompting. Prioritized runpod for now.
        """

        try:
            return self._prompt_runpod(prompt)
        except (ConnectionError, ValueError):
            print("Runpod failed. Trying with gpt client.")
            return self._prompt_gpt(prompt)

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

    def load_model(self, path: str) -> bool:
        if not os.path.exists(path):
            print(f"Path {path} does not exist. Please feed in a correct path.")
            return False

        with open(path, "r", encoding="utf8") as fp:
            data = json.load(fp)

            if len(data) == 0:
                print(f"Data file {path} is empty. Try a different file.")
                return False

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

            return True

    def _coalesce_response(
        self, win_map: Dict[HyperParameter, str], loss_map: Dict[HyperParameter, str]
    ) -> Tuple[ClassificationOutput, str]:

        verdict: ClassificationOutput
        reasonings_str: str

        if len(win_map) == len(loss_map):
            verdict = ClassificationOutput.TIE
        elif len(win_map) > len(loss_map):
            verdict = ClassificationOutput.ACCEPT
        else:
            verdict = ClassificationOutput.REJECT

        merging_prompt = f"""
            You are given the following reasonings to accept a candidate for a job here:
            {' '.join(list(win_map.values()))}
            And the following reasongs to reject that candidate for the job here:
            {' '.join(list(loss_map.values()))}
            
            The job is given here: {self._job_description[postgres_client.DESCRIPTION_DATA_FIELD]}
            
            Given that this candidate should be {verdict.value}ed, Generate a final
            reasoning paragraph that has the same number of sentences as all the reasonings combined.
            Preserve each sentence, and do not summarize or eliminate data. Specifically reiterate that 
            the candidate should be {verdict.value}ed
        """

        reasonings_str = self.prompt_wrapper(merging_prompt)

        return verdict, reasonings_str

    def classify(self, resume_input: str) -> Tuple[ClassificationOutput, str]:
        win_map, loss_map = self._traverse_with_input(self._root, resume_input, {}, {})

        return self._coalesce_response(win_map, loss_map)

    def fit(self, dataset: str):
        """
        Given a dataset, fits a decision tree classifier and uses the decision path as
        context when choosing the final classification decision.
        """

        if not os.path.isfile(SAVED_DOC2VEC):
            corpus = lbl_to_resumeset_multiproc(dataset, set(), disable=False)

        manager = multiprocessing.Manager()

        def __perform_grid_search(rfs: Dict[str, RandomForestClassifier], X, y):
            grid_space = {
                "max_depth": [3, 5, 10, 15],
                "n_estimators": [10, 100, 200, 400, 500, 600],
                "max_features": [2, 3, 4, 5, 6, 7],
                "min_samples_leaf": [2, 3, 4, 5, 6],
                "min_samples_split": [2, 3, 4, 5, 6],
            }

            def __single_grid_search(rf: RandomForestClassifier, category):
                print(f"Performing grid search on {category} classifier")
                grid = GridSearchCV(
                    rf, param_grid=grid_space, cv=3, scoring="accuracy", n_jobs=2
                )
                model_grid = grid.fit(X, y)

                with open(f"{category}-gsrch") as fp:
                    print(
                        f"Best hyperparameters for {category} are: {str(model_grid.best_params_)}"
                    )
                    fp.write(
                        f"Best hyperparameters for {category} are: {str(model_grid.best_params_)}\n"
                    )
                    print(
                        f"Best score for {category} is: {str(model_grid.best_score_)}"
                    )
                    fp.write(
                        f"Best score for {category} is: {str(model_grid.best_score_)}\n"
                    )

            proccesses = []
            for label in rfs:
                classifier = rfs[label]
                print(f"running grid search on label {label}")
                p = multiprocessing.Process(
                    target=__single_grid_search, args=(classifier, label)
                )
                proccesses.append(p)
                p.start()

            for proc in proccesses:
                proc.join()

        def __build_internal_classifier(
            vector_size: int = 50, epochs: int = 1, disable: bool = False
        ) -> Tuple[Dict[str, RandomForestClassifier], Doc2Vec]:
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
                        category_classifiers[splitted_file[-1]] = pickle.load(
                            open(f"{DATAMODELS}{mfile}", "rb")
                        )

                return category_classifiers, pickle.load(open(SAVED_DOC2VEC, "rb"))

            def __tag_documents(category: str, documents, tagged_data):
                for doc in tqdm.tqdm(
                    documents,
                    desc=f"Tagging corpus documents for {category}",
                    disable=disable,
                ):
                    tokenized_doc = word_tokenize(doc.lower())
                    tags = [category]
                    tagged_data.append(TaggedDocument(words=tokenized_doc, tags=tags))

            # Combine all labels for each document
            tagged_data = manager.list()
            proccesses = []
            for category, documents in corpus.items():
                p = multiprocessing.Process(
                    target=__tag_documents, args=(category, documents, tagged_data)
                )
                proccesses.append(p)
                p.start()

            for proc in proccesses:
                proc.join()

            #  Split the data into training and testing sets
            X_train, X_test = train_test_split(
                tagged_data, test_size=0.2, random_state=42
            )

            # Train a Doc2Vec model
            model = Doc2Vec(vector_size=vector_size, window=2, min_count=1, workers=4)
            print("Training Doc2Vec model with corpus")
            model.build_vocab(tagged_data)
            print("Built vocab")

            shuffle(tagged_data)
            model.train(
                list(tagged_data), total_examples=model.corpus_count, epochs=epochs
            )

            print("Trained Doc2Vec Model")

            # Transform the training data using the trained Doc2Vec model
            X_train_vectors = np.array(
                [
                    model.infer_vector(doc.words)
                    for doc in tqdm.tqdm(
                        X_train, desc="Inferring vectors from dataset", disable=disable
                    )
                ]
            )
            print("transformed training data")

            # Train a separate binary classifier for each category
            category_classifiers = {}
            keys = corpus.keys()
            for category in tqdm.tqdm(
                keys, desc=f"Training {len(keys)} decision trees", disable=disable
            ):
                # Extract binary labels for the current category
                y_train = [1 if category in doc.tags else 0 for doc in X_train]

                # Train a Decision Tree classifier
                rf_classifier = RandomForestClassifier()
                # dt_classifier = DecisionTreeClassifier()
                rf_classifier.fit(X_train_vectors, y_train)

                category_classifiers[category] = rf_classifier

            # Transform the testing data using the trained Doc2Vec model
            X_test_vectors = np.array(
                [
                    model.infer_vector(doc.words)
                    for doc in tqdm.tqdm(
                        X_test, desc="Inferring vectors for test set", disable=disable
                    )
                ]
            )

            # Make predictions for each category
            predictions = {}
            for category, classifier in tqdm.tqdm(
                category_classifiers.items(),
                desc="Classifying test set",
                disable=disable,
            ):
                predictions[category] = classifier.predict(X_test_vectors)

            # Evaluate the performance for each category
            for category in corpus.keys():
                y_true = [1 if category in doc.tags else 0 for doc in X_test]
                y_pred = predictions[category]
                accuracy = accuracy_score(y_true, y_pred)
                report = classification_report(
                    y_true, y_pred, target_names=["Not " + category, category]
                )

                try:
                    with open("rf_classifier_report.txt", "a") as fp:
                        print(f"Category: {category}")
                        fp.write(f"Category: {category}\n")

                        print(f"Accuracy: {accuracy}")
                        fp.write(f"Accuracy: {accuracy}\n")

                        print(f"Classification Report:\n {report}")
                        fp.write(f"Classification Report:\n {report}\n")
                except Exception as e:
                    print(f"Error saving dt classifier report data to file: {e}")

            for category, classifier in category_classifiers.items():
                pickle.dump(
                    rf_classifier, open(f"{DATAMODELS}{SAVED_DTMODEL}-{category}", "wb")
                )
            pickle.dump(model, open(SAVED_DOC2VEC, "wb"))

            print("performing grid search")
            __perform_grid_search(category_classifiers, X_train_vectors, y_train)
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

    def _traverse_with_input(
        self,
        cur_node: Node,
        input: str,
        win_map: Dict[HyperParameter, str],
        loss_map: Dict[HyperParameter, str],
    ) -> Tuple[Dict[HyperParameter, str], Dict[HyperParameter, str]]:
        """Traverses the tree with the input, uses comparison function to generate wins or losses"""

        if cur_node is not None:
            pass_heuristic, reason = self._navigate(cur_node, input, 2)

            if pass_heuristic:
                win_map[cur_node.hyperparameter_level] = reason
            else:
                loss_map[cur_node.hyperparameter_level] = reason

            self._traverse_with_input(cur_node.next, input, win_map, loss_map)

        return win_map, loss_map

    def get_navigation_string(self, heuristic: str, input_str: str) -> str:
        """ Returns a crafted heuristic prompt based on the provided args """
        
        return f"""
        You have a candidate and a label. On the bases of the following heuristcs
        here: {heuristic} decide whether the following candidate: {input_str} fits the category 
        of {self._category.name}. When providing a reasoning, only reference the specific heuristics provided,
        all your lines of reasoning should be relevant to the provided heuristic.
        
        Your output should always be modelled as follows:
        reject | accept:<reasoning for why the candidate should be accepted or rejected>
        """

    def _navigate(
        self, node: Node, input_str: str, max_retry: int = 0
    ) -> Tuple[bool, str]:
        navigation_str = self.get_navigation_string(node.heuristic, input_str)

        try:
            res = self._prompter.prompt(navigation_str)
        except Exception:  # Handling runpod failure case
            res = self.prompt_wrapper(navigation_str)

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
                return self._navigate(node, input_str, max_retry - 1)

            raise error_str

    def _generate_heuristic_list(self, consider_keywords: bool) -> List[str]:
        heuristics = []

        def __get_keyword_heuristic(term_count: int = 20) -> str:
            keyword_file = f"{OUTPUT_CSV_DIR}{self._category.name}-full.csv"
            df = pd.read_csv(keyword_file)

            trimmed_df = df.head(term_count)
            term_list = trimmed_df.iloc[:, 0].tolist()
            strterms = ",".join(term_list)

            final_heuristic = f"candidates have the following keywords on their resume for {self._category.name}: {strterms}"

            return final_heuristic

        for hyperparam in self._hyperparam_lst:
            heuristic_prompt = f"""
            To be considered a strong candidate for the position of {self._category.name}, list 
            {self._heuristic_ct} precise qualities regarding the {hyperparam.name} of a resume. You must only generate
            qualities regarding {hyperparam.name}, and nothing else. You must make also make the {self._heuristic_ct} 
            heursitics relevant to the category.
            """

            if self._job_description is not None:
                heuristic_prompt += f"You are given the following information about the job description as well: {self._job_description}. You should focus on and job description's ideal qualities, and reference them when generating heuristics."

            heuristic = self.prompt_wrapper(heuristic_prompt)
            heuristics.append(heuristic)

        if consider_keywords:
            self._hyperparam_lst.append(HyperParameter(KEYWORD_HEURISTIC))
            heuristics.append(__get_keyword_heuristic())
            self._depth += 1

        # print(heuristics)

        return heuristics

    def _construct_linked_list(self, root: Node, idx: int) -> Node:
        hyperparam = self._hyperparam_lst[idx]
        heuristic = self._heuristic_list[idx]

        if root is None:
            root = Node(heuristic=heuristic, level=hyperparam)
        else:
            root.heuristic = heuristic
            root.hyperparameter_level = hyperparam

        if idx < self._depth - 1:
            root.next = Node()
            self._construct_linked_list(root=root.next, idx=idx + 1)

        return root

    def print_list(self):
        """A debug function that prints out the entire list"""

        def __list_print(node: Node):
            if node != None:
                __print_node(node)
                __list_print(node.next)

        def __print_node(node: Node):
            nodestr = f"""
            node hyperparameter: {node.hyperparameter_level.name}
            node heuristic: {node.heuristic}
            """
            print(nodestr)

        __list_print(self._root)
