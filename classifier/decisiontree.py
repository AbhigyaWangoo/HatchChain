from typing import List, Tuple, Dict, Set
from abc import ABC
import pandas as pd
import gensim
import os

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
import nltk
import numpy as np

nltk.download('punkt')

from . import base
from utils.utils import lbl_to_resumeset

DEBUG_HEURISTIC = "This is a sample heuristic for now for debugging purposes"
ROOT = "ROOT"
KEYWORD_HEURISTIC = "Relevant Keywords"
OUTPUT_CSV_DIR = "data/label_keywords/"
DATAMODELS = "data/models/"


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


class TreeClassifier(base.AbstractClassifier):
    """ 
    A classifier loosely based on the Decision tree classifier. Meant to be 
    more explainable in nature that a traditional decision tree. We construct the 
    tree via training data, and enable a heuristic to seperate following nodes. Leaf nodes
    contain classifications.
    """

    def __init__(self, hyperparams: List[str], category: str, _heuristic_ct: int = 5, consider_keywords: bool = True) -> None:
        self._hyperparam_lst: List[HyperParameter] = []
        for param in hyperparams:
            self._hyperparam_lst.append(HyperParameter(param))
        self._depth = len(hyperparams)
        self._heuristic_ct = _heuristic_ct
        self._category = Category(category)
        self._include_keywords = consider_keywords

        # self._heuristic_list = self._generate_heuristic_list(consider_keywords)
        # self._root = self._construct_tree(root=None, idx=0)

    def classify(self, input: str) -> Tuple[bool, str]:
        win_list, loss_list, reasoning_list = self._traverse_with_input(
            self._root, input, [], [], [])
        return len(win_list) > len(loss_list), ' '.join(reasoning_list)

    def fit(self, dataset: str):
        """ 
        Given a dataset, fits a decision tree classifier and uses the decision path as 
        context when choosing the final classification decision. 
        """

        corpus = lbl_to_resumeset(
            dataset, {self._category.name}, disable=False)

        def __build_decision_tree(vector_size: int = 50, epochs: int = 40) -> DecisionTreeClassifier:
            """ Returnes a trained Doc2Vec model """
            documents = corpus[self._category.name]
            
            # Tokenize the documents
            tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]

            # Create TaggedDocument objects
            tagged_data = [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(tokenized_docs)]

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(tagged_data, [self._category.name for _ in tagged_data], test_size=0.2, random_state=42)

            # Train a Doc2Vec model
            vector_size = 50
            model = Doc2Vec(vector_size=vector_size, window=2, min_count=1, workers=4, epochs=100)
            model.build_vocab(tagged_data)
            model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

            # Transform the training data using the trained Doc2Vec model
            X_train_vectors = np.array([model.infer_vector(doc.words) for doc in X_train])
            
            dt_classifier = DecisionTreeClassifier()
            dt_classifier.fit(X_train_vectors, y_train)
            
            X_test_vectors = np.array([model.infer_vector(doc.words) for doc in X_test])
            y_pred = dt_classifier.predict(X_test_vectors)
            accuracy = accuracy_score(y_test, y_pred)
            print("Accuracy on the test set:", accuracy)
            
            # dt_classifier.
            
            return dt_classifier
        
        self._dt_classifier = __build_decision_tree()
        

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

    def _navigate(self, node: Node, input: str) -> Tuple[bool, str]:
        navigation_str = f"""
        You have a candidate and a label. On the bases of the following information
        here: {node.heuristic} decide whether the following candidate: {input} fits the category 
        of {self._category.name}. Your output should always be modelled as follows:

        reject | accept:<reasoning for why the candidate should be accepted or rejected>
        """

        res = self._prompt_gpt(navigation_str)
        reasoning = res.split(":")[-1]
        reject_ct = res.lower().count("reject")
        accept_ct = res.lower().count("accept")

        return accept_ct > reject_ct, reasoning

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

            heuristic = self._prompt_gpt(heuristic_prompt)
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
