from typing import List
from . import base
from abc import ABC

DEBUG_HEURISTIC="This is a sample heuristic for now for debugging purposes"
ROOT="ROOT"

class Category():
    def __init__(self, name: str) -> None:
        self.name=name

class HyperParameter():
    def __init__(self, name: str) -> None:
        self.name=name

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

    def __init__(self, hyperparams: List[str], category: str, _heuristic_ct: int = 5) -> None:
        self._hyperparam_lst: List[HyperParameter] = []
        for param in hyperparams:
            self._hyperparam_lst.append(HyperParameter(param))
        self._depth = len(hyperparams)
        self._heuristic_ct=_heuristic_ct
        self._category = Category(category)
        
        self._heuristic_list = self._generate_heuristic_list()
        self._root = self._construct_tree(root=None, idx=0)

    def classify(self, input: str) -> str:
        
        return super().classify(input)

    def _generate_heuristic_list(self) -> List[str]:
        heuristics = []
        
        for hyperparam in self._hyperparam_lst:
            heuristic_prompt=f"""
            To be considered capable for {self._category.name}, concerning {hyperparam.name}, generate 
            {self._heuristic_ct} precise qualities some input should have that would make them capable of
            being in this category.
            """
        
            heuristic = self._prompt_gpt(heuristic_prompt)
            heuristics.append(heuristic)
        
        return heuristics
            
    def _construct_tree(self, root: Node, idx: int) -> Node:
        hyperparam = self._hyperparam_lst[idx]
        heuristic = self._heuristic_list[idx]
        # heuristic = "This is a sample heuristic for now for debugging purposes"

        if root == None:
            root = Node(heuristic=heuristic, level=hyperparam)
        else:
            root.heuristic = heuristic
            root.hyperparameter_level = hyperparam
            print(f"heurisitc: {heuristic[0:5]}, level: {hyperparam.name}")

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
        