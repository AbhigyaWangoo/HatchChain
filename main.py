from classifier.decisiontree import TreeClassifier
import openai
import chardet
import os
from typing import Dict, Set, List

NOLABEL="nolabel"    
TESTDIR="data/"
LABELS = {'Database_Administrator', 'Project_manager',  'Java_Developer', 'Python_Developer',  'Software_Developer', 'Web_Developer', 'Systems_Administrator', 'Network_Administrator'}

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        return result['encoding']

def read_from(filename: str) -> str:
    encoding=detect_encoding(filename)
    
    with open(filename, "r", encoding=encoding) as fp:
        return fp.readline()

def get_all_labels(file_path: str = 'normlized_classes.txt') -> Set:
    fin=set()
    with open(file_path, encoding=detect_encoding(file_path)) as fp:
        # Iterate through each line in the file
        for line in fp:
            # Process each line as needed
            res=line.split(":")
            lbls=res[-1].split(",")
            fin=fin.union(set(lbls))
    
    print(fin)
    return fin

def run_classifier(dir_path: str, n: int) -> Dict[str, str]:
    """ 
    Returns a dict of {filename: label}
    """

    file_to_lbl = {}
    files = os.listdir(dir_path)
    files.sort()

    lbl=""
    data=""
    total=0.0
    s=0.0

    for file in files:
        _, ext = os.path.splitext(file)

        if ext == ".lab":
            lbl = read_from(dir_path+file).strip() # TODO there are some cases where multiple lables are applicable. Need to test those out as well.
        elif ext == ".txt":
            data = read_from(dir_path+file).strip()

        if lbl != "" and data != "":
            lbls = set(lbl.split("\n"))

            for lbl_line in LABELS:
                try:
                    classifier = TreeClassifier(["Experiences", "Skills"], lbl_line)
                    decision, reason = classifier.classify(data)

                    print(f"FOR CANDIDATE {file}, we {decision} for the position of {lbl_line} because {reason}")
                    if (lbl_line in lbls and decision) or (lbl_line not in lbls and not decision):
                        s+=1.0
                    total+=1.0

                    print(f"Running accuracy= {s/total}\r")

                    if n > 0:
                        n-=1
                    elif n<=0:
                        break
                except openai.error.InvalidRequestError as err:
                    print(f"CONTEXT LENGTH EXCEEDED, CONTINUING. Error: {err}")

            data=""
            lbl=""

    print(f"FINAL accuracy= {s/total}\r")
    return file_to_lbl

def run_classifier_d2(dir_path: str):
    """ Run KNN classifier on a Gaurav's dataset """
    

if __name__ == "__main__":
    run_classifier(TESTDIR, 1)
