from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Tuple
from main import detect_encoding, TESTDIR
import os
from tqdm import tqdm

def get_corpus_for_category(dir_path: str, category: str) -> List[str]:
    """ 
    Given some category, returns all documents related to it
    """
    files = os.listdir(dir_path)
    files.sort()
    corpus = []

    for file in tqdm(files, desc="Processing Files", unit="file", mininterval=1.0):
        base_fname, ext = os.path.splitext(file)

        if ext == ".lab":
            file_path = dir_path+file
            # encoding=detect_encoding(file_path)
            with open(file_path, "r", encoding="utf8") as fp:
                print(file_path)
                lbls = set(fp.readlines())
                print(file_path)

                if category in lbls:
                    datafile = base_fname + ".txt"
                    corpus.append(datafile)

    return corpus


def term_frequency_inverse_document_frequency(corpus: List[str]) -> Dict[str, float]:
    """ a function to calculated the TF IDF scores of each word in the corpus """

    def term_frequency(document: str) -> Dict[float]:
        """ 
        Calculates the frequency of a term according to the below formula:
        TF(t,d)= Total number of terms in document d / Number of times term t appears in document d

        This function returns a dictionary of terms to their TF score.
        """
        document_data = read_from(document)
        terms = document_data.split(" ")

        total_termcount = len(terms)
        term_dict = {}
        for term in terms:
            if term in term_dict:
                term_dict[term]+=1
            else:
                term_dict[term]=1

        tf_scores = {}
        for term in term_dict.items():
            tf_scores[term] = term_dict[term] / total_termcount

        return tf_scores

    def inverse_doc_frequency():
        """
        Calculates the inverse document frequency according to the following formula
        IDF(t,D)=log(Number of documents containing term t / Total number of documents in the corpus N)
        """
        pass

    for document in corpus:
        res = term_frequency(document=document)
        print(res)

cps = get_corpus_for_category(TESTDIR, "Database_Administrator")
# term_frequency_inverse_document_frequency(cps)
