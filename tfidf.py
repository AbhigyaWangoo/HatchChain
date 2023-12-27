from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from typing import List, Dict
from main import TESTDIR, read_from
import os
from tqdm import tqdm
import spacy

def preprocess(dataset: List[str]) -> List[str]:
    # Load the spaCy English model
    nlp = spacy.load("en_core_web_sm")

    # Function to filter non-nouns
    def filter_nouns(text):
        doc = nlp(text)
        return ' '.join([token.text for token in doc if token.pos_ == 'PROPN'])

    # Apply the function to each element in the dataset
    filtered_dataset = [filter_nouns(sentence) for sentence in tqdm(dataset, desc="Removing filter words from dataset")]

    # Display the result
    # for original, filtered in zip(dataset, filtered_dataset):
    #     print(f"Original: {original}")
    #     print(f"Filtered: {filtered}\n")

    return filtered_dataset

def get_corpus_for_category(dir_path: str, category: str) -> List[str]:
    """ 
    Given some category, returns all documents related to it
    """
    files = os.listdir(dir_path)
    files.sort()
    corpus = []

    for file in tqdm(files, desc="Processing Files", unit="file"):
        base_fname, ext = os.path.splitext(file)

        if ext == ".lab":
            file_path = dir_path+file
            with open(file_path, "r", encoding="utf8") as fp:
                lbls = set(fp.readlines())

                if category in lbls:
                    datafile = dir_path + base_fname + ".txt"
                    corpus.append(datafile)

    return corpus


def term_frequency_inverse_document_frequency(corpus: List[str]) -> Dict[str, float]:
    """ a function to calculated the TF IDF scores of each word in the corpus """

    docs = []
    for document in tqdm(corpus, desc="Collecting document data"):
        data = read_from(document)
        docs.append(data)

    docs = preprocess(docs)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()
    df_tfidf = pd.DataFrame(data=tfidf_matrix.toarray(), columns=feature_names)
    
    top_100_terms = df_tfidf.sum().sort_values(ascending=False).to_csv("terms.csv", header=True)


cps = get_corpus_for_category(TESTDIR, "Database_Administrator")
term_frequency_inverse_document_frequency(cps)
