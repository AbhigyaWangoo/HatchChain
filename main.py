from binclassifier import ExplainableClassifier
import chardet

    
TESTFILE="data/ResumeClassifierData/00001.txt"

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        return result['encoding']

def read_from(filename: str) -> str:
    encoding=detect_encoding(filename)
    
    with open(filename, "r", encoding=encoding) as fp:
        return fp.read()

if __name__ == "__main__":
    classifier = ExplainableClassifier(["work experiences", "Skills", "university"])
    data=read_from(TESTFILE)
    
    decision, reasoning = classifier.classify(data, "Database Administrator")
    print(decision)
    print(reasoning)