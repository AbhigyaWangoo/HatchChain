from typing import Dict, Set, List
import os
import chardet
import tqdm

NOLABEL = "nolabel"
TESTDIR = "data/"

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        return result['encoding']


def read_from(filename: str) -> List[str]:
    """ Reads ONE LINE from the provided file."""

    try:
        encoding = detect_encoding(filename)
        with open(filename, "r", encoding=encoding) as fp:
            return fp.readlines()
    except Exception as e:
        print(f"Error reading from file {filename}. Error: {e}")
        return []


def get_all_labels(file_path: str = 'normlized_classes.txt') -> Set:
    fin = set()
    with open(file_path, encoding=detect_encoding(file_path)) as fp:
        # Iterate through each line in the file
        for line in fp:
            # Process each line as needed
            res = line.split(":")
            lbls = res[-1].split(",")
            fin = fin.union(set(lbls))

    print(fin)
    return fin


def lbl_to_resumeset(dir_path: str, label_set: Set, disable: bool=True) -> Dict[str, Set[str]]:
    res = {}
    files = os.listdir(dir_path)
    files.sort()

    for file in tqdm.tqdm(files, desc="Gathering data files into DS", disable=disable):
        base, ext = os.path.splitext(file)

        if ext == ".lab":
            lbls = read_from(dir_path + file)
            lbls_found_set = set(lbls)

            if len(label_set) == 0 or (len(label_set) > 0 and len(lbls_found_set.intersection(label_set)) > 0):
                data = read_from(dir_path + base + ".txt")
                
                if len(data) > 0:
                    data = data[0].strip()
                    for lbl_found in lbls:

                        if lbl_found not in res:
                            res[lbl_found] = set()

                        res[lbl_found].add(data)
    
    return res
