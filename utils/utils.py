from typing import Dict, Set, List, Any
from types import MappingProxyType
import os
import json
import chardet
import tqdm
from typeguard import typechecked
from multiprocessing import Process, Lock, Manager, cpu_count

NOLABEL = "nolabel"
TESTDIR = "data/"
LABELS = {
    "Database_Administrator",
    "Project_manager",
    "Java_Developer",
    "Python_Developer",
    "Software_Developer",
    "Web_Developer",
    "Systems_Administrator",
    "Network_Administrator",
}


def detect_encoding(file_path):
    with open(file_path, "rb") as f:
        result = chardet.detect(f.read())
        return result["encoding"]


def read_from(filename: str) -> List[str]:
    """Reads ALL LINES from the provided file."""

    try:
        encoding = detect_encoding(filename)
        with open(filename, "r", encoding=encoding) as fp:
            return fp.read().splitlines()
    except Exception as e:
        print(f"Error reading from file {filename}. Error: {e}")
        return []


def get_all_labels(file_path: str = "normlized_classes.txt") -> Set:
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


def lbl_to_resumeset(
    dir_path: str,
    label_set: Set,
    files: List[str] = [],
    disable: bool = True,
    n_processes: int = 1,
) -> Dict[str, Set[str]]:
    res = {}
    files = os.listdir(dir_path)
    files.sort()

    for file in tqdm.tqdm(files, desc="Gathering data files into DS", disable=disable):
        base, ext = os.path.splitext(file)

        if ext == ".lab":
            lbls = read_from(dir_path + file)
            lbls_found_set = set(lbls)

            if len(label_set) == 0 or (
                len(label_set) > 0 and len(lbls_found_set.intersection(label_set)) > 0
            ):
                data = read_from(dir_path + base + ".txt")

                if len(data) > 0:
                    data = data[0].strip()
                    for lbl_found in lbls:

                        if lbl_found not in res:
                            res[lbl_found] = set()

                        res[lbl_found].add(data)

    return res


def lbl_to_resumeset_multiproc(
    dir_path: str, label_set: Set, disable: bool = True, process_percentage: float = 0.9
):
    dir_files = os.listdir(dir_path)
    dir_files.sort()
    manager = Manager()
    res = manager.dict()

    proccesses = []
    n_processes = int(cpu_count() * process_percentage)
    chunk_size = len(dir_files) // n_processes
    lock = Lock()

    def __read_chunk(lock: Lock, start_idx: int, end_idx: int, rv_acc) -> None:
        chunk = start_idx // chunk_size

        for idx in tqdm.tqdm(
            range(start_idx, end_idx - 1),
            disable=disable,
            desc=f"Processing chunk {chunk} of {n_processes}",
        ):
            cur_file = dir_files[idx]
            base, ext = os.path.splitext(cur_file)

            if ext == ".lab":
                lbls = read_from(dir_path + cur_file)
                lbls_found_set = set(lbls)

                if len(label_set) == 0 or (
                    len(label_set) > 0
                    and len(lbls_found_set.intersection(label_set)) > 0
                ):
                    data = read_from(dir_path + base + ".txt")

                    if len(data) > 0:
                        data = data[0].strip()
                        for lbl_found in lbls:

                            lock.acquire()
                            if lbl_found not in rv_acc:
                                rv_acc[lbl_found] = manager.list()

                            rv_acc[lbl_found].append(data)

                            lock.release()
            idx += 1

    for idx in range(n_processes):
        p = Process(
            target=__read_chunk,
            args=(lock, idx * chunk_size, (idx + 1) * chunk_size, res),
        )
        proccesses.append(p)
        p.start()

    for process in proccesses:
        process.join()

    return res
