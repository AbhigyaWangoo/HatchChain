from query_engine.src.db import postgres_client, rawtxt_client
import multiprocessing
from classifier import decisiontree
from similarity import cosine
import enum
import json
from typing import Dict, Any, Union
import os
from collections import OrderedDict
import numpy as np

SERVER_ROOT_DATAPATH = "data/server/"
if not os.path.exists(SERVER_ROOT_DATAPATH):
    os.mkdir(SERVER_ROOT_DATAPATH)

CLASSIFIERS_ROOT_DATAPATH = "data/classifiers/"
if not os.path.exists(CLASSIFIERS_ROOT_DATAPATH):
    os.mkdir(CLASSIFIERS_ROOT_DATAPATH)
NUM_SIMILAR = 5

active_classifications = set()
active_classifiers = set()

classifier_condition = multiprocessing.Condition()


class HyperparamEnum(enum.Enum):
    EXPERIENCE = "Experiences"
    SKILLS = "Skills"
    EDUCATION = "Education"


def get_job_metadata(job_id: int) -> Dict[Any, Any]:
    """ 
    Reads all metadata for a job, and returns it. This function first checks to see
    if the local disk has a previously saved metadata file, and if not, queries the
    db.

    job_id: job id to retrieve data from
    """

    job_metadata_file = os.path.join(
        SERVER_ROOT_DATAPATH, f"{str(job_id)}.json")
    job_metadata = {}

    if not os.path.exists(job_metadata_file):
        client = postgres_client.PostgresClient(job_id)
        job_metadata = client.read_job(job_metadata_file)
    else:
        with open(job_metadata_file, 'r', encoding="utf8") as file:
            job_metadata = json.load(file)

    return job_metadata


def get_classifier(job_id: int, save_new_dt: bool = True) -> Union[str, decisiontree.ExplainableTreeClassifier]:
    """
    This function either loads a classifier from local disk, or creates a new one 
    if the classifier has never been created. By default, it will save all newly made
    classifiers to local disk.

    job_id: The id of the job to construct a classifier for
    save_new_dt: Whether or not to save a newly constructed classifier to disk or not.
    """

    if job_id in active_classifiers:
        return f"job {job_id} already has a classifier being created for it. Exiting this call..."

    # Read job data
    job_metadata = get_job_metadata(job_id)
    if job_metadata == {}:
        return f"job {job_id} already has a classifier being created for it. Exiting this call..."

    classifier_path = os.path.join(CLASSIFIERS_ROOT_DATAPATH, f"{job_id}.json")

    # Check local datapath for classifier
    if os.path.exists(classifier_path):
        print("Found classifier in local disk")
        classifier = decisiontree.ExplainableTreeClassifier(
            [], "", classifier_path, consider_keywords=False)
        return classifier

    # Check job data
    if job_metadata['classifier_data'] is not None:
        classifier_metadata = job_metadata['classifier_data']
        with open(classifier_path, "w", encoding="utf8") as fp:
            fp.write(json.dumps(classifier_metadata))

    try:
        with classifier_condition:
            active_classifiers.add(job_id)
            # Never seen classifier before, need to create new one
            hyperparams = [HyperparamEnum.SKILLS.value,
                           HyperparamEnum.EXPERIENCE.value,
                           HyperparamEnum.EDUCATION.value]
            category = job_metadata['title']
            # TODO basic classifier without keywords. Need to up accuracy.
            classifier = decisiontree.ExplainableTreeClassifier(
                hyperparams=hyperparams,
                job_description=job_metadata,
                category=category,
                consider_keywords=False)

            if save_new_dt:
                classifier.save_model(classifier_path)

            active_classifiers.remove(job_id)
            classifier_condition.notify()

        return classifier
    except Exception as e:
        if job_id in active_classifiers:
            active_classifiers.remove(job_id)
        # print(f"Classifier creation on job {job_id} failed, error: {e}")
        return f"Classifier creation on job {job_id} failed, error: {e}"


def set_classifier(job_id: int, classifier_metadata: str):
    """ 
    Sets the classifier in the pgres job table.

    job_id: id of job to update
    classifier: classifier metadata to update in db
    """
    client = postgres_client.PostgresClient(job_id)

    print("Created client")

    client.update_job(postgres_client.CLASSIFIER_DATA_FIELD,
                      classifier_metadata)

    # Small note: Technically, the SERVER_ROOT_DATAPATH/<job_id>.json can potentially have
    # the job metadata, but not the classifier metadata in the classifier_data field if
    # the db didn't have any classifier data when the job metadata was cached by this server.
    print("updated job")


def create_classifier_wrapper(job_id: int):
    """
    A wrapper for the creation of a classifier. see the endpoint /create-classifier for more info.
    """

    # 1. If classifier has been made before, just return (check local, if not, check db). Otherwise, feed the category into classifier
    res = get_classifier(job_id)

    if isinstance(res, str):
        print(res)
    else:
        classifier_path = os.path.join(
            CLASSIFIERS_ROOT_DATAPATH, f"{job_id}.json")
        with open(classifier_path, "r", encoding="utf8") as fp:
            saved_model = json.loads(fp.read())

            # 2. Once classifier framework is constructed, read it from the file, and place data back into db.
            set_classifier(job_id, saved_model)


def create_classification_wrapper(job_id: int, resume_id: int):
    """ 
    This function creates a classification for the resume data provided by the resume id
    and the job id, and updates the value in the job_resumes table. It serves as a 
    wrapper that the async and sync endpoints can call.

    job_id: id to get tree construction for.
    resume_id: id to classify
    """

    if (job_id, resume_id) in active_classifications:
        duplicate_msg = f'''There is already an active classification 
                occurring on resume {resume_id} for job {job_id}'''
        # print(duplicate_msg)
        return {
            "reccommendation": False,
            'reasoning': "",
            'message': duplicate_msg
        }
    while job_id in active_classifiers:
        with classifier_condition:
            classifier_condition.wait()

    active_classifications.add((job_id, resume_id))

    try:
        # 1. declare client
        client = postgres_client.PostgresClient(job_id)

        # 2. Try to read prev job_resume binding.
        # TODO add disk level caching for classification reasoning?
        dbcached = client.read_job_resume(
            resume_id,
            postgres_client.CLASSIFICATION_DATA_FIELD,
            postgres_client.CLASSIFICATION_REASONING_DATA_FIELD)
        accept, reasoning = dbcached[0], dbcached[1]

        if accept is None or reasoning is None:
            print("First time classifying candidate.")

            # 3. Reading client metadata
            candidate_metadata = client.read_candidate(
                resume_id, postgres_client.RESUME_DATA_FIELD)
            strdata = json.dumps(candidate_metadata)

            classifier = get_classifier(job_id, False)
            accept, reasoning = classifier.classify(strdata)

            if accept == decisiontree.ClassificationOutput.TIE:
                print(f"Running tiebreaker on resume {resume_id}")
                res = tiebreak(resume_id, job_id)

                if res == decisiontree.ClassificationOutput.TIE:
                    # Conditional wait
                    accept = False  # TODO remove me
                    pass
                else:
                    accept = res == decisiontree.ClassificationOutput.ACCEPT

            # 4. set job_resumes explanation field, if not run before
            db_update = {
                postgres_client.CLASSIFICATION_DATA_FIELD: accept,
                postgres_client.CLASSIFICATION_REASONING_DATA_FIELD: reasoning
            }
            client.update_job_resume(resume_id, **db_update)

        # 5. Return metadata and success message
        active_classifications.remove((job_id, resume_id))
        return {
            "reccommendation": accept,
            'reasoning': reasoning,
            'message': f'Successfully classified and added reasoning to job table for job id {job_id}'
        }

    except Exception as e:
        active_classifications.remove((job_id, resume_id))
        err_msg = f"Classification on resume {resume_id} failed, error: {e}"
        print(e)

        return {
            "reccommendation": False,
            'reasoning': "",
            'message': err_msg
        }


def get_k_similar(job_id: int,
                  resume_id: int,
                  k: int,
                  threshold: float = 0.55,
                  raw: bool = True) -> OrderedDict[int, decisiontree.ResumeModel]:
    """ 
    This function gets the top k similar resumes from the db and 
    returns their metadata. It will return either the full raw text, or the 
    vectors, dependant on the 'raw' flag.

    job_id: int
    resume_id: int
    k: int
    raw: bool

    returns: A dict of resume_id: (explainable classification, raw text value)
    """
    rv = []

    # 1. Fetch all resume vectors for job id provided. Place into {res_id: vector} dict.
    client = postgres_client.PostgresClient(job_id)
    resumes = client.read_candidates_from_job("", False, True)

    if len(resumes) == 0:
        print("No candidates associated with the job. cant get K similar candidates")
        return rv

    vectors = {}
    classifications = {}
    res_vector = None
    for resume in resumes:
        res_id = resume[0]

        res = client.read_job_resume(
            res_id, postgres_client.VECTOR_DATA_FIELD, postgres_client.CLASSIFICATION_DATA_FIELD)
        vec = res[0]
        classification = res[1]

        if int(res_id) == resume_id:
            res_vector = vec
        else:
            vectors[res_id] = vec
            classifications[res_id] = classification

    if res_vector is None:
        print("Could not find this resume's vector. Aborting for now.")
        return rv

    # 2. Calc cosine similarity with provided resume id and all others.
    similarity_calculator = cosine.CosineSimilarity()
    this_vector = np.array(res_vector)

    # 3. get top k resume ids.
    vectors = similarity_calculator.top_k(this_vector, vectors, k)

    # 4. Use ddb client to retrive from raw text store, or just return jsons
    # retrieved earlier from postgres
    if raw:
        txt_client = rawtxt_client.ResumeDynamoClient(job_id)
        pgres_client = postgres_client.PostgresClient(job_id)

        resume_data = pgres_client.read_candidate(
            resume_id, postgres_client.RESUME_DATA_FIELD)[0]
        raw_txt_data = txt_client.batch_get_resume(
            list(vectors.keys()), "", save_to_file=False, return_txt=True)

        # 5. Return list (in order of closest) with raw text
        final_dict = {}
        for vec in vectors:
            sim = similarity_calculator.compute_similarity(
                this_vector, np.array(vectors[vec]))

            print(f"Similarity between {resume_id} and {vec}: {sim}")

            if sim >= threshold:
                final_dict[vec] = decisiontree.ResumeModel(id=vec,
                                                           name=resume_data[decisiontree.NAME],
                                                           raw_data=raw_txt_data[vec],
                                                           vector=vectors[vec],
                                                           explainable_classification=classifications[vec])

        return final_dict

    return vectors


def tiebreak(resume_id: int, job_id: int) -> decisiontree.ClassificationOutput:
    """
    This is the tiebreaker function. If the |wincount - losscount| <= 1, this function
    will decide whether to push the candidate into or out of the pool depending on
    the k nearest neighbors, and whether a majority of them would be accepted or rejected.

    resume_id: id of the resume to fetch
    job_id: id of the job the resume belongs to
    """

    # 1. call top k function
    # 2. find top k resumes, and find majority acceptance rate
    similar_resumes = get_k_similar(
        job_id, resume_id, k=NUM_SIMILAR, raw=True)

    num_accept = 0
    total = 0
    for resume_id in similar_resumes:
        if similar_resumes[resume_id].explainable_classification is not None:
            total += 1
            if similar_resumes[resume_id].explainable_classification:
                num_accept += 1

    # 3. Reshape reasoning list depending on a) whether we're accepting or
    # rejecting based on tiebreaker, and b) the names of the candidates who also were included in the list.
    if total:
        if num_accept/total > .5:
            return decisiontree.ClassificationOutput.ACCEPT
        elif num_accept/total < .5:
            return decisiontree.ClassificationOutput.REJECT

    print(f"FINAL TIEBREAKER on resume {resume_id}")
    return decisiontree.ClassificationOutput.TIE
