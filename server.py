from fastapi import FastAPI, BackgroundTasks
from query_engine.src.db import postgres_client
from classifier import decisiontree
import enum
import json
import uvicorn
from dotenv import load_dotenv
import os
from typing import Dict, Any

app = FastAPI()
load_dotenv()

EXPLAINABLE_CLASSIFIER_ENDPOINT = os.environ.get(
    "EXPLAINABLE_CLASSIFIER_ENDPOINT")

SERVER_ROOT_DATAPATH = "data/server/"
if not os.path.exists(SERVER_ROOT_DATAPATH):
    os.mkdir(SERVER_ROOT_DATAPATH)

CLASSIFIERS_ROOT_DATAPATH = "data/classifiers/"
if not os.path.exists(CLASSIFIERS_ROOT_DATAPATH):
    os.mkdir(CLASSIFIERS_ROOT_DATAPATH)


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


def get_classifier(job_id: int, save_new_dt: bool = True) -> decisiontree.ExplainableTreeClassifier:
    """
    This function either loads a classifier from local disk, or creates a new one 
    if the classifier has never been created. By default, it will save all newly made
    classifiers to local disk.

    job_id: The id of the job to construct a classifier for
    save_new_dt: Whether or not to save a newly constructed classifier to disk or not.
    """

    # Read job data
    job_metadata = get_job_metadata(job_id)
    if job_metadata == {}:
        return {"Success": False, "Message": f"Could not find job metadata for job {job_id} in local disk or db"}

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

    # Never seen classifier before, need to create new one
    hyperparams = [HyperparamEnum.SKILLS.value,
                   HyperparamEnum.EXPERIENCE.value]
    category = job_metadata['title']
    # TODO basic classifier without keywords. Need to up accuracy.
    classifier = decisiontree.ExplainableTreeClassifier(
        hyperparams, category, consider_keywords=False)

    if save_new_dt:
        classifier.save_model(classifier_path)

    return classifier


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
    get_classifier(job_id)
    classifier_path = os.path.join(CLASSIFIERS_ROOT_DATAPATH, f"{job_id}.json")
    with open(classifier_path, "r", encoding="utf8") as fp:
        saved_model = json.loads(fp.read())

        # 2. Once classifier framework is constructed, read it from the file, and place data back into db.
        set_classifier(job_id, saved_model)


@app.get("/create-classifier")
def create_classifier(job_id: int, background_tasks: BackgroundTasks):
    """ 
    This endpoint retreives the data required from the relevant job, and passes it to the classifier
    to construct a json file. Then, it uploads said file to the pgres table's job table.

    job_id: id to construct tree for.
    """

    background_tasks.add_task(create_classifier_wrapper, job_id)
    return {"message": "Classifier creation task has been added to background tasks."}


@app.get("/create-classifier-sync")
def create_classifier_sync(job_id: int):
    """ 
    The synchronus equivalent of creating a classifier for the provided job. See the 
    /create-classifierfor more info

    job_id: id to construct tree for.
    """

    create_classifier_wrapper(job_id)
    return {"message": "Classifcation complete, job tables should be updated with classifier metadata"}

@app.get("/create-classification")
def create_classification(job_id: int, resume_id: int):
    """ 
    This endpoint creates a classification for the resume data provided by the resume id
    and the job id, and updates the value in the job_resumes table

    job_id: id to get tree construction for.
    resume_id: id to classify
    """

    # 1. Get classifier based on job id
    # 2. get resume from db, and classify resume
    # 3. set job_resumes explanation field


if __name__ == "__main__":
    if EXPLAINABLE_CLASSIFIER_ENDPOINT is None:
        print("Please set the EXPLAINABLE_CLASSIFIER_ENDPOINT var in the .env file. Exiting...")
    else:
        port = EXPLAINABLE_CLASSIFIER_ENDPOINT.split(":")[-1]
        port = int(port.replace("/", ""))
        uvicorn.run(app, host="0.0.0.0", port=port)
