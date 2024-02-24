from fastapi import FastAPI, BackgroundTasks
import uvicorn
from dotenv import load_dotenv
import os

from utils import server_utils

app = FastAPI()
load_dotenv()

EXPLAINABLE_CLASSIFIER_ENDPOINT = os.environ.get("EXPLAINABLE_CLASSIFIER_ENDPOINT")


@app.get("/create-classifier")
def create_classifier(job_id: int, background_tasks: BackgroundTasks):
    """
    This endpoint retreives the data required from the relevant job, and passes it to the classifier
    to construct a json file. Then, it uploads said file to the pgres table's job table.

    job_id: id to construct tree for.
    """

    background_tasks.add_task(server_utils.create_classifier_wrapper, job_id)
    return {"message": "Classifier creation task has been added to background tasks."}


@app.get("/create-classifier-sync")
def create_classifier_sync(job_id: int):
    """
    The synchronus equivalent of creating a classifier for the provided job. See the
    /create-classifierfor more info

    job_id: id to construct tree for.
    """

    server_utils.create_classifier_wrapper(job_id)
    return {
        "message": "Classifcation complete, job tables should be updated with classifier metadata"
    }


@app.get("/create-classification-sync")
def create_classification_sync(job_id: int, resume_id: int):
    """
    This endpoint creates a classification for the resume data provided by the resume id
    and the job id, and updates the value in the job_resumes table

    job_id: id to get tree construction for.
    resume_id: id to classify
    """
    return server_utils.create_classification_wrapper(job_id, resume_id)


@app.get("/create-classification")
def create_classification(
    job_id: int, resume_id: int, background_tasks: BackgroundTasks
):
    """
    This endpoint creates a classification for the resume data provided by the resume id
    and the job id, and updates the value in the job_resumes table

    job_id: id to get tree construction for.
    resume_id: id to classify
    """
    background_tasks.add_task(
        server_utils.create_classification_wrapper, job_id, resume_id
    )
    return {
        "reccommendation": False,
        "reasoning": "",
        "message": f"Started background task for classification of resume {resume_id} under job {job_id}",
    }


if __name__ == "__main__":
    if EXPLAINABLE_CLASSIFIER_ENDPOINT is None:
        print(
            "Please set the EXPLAINABLE_CLASSIFIER_ENDPOINT var in the .env file. Exiting..."
        )
    else:
        port = EXPLAINABLE_CLASSIFIER_ENDPOINT.split(":")[-1]
        port = int(port.replace("/", ""))
        uvicorn.run(app, host="0.0.0.0", port=port)
