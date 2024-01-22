from fastapi import FastAPI
import uvicorn
from dotenv import load_dotenv
import os

app = FastAPI()
load_dotenv()

EXPLAINABLE_CLASSIFIER_ENDPOINT = os.environ.get(
    "EXPLAINABLE_CLASSIFIER_ENDPOINT")


@app.get("/create-classifier")
def create_classifier(job_id: int):
    """ 
    This endpoint retreives the data required from the relevant job, and passes it to the classifier
    to construct a json file. Then, it uploads said file to the pgres table's job table.

    job_id: id to construct tree for.
    """
    # 1. Read job data
    # 2. If classifier has been made before, just return (check local, if not, check db). Otherwise, feed the category into classifier
    # 3. Once classifier framework is constructed, read it from the file, and place data back into db.

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
        print("Please set the RANKER_ENDPOINT var in the .env file. Exiting...")
    else:
        port = EXPLAINABLE_CLASSIFIER_ENDPOINT.split(":")[-1]
        port = int(port.replace("/", ""))
        uvicorn.run(app, host="0.0.0.0", port=port)
