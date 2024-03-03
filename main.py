# from dspy.teleprompt import BootstrapFewShotWithRandomSearch
# from dspy.evaluate import Evaluate
# import dspy
# from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
# from classifier.decisiontree import ExplainableTreeClassifier
# import openai
import os
import pickle
from typing import Dict, Set, List

from utils.utils import read_from, TESTDIR, LABELS

# client = openai.OpenAI()


def run_binclassifier(dir_path: str, n: int = -1) -> Dict[str, str]:
    """
    Returns a dict of {filename: label}
    """

    file_to_lbl = {}
    files = os.listdir(dir_path)
    files.sort()

    total = 0.0
    s = 0.0

    y_pred = []
    for file in files:
        base, ext = os.path.splitext(file)

        if ext == ".lab":
            lbls = read_from(dir_path + file)
            data = read_from(dir_path + base + ".txt")[0].strip()

            for lbl_line in LABELS:
                try:
                    classifier = ExplainableTreeClassifier(
                        ["Experiences", "Skills"], lbl_line
                    )
                    decision, reason = classifier.classify(data)

                    print(
                        f"FOR CANDIDATE {file}, we {decision} for the position of {lbl_line} because {reason}"
                    )
                    if (lbl_line in lbls and decision) or (
                        lbl_line not in lbls and not decision
                    ):
                        s += 1.0
                    total += 1.0

                    print(f"Running accuracy= {s/total}\r")

                    if n > 0:
                        n -= 1
                    elif n <= 0:
                        break
                except openai.BadRequestError:
                    print("CONTEXT LENGTH EXCEEDED, CONTINUING")

            data = ""
            lbl = ""

    print(f"FINAL accuracy= {s/total}\r")
    return file_to_lbl


if __name__ == "__main__":
    from llm.prompt.cov import ChainOfVerification
    from llm.client.mistral import MistralLLMClient

    client = MistralLLMClient()
    prompter = ChainOfVerification(client)

    query="""
        resume: {"name": "ZAKIYA (Kia) HILL", "educations": [{"gpa": "3.54", "degree": "Bachelors", "majors": ["Economics", "Business"], "grad_year": "2020", "institution": "University of Illinois at Urbana-Champaign"}], "experiences": [{"skills": {"training": 1, "POS system": 1, "scheduling": 1, "customer service": 1}, "impacts": ["Processed cash, credit, and WIC payments", "Encouraged sales of credit cards and product protection plans"], "end_date": "Present", "location": "Bolingbrook, IL", "role_title": "Cashier", "start_date": "December 2017", "organization": "Wal-Mart"}, {"skills": {"visual design": 1, "public speaking": 1, "financial budgeting": 1}, "impacts": ["Prepared and conducted presentations about suicide prevention", "Managed a financial budget for an annual fashion show"], "end_date": "April 2018", "location": "Chicago, IL", "role_title": "Suicide Prevention Ambassador", "start_date": "June 2016", "organization": "Live Out Loud Charity"}, {"skills": {"organization": 1, "academic support": 1, "active listening": 1}, "impacts": ["Provided resources & academic support to middle school students", "Actively listened to students\' issues & offered support"], "end_date": "May 2016", "location": "Bolingbrook, IL", "role_title": "Lead Junior Mentor", "start_date": "August 2015", "organization": "Bolingbrook Community Center"}], "general_skills": {"training": 1, "POS system": 1, "scheduling": 1, "organization": 1, "visual design": 1, "public speaking": 1, "academic support": 1, "active listening": 1, "customer service": 1, "financial budgeting": 1}, "years_of_experience": 3}

        job requirements: Based on the job description provided, here are five precise skills-related qualities that can make a candidate strong for the Machine Learning Engineer position, along with heuristics relevant to the category:

        1. **Cloud Expertise (AWS, Azure, GCP)**:
        Heuristic: Demonstrated experience in designing, deploying, and managing AI infrastructure on cloud platforms like AWS, Azure, or GCP.

        Justification: The job description emphasizes the need for designing, implementing, and maintaining AI infrastructure, which requires a strong background in cloud services.

        2. **Proficiency in Containerization (Docker, Kubernetes)**:
        Heuristic: A history of utilizing containerization technologies such as Docker and Kubernetes for deploying AI models and services.

        Justification: Containerization skills are crucial for implementing and managing containerized solutions for deploying AI models, as mentioned in the job description.

        3. **Scripting Proficiency (Python, Bash)**:
        Heuristic: Evidence of strong scripting skills using Python and Bash for automation and infrastructure management.

        Justification: Strong scripting skills are necessary for automation, which is highlighted in the job description as an essential aspect of the role.

        4. **Knowledge of AI Frameworks (TensorFlow, PyTorch)**:
        Heuristic: Experience in applying popular AI frameworks like TensorFlow or PyTorch for AI model training and inference.

        Justification: Knowledge of AI frameworks is vital for designing, implementing, and optimizing the infrastructure for AI model training and inference, as stated in the job description.

        5. **Problem-Solving and Analytical Skills**:
        Heuristic: A track record of troubleshooting complex issues and providing effective solutions in AI infrastructure and model performance.

        Justification: Strong analytical and problem-solving skills are essential for performance optimization, troubleshooting, and resolving infrastructure-related issues affecting AI model performance, as mentioned in the job description.

        Question: Out of the provided job description and resume, respond with \'accept\' if 
        the resume fullfills most of the requirements of the job description. Respond with 
        \'reject\' if it does not. Provide a reasoning for your response. Make a decision based
        upon the provided information, and if it is unclear, reject the candidate. If the 
        resume is missing a few or more requirements, they should be rejected.

        Answer:
    """
    prompter.prompt(query)
