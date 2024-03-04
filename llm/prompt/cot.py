from . import base as prompter
from llm.client import base as llm
from enum import Enum

ZERO_SHOT_PROMPT = "Let's think step by step"

FEW_SHOT_PROMPT = """
Example 1:
Question: You have a candidate and a label. On the bases of the following heuristcs here: 
1. Strong programming skills in languages such as Python, PyTorch, and AWS/GCP are crucial for a machine learning engineer position at ElectroKare. A candidate with a strong background in programming will be able to develop and implement complex algorithms and models quickly and efficiently.
2. Good communication skills are essential for this role, as the machine learning engineer will need to collaborate with cross-functional teams, explain technical concepts to non-technical stakeholders, and present findings to senior management. A candidate with excellent communication skills will be able to effectively convey their ideas and insights to various audiences.
3. Experience in machine learning research, development, and critical thinking abilities are highly desirable for this position. A candidate with a strong background in machine learning will be able to design and implement innovative solutions that drive business growth and improve customer experiences.
4. Knowledge of cloud computing platforms such as AWS or GCP is important for this role, as the machine learning engineer will need to deploy models on these platforms and manage large datasets in a scalable manner. A candidate with experience in cloud computing will be able to leverage the power of cloud technologies to accelerate the development process and improve model performance.
5. The ability to work independently or as part of a team is crucial for success in this role, as the machine learning engineer may need to work on complex projects without direct supervision or collaborate with other engineers on large-scale initiatives. A candidate with strong independent work ethic or teamwork skills will be able to adapt quickly to new challenges and contribute meaningfully to the team's efforts." 
decide whether the following candidate: 
[{"name": "Vardhan Shorewala", "email": "vardhan@berkeley.edu", "links": ["linkedin.com/in/vardhan-shorewala-b85543172/", "github.com/vardhan-shorewala"], "phone": "510-761-1001", "educations": [{"gpa": "3.94", "degree": "Bachelors", "majors": ["Electrical Engineering", "Computer Science", "Business"], "grad_year": "2025", "institution": "University of California, Berkeley"}], "experiences": [{"skills": {"C": 1, "CI": 1, "S3": 1, "AWS": 1, "EC2": 1, "Ruby": 1, "MLOps": 1, "MySQL": 1, "React": 1, "OpenCV": 1, "Fastify": 1, "AkkaHTTP": 1, "Selenium": 1, "Solidity": 1, "AI models": 1, "CD DevOps": 1, "AWS lambda": 1, "MERN stack": 1, "TensorFlow": 1, "UiPath RPA": 1, "Material UI": 1, "Kalman filter": 1, "MetaMask Snap": 1, "AWS Cloud Stack": 1, "Machine Learning": 1, "supply-chain optimization": 1, "GAN-based machine learning models": 1, "Semaphore zero-knowledge cryptography": 1, "bipartite signal processing algorithms": 1}, "impacts": ["Improved match-rate by 35% in supply-chain optimization", "Diminished upstream API calls by 23x, saving upwards of $100,000 monthly", "Validated software architecture with SNS messages, deploying AWS Cloud Stack through CI/CD DevOps for 50K+ vendors"], "end_date": "Aug 2023", "location": "Seattle, WA", "role_title": "Software Engineering Intern", "start_date": "May 2023", "organization": "Amazon"}, {"skills": {"Ruby": 1, "MLOps": 1, "MySQL": 1, "React": 1, "Fastify": 1, "Polygon": 1, "Solidity": 1, "AI models": 1, "MERN stack": 1, "UiPath RPA": 1, "Material UI": 1, "RESTful API": 1, "MetaMask Snap": 1, "Machine Learning": 1, "L1 to L2 transactions": 1, "GAN-based machine learning models": 1, "Semaphore zero-knowledge cryptography": 1}, "impacts": ["Productionized a Voting dApp on Polygon using Solidity and Semaphore zero-knowledge cryptography", "Engineered a React-based dApp and MetaMask Snap for efficient L1 to L2 transactions without bridging contracts for Arbitrium"], "end_date": "Present", "location": "San Francisco, CA", "role_title": "Algorithm Developer", "start_date": "Jan 2023", "organization": "Blockchain at Berkeley"}, {"skills": {"C": 1, "Ruby": 1, "MLOps": 1, "MySQL": 1, "React": 1, "AkkaHTTP": 1, "AI models": 1, "MERN stack": 1, "UiPath RPA": 1, "Material UI": 1, "Machine Learning": 1, "GAN-based machine learning models": 1}, "impacts": ["Designed a robust RESTful API with the MERN stack and employed Material UI and UiPath RPA to create a user-friendly frontend and interface for Astics, streamlining data storage costs by 15%", "Engineered Ruby and MySQL docker container for Hummingbird, which generated transaction data utilizing CycleGANs and ran optimized SQL queries to identify previous fraudulent transactions, catapulting detection accuracy by 20%"], "end_date": "Jul 2023", "location": "San Francisco, CA", "role_title": "Software Developer", "start_date": "Aug 2022", "organization": "Codebase"}, {"skills": {"PCA": 1, "AI models": 1, "Machine Learning": 1}, "impacts": ["Worked under Prof. Bouchard (UCSF/LBNL) to understand biophysical properties of neurons by developing tree and density-based clustering models, trimming interference latency of existing AI models by 65%", "Cut dimensionality variance of features by 45% through statistical methods like PCA, modeling noise from quantitative data"], "end_date": "Aug 2022", "location": "Berkeley, CA", "role_title": "Machine Learning Research Assistant", "start_date": "Jan 2022", "organization": "University of California, Berkeley"}, {"skills": {"AI models": 1, "Machine Learning": 1, "GAN-based machine learning models": 1}, "impacts": ["Devised a novel density-based clustering algorithm for detection of anomalous data points in high-dimensional spaces and big data sets, with ongoing efforts to improve results with GAN-based machine learning models", "Slashed cluster variance by 58% and improved accuracy by 22%, utilizing statistics for data-driven quantitative analytics"], "end_date": "Oct 2022", "location": "Mumbai, India", "role_title": "Machine Learning Research Lead", "start_date": "Sep 2021", "organization": "IEEE"}, {"skills": {"S3": 1, "AWS": 1, "EC2": 1, "MLOps": 1, "OpenCV": 1, "Selenium": 1, "AWS lambda": 1, "TensorFlow": 1, "Kalman filter": 1}, "impacts": ["Implemented real-time user lip coordinate tracking and speech classification with TensorFlow and OpenCV to tackle speech pathology and help individuals with speech impediments, achieving 87% pattern recognition", "Compiled several thousand labeled data points using custom Selenium web scrapers, reducing noise with Kalman filter", "Productionized backend on EC2, with S3 storage buckets and AWS lambda functions, slashing time latencies by around half"], "end_date": "Nov 2021", "location": "Cambridge, MA", "role_title": "Full-Stack Developer", "start_date": "Nov 2021", "organization": "SilverAI"}, {"skills": {"Go": 1, "Flask": 1, "React": 1, "Node.JS": 1, "Postman": 1, "BERT NLP": 1, "Postgres": 1}, "impacts": ["Led team to build a healthcare forum web application on Solana using Go, converting user posts into tradeable NFT tokens", "Incorporated Bootstrap React to create a seamless, personalized frontend and utilized BERT NLP to perform sentiment analysis of comments and posts, boosting concurrent users to several hundreds", "Cut costs by utilizing efficient RESTful Flask and Node.JS APIs on a Heroku server with a Postgres database and conducted rigorous testing using Postman, ensuring robust functionality and stability of data streams"], "end_date": "Aug 2022", "location": "San Francisco, CA", "role_title": "Backend Developer", "start_date": "Aug 2022", "organization": "Faiza"}], "general_skills": {"C": 1, "CD": 1, "CI": 1, "Go": 1, "AWS": 1, "C++": 1, "SQL": 1, "Bash": 1, "Java": 1, "Node": 1, "Ruby": 1, "Flask": 1, "Keras": 1, "Linux": 1, "NumPy": 1, "Docker": 1, "Matlab": 1, "OpenCV": 1, "Pandas": 1, "Python": 1, "VSCode": 1, "Vue.JS": 1, "Postman": 1, "ReactJS": 1, "Seaborn": 1, "HTML & CSS": 1, "JavaScript": 1, "Jupyter Notebooks": 1, "Github Version Controller": 1}, "years_of_experience": 3}] 
fits the category of machine learning engineer. When providing a reasoning, only reference the specific heuristics provided,
all your lines of reasoning should be relevant to the provided heuristic.

Answer:
In order to decide whether to accept or reject Vardhan, we need to assess their data with respect to the heuristics provided.
1. A candidate applying to work as a machine learning engineer at ElectroKare needs experience with Python, Pytorch, and AWS/GCP. Vardhan does have experience in Python and AWS, but there's no indication of specific ML research or experience using Machine learning tools in conjunction with cloud platforms, so he wouldn't fullfill the requirements for this heuristic.
2. Good communication skills are nessecary, and Vardhan has a few indicators that he has good communication skills. He is part of the M.E.T program at berkeley, which emphasizes buisness knowledge. He also has been in several different work environments, including venture capital, software engineering, and consulting teams, making it likely that he has proper communication skills. 
3. Vardhan will need solid experience in machine learning research and development. While he has worked in a machine learning lab before and has had an internship in machine learning, it was nearly 2 years ago, and he has not had much experience since, making it unlikely for him to satisfy this heuristic.
4. Cloud computing platforms like AWS and GCP need to be well known for vardhan to succeed in this role. Vardhan has worked with multiple different AWS platforms, so he most likely will succeed regarding any cloud computing tasks.
5. To be a contributing member of ElectroKare, Vardhan needs to be comfortable working in teams or individually. He has been in a variety of roles that require both collaborative and individual contributions, including software engineering, consulting, and research, thus indivating that he can work independantly and collaborate to a certain degree.
Based on the above thoughts, while Vardhan has a lot of solid experience in software engineering and computer science, his direct experience in machine learning is limited. A candidate who would
excel in this role must have done extensive research on the subject, demonstrating their expertise through their work experience and qualifications, and Vardhan does not quite have the relevant 
qualifications. Based on that, I would reject him for the folr of machine learning engineer.

Example 2:
Question: You have a candidate and a label. On the bases of the following heuristcs here: 
1. Strong programming skills in languages such as Python, PyTorch, and AWS/GCP are crucial for a machine learning engineer position at ElectroKare. A candidate with a strong background in programming will be able to develop and implement complex algorithms and models quickly and efficiently.
2. Good communication skills are essential for this role, as the machine learning engineer will need to collaborate with cross-functional teams, explain technical concepts to non-technical stakeholders, and present findings to senior management. A candidate with excellent communication skills will be able to effectively convey their ideas and insights to various audiences.
3. Experience in machine learning research, development, and critical thinking abilities are highly desirable for this position. A candidate with a strong background in machine learning will be able to design and implement innovative solutions that drive business growth and improve customer experiences.
4. Knowledge of cloud computing platforms such as AWS or GCP is important for this role, as the machine learning engineer will need to deploy models on these platforms and manage large datasets in a scalable manner. A candidate with experience in cloud computing will be able to leverage the power of cloud technologies to accelerate the development process and improve model performance.
5. The ability to work independently or as part of a team is crucial for success in this role, as the machine learning engineer may need to work on complex projects without direct supervision or collaborate with other engineers on large-scale initiatives. A candidate with strong independent work ethic or teamwork skills will be able to adapt quickly to new challenges and contribute meaningfully to the team's efforts." 
decide whether the following candidate: 
{
  "name": "Aayush Joshi",
  "email": "ajoshi16@calpoly.edu",
  "links": [
    "www.linkedin.com/in/aayushjoshi",
    "github.com/aayushjoshi16"
  ],
  "phone": "+1 (415)-812-9146",
  "Github": {
    "aayushjoshi16/CS-Projects": {
      "languages": {
        "C": 18659,
        "Makefile": 159
      },
      "num_forks": 0,
      "num_stars": 0,
      "user_commits": 4,
      "total_commits": 4
    }
  },
  "educations": [
    {
      "gpa": "3.6",
      "degree": "Bachelors",
      "majors": [
        "Computer Science"
      ],
      "grad_year": "2025",
      "institution": "California Polytechnic State University, San Luis Obispo"
    }
  ],
  "experiences": [
    {
      "skills": {
        "SQL": 1,
        "Numpy": 1,
        "Pandas": 1,
        "Python": 1,
        "Documentation": 1,
        "Code Optimization": 1,
        "Data Optimization": 1
      },
      "impacts": [
        "Developed computational techniques for galaxy analysis",
        "Optimized program flow by 30%",
        "Improved space and time complexity by 30%"
      ],
      "end_date": "Present",
      "location": "San Luis Obispo, CA",
      "role_title": "Software Engineering Researcher",
      "start_date": "August 2023",
      "organization": "California Polytechnic State University"
    }
  ],
  "general_skills": {
    "C": 1,
    "C++": 1,
    "Git": 1,
    "SQL": 1,
    "HTML": 1,
    "Java": 1,
    "Unix": 1,
    "Linux": 1,
    "Numpy": 1,
    "Pandas": 1,
    "Python": 1,
    "VS Code": 1,
    "JavaScript": 1,
    "Jupyter Notebooks": 1
  },
  "years_of_experience": 0
}
fits the category of machine learning engineer. When providing a reasoning, only reference the specific heuristics provided,
all your lines of reasoning should be relevant to the provided heuristic.

Answer:
In order to decide whether to accept or reject Aayush, we need to assess their data with respect to the heuristics provided.
1. A candidate applying to work as a machine learning engineer at ElectroKare needs experience with Python, Pytorch, and AWS/GCP. While Aayush has expereince in Python, he has no relevant cloud based skills or pytorch relevant skills. Therefore, he wouldn't fullfill the requirements for this heuristic.
2. Good communication skills are nessecary, But Aayush has no direct indicators of having good communication capabilities. Most of his experience is technical, and while he might have some experinces in somewhat collaborative environments, the data provided doesn't quite showcase his communication capabilities, so he would not be a good fit for this heuristic.
3. Aayush will need solid experience in machine learning research and development. Unfortunately, Aayush has not worked in any Machine learning based roles either in academia or industry, thus making them not a good fit for this heuristic.
4. Cloud computing platforms like AWS and GCP need to be well known for Aayush to succeed in this role. As stated before, however, he has limited to no experience in that field, making him a bad fit for this heuristic as well.
5. To be a contributing member of ElectroKare, Aayush needs to be comfortable working in teams or individually. He might have had some experience collaborating in teams and definately has experience working individually as a lead cybersecurity engineer. Therefore, he would be a good fit for this heuristic.
Based on the above reasonings, Aayush should be rejected for the role of Machine learning engineer. Most of his experience and skills relate to cybersecurity, and while he does have experience in collaborative environments
and has good communicatin skills, his level of skills is not enough to warrant an acceptance to this position. Overall, the heuristics require to succeed in this role are too different from those that Aayush possesses, 
making it logical to reject him for this role.
"""


class CotType(Enum):
    """Enum for type of COT"""

    ZERO_SHOT = 0
    FEW_SHOT_COT = 1


class ChainOfThoughtPrompter(prompter.Prompter):
    """
    A class for chain of thought prompting. Based off paper
    """

    def __init__(
        self, client: llm.AbstractLLM, type: CotType = CotType.ZERO_SHOT
    ) -> None:
        self._client = client
        self._type = type

    def prompt(self, prompt: str) -> str:
        """
        The abstract method for a prompter to execute a prompt
        """
        formatted_prompt = "Q: " + prompt + "\n" + "A:"

        if self._type == CotType.ZERO_SHOT:
            formatted_prompt += ZERO_SHOT_PROMPT
        elif self._type == CotType.FEW_SHOT_COT:
            print("using few shot")
            formatted_prompt += FEW_SHOT_PROMPT

        return self._client.query(formatted_prompt)
