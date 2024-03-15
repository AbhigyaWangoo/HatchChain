import os
import json
import csv

DIRSRC="reports"

def process_directory(directory_path):
    # Iterate through each file in the directory and its subdirectories
    for root, dirs, files in os.walk(directory_path):
        for file_name in files:
            if file_name.endswith('.json'):
                file_path = os.path.join(root, file_name)


                try:
                    with open(file_path, "r", encoding="utf8") as fp:
                        dataset = json.load(fp)
                        dataset_data = dataset["dataset"]

                        for resume_explanation in dataset_data:
                            if resume_explanation["resume"][1].get("links", None):
                                resume_explanation["resume"] = resume_explanation["resume"][1]["links"]
                            elif resume_explanation["resume"][1].get("name", None):
                                resume_explanation["resume"] = resume_explanation["resume"][1]["name"]

                        # print(dataset)# Write the modified JSON
                        ofilename=file_path.split("/")[-1]
                        with open(f"clean-{ofilename}", "w", encoding="utf8") as fp:
                            json.dump(dataset, fp)

                except Exception as e:
                    print(f"error writing customer back: {e}")

def convert_to_csv(json_file: str):
    """Converts the provided json file into a csv"""
    headers=["name", "linkedin links", "reasoning"]
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)

process_directory(DIRSRC)
