import requests
import os
import json
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {"Authorization": f"token {TOKEN}"}

OWNER = "apache"
REPO = "airflow"
PER_PAGE = 100
MAX_ISSUES = 200  # You can change

def safe_get(url, params=None):
    response = requests.get(url, headers=HEADERS, params=params)

    if response.status_code != 200:
        print("Error:", response.status_code)
        print("Response:", response.text[:200])
        return None

    try:
        return response.json()
    except Exception as e:
        print("JSON error:", e)
        return None

def get_issues():
    issues = []
    page = 1

    while len(issues) < MAX_ISSUES:
        url = f"https://api.github.com/repos/{OWNER}/{REPO}/issues"
        params = {
            "state": "all",
            "per_page": PER_PAGE,
            "page": page
        }

        data = safe_get(url, params)
        if not data:
            break


        for issue in data:
            if "pull_request" not in issue:  # skip PRs
                issues.append(issue)

        page += 1
        print(f"Downloaded {len(issues)} issues")

    return issues[:MAX_ISSUES]


import json
import requests

def get_comments(issue_number):
    comments = []
    page = 1

    while True:
        url = f"https://api.github.com/repos/{OWNER}/{REPO}/issues/{issue_number}/comments"
        params = {"per_page": 100, "page": page}

        data = safe_get(url, params)

        if data is None:
            break

        if len(data) == 0:
            break

        comments.extend(data)
        page += 1


    return comments


def main():
    issues = get_issues()

    for issue in tqdm(issues):
        if isinstance(issue, dict) and "number" in issue:
            issue_number = issue["number"]
            issue["comments_data"] = get_comments(issue_number)
        else:
            print(f"Skipping issue: {issue}")

    with open("/Users/luffy_sama/Desktop/Workspace/Try/10/data/raw/airflow_issues.json", "w") as f:
        json.dump(issues, f, indent=2)
        print("TOKEN:", TOKEN)

    print("Download complete.")

if __name__ == "__main__":
    main()