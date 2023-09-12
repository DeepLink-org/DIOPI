import os
import requests
import argparse

def get_run_result(pr_number):
    run_result = {
        'RUN_NV': "F",
        'RUN_CAMB': "F",
        'RUN_ASCEND': "F",
        'RUN_ALL': "F"
    }

    repository = os.environ.get("GITHUB_REPOSITORY")
    token = os.environ.get("GITHUB_TOKEN")

    api_url = f"https://api.github.com/repos/{repository}/pulls/{pr_number}/files"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        pr_files = response.json()
        norunpaths = ["impl/camb_pytorch","impl/cuda","impl/supa","impl/topsrider"]
        for file in pr_files:
            filenames = file["filename"]
            filename = filenames.split("/")[-1]
            if filename.endswith('.md') or '.github/ISSUE_TEMPLATE/' in filenames or filenames.startswith('.img') or filename.startswith('.git') or filename.startswith('CODE_OF_CONDUCT') :
                continue
            elif filenames.startswith('impl'):
                if "impl/camb" in filenames:
                    run_result['RUN_CAMB'] = "T"
                elif "impl/torch" in filenames:
                    run_result['RUN_NV'] = "T"
                elif "impl/ascend" in filenames:
                    run_result['RUN_ASCEND'] = "T"
                elif any(subpath in filenames for subpath in norunpaths):
                    continue
                else:
                    run_result['RUN_ALL'] = "T"
            else:
                run_result['RUN_ALL'] = "T"
    else:
        print("Failed to fetch API")
        exit(1)
    return "ALL_" + run_result['RUN_ALL'] + "-NV_" + run_result['RUN_NV'] + "-CAMB_" + run_result['RUN_CAMB'] + "-ASCEND_" + run_result['RUN_ASCEND']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('prnumber', type=int, help='get pr number')
    args = parser.parse_args()
    pr_number = args.prnumber
    if pr_number == 0:
        RUN_RESULT="ALL_T"
    else:
        RUN_RESULT=get_run_result(pr_number)
    print(RUN_RESULT)