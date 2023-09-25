import os
import requests
import argparse

def get_run_result(pr_number):
    run_result = {
        'NV': False,
        'CAMB': False,
        'ASCEND': False,
        'TOPSRIDER': False,
    }

    repository = os.environ.get("GITHUB_REPOSITORY")
    token = os.environ.get("GITHUB_TOKEN")
    if token == "NONE":
        return "NV_CAMB_ASCEND_TOPSRIDER"
    api_url = f"https://api.github.com/repos/{repository}/pulls/{pr_number}/files"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        pr_files = response.json()
        norunpaths = ["impl/camb_pytorch","impl/cuda","impl/supa"]
        for file in pr_files:
            filenames = file["filename"]
            filename = filenames.split("/")[-1]
            if filename.endswith('.md') or '.github/ISSUE_TEMPLATE/' in filenames or filenames.startswith('.img') or filename.startswith('.git') or filename.startswith('CODE_OF_CONDUCT') :
                continue
            elif filenames.startswith('impl'):
                if "impl/camb" in filenames:
                    run_result['CAMB'] = True
                elif "impl/torch" in filenames:
                    run_result['CAMB'] = True
                elif "impl/ascend" in filenames:
                    run_result['ASCEND'] = True
                elif "impl/topsrider" in filenames:
                    run_result['TOPSRIDER'] = True
                elif any(subpath in filenames for subpath in norunpaths):
                    continue
                else:
                    run_result['CAMB'] = True
                    run_result['NV'] = True
                    run_result['ASCEND'] = True
                    run_result['TOPSRIDER'] = True
            else:
                run_result['CAMB'] = True
                run_result['NV'] = True
                run_result['ASCEND'] = True
                run_result['TOPSRIDER'] = True
    else:
        print("Failed to fetch API")
        exit(1)
    result_string = "_".join([key for key, value in run_result.items() if value])
    return result_string

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('prnumber', type=int, help='get pr number')
    args = parser.parse_args()
    pr_number = args.prnumber
    if pr_number == 0:
        RUN_RESULT="NV_CAMB_ASCEND_TOPSRIDER"
    else:
        RUN_RESULT=get_run_result(pr_number)
    print(RUN_RESULT)