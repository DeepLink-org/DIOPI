import os
import requests
import argparse

def get_run_result(pr_number):
    run_result = {
        'NV': False,
        'CAMB': False,
        'ASCEND': False,
        'TOPSRIDER': False,
        'SUPA': False,
        'KUNLUNXIN': False,
        'GENDATA': False,
    }

    repository = os.environ.get("GITHUB_REPOSITORY")
    api_url = f"https://api.github.com/repos/{repository}/pulls/{pr_number}/files"
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        pr_files = response.json()
        if "diopi_configs.py" in str(pr_files):
            run_result['GENDATA'] = True
        norunpaths = ["impl/camb_pytorch","impl/cuda"]
        for file in pr_files:
            filenames = file["filename"]
            filename = filenames.split("/")[-1]
            if filename.endswith('.md') or '.github/ISSUE_TEMPLATE/' in filenames or filenames.endswith('.img') or filename.endswith('.git') \
                    or filename.endswith('.txt') or filename == 'CODEOWNERS' or filename == 'LICENSE' or filename == '.pre-commit-config.yaml':
                continue
            elif filenames.startswith('impl'):
                if "impl/camb" in filenames:
                    run_result['CAMB'] = True
                elif "impl/torch" in filenames:
                    run_result['NV'] = True
                elif "impl/ascend" in filenames or "impl/ascend_npu" in filenames:
                    run_result['ASCEND'] = True
                elif "impl/topsrider" in filenames:
                    run_result['TOPSRIDER'] = True
                elif "impl/supa" in filenames:
                    run_result['SUPA'] = True
                elif "impl/kunlunxin" in filenames:
                    run_result['KUNLUNXIN'] = True
                elif "impl/droplet" in filenames:
                    run_result['droplet'] = True
                elif any(subpath in filenames for subpath in norunpaths):
                    continue
                else:
                    run_result['CAMB'] = True
                    run_result['NV'] = True
                    run_result['ASCEND'] = True
                    run_result['TOPSRIDER'] = True
                    run_result['SUPA'] = True
                    run_result['KUNLUNXIN'] = True
                    run_result['droplet'] = True
                    break

            else:
                run_result['CAMB'] = True
                run_result['NV'] = True
                run_result['ASCEND'] = True
                run_result['TOPSRIDER'] = True
                run_result['SUPA'] = True
                run_result['KUNLUNXIN'] = True
                run_result['droplet'] = True
                break
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
        RUN_RESULT="NV_CAMB_ASCEND_TOPSRIDER_SUPA_KUNLUNXIN_DROPLET"
    else:
        RUN_RESULT=get_run_result(pr_number)
    print(RUN_RESULT)