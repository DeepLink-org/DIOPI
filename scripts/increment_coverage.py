import sys,os

CONTENT = 'export IS_cover=False\n'

def get_coverage_data(coverage_file):
    coverage_data = {}
    with open(coverage_file, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) > 3:
                file_path = parts[0]
                coverage_rate = float(parts[3]) * 100
                coverage_data[file_path] = coverage_rate
    return coverage_data

def check_coverage(gitdiff_file,coverage_data,require_coverage):
    with open(gitdiff_file, 'r') as file:
        for line in file:
            file_path = line.strip()
            if file_path not in coverage_data:
                with open(os.path.join(coveragedir, 'IS_cover.txt'), 'a') as cover_file:
                    print(f"Warning Cannot get code coverage for  {file_path}")
            else:
                if coverage_data[file_path] < float(require_coverage):
                    with open(os.path.join(coveragedir, 'IS_cover.txt'), 'a') as cover_file:
                        print(f"Error Code coverage is {coverage_data[file_path]} below requirement {require_coverage} :{file_path}")
                        cover_file.write(CONTENT)

if __name__ == '__main__':
    coveragedir, require_coverage = sys.argv[1:3]
    coverage_data = get_coverage_data(coveragedir + "/coverage.csv")
    check_coverage(coveragedir + "/gitdiff.txt", coverage_data, require_coverage)