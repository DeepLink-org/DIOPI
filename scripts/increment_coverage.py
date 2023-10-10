import re
from coverage import Coverage
from coverage.exceptions import NoSource
import sys,os

CONTENT = 'export IS_cover=False\n'

def C_coverage(c_coverage_file, projectdir, require_coverage):
    with open(c_coverage_file, 'r') as file:
        tracefile = file.read()

    lines = tracefile.split("\n")

    for line in lines:
        coverage_percent = re.search(r'\|(.+?)%', line)
        if coverage_percent and float(coverage_percent.group(1)) < int(require_coverage):
            with open(os.path.join(projectdir, 'IS_cover.txt'), 'a') as file:
                file.write(CONTENT)

def python_coverage(python_coverage_file, gitdiff_file, projectdir):
    max_filename_length = 60
    remove_part = os.path.dirname(gitdiff_file)
    cov = Coverage(data_file=python_coverage_file)
    cov.load()

    with open(gitdiff_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.endswith('.py\n'):
            filename = line[:-1]
            file_report = cov.analysis2(filename)
            filename, statements, excluded, missing, missing_formatted = file_report
            if statements:
                total_statements = len(statements)
                total_missing = len(missing)
                total_covered = total_statements - total_missing

                coverage_percent = (total_covered / total_statements) * 100

                coverage_percent = round(coverage_percent, 1)
                if coverage_percent < int(require_coverage):
                    with open(os.path.join(projectdir, 'IS_cover.txt'), 'a') as file:
                        file.write(CONTENT)

                filename = filename.replace(remove_part + '/', '')
                print(f'Python Coverage {filename.ljust(max_filename_length)}: {coverage_percent}%')


if __name__ == '__main__':
    c_coverage_file, projectdir, require_coverage, python_coverage_file, gitdiff_file = sys.argv[1:6]
    if os.path.exists(os.path.join(projectdir, 'increment.info')):
        C_coverage(c_coverage_file, projectdir, require_coverage)
    python_coverage(python_coverage_file, gitdiff_file, projectdir)
