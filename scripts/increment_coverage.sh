set -e
export LANG=en_US.UTF-8
ROOT_DIR=$(dirname "$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)")
repo_name='deeplink.framework'
include='impl'
if [[ $ROOT_DIR == *$repo_name* ]]; then
	include='dipu/torch_dipu/csrc_dipu'
fi
cd $ROOT_DIR && rm -rf coverage && mkdir coverage
echo "entering "$ROOT_DIR
require_coverage=$1

lcov -c -d . --include "*/${ROOT_DIR#/mnt/*/}/${include}/*" -o coverage/coverage.info
newcommit=$(git rev-parse HEAD~1)
oldcommit=$(git merge-base ${newcommit} main)
if [ -z $oldcommit ]; then echo "is not Pull request" && exit 0; fi
git diff $oldcommit $newcommit --name-only | xargs -I {} realpath {} >coverage/gitdiff.txt 2>/dev/null || echo "error can be ignored"
for dir in $(cat coverage/gitdiff.txt); do
	skip=1
	buffer=""
	while IFS= read -r line; do
		if [[ $line == "TN:"* ]]; then
			buffer=$line
			skip=1
		elif [[ $line == *"SF:$dir" ]]; then
			skip=0
			echo "$buffer" >>"coverage/increment.info"
			echo "$line" >>"coverage/increment.info"
		elif [[ $skip -eq 0 ]]; then
			echo "$line" >>"coverage/increment.info"
		fi
		if [[ $line == "end_of_record" ]]; then
			skip=1
		fi
	done <"coverage/coverage.info"
done
cd $ROOT_DIR
echo "export IS_cover=True" >coverage/IS_cover.txt
if [ -f coverage/increment.info ]; then
	lcov --list coverage/increment.info
	lcov --list coverage/increment.info >coverage/increment.txt
	genhtml coverage/increment.info -o coverage/html
else
	echo "No C/C++ in incremental code"
fi
python scripts/increment_coverage.py $ROOT_DIR/coverage/ $require_coverage
source coverage/IS_cover.txt
if [ $IS_cover == 'True' ]; then
	exit 0
else
	echo "coverage does not exceed $require_coverage"
	echo "HTML: ${ROOT_DIR}/coverage/html"
	exit 1
fi
