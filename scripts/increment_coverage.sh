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

remote_count=$(git remote | wc -l)
if [ "$remote_count" -eq 1 ]; then echo "Not from dev repository" && exit 0 ;fi
gcovr --csv --gcov-ignore-errors=no_working_dir_found > coverage/coverage.csv
sed -i '1d' coverage/coverage.csv

commit_hash=$(git rev-parse HEAD)
parent_count=$(git log --pretty=%P -n 1 $commit_hash | wc -w)
if [ $parent_count -eq 1 ]; then echo "is not Pull request" && exit 0; fi
rsync -a ${ROOT_DIR}/../source coverage/ || (echo "cannot find the source dir" && exit 1)
cd coverage/source && git log main --pretty=format:"%H" -n 100 >../gitdiff_main.txt
git reset --hard HEAD~1 &&newcommit=$(git rev-parse HEAD) &&git log  --pretty=format:"%H" -n 100 >../gitdiff_merge.txt
while read -r commit; do
    if grep -Fxq "$commit" ../gitdiff_main.txt; then
        echo $commit
        oldcommit=$commit
        break
    fi
done < ../gitdiff_merge.txt
if [ -z $oldcommit ]; then echo "The Pull request opening time is too long" && exit 0; fi
git diff $oldcommit $newcommit --name-only  > gitdiff.txt 2>/dev/null || echo "error can be ignored"
cd $ROOT_DIR
cat coverage/gitdiff.txt |egrep '\.(cpp|hpp|h)$'|grep "$include/" >coverage/gitdiff_screen.txt || true
if [ ! -s coverage/gitdiff_screen.txt ]; then echo "No C/C++ in incremental code" && exit 0;fi
rm -rf coverage/gitdiff.txt
while IFS= read -r line; do
  if [ -f "$line" ]; then
    echo "$line" >> "coverage/gitdiff.txt"
  fi
done < "coverage/gitdiff_screen.txt"

echo "export IS_cover=True" >coverage/IS_cover.txt
mkdir coverage/html
gcovr -r . --html --html-details --gcov-ignore-errors=no_working_dir_found -o coverage/html/index.html
python scripts/increment_coverage.py $ROOT_DIR/coverage/ $require_coverage
source coverage/IS_cover.txt
if [ $IS_cover == 'True' ]; then
  exit 0
else
  echo "coverage does not exceed $require_coverage"
  echo "HTML: ${ROOT_DIR}/coverage/html"
  exit 1
fi