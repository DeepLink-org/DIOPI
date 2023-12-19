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

gcovr --csv > coverage/coverage.csv
sed -i '1d' coverage/coverage.csv
newcommit=$(git rev-parse HEAD~1)
oldcommit=$(git merge-base ${newcommit} main)
if [ -z $oldcommit ]; then echo "is not Pull request" && exit 0; fi
git diff $oldcommit $newcommit --name-only | xargs -I {} realpath {} >coverage/gitdiff.txt 2>/dev/null || echo "error can be ignored"

cat coverage/gitdiff.txt |egrep '\.(cpp|hpp|h)$'|grep "/$include/" >coverage/gitdiff_screen.txt || true
if [ ! -s coverage/gitdiff_screen.txt ]; then echo "No C/C++ in incremental code" && exit 0;fi
sed -i "s#$ROOT_DIR/##g" coverage/gitdiff_screen.txt
if [[ $ROOT_DIR == */lustre/* ]]; then
  sed -i "s#${ROOT_DIR//\/lustre\//\/cache\/}/##g" coverage/gitdiff_screen.txt
elif [[ $ROOT_DIR == */cache/* ]]; then
  sed -i "s#${ROOT_DIR//\/cache\//\/lustre\/}/##g" coverage/gitdiff_screen.txt
fi

echo "export IS_cover=True" >coverage/IS_cover.txt
mkdir coverage/html
gcovr -r . --html --html-details -o coverage/html/index.html
python scripts/increment_coverage.py $ROOT_DIR/coverage/ $require_coverage
source coverage/IS_cover.txt
if [ $IS_cover == 'True' ]; then
  exit 0
else
  echo "coverage does not exceed $require_coverage"
  echo "HTML: ${ROOT_DIR}/coverage/html"
  exit 1
fi