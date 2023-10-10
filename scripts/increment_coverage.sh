# !/bin/bash
set -e
export LANG=en_US.UTF-8
ROOT_DIR=$(dirname "$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)")
cd $ROOT_DIR
echo "entering "$ROOT_DIR

require_coverage=$1

echo "==============C================"
lcov -c -d . --no-external -o coverage.info
newcommit=`git rev-parse --short HEAD`
git diff $oldcommit $newcommit --name-only | xargs -I {} realpath {} > gitdiff.txt 2>/dev/null || echo "error can be ignored"
for dir in `cat gitdiff.txt`;do
  skip=1
  buffer=""
  while IFS= read -r line; do
      if [[ $line == "TN:"* ]]; then
          buffer=$line
          skip=1
      elif [[ $line == *"SF:$dir" ]]; then
          skip=0
          echo "$buffer" >> "increment.info"
          echo "$line" >> "increment.info"
      elif [[ $skip -eq 0 ]]; then
          echo "$line" >> "increment.info"
      fi
      if [[ $line == "end_of_record" ]]; then
          skip=1
      fi
  done < "coverage.info"
done

echo "=============python============="
cd diopi_test/python
coverage combine
cd $ROOT_DIR
echo "export IS_cover=True" > IS_cover.txt
if [ -f increment.info ];then
    lcov --list increment.info
    lcov --list increment.info > increment.txt
else
    echo "C无增量代码，或测试未覆盖到"
fi
python $ROOT_DIR/scripts/increment_coverage.py $ROOT_DIR/increment.txt $ROOT_DIR $require_coverage $ROOT_DIR/diopi_test/python/.coverage gitdiff.txt
rm -rf coverage.info gitdiff.txt increment.info
source IS_cover.txt
if  [ $IS_cover == 'True' ];then exit 0 ;else exit 1;fi
