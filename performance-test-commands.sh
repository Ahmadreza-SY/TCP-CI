time python main.py history -p ../sample-projects/cpr -l file -o ./cpr-file --language c
echo
time python main.py history -p ../sample-projects/jsoup -l file -o ./jsoup-file --language java
echo
time python main.py history -p ../sample-projects/fastjson -l file -o ./fastjson-file --language java
echo

time python main.py history -p ../sample-projects/cpr -l function -o ./cpr-function --language c
echo
time python main.py history -p ../sample-projects/jsoup -l function -o ./jsoup-function --language java
echo
time python main.py history -p ../sample-projects/fastjson -l function -o ./fastjson-function --language java --since 2014-11-11
echo

time python main.py history -p ../sample-projects/hadoop -l file -o ./hadoop-file --language java --branch trunk --since 2018-11-11
echo
time python main.py history -p ../sample-projects/hadoop -l function -o ./hadoop-function --language java --branch trunk --since 2020-03-11
echo

time python main.py history -p ../sample-projects/ceph -l file -o ./ceph-file --language c --since 2018-11-11
echo
time python main.py history -p ../sample-projects/ceph -l function -o ./ceph-function --language c --since 2020-05-11
echo