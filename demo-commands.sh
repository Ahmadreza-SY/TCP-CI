# Demo Commands:

python main.py history -p ../sample-projects/jsoup -t ../sample-projects/jsoup/src/test -l file -o ./jsoup-file --language java
python main.py release -p ../sample-projects/jsoup -l file -o ./jsoup-file -hist ./jsoup-file --language java -c jsoup-release.txt

python main.py history -p ../sample-projects/cpr -l function -o ./cpr-function --language c
python main.py release -p ../sample-projects/cpr -l function -o ./cpr-function -hist ./cpr-function --language c -c cpr-release.txt

# New Demo Commands
python main.py history -s jhy/jsoup -p ../sample-projects/jsoup -t src/test -l file -o ./jsoup-file --language java
python main.py release -s jhy/jsoup -t src/test -l file -o ./jsoup-file -hist ./jsoup-file --language java -c jsoup-release.txt
python main.py history -s jhy/jsoup -t src/test -l function -o ./jsoup-function --language java
python main.py release -s jhy/jsoup -t src/test -l function -o ./jsoup-function -hist ./jsoup-function --language java -c jsoup-release.txt

python main.py history -s whoshuu/cpr -t test -l file -o ./cpr-file --language c
python main.py release -s whoshuu/cpr -t test -l file -o ./cpr-file -hist ./cpr-file --language c -c cpr-release.txt
python main.py history -s whoshuu/cpr -t test -l function -o ./cpr-function --language c
python main.py release -s whoshuu/cpr -t test -l function -o ./cpr-function -hist ./cpr-function --language c -c cpr-release.txt

python main.py release -p ../sample-projects/jsoup -s jhy/jsoup -t src/test -l file -o ./jsoup-file -hist ./jsoup-file --language java -from 27b656744b273eca3f43111fe20cc15aa978e52d -to e8ae03d25c0bd50d433fbf493834484b068e5479