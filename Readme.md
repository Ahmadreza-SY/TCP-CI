# Code Feature Extraction

## Introduction

This project aims to extract and compute source-code-related features from a software repository. It analyzes the source code based on static and version control history aspects and creates a dependency graph containing the relationships between files/functions and their association weights. The ultimate goal of extracting these features is for solving the test case selection and prioritization problems.

## Environment Setup
### Python Environment
The tool is tested on Python 3.7+. The tool's Python dependencies can be installed via the following command:
```bash
pip install -r requirements.txt
```
### Ruby Environment
The tool is tested on Ruby 2.6.6. The tool's Ruby dependencies can be installed by running the bundler command at the root of the project:
```bash
bundle install
```
### Understand
Understand is a code analysis enterprise software with a wide variety of [supported languages](https://support.scitools.com/t/supported-languages/153) which provides static dependencies available in a source code between files, functions, classes, etc. For more details on this software, visit [this link](https://scitools.com/features). In this project, we utilize Understand to create a static call graph as a base for other features. 

In this section, we explain how to install and set up Understand for our script to obtain a file with `.und` format which is the output of Understand's analysis. Note that this project needs Understand's database for extracting features and won't work without it.

#### Installing Understand's CLI
You can download the latest stable version of Understand from [this link](https://licensing.scitools.com/download). To run this project, you need to add the `und` command to your PATH environment variable so the `und` command is recognized in the shell. `und` is located in the `bin` directory of Understand's software.

```bash
export PATH="$PATH:/path/to/understand/scitools/bin/linux64"
```

Finally, run the following command to make sure `und` is successfully installed:

```bash
$ und version
(Build 1029)
```

#### Adding Understand Python Package/Library
Unlike typical projects, Understand doesn't provide its Python library in the well-known pip package installer, and you need to manually add the package to your Python environment. The instructions for adding the package are explained in [this link](https://support.scitools.com/t/getting-started-with-the-python-api/51).

## Usage Instructions
After setting up the environment, you're ready to run our script.

This project consists of two sub-commands: `history` and `release`. The `history` sub-command extracts code dependencies based on the history of the code and static analysis. The `release` sub-command depends on the output of the `history` command and extracts changed entities in a new CI build/cycle.

### Arguments

Common arguments:

Argument Name | Description | Required
--- | --- | ---
-h, --help | Shows the help message and exits. | No
-p, --project-path | Path to project's source code which is a git repository. | No
-s, --project-slug | The project's GitHub slug, e.g., apache/commons. | No
--language | The main programming language of the project. We currently support Java and C/C++ languages and this argument's value for these languages are "java" and "c" respectively. | Yes
-l, --level | Specifies the granularity of feature extraction. It can be one of the two "file" or "function" values. | Yes
-t, --test-path | Specifies the relative root directory of the test source code. | No
-o, --output-dir | Specifies the directory to save resulting datasets. The directory would be created if it doesn't exist. | Yes

#### Notes
- At least one of the `--project-path` or `--project-slug` arguments should be provided since the tool requires the source code for analysis.
- The `--test-path` is only optional for Java projects since test source is found under `src/test` directories in Java projects.

`history` sub-command arguments:

Argument Name | Description | Required
--- | --- | ---
--branch | The git branch to analyze. The default value is "master". | No
--since | The start date of commits to analyze with the format of YYYY-MM-DD. Not providing this argument means analyzing all commits. | No

`release` sub-command arguments:

Argument Name | Description | Required
--- | --- | ---
-from, --from-commit | Hash of the start commit of this release. | Yes
-to, --to-commit | Hash of the last commit of this release. | Yes
-hist, --histories-dir | Path to outputs of the history command. | Yes

### Usage Examples
This example analyzes ceph's nautilus branch in the file granularity level for only commits since 2020-04-16:
```
python main.py history -s ceph/ceph -l file -o ./ceph-file --branch nautilus --since 2020-04-16
```
This example analyzes ceph's master branch in the function granularity level for all available commits:
```
python main.py history -p ./sample-projects/ceph -l function -o ./ceph-function
```
This example extracts changed entities from a new release of ceph project:
```
python main.py release -p ./sample-projects/ceph -l function -o ./ceph-function -hist ./ceph-function --language c -from 19f492014bcab54e4bafae4c52576de390bdbe47 -to 7efcc72483543cdbeae268b42ff33491a258626c
```

## Outputs
The `history` sub-command creates the following files as its outputs: 

- `metadata.csv`: Contains all entities (files/functions) of the source and test code with all their complexity features.
- `dep.csv`: Contains the dependency graph for only the source entities of the system under test.
- `tar.csv`: Contains the dependency graph for test case and the system under test dependencies.
- `exe.csv`: Contains the execution history of the test cases.
- `commits.csv`: Contains metadata and metrics of all analyzed commits.
- `contributors.csv`: Contains the process metrics related to contributors' experience.

The `release` sub-command outputs the `release_changes.csv` file.

We used meaningful names for the output files and columns, therefore we only discuss the following files.

### Metadata File
The `metadata.csv` file represents the features available for each entity (file or function depending on the used granularity level).
In addition to the metrics provided by Understand's analysis, it includes id, unique name, path, and EntityType columns. 
You can read the description of all Understand metrics in [this link](https://support.scitools.com/t/what-metrics-does-undertand-have/66).
Here's a sample of ceph's extracted metadata in the file granularity:

|Id|Name|FilePath|EntityType|AltAvgLineBlank|AltAvgLineCode|AltAvgLineComment|AltCountLineBlank|AltCountLineCode|AltCountLineComment|AvgCyclomatic|AvgCyclomaticModified|AvgCyclomaticStrict|AvgEssential|AvgLine|AvgLineBlank|AvgLineCode|AvgLineComment|CountDeclClass|CountDeclFunction|CountLine|CountLineBlank|CountLineCode|CountLineCodeDecl|CountLineCodeExe|CountLineComment|CountLineInactive|CountLinePreprocessor|CountSemicolon|CountStmt|CountStmtDecl|CountStmtEmpty|CountStmtExe|MaxCyclomatic|MaxCyclomaticModified|MaxCyclomaticStrict|MaxEssential|MaxNesting|RatioCommentToCode|SumCyclomatic|SumCyclomaticModified|SumCyclomaticStrict|SumEssential|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|1|ClusterWatcher.cc|src/tools/cephfs_mirror/ClusterWatcher.cc|SRC|2|14|0|28|123|7|3|3|3|0|16|2|14|0|0|8|156|28|112|40|10|7|0|11|50|111|87|0|24|8|8|8|1|3|0.06|27|27|27|8|
|20|debug.h|src/common/debug.h|SRC|0|0|0|10|10|15|0|0|0|0|0|0|0|0|0|0|35|10|0|0|0|15|0|10|0|0|0|0|0|0|0|0|0|0|0.00|0|0|0|0|
|724|ClientSnapRealm.cc|src/client/ClientSnapRealm.cc|SRC|2|21|0|4|23|2|4|4|4|1|23|2|21|0|0|1|29|4|21|4|18|2|0|2|21|28|4|0|24|4|4|4|1|1|0.10|4|4|4|1|
|725|ClientSnapRealm.h|src/client/ClientSnapRealm.h|SRC|0|18|0|14|44|4|5|5|5|1|18|0|18|0|1|4|60|14|38|0|0|4|0|6|21|32|27|0|5|2|2|2|1|1|0.11|5|5|5|4|
|769|actions.hpp|src/rbd_replay/actions.hpp|SRC|0|105|0|83|212|49|34|34|34|1|105|0|105|0|15|34|344|83|203|0|0|49|0|9|77|205|181|0|24|1|1|1|1|0|0.24|34|34|34|34|

### Dependency Graph Files
The `dep.csv` file represents the static and historical dependencies between the entities found in `metadata.csv`.
This file is created of three columns which are `entity_id`, `dependencies`, `weights`.

Each row demonstrates the dependencies of an entity with the id of `entity_id` to a list of other entities (the `dependencies` column).
These static dependencies are extracted from Understand's database.
A file has a dependency on a second file if it includes/imports the second file.
A function has a dependency on a second function if it calls the second function.

It also contains the association weights for each dependency which can be found in the `weights` column.
The association weights are extracted using the git history and Apriori algorithm.
For each dependency, the association weight is either a single zero or four real numbers. 

A single zero means although Understand detected a static relation, there is no historical dependency between these two entities.
In other words, among all analyzed commits, there is no commit in which both entities have changed.

On the other hand, the four real numbers represent `support`, `forward_confidence`, `backward_confidence`, and `lift`.
These metrics are popular among the association rule mining algorithms.
Note that assuming A has a dependency to B, `forward_confidence` is the confidence for A given B (A|B) and `backward_confidence` is the confidence for B given A (B|A).
For details about these association metrics, visit [this link](https://www.kdnuggets.com/2016/04/association-rules-apriori-algorithm-tutorial.html).

Here's how the dependency graph file looks like:

|entity_id|dependencies|weights|
|---|---|---|
|933|[317, 620]|[[0.014, 0.33, 0.25, 6.03], [0.0092, 0.22, 0.17, 4.02]]|
|1013|[272]|[0]|
|21|[22]|[[0.005, 1.0, 0.5, 108.5]]|
|18298|[1887, 431]|[0, [0.0061, 0.22, 0.5, 18.22]]|

Additionally, the `tar.csv` file represents the dependencies between the test code and source code. The idea behind this file is the same as `dep.csv` with the similar idea of association weights and dependencies. However, it contains the following three columns which are slightly different from `dep.csv` columns.

- `entity_id`: The id of a source code entity that is executed (targeted) by test cases.
- `targeted_by_tests`: A list of test case ids which execute (target) the `entity_id`.
- `weights`: The weights of the target relation between each test case in `targeted_by_tests` and the `entity_id`.

### Execution History File
The `exe.csv` file represents the execution history of all available test cases in the project. We assign each CI build/cycle a build_id, and for each build_id we have multiple jobs in which the test cases are executed. The columns of this file are briefly explained in the following list.

- `test`: Id of the test case which was executed.
- `build`: Id of the build in which the test case was executed. The build metadata is available in the `full_builds.csv` file.
- `job`: Id of the job in which the test case was executed. The jobs metadata is available in the `full_jobs.csv` file.
- `test_result`: The result of the test case execution. The value of 0 means passed, 1 means failed due to exception, 2 means failed due to assertion, and 3 means failed due to an unknown reason.
- `duration`: The duration of the test case execution in milliseconds.

### Release Changes File
The `release_changes.csv` file indicates which enities have been directly and indirectly changed in a release (set of commits).

The entities which are directly changed are indicated with the weight value of 1, however, entities with the weight value of 0 or a list of four real numbers are indirectly changed, and the meaning of their weights is completely explained in the previous section.

The following table demonstrates a sample of release data:

|entity_id|weight|
|---|---|
|1763|1|
|48641|[0.0011, 0.333, 0.014, 4.32]|
|3792|1|
|1024|0|