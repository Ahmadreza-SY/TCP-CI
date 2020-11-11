# Code Feature Extraction

## Introduction

This project aims to extract and compute source-code-related features from a software repository. It analyzes the source code based on static and version control history aspects and creates a dependency graph containing the relationships between files/functions and their association weights. The ultimate goal of extracting these features is for sloving the  test case selection and prioritization problems.

## Instructions for Obtaining Understand Database
### Understand
Understand is a code analysis enterprise software which provides static dependencies available in a source code between files, functions, classes, etc. For more details on this software, visit [this link](https://scitools.com/features). In this project, we utilize Understand to create a static call graph as a base for other features. 

In this section, we explain how to obtain the Understand database, a file with `.und` format which is the output of Understand's analysis. Note that this project needs Understand's database for extracting features and won't work without it.

### Installing Understand's CLI
You can download the latest stable version of Understand from [this link](https://licensing.scitools.com/download). In order to run this project, you need to add the `und` command to your PATH environment variable so the `und` command is recognized in the shell. `und` is located in the `bin` directory of Understand's software.

```bash
export PATH="$PATH:/path/to/understand/scitools/bin/linux64"
```

Finally, run the following command to make sure `und` is successfully installed:

```bash
$ und version
(Build 1029)
```

### Adding Understand Python Package/Library
Unlike typical projects, Understand doesn't provide its Python library in the well-known pip package installer, and you need to manually add the package to your Python environment. The instructions of adding the package are explained in [this link](https://scitools.com/support/python-api/).

## Usage Instructions
### Arguments
After creating Understand's database, you're ready to run our script. The following table demonstrates available arguments to run the script:

Argument Name | Description | Required
--- | --- | ---
-h, --help | Shows the help message and exits. | No
-p, --project-path | Path to project's source code which is a git repository. | Yes
--language | The main programming language of the project. We currently support Java and C/C++ languages and this argument's value for these languages are "java" and "c" respectively. | Yes
-l, --level | Specifies the granularity of feature extraction. It can be one of the two "file" or "function" values. | Yes
-o, --ouput-dir | Specifies the directory to save resulting datasets. The directory would be created if it doesn't exist. | Yes
--branch | The git branch to analyze. The default value is "master". | No
--since | The start date of commits to analyze with the format of YYYY-MM-DD. Not providing this argument means to analyze all commits. | No

### Usage Examples
- This example analyzes ceph's nautilus branch in the file granularity level for only commits since 2020-04-16:
```
python main.py -p ../sample-projects/ceph -l file -o ./ceph-file --branch nautilus --since 2020-04-16
```
- This example analyzes ceph's master branch in the function granularity level for all available commits:
```
python main.py -p ../sample-projects/ceph -l function -o ./ceph-function
```

## Outputs
Currently, this project creates two .csv files as its final results which are `metadata.csv` and `dep_graph.csv`.

### Metadata File
The `metadata.csv` file represents the features available for each individual entity (file or function depending on the used granularity level).
In addition to the metrics provided by Understand's analysis, it includes id, unique name, and path columns. 
You can read the description of all Understand metrics in [this link](https://scitools.com/support/metrics_list).
Here's a sample of ceph's extracted metadata in the file granularity:

|Id|Name|FilePath|AltAvgLineBlank|AltAvgLineCode|AltAvgLineComment|AltCountLineBlank|AltCountLineCode|AltCountLineComment|AvgCyclomatic|AvgCyclomaticModified|AvgCyclomaticStrict|AvgEssential|AvgLine|AvgLineBlank|AvgLineCode|AvgLineComment|CountDeclClass|CountDeclFunction|CountLine|CountLineBlank|CountLineCode|CountLineCodeDecl|CountLineCodeExe|CountLineComment|CountLineInactive|CountLinePreprocessor|CountSemicolon|CountStmt|CountStmtDecl|CountStmtEmpty|CountStmtExe|MaxCyclomatic|MaxCyclomaticModified|MaxCyclomaticStrict|MaxEssential|MaxNesting|RatioCommentToCode|SumCyclomatic|SumCyclomaticModified|SumCyclomaticStrict|SumEssential|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|1|ClusterWatcher.cc|src/tools/cephfs_mirror/ClusterWatcher.cc|2|14|0|28|123|7|3|3|3|0|16|2|14|0|0|8|156|28|112|40|10|7|0|11|50|111|87|0|24|8|8|8|1|3|0.06|27|27|27|8|
|20|debug.h|src/common/debug.h|0|0|0|10|10|15|0|0|0|0|0|0|0|0|0|0|35|10|0|0|0|15|0|10|0|0|0|0|0|0|0|0|0|0|0.00|0|0|0|0|
|724|ClientSnapRealm.cc|src/client/ClientSnapRealm.cc|2|21|0|4|23|2|4|4|4|1|23|2|21|0|0|1|29|4|21|4|18|2|0|2|21|28|4|0|24|4|4|4|1|1|0.10|4|4|4|1|
|725|ClientSnapRealm.h|src/client/ClientSnapRealm.h|0|18|0|14|44|4|5|5|5|1|18|0|18|0|1|4|60|14|38|0|0|4|0|6|21|32|27|0|5|2|2|2|1|1|0.11|5|5|5|4|
|769|actions.hpp|src/rbd_replay/actions.hpp|0|105|0|83|212|49|34|34|34|1|105|0|105|0|15|34|344|83|203|0|0|49|0|9|77|205|181|0|24|1|1|1|1|0|0.24|34|34|34|34|

This is a another sample of ceph's extracted metadata for function granularity:

|Id|Name|FullName|FilePath|Parameters|AltCountLineBlank|AltCountLineCode|AltCountLineComment|CountInput|CountLine|CountLineBlank|CountLineCode|CountLineCodeDecl|CountLineCodeExe|CountLineComment|CountLineInactive|CountLinePreprocessor|CountOutput|CountPath|CountPathLog|CountSemicolon|CountStmt|CountStmtDecl|CountStmtEmpty|CountStmtExe|Cyclomatic|CyclomaticModified|CyclomaticStrict|Essential|Knots|MaxEssentialKnots|MaxNesting|MinEssentialKnots|RatioCommentToCode|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|15|get_ostream|ceph::logging::MutableEntry::get_ostream|src/log/Entry.h||0|3|0|7668|3|0|3|1|1|0|0|0|2|1|0|1|1|0|0|1|1|1|1|1|0|0|0|0|0.00|
|71|handle_fsmap|cephfs::mirror::ClusterWatcher::handle_fsmap|src/tools/cephfs_mirror/ClusterWatcher.cc|const int &|11|69|3|1|83|11|69|25|0|3|0|0|3|8|1|34|55|49|0|6|7|7|7|1|0|0|2|0|0.04|
|163|append|ceph::buffer::v15_2_0::list::append|src/common/buffer.cc|const ceph::buffer::v15_2_0::ptr &|0|4|0|110|4|0|4|1|1|0|0|0|1|1|0|1|1|0|0|1|1|1|1|1|0|0|0|0|0.00|
|177|clone_range|SloppyCRCMap::clone_range|src/common/SloppyCRCMap.cc|"uint64_t,uint64_t,uint64_t,const SloppyCRCMap &,std::ostream *"|0|43|1|10|44|0|43|8|33|1|0|0|10|63|2|21|31|5|0|26|11|11|11|1|2|0|4|0|0.02|

### Dependency Graph File
The `dep_graph.csv` file represents the static and historical dependencies between the entities found in `metadata.csv`.
This file is created of three columns which are `entity_id`, `dependencies`, `weights`.

Each row demonstrates the dependencies of an entity with the id of `entity_id` to a list of other entities (the `dependencies` column).
These static dependencies are extracted from Understand's database.
A file has a dependency to a second file if it includes/imports the second file.
A function has a dependency to a second function if it calls the second function.

It also contains the association weights for each dependency which can be found in the `weights` column.
The association weights are extracted using the git history and Apriori algorithm.
For each dependency, the association weight is either a single zero or four real numbers. 

A single zero means although Understand detected a static relation, but there is no historical dependency between these two entities.
In other words, among all analyzed commits, there is no commit in which these both enities have changed.

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
