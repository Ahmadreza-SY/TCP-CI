# Repository Feature Extraction

## Introduction

This project aims to extract and compute all available source-code-related features from a software repository. The extracted feature will be used for test case prioritization and selection tasks.

## Usage

```
usage: main.py [-h] -d DB_PATH -l {function,file} -o OUTPUT_DIR
               [--language {c}]

optional arguments:
  -h, --help            show this help message and exit
  -d DB_PATH, --db-path DB_PATH
                        Understand's database path with the .udb format.
  -l {function,file}, --level {function,file}
                        Specifies the granularity of feature extraction.
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Specifies the directory to save resulting datasets.
  --language {c}        Project's main language
```