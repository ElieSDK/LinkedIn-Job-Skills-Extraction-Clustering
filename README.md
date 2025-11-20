# LinkedIn Job Postings Skill Extraction & Clustering

This project contains a workflow for analyzing LinkedIn job
posting data.\
It focuses on cleaning job descriptions, extracting technical skills,
and grouping similar roles using a clustering approach.

## Overview

The repository includes:

-   `job_skill_clustering.py` --- main script for loading data, cleaning
    text, extracting skills, and running K-Means clustering.
-   A Jupyter notebook showing the steps and output more
    interactively.


## Dataset

The original dataset used in this project can be downloaded from Kaggle:

[LinkedIn Job Postings Dataset](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings/data)


## Features

-   Cleans raw job description text (lowercasing, removing URLs,
    tokenizing, etc.)
-   Extracts skills from a predefined list (Python, SQL, AWS, Docker,
    etc.)
-   Normalizes common abbreviations and variants
-   Generates a TF-IDF matrix for skill sets
-   Clusters job postings based on extracted skills
-   Optional salary analysis if a `normalized_salary` column is present

## Requirements

-   Python 3.x
-   pandas
-   nltk
-   scikit-learn

You may also need to download NLTK resources:

``` python
nltk.download('punkt')
nltk.download('stopwords')
```

## How to Run

1.  Update the file paths at the top of the script if needed.
2.  Run:

``` bash
python job_skill_clustering.py
```

Or open the Jupyter notebook to inspect the steps interactively.

## Output

-   Cleaned description text
-   Extracted and normalized skills per job posting
-   Cluster labels for each job
-   Overview of salary patterns per cluster (if salary data is
    available)
