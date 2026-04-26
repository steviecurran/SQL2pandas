## SQL to Pandas: Practical Data Analysis

This project demonstrates how common SQL-style analytical queries can be implemented in pandas, applied to a transactional dataset to generate insights.

The focus is on translating structured query logic into flexible Python workflows, while maintaining a clear link between data manipulation and business interpretation.

### Motivation

Many analysts are fluent in SQL but need to transition to Python-based data workflows for more flexible analysis and modelling.

### Overview

The notebook walks through a typical analytical workflow:
- filtering and selecting data
- aggregating metrics (e.g. revenue by customer)
- ranking and sorting results
- analysing distributions and concentration

Each step is framed using SQL-style queries alongside their pandas equivalents.

![](https://raw.githubusercontent.com/steviecurran/SQL2pandas/refs/heads/main/Jupyter_example.png)

**Example**

    SQL:
    SELECT country, AVG(revenue)
    FROM sales
    GROUP BY country;

Equivalent pandas:

    df.groupby("country")["revenue"].mean()

### Key Analyses

**Business Problem 1**

A Pareto-style analysis is used to examine how revenue is distributed across customers.

The analysis shows that approximately X% of customers generate 80% of total revenue, providing a data-driven view of concentration rather than assuming a standard 80/20 split.

**Business Problem 2**

Relational databases - combine the datasets to compare the salaries of each department.

![](https://github.com/steviecurran/SQL2pandas/blob/main/dept_salaries.png)

## Interpretation

This type of analysis is commonly used in:
- customer analytics
- product performance tracking
- financial reporting

and helps inform decisions around targeting, retention, and resource allocation.

### Tools
- Python (pandas, NumPy, matplotlib)
- SQL-style query logic
- Exploratory data analysis

### Features

- SQL-to-pandas translations for common analytical queries  
- End-to-end workflow: filtering → aggregation → ranking → interpretation  
- Pareto analysis for revenue concentration  
- Practical examples for analysts transitioning from SQL to Python  

For full interactivity (e.g. navigation and outputs), run the notebook locally.

### Summary

This project illustrates how SQL-style queries can be translated into pandas workflows to support practical, real-world data analysis, combining structured querying with flexible exploration and interpretation.

### Related Projects

- [sql2csv](https://github.com/steviecurran/sql2csv): Convert SQL outputs to CSV via shell scripting  

