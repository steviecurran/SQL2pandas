## SQL to Pandas: Practical Data Analysis

This project demonstrates how common SQL-style analytical queries can be implemented in pandas, applied to a transactional dataset to generate insights.

The focus is on translating structured query logic into flexible Python workflows, while maintaining a clear link between data manipulation and business interpretation.

###Motivation

Many analysts are fluent in SQL but need to transition to Python data workflows.

### Overview

The notebook walks through a typical analytical workflow:
- filtering and selecting data
- aggregating metrics (e.g. revenue by customer)
- ranking and sorting results
- analysing distributions and concentration

Each step is framed using SQL-style queries alongside their pandas equivalents.

**Example**

    SQL:
    SELECT country, AVG(revenue)
    FROM sales
    GROUP BY country;

Equivalent pandas:

    df.groupby("country")["revenue"].mean()

### Key Analysis

A Pareto-style analysis is used to examine how revenue is distributed across customers.

The results quantify what proportion of customers generate 80% of total revenue, providing a data-driven view of concentration rather than assuming a standard 80/20 split.

## Interpretation

This type of analysis is commonly used in:
- customer analytics
- product performance tracking
- financial reporting

and helps inform decisions around targeting, retention, and resource allocation.

###Tools
- Python (pandas, NumPy, matplotlib)
- SQL-style query logic
- Exploratory data analysis

###Features

 - SQL SELECT / WHERE / GROUP BY equivalents
 - Translation helper functions
 -  Examples for analysts moving to pandas

![](https://raw.githubusercontent.com/steviecurran/SQL2pandas/refs/heads/main/Jupyter_example.png)

I suggest you download the Jupyter notebook and run it locally to get the full functionality (e.g. the table of contents) and let me know if there's anything you want added. 

###Summary

This project illustrates how SQL-style queries can be translated into pandas workflows to support practical, real-world data analysis, combining structured querying with flexible exploration and interpretation.

See https://github.com/steviecurran/sql2csv to convert .sql to .csv, via the C shell
