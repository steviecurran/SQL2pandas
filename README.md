## SQL2pandas ##


Translate SQL queries into equivalent pandas operations.

**Motivation**

Many analysts are fluent in SQL but need to transition to Python data workflows.

**Example**

SQL:
SELECT country, AVG(revenue)
FROM sales
GROUP BY country;

Equivalent pandas:
df.groupby("country")["revenue"].mean()

**Features**

• SQL SELECT / WHERE / GROUP BY equivalents
• Translation helper functions
• Examples for analysts moving to pandas


![](https://raw.githubusercontent.com/steviecurran/SQL2pandas/refs/heads/main/Jupyter_example.png)

I suggest you download the Jupyter notebook and run it locally to get the full functionality (e.g. the table of contents) and let me know if there's anything you want added. 

See https://github.com/steviecurran/sql2csv to convert .sql to .csv, via the C shell
