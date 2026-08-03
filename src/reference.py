import pandas as pd


def reference_table() -> pd.DataFrame:
    rows = [
        ("SELECT", "SELECT a, b FROM data;", "data[['a', 'b']]"),
        ("WHERE", "SELECT * FROM data WHERE value > 10;", "data[data['value'] > 10]"),
        ("IN", "WHERE region IN ('West', 'East')", "data[data['region'].isin(['West', 'East'])]"),
        ("LIKE", "WHERE name LIKE '%tech%'", "data[data['name'].str.contains('tech', case=False, na=False)]"),
        ("GROUP BY", "SELECT region, SUM(value) FROM data GROUP BY region;", "data.groupby('region')['value'].sum().reset_index()"),
        ("HAVING", "GROUP BY region HAVING SUM(value) > 100", "grouped.loc[lambda x: x['sum_value'] > 100]"),
        ("INNER JOIN", "a INNER JOIN b ON a.id = b.id", "a.merge(b, on='id', how='inner')"),
        ("LEFT JOIN", "a LEFT JOIN b ON a.id = b.id", "a.merge(b, on='id', how='left')"),
        ("FULL OUTER JOIN", "a FULL OUTER JOIN b ON a.id = b.id", "a.merge(b, on='id', how='outer')"),
        ("UNION ALL", "SELECT * FROM a UNION ALL SELECT * FROM b", "pd.concat([a, b], ignore_index=True)"),
        ("CASE WHEN", "CASE WHEN value > 100 THEN 'High' ELSE 'Low' END", "np.where(data['value'] > 100, 'High', 'Low')"),
        ("ROW_NUMBER", "ROW_NUMBER() OVER (PARTITION BY region ORDER BY value DESC)", "data.groupby('region').cumcount().add(1)"),
        ("RANK", "RANK() OVER (ORDER BY value DESC)", "data['value'].rank(method='min', ascending=False)"),
        ("DENSE_RANK", "DENSE_RANK() OVER (ORDER BY value DESC)", "data['value'].rank(method='dense', ascending=False)"),
        ("LAG", "LAG(value, 1) OVER (ORDER BY date)", "data['value'].shift(1)"),
        ("LEAD", "LEAD(value, 1) OVER (ORDER BY date)", "data['value'].shift(-1)"),
        ("Rolling mean", "AVG(value) OVER (ORDER BY date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)", "data['value'].rolling(3, min_periods=1).mean()"),
    ]
    return pd.DataFrame(rows, columns=["Operation", "SQL", "pandas"])
