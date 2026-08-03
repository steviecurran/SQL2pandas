from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from src.datasets import load_builtin_tables, normalise_table_name, read_delimited
from src.examples import (
    customer_revenue_analysis,
    department_salary_analysis,
    department_salary_histograms,
    pareto_plot,
    revenue_histogram,
    salary_business_pandas,
    salary_business_sql,
)
from src.joins import join_key_compatibility, join_tables
from src.operations import (
    case_when,
    group_aggregate,
    membership_filter,
    select_columns,
    where_filter,
)
from src.reference import reference_table
from src.unions import union_tables
from src.windows import WINDOW_FUNCTIONS, apply_window


DATA_DIR = PROJECT_ROOT / "data"

st.set_page_config(
    page_title="SQL to Pandas Workbench",
    page_icon="🔄",
    layout="wide",
)

st.title("SQL to Pandas Workbench")

st.markdown(
    """
<style>
[data-testid="stRadio"] label,
[data-testid="stRadio"] [data-testid="stWidgetLabel"] p,
div[data-testid="stWidgetLabel"] p {
    font-size: 20px !important;
    font-weight: bold !important;
}
[data-testid="stMetricLabel"] p {
    font-size: 1.15rem !important;
    font-weight: 600 !important;
}
.code-heading {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 5px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
### Build a relational operation and see the equivalent SQL and pandas workflow

Use built-in datasets or upload your own data. The workbench generates readable
SQL and pandas code, executes the pandas operation, and explains the result.
"""
)


@st.cache_data
def builtins():
    return load_builtin_tables(DATA_DIR)


def show_translation(translation: dict, key: str):
    code_cols = st.columns(2)

    with code_cols[0]:
        st.markdown(
            "<div class='code-heading'>SQL</div>",
            unsafe_allow_html=True,
        )
        st.code(translation["sql"], language="sql")

    with code_cols[1]:
        st.markdown(
            "<div class='code-heading'>pandas</div>",
            unsafe_allow_html=True,
        )
        st.code(translation["pandas"], language="python")

    st.markdown("### Result")

    result = translation["result"]
    metrics = st.columns(2)
    metrics[0].metric("Rows returned", f"{len(result):,}")
    metrics[1].metric("Columns returned", f"{len(result.columns):,}")

    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download result as CSV",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name=f"{key}_result.csv",
        mime="text/csv",
        key=f"{key}_download",
    )

    with st.expander("What happened?"):
        st.markdown(translation["explanation"])


def table_loader(prefix: str, available: dict[str, pd.DataFrame]):
    method = st.radio(
        "Data source",
        ["Built-in", "Upload", "URL"],
        horizontal=True,
        key=f"{prefix}_method",
    )

    if method == "Built-in":
        name = st.selectbox(
            "Table",
            list(available.keys()),
            key=f"{prefix}_builtin",
        )
        return name, available[name].copy()

    if method == "Upload":
        uploaded = st.file_uploader(
            "Upload CSV, TXT, DAT or TSV",
            type=["csv", "txt", "dat", "tsv"],
            key=f"{prefix}_upload",
        )
        if uploaded is None:
            return None, None
        name = normalise_table_name(Path(uploaded.name).stem)
        return name, read_delimited(uploaded)

    url = st.text_input(
        "Dataset URL",
        key=f"{prefix}_url",
    )
    if not url:
        return None, None
    return normalise_table_name(prefix), read_delimited(url)


available_tables = builtins()

tabs = st.tabs(
    [
        "Query builder",
        "JOIN workbench",
        "Window functions",
        "CASE & UNION",
        "End-to-End Business Analysis",
        "SQL ↔ pandas reference",
    ]
)

# Query builder
with tabs[0]:
    st.markdown("## Query builder")

    table_name, data = table_loader("query", available_tables)

    if data is None:
        st.info("Choose a dataset to continue.")
    else:
        st.caption(
            f"Table: `{table_name}` | Rows: `{len(data):,}` | "
            f"Columns: `{len(data.columns)}`"
        )
        with st.expander("Preview source data"):
            st.dataframe(data.head(100), use_container_width=True, hide_index=True)

        operation = st.selectbox(
            "Operation",
            [
                "SELECT columns",
                "WHERE",
                "IN",
                "NOT IN",
                "GROUP BY aggregate",
                "GROUP BY aggregate + HAVING",
            ],
        )

        translation = None

        if operation == "SELECT columns":
            cols = st.multiselect(
                "Columns",
                list(data.columns),
                default=list(data.columns[: min(3, len(data.columns))]),
            )
            if st.button("Run operation", type="primary", key="run_select"):
                translation = select_columns(data, table_name, cols)

        elif operation == "WHERE":
            row = st.columns([1.2, 0.8, 1.2])
            with row[0]:
                col = st.selectbox("Column", list(data.columns))
            with row[1]:
                operator = st.selectbox("Operator", ["=", "!=", ">", ">=", "<", "<="])
            with row[2]:
                default_value = str(data[col].dropna().iloc[0]) if data[col].notna().any() else ""
                value = st.text_input("Value", value=default_value)
            if st.button("Run operation", type="primary", key="run_where"):
                translation = where_filter(data, table_name, col, operator, value)

        elif operation in {"IN", "NOT IN"}:
            col = st.selectbox("Column", list(data.columns), key="membership_col")
            values = data[col].dropna().astype(str).unique().tolist()
            selected = st.multiselect(
                "Values",
                values,
                default=values[: min(3, len(values))],
            )
            if st.button("Run operation", type="primary", key="run_membership"):
                translation = membership_filter(
                    data,
                    table_name,
                    col,
                    selected,
                    negate=operation == "NOT IN",
                )

        else:
            numeric = data.select_dtypes(include=np.number).columns.tolist()
            if not numeric:
                st.error("This operation requires a numeric column.")
                st.stop()

            row = st.columns(3)
            with row[0]:
                group_cols = st.multiselect(
                    "Group columns",
                    list(data.columns),
                    default=[data.columns[0]],
                )
            with row[1]:
                value_col = st.selectbox("Value column", numeric)
            with row[2]:
                function = st.selectbox(
                    "Function",
                    ["SUM", "AVG", "MIN", "MAX", "COUNT", "MEDIAN"],
                )

            having_operator = None
            having_value = None
            if "HAVING" in operation:
                having_cols = st.columns(2)
                with having_cols[0]:
                    having_operator = st.selectbox(
                        "HAVING operator",
                        [">", ">=", "<", "<=", "=", "!="],
                    )
                with having_cols[1]:
                    having_value = st.number_input("HAVING value", value=0.0)

            if st.button("Run operation", type="primary", key="run_group"):
                translation = group_aggregate(
                    data,
                    table_name,
                    group_cols,
                    value_col,
                    function,
                    having_operator,
                    having_value,
                )

        if translation is not None:
            st.session_state["query_translation"] = translation

        if "query_translation" in st.session_state:
            show_translation(
                st.session_state["query_translation"],
                "query",
            )

# JOIN workbench
with tabs[1]:
    st.markdown("## JOIN workbench")
    st.caption(
        "Load two tables, choose a join type and keys, and inspect matched and unmatched rows."
    )

    loader_cols = st.columns(2)
    with loader_cols[0]:
        st.markdown("### Left table")
        left_name, left = table_loader("left", available_tables)
    with loader_cols[1]:
        st.markdown("### Right table")
        right_name, right = table_loader("right", available_tables)

    if left is None or right is None:
        st.info("Choose both tables to configure a join.")
    else:
        preview_cols = st.columns(2)
        with preview_cols[0]:
            st.dataframe(left.head(20), use_container_width=True, hide_index=True)
        with preview_cols[1]:
            st.dataframe(right.head(20), use_container_width=True, hide_index=True)

        join_type = st.radio(
            "Join type",
            ["INNER", "LEFT", "RIGHT", "FULL OUTER", "CROSS"],
            horizontal=True,
        )

        left_keys = []
        right_keys = []

        if join_type != "CROSS":
            key_cols = st.columns(2)
            common = [c for c in left.columns if c in right.columns]
            with key_cols[0]:
                left_keys = st.multiselect(
                    "Left join key(s)",
                    list(left.columns),
                    default=common[:1],
                )
            with key_cols[1]:
                right_keys = st.multiselect(
                    "Right join key(s)",
                    list(right.columns),
                    default=common[:1],
                )

            with st.expander("Suggested compatible keys"):
                compatibility = join_key_compatibility(left, right)
                st.dataframe(
                    compatibility,
                    use_container_width=True,
                    hide_index=True,
                )

        if st.button("Run join", type="primary", key="run_join"):
            try:
                st.session_state["join_translation"] = join_tables(
                    left,
                    right,
                    left_name,
                    right_name,
                    join_type,
                    left_keys,
                    right_keys,
                )
            except Exception as error:
                st.error(str(error))

        if "join_translation" in st.session_state:
            translation = st.session_state["join_translation"]
            d = translation["diagnostics"]
            metrics = st.columns(6)
            metrics[0].metric("Left rows", f"{d['left_rows']:,}")
            metrics[1].metric("Right rows", f"{d['right_rows']:,}")
            metrics[2].metric("Output rows", f"{d['output_rows']:,}")
            metrics[3].metric("Matched", f"{d['matched_rows']:,}")
            metrics[4].metric("Left only", f"{d['left_only_rows']:,}")
            metrics[5].metric("Right only", f"{d['right_only_rows']:,}")
            show_translation(translation, "join")

# Windows
with tabs[2]:
    st.markdown("## Window functions")

    table_name, data = table_loader("window", available_tables)

    if data is None:
        st.info("Choose a dataset to continue.")
    else:
        numeric = data.select_dtypes(include=np.number).columns.tolist()
        if not numeric:
            st.error("The selected dataset has no numeric columns.")
        else:
            row = st.columns(3)
            with row[0]:
                function = st.selectbox("Window function", WINDOW_FUNCTIONS)
            with row[1]:
                value_col = st.selectbox("Value column", numeric)
            with row[2]:
                order_col = st.selectbox("Order column", list(data.columns))

            partition_cols = st.multiselect(
                "Partition by",
                list(data.columns),
            )

            direction = st.radio(
                "Order",
                ["Ascending", "Descending"],
                horizontal=True,
            )

            offset = 1
            window_size = 3
            if function in {"LAG", "LEAD"}:
                offset = st.slider("Offset", 1, 20, 1)
            if function == "ROLLING MEAN":
                window_size = st.slider("Window size", 2, 50, 3)

            output_col = st.text_input(
                "Output column",
                value=function.lower().replace(" ", "_"),
            )

            if st.button("Run window function", type="primary"):
                try:
                    st.session_state["window_translation"] = apply_window(
                        data,
                        table_name,
                        function,
                        value_col,
                        partition_cols,
                        order_col,
                        direction == "Ascending",
                        offset=offset,
                        window_size=window_size,
                        output_column=output_col,
                    )
                except Exception as error:
                    st.error(str(error))

            if "window_translation" in st.session_state:
                show_translation(
                    st.session_state["window_translation"],
                    "window",
                )

# CASE and UNION
with tabs[3]:
    st.markdown("## CASE & UNION")

    subtab_case, subtab_union = st.tabs(["CASE WHEN", "UNION"])

    with subtab_case:
        table_name, data = table_loader("case", available_tables)

        if data is None:
            st.info("Choose a dataset to continue.")
        else:
            row = st.columns(3)
            with row[0]:
                column = st.selectbox("Column", list(data.columns), key="case_column")
            with row[1]:
                operator = st.selectbox(
                    "Operator",
                    ["=", "!=", ">", ">=", "<", "<="],
                    key="case_operator",
                )
            with row[2]:
                default_value = str(data[column].dropna().iloc[0]) if data[column].notna().any() else ""
                threshold = st.text_input("Comparison value", value=default_value)

            labels = st.columns(3)
            with labels[0]:
                true_label = st.text_input("THEN label", value="High")
            with labels[1]:
                false_label = st.text_input("ELSE label", value="Low")
            with labels[2]:
                output_col = st.text_input("Output column", value="category")

            if st.button("Run CASE", type="primary"):
                try:
                    st.session_state["case_translation"] = case_when(
                        data,
                        table_name,
                        column,
                        operator,
                        threshold,
                        true_label,
                        false_label,
                        output_col,
                    )
                except Exception as error:
                    st.error(str(error))

            if "case_translation" in st.session_state:
                show_translation(
                    st.session_state["case_translation"],
                    "case",
                )

    with subtab_union:
        cols = st.columns(2)
        with cols[0]:
            first_name, first = table_loader("union_first", available_tables)
        with cols[1]:
            second_name, second = table_loader("union_second", available_tables)

        union_all = st.radio(
            "Union type",
            ["UNION", "UNION ALL"],
            horizontal=True,
        ) == "UNION ALL"

        if first is None or second is None:
            st.info("Choose both tables.")
        else:
            st.caption(
                "The tables must have identical columns in identical order."
            )
            if st.button("Run UNION", type="primary"):
                try:
                    st.session_state["union_translation"] = union_tables(
                        first,
                        second,
                        first_name,
                        second_name,
                        union_all,
                    )
                except Exception as error:
                    st.error(str(error))

            if "union_translation" in st.session_state:
                show_translation(
                    st.session_state["union_translation"],
                    "union",
                )

# Business insight demos
with tabs[4]:
    st.markdown("## End-to-End Business Analysis")

    st.markdown(
        """
These examples show the full analytical path from a business question to a
decision-ready result.

Each case includes:

1. the business problem
2. the relational operations required
3. the equivalent SQL and pandas workflows
4. interactive results and visualisation
5. a plain-English business interpretation
"""
    )

    business_problem_1, business_problem_2 = st.tabs(
        [
            "1. Customer revenue concentration",
            "2. Salaries by department",
        ]
    )

    with business_problem_1:

        st.markdown(
            """
### Business problem 1: Which customers generate most revenue?

#### 1. Business problem

Is revenue broadly distributed across the customer base, or concentrated
among a relatively small group?

A business may use this analysis to identify high-value customers, prioritise
retention activity, design differentiated service levels and understand
revenue-concentration risk.

#### 2. Analytical approach

The workflow:

- groups all transactions by customer
- sums total revenue for each customer
- ranks customers from highest to lowest revenue
- calculates cumulative revenue contribution
- identifies how many customers are required to explain a selected proportion
  of total revenue

The histogram shows the distribution of customer-level revenue. The Pareto
chart combines sorted customer revenue with the cumulative share of total
revenue.
"""
        )

        transactions = available_tables[
            "transactions"
        ]

        analysis_controls = st.columns(
            [2, 1]
        )

        with analysis_controls[0]:

            target_revenue = st.slider(
                "Target cumulative revenue (%)",
                min_value=50,
                max_value=99,
                value=80,
                step=1,
                help=(
                    "Choose the proportion of total "
                    "revenue you would like to explain."
                ),
            )

        revenue, summary = (
            customer_revenue_analysis(
                transactions,
                target_revenue=target_revenue,
            )
        )

        with analysis_controls[1]:

            maximum_bin_width = max(
                100.0,
                float(
                    revenue[
                        "total_revenue"
                    ].max()
                ),
            )

            bin_width = st.number_input(
                "Bin width",
                min_value=10.0,
                max_value=maximum_bin_width,
                value=100.0,
                step=10.0,
                help=(
                    "Controls the width of each "
                    "customer-revenue interval "
                    "in the histogram."
                ),
            )

        metrics = st.columns(4)

        metrics[0].metric(
            "Transactions",
            f"{len(transactions):,}",
        )

        metrics[1].metric(
            "Unique customers",
            f"{revenue['customer_id'].nunique():,}",
        )

        metrics[2].metric(
            (
                f"Customers needed to reach "
                f"{target_revenue}% revenue"
            ),
            f"{summary['customers_required']:,}",
            help=(
                "The number of highest-revenue customers "
                "whose combined revenue first reaches the "
                "selected cumulative target."
            ),
        )

        metrics[3].metric(
            "Share of all customers",
            (
                f"{summary['percentage_customers']:.1f}%"
            ),
            help=(
                "The required customers expressed as a "
                "percentage of all unique customers, not "
                "of all transactions."
            ),
        )

        concentration_label = (
            "high"
            if summary[
                "percentage_customers"
            ] < 30
            else (
                "moderate"
                if summary[
                    "percentage_customers"
                ] < 50
                else "low"
            )
        )

        st.markdown(
            "#### 3. SQL and pandas workflow"
        )

        with st.expander(
            "Show workflow",
            expanded=False,
        ):

            st.markdown(
                f"""
1. Start with transaction-level data.
2. Group all transactions by `customer_id`.
3. Sum `transaction_value` for each customer.
4. Sort customers by total revenue.
5. Calculate cumulative revenue and cumulative percentage.
6. Count how many customers are required to reach
   {target_revenue}% of total revenue.
"""
            )

            code_columns = st.columns(2)

            with code_columns[0]:

                st.markdown("#### SQL")

                st.code(
                    """SELECT
    customer_id,
    SUM(transaction_value) AS total_revenue
FROM transactions
GROUP BY customer_id
ORDER BY total_revenue DESC;""",
                    language="sql",
                )

            with code_columns[1]:

                st.markdown("#### pandas")

                st.code(
                    """revenue = (
    transactions
    .groupby("customer_id")
    ["transaction_value"]
    .sum()
    .reset_index(name="total_revenue")
    .sort_values(
        "total_revenue",
        ascending=False,
    )
)

revenue["cumulative_percentage"] = (
    100
    * revenue["total_revenue"].cumsum()
    / revenue["total_revenue"].sum()
)""",
                    language="python",
                )

        st.markdown(
            "#### 4. Results and visualisation"
        )

        plot_cols = st.columns(2)

        with plot_cols[0]:

            histogram_figure = (
                revenue_histogram(
                    revenue,
                    bin_width=bin_width,
                )
            )

            st.pyplot(
                histogram_figure,
                use_container_width=False,
            )

        with plot_cols[1]:

            pareto_figure = (
                pareto_plot(
                    revenue,
                    target_revenue=
                        target_revenue,
                )
            )

            st.pyplot(
                pareto_figure,
                use_container_width=False,
            )

        st.caption(
            "The histogram summarises customer-level revenue. "
            "The Pareto chart sorts customers by revenue and shows "
            "their cumulative contribution to the total. "
            "The red dotted line follows the selected revenue target."
        )

        st.markdown(
            f"""
#### 5. Business interpretation

Approximately
**{summary['percentage_customers']:.1f}% of customers**
generate **{target_revenue}% of total revenue**. This suggests
**{concentration_label} revenue concentration** in the current dataset.

Potential next steps include:

- profiling the highest-value customers
- checking whether revenue is concentrated in one region or channel
- comparing customer value with retention or service cost
- monitoring whether concentration changes over time
- repeating the analysis at different cumulative-revenue targets
"""
        )

        with st.expander(
            "Customer revenue table"
        ):

            st.dataframe(
                revenue,
                use_container_width=True,
                hide_index=True,
            )


    with business_problem_2:

        st.markdown(
            """
### Business problem 2: How do salaries compare between departments?

This reproduces the second business problem from the original notebook.

**Business question:** Do departments differ in salary level, spread and
workforce size?

The analysis joins four related tables:

- `employees`: employee names and identifiers
- `salaries`: salary records and validity dates
- `dept_emp`: employee-to-department assignments
- `departments`: department names

It then calculates the number of salary records, minimum, average, median and
maximum salary for each department. The small-multiple histograms reveal
features hidden by averages alone, including spread, skewness and overlap
between departments.
"""
        )

        employees = available_tables[
            "employees"
        ]

        salaries = available_tables[
            "salaries"
        ]

        dept_emp = available_tables[
            "dept_emp"
        ]

        departments = available_tables[
            "departments"
        ]

        control_columns = st.columns(
            [1.1, 1.2, 1.2]
        )

        with control_columns[0]:

            employee_scope = st.selectbox(
                "Employee scope",
                [
                    "Current employees",
                    "All salary records",
                ],
            )

        salaries_dates = pd.to_datetime(
            salaries["from_date"],
            errors="coerce",
        )

        minimum_date = salaries_dates.min().date()
        maximum_date = salaries_dates.max().date()

        with control_columns[1]:

            start_date = st.date_input(
                "Salary records from",
                value=minimum_date,
                min_value=minimum_date,
                max_value=maximum_date,
            )

        with control_columns[2]:

            end_date = st.date_input(
                "Salary records to",
                value=maximum_date,
                min_value=minimum_date,
                max_value=maximum_date,
            )

        bins = st.slider(
            "Histogram bins",
            min_value=5,
            max_value=30,
            value=10,
            step=1,
        )

        if start_date > end_date:

            st.error(
                "The start date must be before "
                "the end date."
            )

        else:

            joined_salary, salary_summary = (
                department_salary_analysis(
                    employees=employees,
                    salaries=salaries,
                    dept_emp=dept_emp,
                    departments=departments,
                    current_only=(
                        employee_scope
                        == "Current employees"
                    ),
                    start_date=str(start_date),
                    end_date=str(end_date),
                )
            )

            salary_metrics = st.columns(4)

            salary_metrics[0].metric(
                "Salary records",
                f"{len(joined_salary):,}",
            )

            salary_metrics[1].metric(
                "Employees represented",
                (
                    f"{joined_salary['emp_no'].nunique():,}"
                ),
            )

            salary_metrics[2].metric(
                "Departments",
                (
                    f"{joined_salary['dept_no'].nunique():,}"
                ),
            )

            highest_department = (
                salary_summary.iloc[0]
                if not salary_summary.empty
                else None
            )

            salary_metrics[3].metric(
                "Highest average salary",
                (
                    highest_department[
                        "dept_name"
                    ]
                    if highest_department is not None
                    else "Not available"
                ),
            )

            if salary_summary.empty:

                st.warning(
                    "No salary records match the "
                    "selected settings."
                )

            else:

                st.markdown(
                    f"""
**Finding:** **{highest_department['dept_name']}** has the highest
average salary in the selected data at approximately
**${highest_department['average_salary']:,.0f}**.

The plot should not be interpreted only by comparing the centres of the
histograms. A department may have a similar average but a much wider salary
distribution, indicating greater variation in role level, seniority or pay.
Differences may also reflect department composition and should not be treated
as evidence of unfairness without further analysis.
"""
                )

                salary_figure = (
                    department_salary_histograms(
                        joined_salary,
                        bins=bins,
                    )
                )

                st.pyplot(
                    salary_figure,
                    use_container_width=False,
                )

                st.markdown(
                    """
The coloured outline distinguishes departments, while each legend reports
the mean (`μ`) and standard deviation (`σ`) in thousands. Using identical
salary units and a common grid makes it easier to compare both typical salary
and distribution shape across departments.
"""
                )

                with st.expander(
                    "Department salary summary"
                ):

                    formatted_summary = (
                        salary_summary.copy()
                    )

                    st.dataframe(
                        formatted_summary,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "minimum_salary":
                                st.column_config.NumberColumn(
                                    "Minimum salary",
                                    format="$%d",
                                ),
                            "average_salary":
                                st.column_config.NumberColumn(
                                    "Average salary",
                                    format="$%d",
                                ),
                            "median_salary":
                                st.column_config.NumberColumn(
                                    "Median salary",
                                    format="$%d",
                                ),
                            "maximum_salary":
                                st.column_config.NumberColumn(
                                    "Maximum salary",
                                    format="$%d",
                                ),
                            "salary_std":
                                st.column_config.NumberColumn(
                                    "Salary standard deviation",
                                    format="$%d",
                                ),
                        },
                    )

                with st.expander(
                    "How the analysis works"
                ):

                    st.markdown(
                        """
1. Join employees to salary records using `emp_no`.
2. Join employee records to department assignments.
3. Join department codes to department names.
4. Remove rows without a salary or department.
5. Optionally retain only current employee and salary records.
6. Restrict salary records to the selected date range.
7. Group records by department.
8. calculate counts and salary summary statistics.
9. Plot one salary histogram for each department.
"""
                    )

                    code_columns = st.columns(2)

                    with code_columns[0]:

                        st.markdown("#### SQL")

                        st.code(
                            salary_business_sql(),
                            language="sql",
                        )

                    with code_columns[1]:

                        st.markdown("#### pandas")

                        st.code(
                            salary_business_pandas(),
                            language="python",
                        )

                with st.expander(
                    "Joined employee salary records"
                ):

                    st.dataframe(
                        joined_salary[
                            [
                                "emp_no",
                                "first_name",
                                "last_name",
                                "salary",
                                "dept_no",
                                "dept_name",
                                "from_date",
                                "to_date",
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )


# Reference
with tabs[5]:
    st.markdown("## SQL ↔ pandas reference")

    reference = reference_table()
    search = st.text_input(
        "Search operations",
        placeholder="e.g. join, rank, null",
    )

    if search:
        mask = (
            reference.astype(str)
            .apply(
                lambda col: col.str.contains(
                    search,
                    case=False,
                    na=False,
                )
            )
            .any(axis=1)
        )
        reference = reference[mask]

    st.dataframe(
        reference,
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Scope and limitations"):
        st.markdown(
            """
- The workbench generates a curated set of operations rather than parsing arbitrary SQL.
- SQL dialects vary, so the examples use broadly familiar syntax.
- pandas and SQL can differ in their treatment of missing values, ordering and duplicates.
- Generated code should be validated against the target database and business rules.
"""
        )
