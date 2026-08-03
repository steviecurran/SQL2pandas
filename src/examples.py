from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def customer_revenue_analysis(
    transactions: pd.DataFrame,
    target_revenue: float = 80.0,
):
    """
    Aggregate revenue by customer and calculate how many customers are
    required to explain the selected percentage of total revenue.
    """

    if not 0 < target_revenue <= 100:
        raise ValueError(
            "target_revenue must be between 0 and 100."
        )

    revenue = (
        transactions
        .groupby("customer_id")[
            "transaction_value"
        ]
        .sum()
        .reset_index(
            name="total_revenue"
        )
        .sort_values(
            "total_revenue",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    revenue[
        "cumulative_revenue"
    ] = (
        revenue[
            "total_revenue"
        ]
        .cumsum()
    )

    revenue[
        "cumulative_percentage"
    ] = (
        100
        * revenue[
            "cumulative_revenue"
        ]
        / revenue[
            "total_revenue"
        ].sum()
    )

    customers_required = int(
        (
            revenue[
                "cumulative_percentage"
            ]
            < target_revenue
        ).sum()
        + 1
    )

    customers_required = min(
        customers_required,
        len(revenue),
    )

    summary = {
        "target_revenue":
            float(target_revenue),
        "customers_required":
            customers_required,
        "percentage_customers":
            (
                100
                * customers_required
                / len(revenue)
            ),
    }

    return revenue, summary


def revenue_histogram(
    revenue: pd.DataFrame,
    bin_width: float = 100.0,
):
    """
    Plot customer-level revenue using a user-selected bin width.
    """

    values = revenue["total_revenue"].dropna()

    if values.empty:
        raise ValueError(
            "No customer-revenue values are available."
        )

    if bin_width <= 0:
        raise ValueError(
            "Bin width must be greater than zero."
        )

    minimum = float(values.min())
    maximum = float(values.max())

    minimum_boundary = (
        np.floor(minimum / bin_width)
        * bin_width
    )

    maximum_boundary = (
        np.ceil(maximum / bin_width)
        * bin_width
    )

    if minimum_boundary == maximum_boundary:
        maximum_boundary += bin_width

    bins = np.arange(
        minimum_boundary,
        maximum_boundary + bin_width,
        bin_width,
    )

    mean = float(values.mean())
    standard_deviation = float(
        values.std(ddof=1)
    )

    figure, axis = plt.subplots(
        figsize=(6.8, 4.8)
    )

    axis.hist(
        values,
        bins=bins,
        color="silver",
        edgecolor="black",
        linewidth=1.8,
        label=(
            f"μ = {mean:.0f}, "
            f"σ = {standard_deviation:.0f}"
        ),
    )

    axis.set_xlabel(
        "Customer revenue",
        fontsize=14,
    )

    axis.set_ylabel(
        "Number",
        fontsize=14,
    )

    axis.legend(
        fontsize=12,
        loc="best",
    )

    axis.tick_params(
        direction="in",
        pad=7,
        length=6,
        width=1.5,
        labelsize=12,
        which="major",
        right=True,
        top=True,
    )

    axis.tick_params(
        direction="in",
        pad=7,
        length=3,
        width=1.2,
        which="minor",
        right=True,
        top=True,
    )

    for spine in axis.spines.values():
        spine.set_linewidth(1.8)

    figure.tight_layout()

    return figure


def pareto_plot(
    revenue: pd.DataFrame,
    target_revenue: float = 80.0,
):
    """
    Plot sorted customer revenue with cumulative contribution on a second axis.
    """

    sorted_revenue = (
        revenue
        .sort_values(
            "total_revenue",
            ascending=False,
        )
        .reset_index(drop=True)
        .copy()
    )

    x = np.arange(
        len(sorted_revenue)
    )

    cumulative_percentage = (
        100
        * sorted_revenue[
            "total_revenue"
        ].cumsum()
        / sorted_revenue[
            "total_revenue"
        ].sum()
    )

    figure, revenue_axis = plt.subplots(
        figsize=(6.8, 4.8)
    )

    revenue_axis.bar(
        x,
        sorted_revenue[
            "total_revenue"
        ],
        color="silver",
        edgecolor="black",
        linewidth=0.7,
        width=0.9,
    )

    revenue_axis.set_xlabel(
        "Customers [sorted]",
        fontsize=14,
    )

    revenue_axis.set_ylabel(
        "Customer revenue",
        fontsize=14,
    )

    revenue_axis.tick_params(
        direction="in",
        pad=7,
        length=6,
        width=1.5,
        labelsize=12,
        which="major",
        right=False,
        top=True,
    )

    revenue_axis.tick_params(
        direction="in",
        pad=7,
        length=3,
        width=1.2,
        which="minor",
        right=False,
        top=True,
    )

    for spine in revenue_axis.spines.values():
        spine.set_linewidth(1.8)

    cumulative_axis = (
        revenue_axis.twinx()
    )

    cumulative_axis.plot(
        x,
        cumulative_percentage,
        color="blue",
        linewidth=2.2,
    )

    target_position = int(
        (
            cumulative_percentage
            < target_revenue
        ).sum()
    )

    target_position = min(
        target_position,
        len(cumulative_percentage) - 1,
    )

    target_customer_count = (
        target_position + 1
    )

    target_share_customers = (
        100.0
        * target_customer_count
        / len(sorted_revenue)
    )

    target_cumulative_value = float(
        cumulative_percentage.iloc[
            target_position
        ]
    )

    cumulative_axis.axhline(
        target_revenue,
        color="red",
        linestyle=":",
        linewidth=2,
        label=(
            f"{target_revenue:.0f}% "
            "revenue target"
        ),
    )

    cumulative_axis.axvline(
        target_position,
        color="red",
        linestyle=":",
        linewidth=2,
    )

    cumulative_axis.scatter(
        target_position,
        target_cumulative_value,
        color="red",
        s=55,
        zorder=5,
    )

    cumulative_axis.annotate(
        (
            f"{target_customer_count} customers "
            f"({target_share_customers:.1f}%)\n"
            f"→ {target_revenue:.0f}% revenue"
        ),
        xy=(
            target_position,
            target_cumulative_value,
        ),
        xytext=(14, -42),
        textcoords="offset points",
        fontsize=11,
        fontweight="bold",
        color="red",
        arrowprops={
            "arrowstyle": "->",
            "color": "red",
            "linewidth": 1.5,
        },
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "red",
            "alpha": 0.9,
        },
    )

    cumulative_axis.set_ylabel(
        "Cumulative Total [%]",
        fontsize=14,
    )

    cumulative_axis.legend(
        fontsize=11,
        loc="lower right",
    )

    cumulative_axis.set_ylim(
        0,
        105,
    )

    cumulative_axis.tick_params(
        direction="in",
        pad=7,
        length=6,
        width=1.5,
        labelsize=12,
        which="major",
        left=False,
        top=True,
    )

    cumulative_axis.tick_params(
        direction="in",
        pad=7,
        length=3,
        width=1.2,
        which="minor",
        left=False,
        top=True,
    )

    for spine in cumulative_axis.spines.values():
        spine.set_linewidth(1.8)

    figure.tight_layout()

    return figure


def department_salary_analysis(
    employees: pd.DataFrame,
    salaries: pd.DataFrame,
    dept_emp: pd.DataFrame,
    departments: pd.DataFrame,
    current_only: bool = True,
    start_date: str | None = None,
    end_date: str | None = None,
):
    """
    Recreate Business Problem 2 from the original notebook:
    join employee, salary and department tables, then compare salary
    distributions and summary statistics across departments.
    """

    salary_data = salaries.copy()
    department_history = dept_emp.copy()

    salary_data["from_date"] = pd.to_datetime(
        salary_data["from_date"],
        errors="coerce",
    )
    salary_data["to_date_parsed"] = pd.to_datetime(
        salary_data["to_date"].replace(
            "9999-01-01",
            "2099-01-01",
        ),
        errors="coerce",
    )

    department_history["from_date"] = pd.to_datetime(
        department_history["from_date"],
        errors="coerce",
    )
    department_history["to_date_parsed"] = pd.to_datetime(
        department_history["to_date"].replace(
            "9999-01-01",
            "2099-01-01",
        ),
        errors="coerce",
    )

    if current_only:
        salary_data = salary_data[
            salary_data["to_date"] == "9999-01-01"
        ].copy()

        department_history = department_history[
            department_history["to_date"] == "9999-01-01"
        ].copy()

    joined = (
        employees.merge(
            salary_data[
                [
                    "emp_no",
                    "salary",
                    "from_date",
                    "to_date",
                    "to_date_parsed",
                ]
            ],
            on="emp_no",
            how="left",
        )
        .merge(
            department_history[
                [
                    "emp_no",
                    "dept_no",
                ]
            ],
            on="emp_no",
            how="left",
        )
        .merge(
            departments,
            on="dept_no",
            how="left",
        )
        .dropna(
            subset=[
                "salary",
                "dept_no",
                "dept_name",
            ]
        )
    )

    if start_date is not None:
        joined = joined[
            joined["from_date"]
            >= pd.Timestamp(start_date)
        ]

    if end_date is not None:
        joined = joined[
            joined["from_date"]
            <= pd.Timestamp(end_date)
        ]

    summary = (
        joined.groupby(
            [
                "dept_no",
                "dept_name",
            ],
            dropna=False,
        )["salary"]
        .agg(
            staff_records="count",
            minimum_salary="min",
            average_salary="mean",
            median_salary="median",
            maximum_salary="max",
            salary_std="std",
        )
        .reset_index()
        .sort_values(
            "average_salary",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    money_columns = [
        "minimum_salary",
        "average_salary",
        "median_salary",
        "maximum_salary",
        "salary_std",
    ]

    summary[money_columns] = (
        summary[money_columns]
        .round(0)
    )

    return joined, summary


def department_salary_histograms(
    joined: pd.DataFrame,
    bins: int = 10,
):
    """
    Create the 3 × 3 salary-distribution plot used in Business Problem 2.
    """

    departments = (
        joined[
            [
                "dept_no",
                "dept_name",
            ]
        ]
        .drop_duplicates()
        .sort_values("dept_no")
        .reset_index(drop=True)
    )

    figure, axes = plt.subplots(
        3,
        3,
        figsize=(11, 9),
    )

    axes = axes.flatten()

    for index, axis in enumerate(axes):

        if index >= len(departments):

            axis.set_visible(False)
            continue

        dept_no = departments.loc[
            index,
            "dept_no",
        ]

        dept_name = departments.loc[
            index,
            "dept_name",
        ]

        values = (
            joined.loc[
                joined["dept_no"] == dept_no,
                "salary",
            ]
            .dropna()
            / 1000.0
        )

        mean = float(values.mean())
        standard_deviation = float(
            values.std(ddof=1)
        )

        axis.hist(
            values,
            bins=bins,
            color="silver",
            edgecolor=f"C{index}",
            linewidth=2,
            label=(
                rf"$\mu={mean:.0f}$k, "
                rf"$\sigma={standard_deviation:.0f}$k"
            ),
        )

        axis.set_xlabel(
            f"{dept_name} [{dept_no}]"
        )

        if index % 3 == 0:
            axis.set_ylabel(
                "Number of salary records"
            )

        axis.legend(
            fontsize=8,
            loc="upper right",
        )

        axis.tick_params(
            direction="in",
            right=True,
            top=True,
        )

        for spine in axis.spines.values():
            spine.set_linewidth(1.5)

    figure.suptitle(
        "Salary distributions by department",
        y=1.01,
    )

    figure.tight_layout()

    return figure


def salary_business_sql() -> str:
    return """WITH employee_salaries AS (
    SELECT
        e.emp_no,
        e.first_name,
        e.last_name,
        s.salary,
        de.dept_no,
        d.dept_name,
        s.from_date,
        s.to_date
    FROM employees AS e
    LEFT JOIN salaries AS s
        ON e.emp_no = s.emp_no
    LEFT JOIN dept_emp AS de
        ON e.emp_no = de.emp_no
    LEFT JOIN departments AS d
        ON de.dept_no = d.dept_no
    WHERE s.salary IS NOT NULL
)
SELECT
    dept_no,
    dept_name,
    COUNT(*) AS staff_records,
    MIN(salary) AS minimum_salary,
    AVG(salary) AS average_salary,
    MAX(salary) AS maximum_salary
FROM employee_salaries
GROUP BY dept_no, dept_name
ORDER BY average_salary DESC;"""


def salary_business_pandas() -> str:
    return """joined = (
    employees
    .merge(salaries, on="emp_no", how="left")
    .merge(dept_emp, on="emp_no", how="left")
    .merge(departments, on="dept_no", how="left")
    .dropna(subset=["salary"])
)

summary = (
    joined
    .groupby(["dept_no", "dept_name"])["salary"]
    .agg(
        staff_records="count",
        minimum_salary="min",
        average_salary="mean",
        maximum_salary="max",
    )
    .reset_index()
    .sort_values(
        "average_salary",
        ascending=False,
    )
)"""
