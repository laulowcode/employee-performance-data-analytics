"""
Analytics / Semantic Layer
==========================
Reads from dataset/staging and produces:

  Dimensions
  ----------
  dim_employee        – one row per employee (latest snapshot)
  dim_store           – one row per store
  dim_manager         – one row per manager
  dim_date            – calendar spine (monthly grain)

  Fact table
  ----------
  fact_employee_monthly  – grain: employee × year_month
                           joins monthly_performance + role_kpis + employee attributes

  Analytical views  (answer the 10 questions)
  ------------------
  v_turnover_by_department      Q1  – headcount, exits, turnover rate
  v_salary_by_job_level         Q2  – avg salary by job level × department
  v_performance_by_month        Q3  – avg performance rating by year-month
  v_manager_team_performance    Q4  – top managers by avg team performance
  v_training_vs_performance     Q5  – training hours bucket × avg performance
  v_store_revenue               Q6  – store revenue, achievement %, satisfaction
  v_satisfaction_by_department  Q7  – avg satisfaction + engagement by dept
  v_productivity_by_role        Q8  – avg productivity index by job role
  v_promotion_candidates        Q9  – employees ranked by composite promotion score
  v_age_vs_performance          Q10 – avg performance by age band
"""

import pathlib

import pandas as pd


PROJECT_ROOT = pathlib.Path(__file__).parent
STAGING_DIR = PROJECT_ROOT / "dataset" / "staging"
ANALYTICS_DIR = PROJECT_ROOT / "dataset" / "analytics"


def _ensure_analytics_dir() -> None:
    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)


def write_analytics(df: pd.DataFrame, name: str) -> None:
    _ensure_analytics_dir()
    df.to_parquet(ANALYTICS_DIR / f"{name}.parquet", index=False)
    df.to_csv(ANALYTICS_DIR / f"{name}.csv", index=False)
    print(f"  [analytics] {name}: {len(df):,} rows -> dataset/analytics/{name}.*")


# ---------------------------------------------------------------------------
# Load staging
# ---------------------------------------------------------------------------

def load_staging() -> dict[str, pd.DataFrame]:
    print("Loading staging tables...")
    return {
        "employees": pd.read_parquet(STAGING_DIR / "employees.parquet"),
        "stores": pd.read_parquet(STAGING_DIR / "stores.parquet"),
        "monthly_performance": pd.read_parquet(STAGING_DIR / "monthly_performance.parquet"),
        "role_kpis": pd.read_parquet(STAGING_DIR / "role_kpis.parquet"),
        "business_outcomes": pd.read_parquet(STAGING_DIR / "business_outcomes.parquet"),
    }


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------

def build_dim_employee(employees: pd.DataFrame) -> pd.DataFrame:
    dim = employees[[
        "employee_id", "full_name", "age", "age_band", "education_level",
        "hire_date", "exit_date", "is_active", "tenure_days",
        "department", "job_role", "job_level", "employment_type",
        "base_salary_annual",
        "store_id", "manager_id", "manager_name", "manager_status",
    ]].copy()
    return dim


def build_dim_store(stores: pd.DataFrame) -> pd.DataFrame:
    return stores.copy()


def build_dim_manager(employees: pd.DataFrame) -> pd.DataFrame:
    dim = (
        employees[["manager_id", "manager_name", "manager_status", "department"]]
        .drop_duplicates(subset=["manager_id"])
        .reset_index(drop=True)
    )
    return dim


def build_dim_date(monthly_performance: pd.DataFrame) -> pd.DataFrame:
    """Calendar spine at monthly grain derived from the data range."""
    months = monthly_performance["year_month"].dropna().unique()
    dim = pd.DataFrame({"year_month": sorted(months)})
    dim["year"] = dim["year_month"].dt.year
    dim["month"] = dim["year_month"].dt.month
    dim["month_name"] = dim["year_month"].dt.strftime("%B")
    dim["quarter"] = dim["year_month"].dt.quarter
    dim["year_month_label"] = dim["year_month"].dt.strftime("%Y-%m")
    return dim


# ---------------------------------------------------------------------------
# Fact table
# ---------------------------------------------------------------------------

def build_fact_employee_monthly(
    monthly_performance: pd.DataFrame,
    role_kpis: pd.DataFrame,
    employees: pd.DataFrame,
) -> pd.DataFrame:
    """
    Grain: employee_id × year_month
    Joins monthly_performance with role_kpis, then enriches with
    employee dimension attributes needed for most analytical queries.
    """
    fact = monthly_performance.merge(
        role_kpis.drop(columns=["year", "month"]),
        on=["employee_id", "year_month"],
        how="left",
    )

    emp_attrs = employees[[
        "employee_id", "department", "job_role", "job_level",
        "employment_type", "store_id", "manager_id",
        "age", "age_band", "base_salary_annual", "is_active",
    ]]
    fact = fact.merge(emp_attrs, on="employee_id", how="left")

    col_order = [
        "employee_id", "year_month", "year", "month",
        "department", "job_role", "job_level", "employment_type",
        "store_id", "manager_id", "age", "age_band",
        "base_salary_annual", "is_active",
        "performance_rating", "training_hours", "overtime_hours",
        "absenteeism_days", "promotion_flag", "salary_increase_flag",
        "monthly_bonus", "benefits_cost",
        "employee_satisfaction", "engagement_index", "manager_evaluation",
        "kpi_1_name", "kpi_1_value",
        "kpi_2_name", "kpi_2_value",
        "kpi_3_name", "kpi_3_value",
        "productivity_index",
    ]
    return fact[col_order]


# ---------------------------------------------------------------------------
# Analytical views
# ---------------------------------------------------------------------------

def build_v_turnover_by_department(employees: pd.DataFrame) -> pd.DataFrame:
    """Q1 – Turnover rate per department."""
    total = employees.groupby("department")["employee_id"].count().rename("total_employees")
    exited = (
        employees[~employees["is_active"]]
        .groupby("department")["employee_id"]
        .count()
        .rename("resigned_employees")
    )
    df = pd.concat([total, exited], axis=1).fillna(0).reset_index()
    df["resigned_employees"] = df["resigned_employees"].astype(int)
    df["turnover_rate_pct"] = (df["resigned_employees"] / df["total_employees"] * 100).round(2)
    return df.sort_values("turnover_rate_pct", ascending=False).reset_index(drop=True)


def build_v_salary_by_job_level(employees: pd.DataFrame) -> pd.DataFrame:
    """Q2 – Average salary by job level × department."""
    return (
        employees.groupby(["job_level", "department"])["base_salary_annual"]
        .agg(avg_salary="mean", median_salary="median", employee_count="count")
        .round(2)
        .reset_index()
        .sort_values(["job_level", "avg_salary"], ascending=[True, False])
        .reset_index(drop=True)
    )


def build_v_performance_by_month(fact: pd.DataFrame) -> pd.DataFrame:
    """Q3 – Average performance rating by year-month."""
    return (
        fact.groupby(["year", "month", "year_month"])["performance_rating"]
        .agg(avg_performance="mean", employee_count="count")
        .round(3)
        .reset_index()
        .sort_values("year_month")
        .reset_index(drop=True)
    )


def build_v_manager_team_performance(fact: pd.DataFrame, employees: pd.DataFrame) -> pd.DataFrame:
    """Q4 – Top 10 managers by average team performance."""
    mgr_map = employees[["manager_id", "manager_name", "manager_status"]].drop_duplicates("manager_id")
    df = (
        fact.groupby("manager_id")
        .agg(
            avg_performance=("performance_rating", "mean"),
            avg_engagement=("engagement_index", "mean"),
            avg_satisfaction=("employee_satisfaction", "mean"),
            team_size=("employee_id", "nunique"),
        )
        .round(3)
        .reset_index()
        .merge(mgr_map, on="manager_id", how="left")
        .sort_values("avg_performance", ascending=False)
        .reset_index(drop=True)
    )
    return df


def build_v_training_vs_performance(fact: pd.DataFrame) -> pd.DataFrame:
    """Q5 – Avg performance per training-hours bucket."""
    fact = fact.copy()
    fact["training_bucket"] = pd.cut(
        fact["training_hours"],
        bins=[0, 5, 10, 20, 40, 200],
        labels=["0-5h", "6-10h", "11-20h", "21-40h", "40h+"],
        right=True,
    ).astype(str)
    return (
        fact.groupby("training_bucket")
        .agg(
            avg_performance=("performance_rating", "mean"),
            avg_productivity=("productivity_index", "mean"),
            employee_months=("employee_id", "count"),
        )
        .round(3)
        .reset_index()
    )


def build_v_store_revenue(
    business_outcomes: pd.DataFrame,
    stores: pd.DataFrame,
) -> pd.DataFrame:
    """Q6 – Total revenue, achievement % and satisfaction per store."""
    df = (
        business_outcomes.groupby("store_id")
        .agg(
            total_sales_target=("sales_target", "sum"),
            total_sales_actual=("sales_actual", "sum"),
            avg_customer_satisfaction=("customer_satisfaction", "mean"),
            avg_nps_score=("nps_score", "mean"),
            avg_waste_pct=("waste_percentage", "mean"),
            avg_on_time_delivery=("on_time_delivery", "mean"),
        )
        .round(2)
        .reset_index()
    )
    df["overall_achievement_pct"] = (
        df["total_sales_actual"] / df["total_sales_target"] * 100
    ).round(2)
    df = df.merge(stores[["store_id", "store_name", "city", "store_type"]], on="store_id", how="left")
    return df.sort_values("total_sales_actual", ascending=False).reset_index(drop=True)


def build_v_satisfaction_by_department(fact: pd.DataFrame) -> pd.DataFrame:
    """Q7 – Avg employee satisfaction and engagement by department."""
    return (
        fact.groupby("department")
        .agg(
            avg_satisfaction=("employee_satisfaction", "mean"),
            avg_engagement=("engagement_index", "mean"),
            avg_manager_eval=("manager_evaluation", "mean"),
            employee_months=("employee_id", "count"),
        )
        .round(3)
        .reset_index()
        .sort_values("avg_satisfaction", ascending=False)
        .reset_index(drop=True)
    )


def build_v_productivity_by_role(fact: pd.DataFrame) -> pd.DataFrame:
    """Q8 – Average productivity index by job role."""
    return (
        fact.groupby("job_role")
        .agg(
            avg_productivity=("productivity_index", "mean"),
            avg_performance=("performance_rating", "mean"),
            employee_months=("employee_id", "count"),
        )
        .round(3)
        .reset_index()
        .sort_values("avg_productivity", ascending=False)
        .reset_index(drop=True)
    )


def build_v_promotion_candidates(
    fact: pd.DataFrame,
    employees: pd.DataFrame,
) -> pd.DataFrame:
    """
    Q9 – Employees most likely to be promoted.
    Composite score = avg(performance_rating/5 + satisfaction/10 + productivity_index/2)
    normalised to 0–100.  Only active employees are included.
    """
    summary = (
        fact.groupby("employee_id")
        .agg(
            avg_performance=("performance_rating", "mean"),
            avg_satisfaction=("employee_satisfaction", "mean"),
            avg_productivity=("productivity_index", "mean"),
            total_promotions=("promotion_flag", "sum"),
            months_observed=("year_month", "count"),
        )
        .reset_index()
    )
    summary["promotion_score"] = (
        (summary["avg_performance"] / 5) * 40
        + (summary["avg_satisfaction"] / 10) * 30
        + (summary["avg_productivity"] / 2) * 30
    ).round(2)

    active_employees = employees[employees["is_active"]][
        ["employee_id", "full_name", "department", "job_role", "job_level", "base_salary_annual"]
    ]
    df = summary.merge(active_employees, on="employee_id", how="inner")
    return df.sort_values("promotion_score", ascending=False).reset_index(drop=True)


def build_v_age_vs_performance(fact: pd.DataFrame) -> pd.DataFrame:
    """Q10 – Average performance by age band."""
    return (
        fact.groupby("age_band")
        .agg(
            avg_performance=("performance_rating", "mean"),
            avg_productivity=("productivity_index", "mean"),
            avg_satisfaction=("employee_satisfaction", "mean"),
            employee_months=("employee_id", "count"),
        )
        .round(3)
        .reset_index()
        .sort_values("age_band")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_analytics_layer() -> None:
    print("Building analytics layer...")
    stg = load_staging()

    emp = stg["employees"]
    stores = stg["stores"]
    mp = stg["monthly_performance"]
    kpis = stg["role_kpis"]
    biz = stg["business_outcomes"]

    # Dimensions
    write_analytics(build_dim_employee(emp), "dim_employee")
    write_analytics(build_dim_store(stores), "dim_store")
    write_analytics(build_dim_manager(emp), "dim_manager")
    write_analytics(build_dim_date(mp), "dim_date")

    # Fact table
    fact = build_fact_employee_monthly(mp, kpis, emp)
    write_analytics(fact, "fact_employee_monthly")

    # Analytical views
    write_analytics(build_v_turnover_by_department(emp), "v_turnover_by_department")
    write_analytics(build_v_salary_by_job_level(emp), "v_salary_by_job_level")
    write_analytics(build_v_performance_by_month(fact), "v_performance_by_month")
    write_analytics(build_v_manager_team_performance(fact, emp), "v_manager_team_performance")
    write_analytics(build_v_training_vs_performance(fact), "v_training_vs_performance")
    write_analytics(build_v_store_revenue(biz, stores), "v_store_revenue")
    write_analytics(build_v_satisfaction_by_department(fact), "v_satisfaction_by_department")
    write_analytics(build_v_productivity_by_role(fact), "v_productivity_by_role")
    write_analytics(build_v_promotion_candidates(fact, emp), "v_promotion_candidates")
    write_analytics(build_v_age_vs_performance(fact), "v_age_vs_performance")

    print("Analytics layer complete.\n")


if __name__ == "__main__":
    build_analytics_layer()
