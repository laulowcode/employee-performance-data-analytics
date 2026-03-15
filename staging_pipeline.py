import pathlib
from typing import Optional

import pandas as pd


PROJECT_ROOT = pathlib.Path(__file__).parent
RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
STAGING_DIR = PROJECT_ROOT / "dataset" / "staging"


def _ensure_staging_dir() -> None:
    STAGING_DIR.mkdir(parents=True, exist_ok=True)


def _parse_date_dmy(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%d/%m/%Y", errors="coerce")


def _parse_yearmonth(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%Y-%m", errors="coerce")


def write_staging(df: pd.DataFrame, name: str) -> None:
    _ensure_staging_dir()
    df.to_parquet(STAGING_DIR / f"{name}.parquet", index=False)
    df.to_csv(STAGING_DIR / f"{name}.csv", index=False)
    print(f"  [staging] {name}: {len(df):,} rows -> dataset/staging/{name}.*")


# ---------------------------------------------------------------------------
# Employees
# ---------------------------------------------------------------------------

def load_raw_employees(path: Optional[pathlib.Path] = None) -> pd.DataFrame:
    df = pd.read_csv(path or RAW_DIR / "employees.csv")
    df["Hire_Date"] = _parse_date_dmy(df["Hire_Date"])
    df["Exit_Date"] = _parse_date_dmy(df["Exit_Date"])
    return df


def transform_employees(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    today = pd.Timestamp("today").normalize()
    df["is_active"] = df["Exit_Date"].isna()
    df["tenure_days"] = (df["Exit_Date"].fillna(today) - df["Hire_Date"]).dt.days
    age = df["Age"]
    df["age_band"] = pd.cut(
        age,
        bins=[0, 25, 35, 45, 55, 100],
        labels=["<25", "25-34", "35-44", "45-54", "55+"],
        right=False,
    ).astype(str)
    df.rename(columns={
        "Employee_Id": "employee_id",
        "Full_Name": "full_name",
        "Age": "age",
        "Education_Level": "education_level",
        "Hire_Date": "hire_date",
        "Exit_Date": "exit_date",
        "Department": "department",
        "Job_Role": "job_role",
        "Job_Level": "job_level",
        "Employment_Type": "employment_type",
        "Base_Salary_Annual": "base_salary_annual",
        "Store_Location": "store_location",
        "Store_Location_Latitude": "store_location_latitude",
        "Store_Location_Longitude": "store_location_longitude",
        "Store_Id": "store_id",
        "Manager_Id": "manager_id",
        "Manager_Name": "manager_name",
        "Manager_Status": "manager_status",
    }, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------

def load_raw_stores(path: Optional[pathlib.Path] = None) -> pd.DataFrame:
    df = pd.read_csv(path or RAW_DIR / "stores.csv")
    df["Opening_Date"] = _parse_date_dmy(df["Opening_Date"])
    return df


def transform_stores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.rename(columns={
        "Store_Id": "store_id",
        "Store_Name": "store_name",
        "City": "city",
        "City_Latitude": "city_latitude",
        "City_Longitude": "city_longitude",
        "Store_Type": "store_type",
        "Opening_Date": "opening_date",
    }, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Monthly Performance
# ---------------------------------------------------------------------------

def load_raw_monthly_performance(path: Optional[pathlib.Path] = None) -> pd.DataFrame:
    df = pd.read_csv(path or RAW_DIR / "monthly_performance.csv")
    df["Year_Month"] = _parse_yearmonth(df["Year_Month"])
    return df


def transform_monthly_performance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["Year_Month"].dt.year
    df["month"] = df["Year_Month"].dt.month
    df["Promotion_Flag"] = df["Promotion_Flag"].astype(bool)
    df["Salary_Increase_Flag"] = df["Salary_Increase_Flag"].astype(bool)
    df.rename(columns={
        "Employee_Id": "employee_id",
        "Year_Month": "year_month",
        "Performance_Rating": "performance_rating",
        "Training_Hours": "training_hours",
        "Overtime_Hours": "overtime_hours",
        "Absenteeism_Days": "absenteeism_days",
        "Promotion_Flag": "promotion_flag",
        "Salary_Increase_Flag": "salary_increase_flag",
        "Monthly_Bonus": "monthly_bonus",
        "Benefits_Cost": "benefits_cost",
        "Employee_Satisfaction": "employee_satisfaction",
        "Engagement_Index": "engagement_index",
        "Manager_Evaluation": "manager_evaluation",
    }, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Role KPIs
# ---------------------------------------------------------------------------

def load_raw_role_kpis(path: Optional[pathlib.Path] = None) -> pd.DataFrame:
    df = pd.read_csv(path or RAW_DIR / "role_kpis.csv")
    df["Year_Month"] = _parse_yearmonth(df["Year_Month"])
    return df


def transform_role_kpis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["Year_Month"].dt.year
    df["month"] = df["Year_Month"].dt.month
    df.rename(columns={
        "Employee_Id": "employee_id",
        "Year_Month": "year_month",
        "Kpi_1_Value": "kpi_1_value",
        "Kpi_1_Name": "kpi_1_name",
        "Kpi_2_Value": "kpi_2_value",
        "Kpi_2_Name": "kpi_2_name",
        "Kpi_3_Value": "kpi_3_value",
        "Kpi_3_Name": "kpi_3_name",
        "Productivity_Index": "productivity_index",
    }, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Business Outcomes
# ---------------------------------------------------------------------------

def load_raw_business_outcomes(path: Optional[pathlib.Path] = None) -> pd.DataFrame:
    df = pd.read_csv(path or RAW_DIR / "business_outcomes.csv")
    df["Year_Month"] = _parse_yearmonth(df["Year_Month"])
    return df


def transform_business_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["Year_Month"].dt.year
    df["month"] = df["Year_Month"].dt.month
    df["sales_achievement_pct"] = (df["Sales_Actual"] / df["Sales_Target"] * 100).round(2)
    df.rename(columns={
        "Store_Id": "store_id",
        "Department": "department",
        "Year_Month": "year_month",
        "Sales_Target": "sales_target",
        "Sales_Actual": "sales_actual",
        "Customer_Satisfaction": "customer_satisfaction",
        "Nps_Score": "nps_score",
        "Waste_Percentage": "waste_percentage",
        "On_Time_Delivery": "on_time_delivery",
    }, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_staging_layer() -> None:
    print("Building staging layer...")

    write_staging(transform_employees(load_raw_employees()), "employees")
    write_staging(transform_stores(load_raw_stores()), "stores")
    write_staging(transform_monthly_performance(load_raw_monthly_performance()), "monthly_performance")
    write_staging(transform_role_kpis(load_raw_role_kpis()), "role_kpis")
    write_staging(transform_business_outcomes(load_raw_business_outcomes()), "business_outcomes")

    print("Staging layer complete.\n")


if __name__ == "__main__":
    build_staging_layer()
