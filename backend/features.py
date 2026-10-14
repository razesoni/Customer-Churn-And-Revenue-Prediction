"""Feature engineering shared by notebook training and legacy model serving."""
import numpy as np
import pandas as pd

REQUIRED_COLS = [
    "AccountAge", "MonthlyCharges", "TotalCharges", "ViewingHoursPerWeek",
    "AverageViewingDuration", "ContentDownloadsPerMonth", "UserRating", "SupportTicketsPerMonth",
]

ENGINEERED_COLUMNS = [
    "ChargesPerMonthAccount",
    "DownloadsPerViewingHour",
    "SupportTicketsPerMonthAge",
    "EngagementScore",
    "ValueRatio",
    "ActivityIntensity",
    "CostPerRating",
    "SupportRisk",
    "CostPerViewingHour",
    "ValueIndex",
    "SupportTicketsPerViewingHour",
    "UnproductiveViewing",
    "DownloadsPerSupportTicket",
    "IsNewUser",
    "IsLoyalUser",
]


def _safe_div(a, b):
    return (a / b).replace([np.inf, -np.inf], np.nan)


def _flag(condition, source):
    return condition.astype(float).where(source.notna())


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLS if column not in df.columns]
    if missing:
        raise ValueError(f"engineer_features: missing required columns {missing}")

    df = df.copy()
    age, hours, rating = df["AccountAge"], df["ViewingHoursPerWeek"], df["UserRating"]
    tickets, charges, total = df["SupportTicketsPerMonth"], df["MonthlyCharges"], df["TotalCharges"]
    downloads, duration = df["ContentDownloadsPerMonth"], df["AverageViewingDuration"]

    df["ChargesPerMonthAccount"] = _safe_div(total, age + 1)
    df["CostPerViewingHour"] = _safe_div(charges, hours + 1)
    df["CostPerRating"] = _safe_div(charges, rating + 0.1)
    df["ValueRatio"] = _safe_div(total, hours * age + 1)
    df["ValueIndex"] = _safe_div(hours * rating, charges + 1)
    df["ActivityIntensity"] = hours * duration
    df["EngagementScore"] = _safe_div(hours * rating, tickets + 1)
    df["DownloadsPerViewingHour"] = _safe_div(downloads, hours + 1)
    df["SupportTicketsPerMonthAge"] = _safe_div(tickets, age + 1)
    df["SupportTicketsPerViewingHour"] = _safe_div(tickets, hours + 1)
    df["DownloadsPerSupportTicket"] = _safe_div(downloads, tickets + 1)
    df["SupportRisk"] = tickets * (charges / 10)
    df["UnproductiveViewing"] = duration * (tickets + 1)
    df["IsNewUser"] = _flag(age <= 12, age)
    df["IsLoyalUser"] = _flag(age >= 48, age)

    return df
