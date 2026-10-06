"""Feature engineering — an exact copy of `engineer_features` from the notebook.

The model was trained on these columns, so this must stay in sync with training.
"""
import pandas as pd

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
    "TotalChargesPerMonth",
    "ValueIndex",
    "SupportTicketsPerViewingHour",
    "UnproductiveViewing",
    "DownloadsPerSupportTicket",
    "IsNewUser",
    "IsLoyalUser",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ChargesPerMonthAccount"] = df["TotalCharges"] / (df["AccountAge"] + 1)
    df["DownloadsPerViewingHour"] = df["ContentDownloadsPerMonth"] / (df["ViewingHoursPerWeek"] + 1)
    df["SupportTicketsPerMonthAge"] = df["SupportTicketsPerMonth"] / (df["AccountAge"] + 1)
    df["EngagementScore"] = (df["ViewingHoursPerWeek"] * df["UserRating"]) / (df["SupportTicketsPerMonth"] + 1)
    df["ValueRatio"] = df["TotalCharges"] / (df["ViewingHoursPerWeek"] * df["AccountAge"] + 1)
    df["ActivityIntensity"] = df["ViewingHoursPerWeek"] * df["AverageViewingDuration"]
    df["CostPerRating"] = df["MonthlyCharges"] / (df["UserRating"] + 0.1)
    df["SupportRisk"] = df["SupportTicketsPerMonth"] * (df["MonthlyCharges"] / 10)
    df["CostPerViewingHour"] = df["MonthlyCharges"] / (df["ViewingHoursPerWeek"] + 1)
    df["TotalChargesPerMonth"] = df["TotalCharges"] / (df["AccountAge"] + 1)
    df["ValueIndex"] = (df["ViewingHoursPerWeek"] * df["UserRating"]) / (df["MonthlyCharges"] + 1)
    df["SupportTicketsPerViewingHour"] = df["SupportTicketsPerMonth"] / (df["ViewingHoursPerWeek"] + 1)
    df["UnproductiveViewing"] = df["AverageViewingDuration"] * (df["SupportTicketsPerMonth"] + 1)
    df["DownloadsPerSupportTicket"] = df["ContentDownloadsPerMonth"] / (df["SupportTicketsPerMonth"] + 1)
    df["IsNewUser"] = (df["AccountAge"] <= 12).astype(int)
    df["IsLoyalUser"] = (df["AccountAge"] >= 48).astype(int)

    return df
