import glob
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV


def load_daily_features(files):
    dfs = []
    for path in files:
        df = pd.read_csv(path, sep=";", dtype={"CO2 ppm": float, "HUMIDITY %": float})
        df["recorded"] = pd.to_datetime(
            df["recorded"],
            errors="coerce"
        )
        df = df.dropna(subset=["recorded", "CO2 ppm", "HUMIDITY %"])
        dfs.append(df)

    # Concatenate all years
    data = pd.concat(dfs, ignore_index=True)

    # Extract just the date
    data["date"] = data["recorded"].dt.date

    # Aggregate per day
    daily = (
        data.groupby("date")
        .agg(
            CO2_max=("CO2 ppm", "max"),
            CO2_mean=("CO2 ppm", "mean"),
            HUM_min=("HUMIDITY %", "min"),
            HUM_mean=("HUMIDITY %", "mean"),
        )
        .reset_index()
    )
    return daily


if __name__ == "__main__":
    files = glob.glob("dataset/*")
    daily = load_daily_features(files)
    iso = IsolationForest(
        contamination=0.01,
        n_estimators=100,
        random_state=42
    )
    iso.fit(daily[["CO2_max", "CO2_mean", "HUM_min", "HUM_mean"]])
    daily["party_day"] = (
        iso.predict(daily[["CO2_max", "CO2_mean", "HUM_min", "HUM_mean"]]) == -1
    )
    print(
        "Predicted party days: ",
        daily.loc[daily["party_day"], "date"].tolist()
    )
