from datetime import datetime
import pandas as pd

time_measures = {}


def tik(name):
    global time_measures
    key = name
    if name in time_measures:
        key = f"$TMP{name}"
    time_measures[key] = datetime.now()


def tok(name):
    global time_measures
    key = name
    prev_value = 0.0
    if type(time_measures[name]) is not datetime:
        key = f"$TMP{name}"
        prev_value = time_measures[name]
    time_measures[name] = (
        prev_value + (datetime.now() - time_measures[key]).total_seconds()
    )


def save_time_measures(output_path):
    global time_measures
    keys = list(time_measures.keys())
    for key in keys:
        if key.startswith("$TMP"):
            time_measures.pop(key)
    process_name = list(time_measures.keys())
    duration = list(time_measures.values())
    df = pd.DataFrame({"ProcessName": process_name, "Duration": duration})
    df.to_csv(f"{output_path}/time_measure.csv", index=False)
