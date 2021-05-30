from datetime import datetime
import pandas as pd

time_measures = {}
build_time_measures = {}


def internal_tik(name, tm):
    key = name
    if name in tm:
        key = f"$TMP{name}"
    tm[key] = datetime.now()


def internal_tok(name, tm):
    key = name
    prev_value = 0.0
    if type(tm[name]) is not datetime:
        key = f"$TMP{name}"
        prev_value = tm[name]
    tm[name] = prev_value + (datetime.now() - tm[key]).total_seconds()


def tik(name, build=None):
    global time_measures
    global build_time_measures
    if build is not None:
        tm = build_time_measures.setdefault(build, {})
        internal_tik(name, tm)
    else:
        internal_tik(name, time_measures)


def tok(name, build=None):
    global time_measures
    global build_time_measures
    if build is not None:
        tm = build_time_measures.setdefault(build, {})
        internal_tok(name, tm)
    else:
        internal_tok(name, time_measures)


def tik_list(names, build=None):
    for name in names:
        tik(name, build)


def tok_list(names, build=None):
    for name in names:
        tok(name, build)


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

    global build_time_measures
    builds = []
    process_names = []
    durations = []
    for build, pnames in build_time_measures.items():
        for pname, d in pnames.items():
            if pname.startswith("$TMP"):
                continue
            builds.append(build)
            process_names.append(pname)
            durations.append(d)
    df = pd.DataFrame(
        {"Build": builds, "ProcessName": process_names, "Duration": durations}
    )
    df.to_csv(f"{output_path}/build_time_measure.csv", index=False)
