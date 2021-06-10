from datetime import datetime
import pandas as pd

time_measures = {}
build_time_measures = {}
commit_time_measures = {}


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


def tik(name, build=None, commit=None):
    if build is not None and commit is not None:
        print("Only one of build or commit args should be provided!")
        return
    global time_measures
    global build_time_measures
    global commit_time_measures
    if build is not None:
        tm = build_time_measures.setdefault(build, {})
        internal_tik(name, tm)
    elif commit is not None:
        tm = commit_time_measures.setdefault(commit, {})
        internal_tik(name, tm)
    else:
        internal_tik(name, time_measures)


def tok(name, build=None, commit=None):
    if build is not None and commit is not None:
        print("Only one of build or commit args should be provided!")
        return
    global time_measures
    global build_time_measures
    global commit_time_measures
    if build is not None:
        tm = build_time_measures.setdefault(build, {})
        internal_tok(name, tm)
    elif commit is not None:
        tm = commit_time_measures.setdefault(commit, {})
        internal_tok(name, tm)
    else:
        internal_tok(name, time_measures)


def tik_list(names, build=None, commit=None):
    for name in names:
        tik(name, build, commit)


def tok_list(names, build=None, commit=None):
    for name in names:
        tok(name, build, commit)


def create_time_measures_df():
    global time_measures
    keys = list(time_measures.keys())
    for key in keys:
        if key.startswith("$TMP"):
            time_measures.pop(key)
    process_name = list(time_measures.keys())
    duration = list(time_measures.values())
    return pd.DataFrame({"ProcessName": process_name, "Duration": duration})


def create_build_time_measures_df(commit_build_map):
    global build_time_measures
    global commit_time_measures
    for commit, pnames in commit_time_measures.items():
        for pname, d in pnames.items():
            if pname.startswith("$TMP"):
                continue
            if commit in commit_build_map:
                build_time = build_time_measures[commit_build_map[commit]]
                if pname not in build_time:
                    build_time[pname] = 0.0
                build_time[pname] = build_time[pname] + d

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
    return pd.DataFrame(
        {"Build": builds, "ProcessName": process_names, "Duration": durations}
    )


def create_feature_group_time_df(time_df, build_time_df, valid_builds):
    feature_groups = [
        "TES_COM",
        "TES_PRO",
        "TES_CHN",
        "REC",
        "COV",
        "COD_COV_COM",
        "COD_COV_PRO",
        "COD_COV_CHN",
        "DET_COV",
    ]
    valid_build_time_df = build_time_df[build_time_df["Build"].isin(valid_builds)]
    build_names = valid_build_time_df.groupby("Build")["ProcessName"].apply(list)
    build_durations = valid_build_time_df.groupby("Build")["Duration"].apply(list)
    builds = []
    features = []
    preprocessings = []
    measurements = []
    total = []
    for build, durations in build_durations.iteritems():
        for fg in feature_groups:
            preprocessing = 0.0
            fgp_df = time_df[time_df["ProcessName"] == f"{fg}_P"]
            if len(fgp_df) > 0:
                preprocessing += fgp_df["Duration"].values.tolist()[0]
            try:
                i = build_names[build].index(f"{fg}_P")
                preprocessing += durations[i]
            except:
                pass

            measurement = 0.0
            try:
                i = build_names[build].index(f"{fg}_M")
                measurement += durations[i]
            except:
                pass
            builds.append(build)
            features.append(fg)
            preprocessings.append(preprocessing)
            measurements.append(measurement)
            total.append(preprocessing + measurement)

    return pd.DataFrame(
        {
            "Build": builds,
            "FeatureGroup": features,
            "PreprocessingTime": preprocessings,
            "MeasurementTime": measurements,
            "TotalTime": total,
        }
    )


def save_time_measures(output_path):
    time_df = create_time_measures_df()
    time_df.to_csv(f"{output_path}/time_measure.csv", index=False)

    valid_builds = (
        pd.read_csv(f"{output_path}/dataset.csv", usecols=["Build"])["Build"]
        .unique()
        .tolist()
    )
    builds_df = pd.read_csv(
        f"{output_path}/full_builds.csv", usecols=["id", "commit_hash"], sep="\t"
    )
    commit_build_map = {}
    for _, row in builds_df.iterrows():
        if row["id"] in valid_builds:
            commit_build_map[row["commit_hash"]] = row["id"]
    build_time_df = create_build_time_measures_df(commit_build_map)
    feature_group_time_df = create_feature_group_time_df(
        time_df, build_time_df, valid_builds
    )
    feature_group_time_df.to_csv(f"{output_path}/feature_group_time.csv", index=False)
