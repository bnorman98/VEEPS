from common.preprocess import *
import pandas as pd
import numpy as np


from sklearn.preprocessing import MinMaxScaler

from imblearn.combine import SMOTEENN

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--full", help="full.csv")
parser.add_argument("--edge", help="e.csv")  # edge and flow
parser.add_argument("--leader", help="leader.csv")
parser.add_argument("--network", help="net.xml")
parser.add_argument("-o", "--output", help="net.xml")

args = parser.parse_args()


ROLLING_WINDOW = 300


DF_FULL_PATH = args.full

DF_FLOW_PATH = args.edge

DF_LEADER_PATH = args.leader

DF_EDGE_DATA_PATH = args.edge

NETWORK_XML_PATH = args.network

SAVING_PATH = args.output


df_full = pd.read_csv(
    DF_FULL_PATH,
    sep=";",
)

df_flow = read_edge_flow_csv(DF_FLOW_PATH)
df_vehicle, df_lane = preprocess_vehicle_df(df_full)
df_leader = read_leader_csv(path=DF_LEADER_PATH)

df_vehicle = df_vehicle[df_vehicle.vehicle_id.str.contains("veh")]

df_leader = pd.merge(
    df_leader,
    df_vehicle.rename(columns={"vehicle_id": "vehicle_leaderID"}),
    on=["data_timestep", "vehicle_leaderID"],
    how="left",
)[
    [
        "data_timestep",
        "vehicle_id",
        "vehicle_leaderGap",
        "vehicle_leaderID",
        "vehicle_leaderSpeed",
        "lane_id_x",
        "lane_id_y",
    ]
].rename(
    columns={"lane_id_x": "lane_id"}
)

df_leader = df_leader[df_leader["lane_id"] == df_leader["lane_id_y"]]


df_vehicle = pd.merge(
    df_vehicle, df_leader, on=["data_timestep", "vehicle_id", "lane_id"], how="inner"
)[
    [
        "data_timestep",
        "vehicle_id",
        "lane_id",
        "vehicle_speed",
        "vehicle_type",
        "vehicle_pos",
        "edge_id",
        "lane_maxspeed",
        "vehicle_leaderID",
        "vehicle_leaderGap",
        "vehicle_leaderSpeed",
    ]
]


df_vehicle = df_vehicle[~df_vehicle["vehicle_leaderID"].isna()]

df_vehicle = df_vehicle[
    (df_vehicle["data_timestep"] < df_vehicle["data_timestep"].max() * 0.8)
    & (df_vehicle["data_timestep"] > df_vehicle["data_timestep"].max() * 0.2)
]

df_vehicle

df_vehicle["optimal_following_distance"] = optimal_following_distance(
    df_vehicle["lane_maxspeed"], df_vehicle["lane_maxspeed"]
)
df_vehicle["following_distance_ratio"] = (
    df_vehicle["vehicle_leaderGap"] / df_vehicle["optimal_following_distance"]
)

df_vehicle["congested_pair"] = (df_vehicle["following_distance_ratio"] < 0.8).astype(
    int
)

df_vehicle["vehicle_speed"] = df_vehicle["vehicle_speed"] / df_vehicle["lane_maxspeed"]
df_vehicle["vehicle_leaderSpeed"] = (
    df_vehicle["vehicle_leaderSpeed"] / df_vehicle["lane_maxspeed"]
)


df_agg = (
    df_vehicle.groupby(by=["edge_id", "data_timestep"])
    .agg(
        vehicle_speed_mean=("vehicle_speed", "mean"),
        FDR_mean=("following_distance_ratio", "mean"),
        FDR_median=("following_distance_ratio", "median"),
        FDR_std=("following_distance_ratio", "std"),
        FDR_skew=("following_distance_ratio", "skew"),
        vehicle_nr=("vehicle_id", "count"),
        lane_maxspeed=("lane_maxspeed", "max"),
    )
    .reset_index()
)

df_agg = pd.merge(
    df_agg,
    df_lane[["data_timestep", "edge_id", "lane_occupancy"]],
    on=["data_timestep", "edge_id"],
    how="left",
)

df_agg["interval_begin"] = np.floor(df_agg["data_timestep"] / 60) * 60


df_agg["FDR_median_rolling_std"] = df_agg.groupby("edge_id")["FDR_median"].transform(
    lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).std()
)
df_agg["FDR_median_rolling_median"] = df_agg.groupby("edge_id")["FDR_mean"].transform(
    lambda x: x.ewm(com=0.5, min_periods=ROLLING_WINDOW).mean()
)
df_agg["FDR_skew_rolling_std"] = df_agg.groupby("edge_id")["FDR_skew"].transform(
    lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).std()
)

df_agg["SPI"] = df_agg["vehicle_speed_mean"] / df_agg["lane_maxspeed"]

df_edge_data = pd.read_csv(
    DF_EDGE_DATA_PATH,
    sep=";",
)[["interval_begin", "interval_end", "edge_entered", "edge_left", "edge_id"]]

df_edge_data["leaving_flow"] = df_edge_data["edge_left"] / abs(
    df_edge_data["interval_begin"] - df_edge_data["interval_end"]
)
df_edge_data["entering_flow"] = df_edge_data["edge_entered"] / abs(
    df_edge_data["interval_begin"] - df_edge_data["interval_end"]
)

df_edge_data["edge_id"] = df_edge_data["edge_id"].astype(str)
df_agg = pd.merge(
    df_agg,
    df_edge_data[["interval_begin", "edge_id", "leaving_flow", "entering_flow"]],
    how="left",
    on=["edge_id", "interval_begin"],
)


EA_in, EA_out, edge_id_dict = get_edge_adj_mtx(NETWORK_XML_PATH)
edge_id_dict_inverted = dict((v, k) for k, v in edge_id_dict.items())
df_agg["edge_id"] = df_agg["edge_id"].apply(lambda x: edge_id_dict[x])

df_agg["FDR_mean_incoming"] = df_agg["FDR_mean"] * df_agg["entering_flow"]
df_agg["FDR_mean_outgoing"] = df_agg["FDR_mean"] * df_agg["leaving_flow"] * -1


def create_adjacent_feature(df, feature_name, EA):

    adjacency_list = []
    for edge_id in range(EA.shape[0]):
        adjacent_lanes = np.where(EA[edge_id] == 1)[0]
        for adj_lane in adjacent_lanes:
            adjacency_list.append((edge_id, adj_lane))
    adj_df = pd.DataFrame(adjacency_list, columns=["edge_id", "adjacent_lane_id"])
    df_neigh = df[["edge_id", "data_timestep", feature_name]]
    expanded_df = adj_df.merge(df_neigh, left_on="adjacent_lane_id", right_on="edge_id")
    expanded_df = (
        expanded_df.drop(columns=["adjacent_lane_id", "edge_id_y"])
        .rename({"edge_id_x": "edge_id"}, axis=1)
        .groupby(["edge_id", "data_timestep"])[feature_name]
        .mean()
    )
    return (
        pd.merge(df_agg, expanded_df, how="left", on=["edge_id", "data_timestep"])
        .drop(columns=[feature_name + "_x"])
        .rename({feature_name + "_y": feature_name})
    )


df_agg = create_adjacent_feature(df_agg, "FDR_mean_incoming", EA_in)
df_agg = create_adjacent_feature(df_agg, "FDR_mean_outgoing", EA_in)

df_agg["isCongested"] = df_agg["lane_occupancy"] > 0.3


df_agg.to_csv(SAVING_PATH)

print(df_agg.isCongested.value_counts())
