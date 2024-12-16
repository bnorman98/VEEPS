import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import os
import sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import sumolib


def stopping_distance(speed, decel, efficiency=1, reaction_time=2):
    # return speed*2
    return (speed**2 * (1 / efficiency)) / (2 * decel) + reaction_time * speed


def optimal_following_distance(vehicle_speeds, leader_speeds, eps=2):
    follower_stop_distances = stopping_distance(
        vehicle_speeds, decel=3, reaction_time=1.5
    )
    leader_stop_distances = stopping_distance(leader_speeds, decel=3, reaction_time=0)

    optimal_distances = np.maximum(
        follower_stop_distances - leader_stop_distances + eps, eps
    )

    return optimal_distances


def read_edge_flow_csv(path):
    df_flow = pd.read_csv(
        path,
        sep=";",
    )[["interval_begin", "interval_end", "edge_entered", "edge_left", "edge_id"]]
    df_flow["leaving_flow"] = df_flow["edge_left"] / abs(
        df_flow["interval_begin"] - df_flow["interval_end"]
    )
    df_flow["entering_flow"] = df_flow["edge_entered"] / abs(
        df_flow["interval_begin"] - df_flow["interval_end"]
    )
    df_flow["edge_id"] = df_flow["edge_id"].astype(str)

    return df_flow[["interval_begin", "edge_id", "leaving_flow", "entering_flow"]]


def preprocess_vehicle_df(df_full):
    df_vehicle = df_full.loc[~df_full["vehicle_id"].isna()][
        [
            "data_timestep",
            "vehicle_id",
            "vehicle_lane",
            "edge_id",
            "vehicle_speed",
            "vehicle_type",
            "vehicle_pos",
        ]
    ].rename({"vehicle_lane": "lane_id"}, axis=1)

    df_lane = df_full.loc[~df_full["lane_id"].isna()][
        [
            "data_timestep",
            "lane_id",
            "edge_id",
            "lane_occupancy",
            "lane_meanspeed",
            "lane_maxspeed",
        ]
    ]

    df_vehicle = (
        pd.merge(
            df_vehicle,
            df_lane[["edge_id", "data_timestep", "lane_maxspeed", "lane_id"]],
            how="left",
            on=["data_timestep", "lane_id"],
        )
        .drop(columns=["edge_id_x"])
        .rename(columns={"edge_id_y": "edge_id"})
    )

    # remove junctions
    df_vehicle = df_vehicle[~df_vehicle["edge_id"].str.contains(":")]

    # df_vehicle["lane_meanspeedr"] = (
    #    df_vehicle["lane_meanspeed"] / df_vehicle["lane_maxspeed"]
    # )
    # df_vehicle["vehicle_speedr"] = (
    #    df_vehicle["vehicle_speed"] / df_vehicle["lane_maxspeed"]
    # )

    scaler = MinMaxScaler(feature_range=(0, 1))
    df_vehicle["vehicle_pos"] = scaler.fit_transform(df_vehicle[["vehicle_pos"]])
    df_vehicle[(df_vehicle["vehicle_pos"] > 0.15) & (df_vehicle["vehicle_pos"] < 0.85)]
    return df_vehicle, df_lane


def read_leader_csv(path):
    df_leader = pd.read_csv(
        path,
        sep=";",
    )

    df_leader = df_leader.loc[~df_leader["vehicle_leaderID"].isna()][
        [
            "timestep_time",
            "vehicle_id",
            "vehicle_leaderGap",
            "vehicle_leaderID",
            "vehicle_leaderSpeed",
            "vehicle_lane",
        ]
    ].rename({"timestep_time": "data_timestep", "vehicle_lane": "lane_id"}, axis=1)

    return df_leader


def get_edge_adj_mtx(net_file_path):
    net = sumolib.net.readNet(net_file_path)
    edge_list = net.getEdges()
    edge_id_dict = {
        edge_id: i for i, edge_id in enumerate([edge.getID() for edge in edge_list])
    }
    EA_in = np.zeros((len(edge_list), len(edge_list)))
    EA_out = np.zeros((len(edge_list), len(edge_list)))

    # print(edge_id_dict)

    for edge in edge_list:
        source_node = edge.getFromNode()
        target_node = edge.getToNode()
        incoming_edge_id_list = [str(e.getID()) for e in edge.getIncoming()]
        outgoing_edge_id_list = [str(e.getID()) for e in edge.getOutgoing()]

        # print(edge.getID(), incoming_edge_id_list, outgoing_edge_id_list)
        for incoming_edge_id in incoming_edge_id_list:
            if incoming_edge_id != edge.getID():
                EA_in[edge_id_dict[edge.getID()], edge_id_dict[incoming_edge_id]] = 1
        for outgoing_edge_id in outgoing_edge_id_list:
            if outgoing_edge_id != edge.getID():
                EA_out[edge_id_dict[edge.getID()], edge_id_dict[outgoing_edge_id]] = -1
        # print()
    # TODO: ADD COMMENT TO SELECTION
    # print(
    #    edge.getID(),
    #   source_node.getID(),
    ##    "\n",
    #    incoming_edge_id_list,
    #    "\n",
    #    target_node.getID(),
    #    outgoing_edge_id_list,
    # )

    return EA_in, EA_out, edge_id_dict
