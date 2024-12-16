import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from pandas.plotting import parallel_coordinates

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


plt.style.use("ggplot")


def stopping_distance(speed, decel, efficiency=0.9, reaction_time=2):
    # return speed*2
    return (speed**2 * (1 / efficiency)) / (2 * decel) + reaction_time * speed


def optimal_following_distance(vehicle_speeds, leader_speeds, eps=2):
    follower_stop_distances = stopping_distance(
        vehicle_speeds, decel=5, reaction_time=3
    )
    leader_stop_distances = stopping_distance(leader_speeds, decel=5, reaction_time=0)

    optimal_distances = np.maximum(
        follower_stop_distances - leader_stop_distances + eps, eps
    )

    return optimal_distances


def data_preprocessing(df_full, df_leader):

    df_vehicle = df_full.loc[~df_full["vehicle_id"].isna()][
        [
            "data_timestep",
            "vehicle_id",
            "vehicle_lane",
            "vehicle_speed",
            "vehicle_type",
            "vehicle_pos",
        ]
    ]
    df_vehicle = df_vehicle.rename({"vehicle_lane": "lane_id"}, axis=1)
    df_lane = df_full.loc[~df_full["lane_id"].isna()][
        [
            "data_timestep",
            "lane_id",
            "lane_occupancy",
            "lane_meanspeed",
            "lane_maxspeed",
        ]
    ]

    df_vehicle = pd.merge(
        df_vehicle, df_lane, how="left", on=["data_timestep", "lane_id"]
    )
    df_vehicle["lane_meanspeedr"] = (
        df_vehicle["lane_meanspeed"] / df_vehicle["lane_maxspeed"]
    )
    df_vehicle["vehicle_speedr"] = (
        df_vehicle["vehicle_speed"] / df_vehicle["lane_maxspeed"]
    )

    scaler = MinMaxScaler(feature_range=(0, 1))
    df_vehicle["vehicle_pos"] = scaler.fit_transform(df_vehicle[["vehicle_pos"]])

    df_leader = df_leader.loc[~df_leader["vehicle_leaderID"].isna()][
        [
            "timestep_time",
            "vehicle_id",
            "vehicle_leaderGap",
            "vehicle_leaderID",
            "vehicle_leaderSpeed",
        ]
    ]
    df_leader = df_leader.rename({"timestep_time": "data_timestep"}, axis=1)

    df_leader = pd.merge(
        df_leader,
        df_vehicle.rename(
            columns={
                "vehicle_id": "vehicle_leaderID",
                "lane_id": "lane_id_leader",
                "lane_maxspeed": "lane_maxspeed_leader",
                "vehicle_speedr": "vehicle_speedr_leader",
            }
        )[
            [
                "data_timestep",
                "vehicle_leaderID",
                "lane_id_leader",
                "lane_maxspeed_leader",
                "vehicle_speedr_leader",
            ]
        ],
        how="left",
        on=["data_timestep", "vehicle_leaderID"],
    )

    df_vehicle = pd.merge(
        df_vehicle, df_leader, how="left", on=["data_timestep", "vehicle_id"]
    )
    df_vehicle = df_vehicle[~df_vehicle["vehicle_leaderID"].isna()]

    df_vehicle["optimal_following_distance"] = optimal_following_distance(
        df_vehicle["lane_maxspeed"], df_vehicle["lane_maxspeed_leader"]
    )
    df_vehicle["following_distance_ratio"] = (
        df_vehicle["vehicle_leaderGap"] / df_vehicle["optimal_following_distance"]
    )
    df_vehicle["isCongested"] = df_vehicle["lane_occupancy"] > 0.3

    df_vehicle["congested_pair"] = (df_vehicle["following_distance_ratio"] < 1).astype(
        int
    )

    return df_vehicle, df_lane


def aggregate(df_vehicle):
    df_agg = df_vehicle.groupby(by=["data_timestep", "lane_id"]).agg(
        {
            "following_distance_ratio": "mean",
            "lane_meanspeedr": "mean",
            "isCongested": "max",
            "congested_pair": "sum",
        }
    )
    df_agg = df_agg.reset_index().iloc[:, 2:]
    df_agg = df_agg.dropna(how="any")

    scaler = MinMaxScaler(feature_range=(0, 1))
    df_agg[df_agg.columns] = scaler.fit_transform(df_agg[df_agg.columns])
    return df_agg


def get_edge_adj_mtx(net_file_path):
    net = sumolib.net.readNet(net_file_path)
    edge_list = net.getEdges()
    edge_id_dict = {
        edge_id: i for i, edge_id in enumerate([edge.getID() for edge in edge_list])
    }
    EA_in = np.zeros((len(edge_list), len(edge_list)))
    EA_out = np.zeros((len(edge_list), len(edge_list)))

    print(edge_id_dict)

    for edge in edge_list:
        source_node = edge.getFromNode()
        target_node = edge.getToNode()
        incoming_edge_id_list = [str(e.getID()) for e in edge.getIncoming()]
        outgoing_edge_id_list = [str(e.getID()) for e in edge.getOutgoing()]

        print(edge.getID(), incoming_edge_id_list, outgoing_edge_id_list)
        for incoming_edge_id in incoming_edge_id_list:
            if incoming_edge_id != edge.getID():
                EA_in[edge_id_dict[edge.getID()], edge_id_dict[incoming_edge_id]] = 1
        for outgoing_edge_id in outgoing_edge_id_list:
            if outgoing_edge_id != edge.getID():
                EA_out[edge_id_dict[edge.getID()], edge_id_dict[outgoing_edge_id]] = -1
        print()
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


EA_in, EA_out, edge_id_dict = get_edge_adj_mtx()
edge_id_dict_inverted = dict((v, k) for k, v in edge_id_dict.items())
