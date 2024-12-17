# VEEPS - V2X-Based Traffic Congestion Prediction System
To get more information contact corresponding author: bnorman@hit.bme.hu

This repository contains the source code for the proposed system VEEPS.
The goal of VEEPS is to forecast traffic flow utilizing vehicular sensor information and V2X.
In this repository, we share the source code of VEEPS and an example data set.
The generation of the data set is shown in manuscript.
The data set is exchangable.

![alt text](https://github.com/bnorman98/VEEPS/blob/main/VEEPS_Arch.jpg?raw=true)


VEEPS uses the following distance ratio which is a novel metric to monitor the actual following distance compared to the required following distance.



To use that, run 

```
    python data_processer.py \
        --full <path to full output> \
        --edge <path to edge output> \
        --leader <path to additional edgeData file output> \
        --network <path to network file> \
        --output <path>

```
data_processer.py helps to process SUMO data into the required format.
This data_processer module is published to let researchers prepare SUMO data to the required format.
The exact simulation parameters are described in the article.
The simulation environment (part of Buda, Hungary) is also published. 
(Without the traffic data due to the size.)
