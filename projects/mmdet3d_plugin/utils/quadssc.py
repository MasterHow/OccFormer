import numpy as np

# quadssc_class_frequencies = np.array(       # new static mapped
#     [
#         # 14943763,   # unlabel
#         663026077,  # empty
#         1779043,    # car
#         1123370,    # person
#         62837236,   # road
#         7986225,    # building
#         17658546,   # vegetation
#         10102778,   # terrain
#     ]
# )
# unlabeled: 8168482
# bus: 194008
# car: 1779043
# terrain: 10102778
# traffic sign: 83353
# vegetation: 17658546
# road: 62837236
# building: 7986225
# sidewalk: 3569596
# bicycle: 555677
# pole: 239088
# person: 1123370
# fence: 2131218
# trunk: 93682
# parking: 2341

quadssc_class_frequencies = np.array(       # new finally mapped
    [
        713501325,  # empty
        31872,    # car
        641280,    # person
        29380702,   # road
        4091494,    # building
        8309282,   # vegetation
        4519974,   # terrain
    ]
)
# unlabeled: 15491626
# empty: 713501325
# bus: 113323
# terrain: 4519974
# traffic sign: 40583
# vegetation: 8309282
# road: 29380702
# building: 4091494
# sidewalk: 1903075
# bicycle: 281848
# pole: 135337
# person: 641280
# fence: 1054251
# car: 31872
# trunk: 53250
# parking: 1498


kitti_class_names = [
    "empty",
    "car",
    "person",
    "road",
    "building",
    "vegetation",
    "terrain",
]