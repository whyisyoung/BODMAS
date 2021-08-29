
config = {
    'gbdt_params': {
        "boosting": "gbdt",
        "objective": "binary",
        "num_iterations": 1000,
        "learning_rate": 0.05,
        "num_leaves": 2048,
        "max_depth": 15,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.5,
        "verbosity": -1 # 1 means INFO, > 1 means DEBUG, 0 means Error(WARNING), <0 means Fatal
    },
    # WARNING: change the following lines to your location
    'sophos_model_folder': '/home/datashare/sophos/baselines/checkpoints/lightGBM/',
    'sophos_features_folder': '/home/datashare/sophos/lightGBM-features/'
}
