############################################################
###### GBDT random
mkdir -p logs/ember_drift_random/  &&                   \
nohup python -u concept_drift_ember.py                  \
                --setting-name ember_drift_random       \
                --classifier gbdt                       \
                --month-interval 1                      \
                --testing-time 2019-10,2020-09          \
                --ember-ratio 1.0                       \
                --seed 5                                \
                --sample-ratio 0.01                     \
                --retrain 0                             \
                --quiet 0                               \
                > logs/ember_drift_random/$(date "+%m.%d-%H.%M.%S")_ember_1.0_gbdt_ratio_0.01_seed5.log &


# mkdir -p logs/ember_drift_random/  &&                         \
# nohup python -u concept_drift_ember.py                        \
#                 --setting-name ember_drift_random             \
#                 --classifier gbdt                       \
#                 --month-interval 1                      \
#                 --testing-time 2019-10,2020-09          \
#                 --ember-ratio 1.0                       \
#                 --seed 10                                \
#                 --sample-ratio 0.01                     \
#                 --retrain 0                             \
#                 --quiet 0                               \
#                 > logs/ember_drift_random/$(date "+%m.%d-%H.%M.%S")_ember_1.0_gbdt_ratio_0.01_seed10.log &


# mkdir -p logs/ember_drift_random/  &&                         \
# nohup python -u concept_drift_ember.py                        \
#                 --setting-name ember_drift_random             \
#                 --classifier gbdt                       \
#                 --month-interval 1                      \
#                 --testing-time 2019-10,2020-09          \
#                 --ember-ratio 1.0                       \
#                 --seed 15                                \
#                 --sample-ratio 0.01                     \
#                 --retrain 0                             \
#                 --quiet 0                               \
#                 > logs/ember_drift_random/$(date "+%m.%d-%H.%M.%S")_ember_1.0_gbdt_ratio_0.01_seed15.log &

# mkdir -p logs/ember_drift_random/  &&                         \
# nohup python -u concept_drift_ember.py                        \
#                 --setting-name ember_drift_random             \
#                 --classifier gbdt                       \
#                 --month-interval 1                      \
#                 --testing-time 2019-10,2020-09          \
#                 --ember-ratio 1.0                       \
#                 --seed 20                                \
#                 --sample-ratio 0.01                     \
#                 --retrain 0                             \
#                 --quiet 0                               \
#                 > logs/ember_drift_random/$(date "+%m.%d-%H.%M.%S")_ember_1.0_gbdt_ratio_0.01_seed20.log &







############################################################
##### GBDT Transcend
# NOTE: due to pairwise comparison, we only randomly selected 30% Ember to get the non-comformity measure
# mkdir -p logs/ember_drift_transcend/  &&                      \
# nohup python -u concept_drift_ember.py                        \
#                 --setting-name ember_drift_transcend          \
#                 --classifier gbdt                       \
#                 --month-interval 1                      \
#                 --testing-time 2019-10,2020-09          \
#                 --ember-ratio 0.3                       \
#                 --sample-ratio 0.05                     \
#                 --retrain 0                             \
#                 --quiet 0                               \
#                 > logs/ember_drift_transcend/$(date "+%m.%d-%H.%M.%S")_ember_0.3_gbdt_ratio_0.05_.log &






############################################################
##### GBDT Probability
# mkdir -p logs/ember_drift_probability/  &&                     \
# nohup python -u concept_drift_ember.py                        \
#                 --setting-name ember_drift_probability         \
#                 --classifier gbdt                       \
#                 --month-interval 1                      \
#                 --testing-time 2019-10,2020-09          \
#                 --ember-ratio 1.0                       \
#                 --sample-ratio 0.01                     \
#                 --retrain 0                             \
#                 --quiet 0                               \
#                 > logs/ember_drift_probability/gbdt_ratio_0.01_ember_1.0_$(date "+%m.%d-%H.%M.%S").log &
