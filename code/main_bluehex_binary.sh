########################################################################
# binary classifier using the first month of Blue Hexagon data
# as training set and test on the rest of the months
# WARNING: it's better NOT to run all the 5 seeds on a single machine, otherwise may lead to Memory Error

mkdir -p logs/bluehex_diversity_no/  &&                     \
seed=(2) # (0 1 2 3 4)
for i in ${!seed[@]}; do
    nohup python -u bluehex_main.py                         \
                    --training-set bluehex                  \
                    --diversity no                          \
                    --setting-name bluehex_diversity_no     \
                    --classifier gbdt                       \
                    --testing-time 2019-10,2020-09          \
                    --retrain 0                             \
                    --seed ${seed[$i]}                      \
                    --quiet 0                               \
                    > logs/bluehex_diversity_no/gbdt_seed_${seed[$i]}_$(date "+%m.%d-%H.%M.%S").log &
done
