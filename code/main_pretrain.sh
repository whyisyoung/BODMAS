
################################################################
########   binary, classifier: GBDT, train_set: ember  ########
# test
mkdir -p logs/pretrain_ember/
nohup python -u fabric_pretrain.py angel run_pretrain.sh ember gbdt 0 > logs/pretrain_ember/seed_0_gbdt_$(date "+%m.%d-%H.%M.%S").log &

# hosts=("beast" "bishop" "colossus" "cyclops")
# seed=(1 2 3 4)
# for i in ${!hosts[@]}; do
#     nohup python -u fabric_pretrain.py ${hosts[$i]} run_pretrain.sh ember gbdt ${seed[$i]} > logs/pretrain_ember/seed_${seed[$i]}_gbdt_$(date "+%m.%d-%H.%M.%S").log &
# done







################################################################
########   binary, classifier: GBDT, train_set: sophos  ########
# test
# mkdir -p logs/pretrain_sophos/
# nohup python -u fabric_pretrain.py cyclops run_pretrain.sh sophos gbdt 0 > logs/pretrain_sophos/seed_0_gbdt_$(date "+%m.%d-%H.%M.%S").log &

# hosts=("beast" "bishop" "colossus" "cyclops")
# seed=(1 2 3 4)
# hosts=("nightcrawler" "wolverine" "nightcrawler" "cyclops")
# seed=(1 2 3 4)
# for i in ${!hosts[@]}; do
#     nohup python -u fabric_pretrain.py ${hosts[$i]} run_pretrain.sh sophos gbdt ${seed[$i]} > logs/pretrain_sophos/seed_${seed[$i]}_gbdt_$(date "+%m.%d-%H.%M.%S").log &
# done




################################################################
########   binary, classifier: GBDT, train_set: ucsb  ########
# mkdir -p logs/pretrain_ucsb/
# hosts=("wolverine" "jubilee" "nightcrawler" "beast" "bishop")
# seed=(0 1 2 3 4)
# for i in ${!hosts[@]}; do
#     nohup python -u fabric_pretrain.py ${hosts[$i]} run_pretrain.sh ucsb gbdt ${seed[$i]} > logs/pretrain_ucsb/seed_${seed[$i]}_gbdt_$(date "+%m.%d-%H.%M.%S").log &
# done

