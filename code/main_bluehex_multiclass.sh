################################################################
########   multi-class, classifier: GBDT  ########
# test
mkdir -p logs/bluehex_multiclass/ && \
nohup python -u fabric_multiclass.py bishop run_multiclass.sh gbdt 5 > logs/bluehex_multiclass/gbdt_5_$(date "+%m.%d-%H.%M.%S").log &

# hosts=("beast" "storm" "cyclops" "nightcrawler" "wolverine" "jubilee")
# family=(10 20 40 60 80 100)
# for i in ${!hosts[@]}; do
#     nohup python -u fabric_multiclass.py ${hosts[$i]} run_multiclass.sh gbdt ${family[$i]} > logs/bluehex_multiclass/gbdt_${family[$i]}_$(date "+%m.%d-%H.%M.%S").log &
# done





################################################################
########     multi-class, classifier: RF  ########
# hosts=("angel" "colossus" "bishop" "beast" "jubilee" "cyclops" "nightcrawler")
# family=(5 10 20 40 60 80 100)
# for i in ${!hosts[@]}; do
#     nohup python -u fabric_multiclass.py ${hosts[$i]} run_multiclass.sh rf ${family[$i]} > logs/bluehex_multiclass/rf_${family[$i]}_$(date "+%m.%d-%H.%M.%S").log &
# done





################################################################
########    multi-class, classifier: MLP  ########
# hosts=("angel" "colossus" "bishop" "beast" "wolverine" "cyclops" "nightcrawler")
# family=(5 10 20 40 60 80 100)
# for i in ${!hosts[@]}; do
#     nohup python -u fabric_multiclass.py ${hosts[$i]} run_multiclass.sh mlp ${family[$i]} > logs/bluehex_multiclass/mlp_${family[$i]}_$(date "+%m.%d-%H.%M.%S").log &
# done

