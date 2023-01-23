for seed in 1 2 3 4 5
do

mkdir "$1_split${seed}"

python train.py --dataset_path ./data/$1_simulation_cpa.h5ad \
--dataset $1 \
--split_key "split${seed}" \
--save_dir "$1_split${seed}" \
--cuda $2 \
#--emb "kg"
done


## example script
# bash cpa.sh jost2020_hvg 5
# bash cpa.sh tian2019_ipsc_hvg 5
# bash cpa.sh replogle2020_hvg 2
# bash cpa.sh replogle_rpe1_gw_filtered_hvg 2
# bash cpa.sh replogle_k562_essential_filtered_hvg 7

## to run CPA+KG, simply add a flag --emb "kg"
