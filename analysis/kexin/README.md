
```
python train.py --dataset Norman2019 \
                 --split single_only \
                 --device cuda:0 \
                 --batch_size 64 \
                 --model GNN_Disentangle_AE \
                 --node_hidden_size 8 \
                 --max_epochs 1 \
                 --model_backend GAT \
                 --gnn_num_layers 4 \
                 --loss_mode l2 \
                 --focal_gamma 2
```