####### use Infiltration as the unseen family #######
mkdir -p logs/IDS_new_Infilteration/ &&                     \
nohup python -u main.py 	                                \
            --data IDS_new_Infilteration                    \
            --newfamily-label 3                             \
            -c mlp                                          \
            --mlp-retrain 0                                 \
            --mlp-hidden 30                                 \
            --mlp-lr 0.001                                  \
            --mlp-batch-size 256                            \
            --mlp-epochs 30                                 \
            --mlp-dropout 0.2                               \
            --cae-hidden 64-32-16                           \
            --cae-lr 0.0001                                 \
            --cae-batch-size 512                            \
            --cae-lambda-1 1e-1                             \
            --similar-ratio 0.25                            \
            --margin 10.0                                   \
            --display-interval 1                            \
            --mad-threshold 0                               \
            --pure-ae 1                                     \
            > logs/IDS_new_Infilteration/mlp_$(date "+%m.%d-%H.%M.%S").log &

###### use DoS-Hulk as the unseen family #######
mkdir -p logs/IDS_new_Hulk/ &&                              \
nohup python -u main.py 	                                \
            --data IDS_new_Hulk                             \
            --newfamily-label 2                             \
            -c mlp                                          \
            --mlp-retrain 0                                 \
            --mlp-hidden 30                                 \
            --mlp-lr 0.001                                  \
            --mlp-batch-size 256                            \
            --mlp-epochs 30                                 \
            --mlp-dropout 0.2                               \
            --cae-hidden 64-32-16                           \
            --cae-lr 0.0001                                 \
            --cae-batch-size 512                            \
            --cae-lambda-1 1e-1                             \
            --similar-ratio 0.25                            \
            --margin 10.0                                   \
            --display-interval 1                            \
            --mad-threshold 0                               \
            --pure-ae 1                                     \
            > logs/IDS_new_Hulk/mlp_$(date "+%m.%d-%H.%M.%S").log &

####### use SSH-Bruteforce as the unseen family #######
mkdir -p logs/IDS_new_SSH/ &&                               \
nohup python -u main.py 	                                \
            --data IDS_new_SSH                              \
            --newfamily-label 1                             \
            -c mlp                                          \
            --mlp-retrain 0                                 \
            --mlp-hidden 30                                 \
            --mlp-lr 0.001                                  \
            --mlp-batch-size 256                            \
            --mlp-epochs 30                                 \
            --mlp-dropout 0.2                               \
            --cae-hidden 64-32-16                           \
            --cae-lr 0.0001                                 \
            --cae-batch-size 512                            \
            --cae-lambda-1 1e-1                             \
            --similar-ratio 0.25                            \
            --margin 10.0                                   \
            --display-interval 1                            \
            --mad-threshold 0                               \
            --pure-ae 1                                     \
            > logs/IDS_new_SSH/mlp_$(date "+%m.%d-%H.%M.%S").log &
