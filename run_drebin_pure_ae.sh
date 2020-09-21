###### use family 7 as the unseen family #######
mkdir -p logs/pure_ae/drebin_new_7/ &&                      \
nohup python -u main.py 	                                \
            --data drebin_new_7                             \
            --newfamily-label 7                             \
            -c mlp                                          \
            --mlp-retrain 0                                 \
            --mlp-hidden 100-30                             \
            --mlp-lr 0.001                                  \
            --mlp-batch-size 32                             \
            --mlp-epochs 50                                 \
            --mlp-dropout 0.2                               \
            --cae-hidden 512-128-32                         \
            --cae-lr 0.0001                                 \
            --cae-batch-size 64                             \
            --cae-lambda-1 1e-1                             \
            --similar-ratio 0.25                            \
            --margin 10.0                                   \
            --display-interval 10                           \
            --mad-threshold 0                               \
            --pure-ae 1                                     \
            > logs/pure_ae/drebin_new_7/mlp_$(date "+%m.%d-%H.%M.%S").log &

####### use family 6 as the unseen family #######
mkdir -p logs/pure_ae/drebin_new_6/ &&                      \
nohup python -u main.py 	                                \
            --data drebin_new_6                             \
            --newfamily-label 6                             \
            -c mlp                                          \
            --mlp-retrain 0                                 \
            --mlp-hidden 100-30                             \
            --mlp-lr 0.001                                  \
            --mlp-batch-size 32                             \
            --mlp-epochs 50                                 \
            --mlp-dropout 0.2                               \
            --cae-hidden 512-128-32                         \
            --cae-lr 0.0001                                 \
            --cae-batch-size 64                             \
            --cae-lambda-1 1e-1                             \
            --similar-ratio 0.25                            \
            --margin 10.0                                   \
            --display-interval 10                           \
            --mad-threshold 0                               \
            --pure-ae 1                                     \
            > logs/pure_ae/drebin_new_6/mlp_$(date "+%m.%d-%H.%M.%S").log &

####### use family 5 as the unseen family #######
mkdir -p logs/pure_ae/drebin_new_5/ &&                      \
nohup python -u main.py 	                                \
            --data drebin_new_5                             \
            --newfamily-label 5                             \
            -c mlp                                          \
            --mlp-retrain 0                                 \
            --mlp-hidden 100-30                             \
            --mlp-lr 0.001                                  \
            --mlp-batch-size 32                             \
            --mlp-epochs 50                                 \
            --mlp-dropout 0.2                               \
            --cae-hidden 512-128-32                         \
            --cae-lr 0.0001                                 \
            --cae-batch-size 64                             \
            --cae-lambda-1 1e-1                             \
            --similar-ratio 0.25                            \
            --margin 10.0                                   \
            --display-interval 10                           \
            --mad-threshold 0                               \
            --pure-ae 1                                     \
            > logs/pure_ae/drebin_new_5/mlp_$(date "+%m.%d-%H.%M.%S").log &


###### use family 4 as the unseen family #######
mkdir -p logs/pure_ae/drebin_new_4/ &&                      \
nohup python -u main.py 	                                \
            --data drebin_new_4                             \
            --newfamily-label 4                             \
            -c mlp                                          \
            --mlp-retrain 0                                 \
            --mlp-hidden 100-30                             \
            --mlp-lr 0.001                                  \
            --mlp-batch-size 32                             \
            --mlp-epochs 50                                 \
            --mlp-dropout 0.2                               \
            --cae-hidden 512-128-32                         \
            --cae-lr 0.0001                                 \
            --cae-batch-size 64                             \
            --cae-lambda-1 1e-1                             \
            --similar-ratio 0.25                            \
            --margin 10.0                                   \
            --display-interval 10                           \
            --mad-threshold 0                               \
            --pure-ae 1                                     \
            > logs/pure_ae/drebin_new_4/mlp_$(date "+%m.%d-%H.%M.%S").log &

####### use family 3 as the unseen family #######
mkdir -p logs/pure_ae/drebin_new_3/ &&                      \
nohup python -u main.py 	                                \
            --data drebin_new_3                             \
            --newfamily-label 3                             \
            -c mlp                                          \
            --mlp-retrain 0                                 \
            --mlp-hidden 100-30                             \
            --mlp-lr 0.001                                  \
            --mlp-batch-size 32                             \
            --mlp-epochs 50                                 \
            --mlp-dropout 0.2                               \
            --cae-hidden 512-128-32                         \
            --cae-lr 0.0001                                 \
            --cae-batch-size 64                             \
            --cae-lambda-1 1e-1                             \
            --similar-ratio 0.25                            \
            --margin 10.0                                   \
            --display-interval 10                           \
            --mad-threshold 0                               \
            --pure-ae 1                                     \
            > logs/pure_ae/drebin_new_3/mlp_$(date "+%m.%d-%H.%M.%S").log &

####### use family 2 as the unseen family #######
mkdir -p logs/pure_ae/drebin_new_2/ &&                      \
nohup python -u main.py 	                                \
            --data drebin_new_2                             \
            --newfamily-label 2                             \
            -c mlp                                          \
            --mlp-retrain 0                                 \
            --mlp-hidden 100-30                             \
            --mlp-lr 0.001                                  \
            --mlp-batch-size 32                             \
            --mlp-epochs 50                                 \
            --mlp-dropout 0.2                               \
            --cae-hidden 512-128-32                         \
            --cae-lr 0.0001                                 \
            --cae-batch-size 64                             \
            --cae-lambda-1 1e-1                             \
            --similar-ratio 0.25                            \
            --margin 10.0                                   \
            --display-interval 10                           \
            --mad-threshold 0                               \
            --pure-ae 1                                     \
            > logs/pure_ae/drebin_new_2/mlp_$(date "+%m.%d-%H.%M.%S").log &

####### use family 1 as the unseen family #######
mkdir -p logs/pure_ae/drebin_new_1/ &&                      \
nohup python -u main.py 	                                \
            --data drebin_new_1                             \
            --newfamily-label 1                             \
            -c mlp                                          \
            --mlp-retrain 0                                 \
            --mlp-hidden 100-30                             \
            --mlp-lr 0.001                                  \
            --mlp-batch-size 32                             \
            --mlp-epochs 50                                 \
            --mlp-dropout 0.2                               \
            --cae-hidden 512-128-32                         \
            --cae-lr 0.0001                                 \
            --cae-batch-size 64                             \
            --cae-lambda-1 1e-1                             \
            --similar-ratio 0.25                            \
            --margin 10.0                                   \
            --display-interval 10                           \
            --mad-threshold 0                               \
            --pure-ae 1                                     \
            > logs/pure_ae/drebin_new_1/mlp_$(date "+%m.%d-%H.%M.%S").log &

####### use family 0 as the unseen family #######
mkdir -p logs/pure_ae/drebin_new_0/ &&                      \
nohup python -u main.py 	                                \
            --data drebin_new_0                             \
            --newfamily-label 0                             \
            -c mlp                                          \
            --mlp-retrain 0                                 \
            --mlp-hidden 100-30                             \
            --mlp-lr 0.001                                  \
            --mlp-batch-size 32                             \
            --mlp-epochs 50                                 \
            --mlp-dropout 0.2                               \
            --cae-hidden 512-128-32                         \
            --cae-lr 0.0001                                 \
            --cae-batch-size 64                             \
            --cae-lambda-1 1e-1                             \
            --similar-ratio 0.25                            \
            --margin 10.0                                   \
            --display-interval 10                           \
            --mad-threshold 0                               \
            --pure-ae 1                                     \
            > logs/pure_ae/drebin_new_0/mlp_$(date "+%m.%d-%H.%M.%S").log &
