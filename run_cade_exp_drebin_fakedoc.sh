#### use our distance-based method as the explanation #######
mkdir -p logs/drebin_new_7/ &&                              \
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
            --mad-threshold 3.5                             \
            --stage explanation                             \
            --exp-lambda-1 1e-3                             \
            --exp-method distance_mm1                       \
            > logs/drebin_new_7/mlp_$(date "+%m.%d-%H.%M.%S")_exp_distance_mm1.log &

