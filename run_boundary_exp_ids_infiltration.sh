#### use boundary-based method as the explanation #######
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
            --mad-threshold 3.5                             \
            --stage explanation                             \
            --exp-lambda-1 1e-3                             \
            --exp-method approximation_loose                \
            > logs/IDS_new_Infilteration/mlp_$(date "+%m.%d-%H.%M.%S")_exp_approximation_loose.log &
