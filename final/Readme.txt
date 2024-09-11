# 111061702 常安彥
# Pattern Recognition Final Project

# Dataset (all 16k audios)
    # Save AudioSet audios in ./AST_MLP/AudioSet/data/audios/
    # Save Respiratory Sound audios in ./AST_MLP/Resp/data/audios/

# Run pretrain steps
    # cd ./AST_MLP/AudioSet/
    # python prep_audioset.py
    # bash run_pretrain.sh
    # move Pretrain.pth under ./AST_MLP/models

# Run Transformer training steps
    # cd ./AST_MLP/Resp/
    # python prep_resp.py 
    # bash run_resp_patch.sh (can change the parameters)
    # The model will be placed under esc folder

# Get Embedding Features
    # cd ./AST_MLP/
    # python getEmbedding.py (change the path of model to the previous model)

# Train with SVM and Voting classifier
    # cd ./ML
    # python svm.py
    # python ensemble.py
    # The result pictures will be saved