for category in "Toys_and_Games" "Sports_and_Outdoors" "CDs_and_Vinyl" "Video_Games" "Books"
do
    train_file=$(ls -f ./data/Amazon/train/${category}*11.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    test_file=$(ls -f ./data/Amazon/test/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)
    echo ${train_file} ${test_fie} ${info_file} ${eval_file}
    python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 \
        --base_model YOUR_BASE_MODEL_PATH \
        --train_file ${train_file} \
        --eval_file ${eval_file} \
        --output_dir ./output_dir/${category} \
        --category ${category} 
    cp YOUR_BASE_MODEL_PATH*token* ./output_dir/${category}/
done