### DecodingMatters

This is the raw implementation of our paper **[Decoding Matters: Addressing Amplification Bias and Homogeneity Issue for LLM-based Recommendation](https://arxiv.org/abs/2406.14900)**

### Reproduce
To reproduce our results, you need to conduct the following pipeline.

```bash
# Take the book dataset as an example
# Download the dataset
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Books.json.gz
wget wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Books.json.gz
# Unzip
gunzip Books.json.gz
gunzip meta_Books.json.gz
# Preprocess
python ./code/preprocess.py --category "Books"
# Train
bash run.sh  # You only need to change the category parameter in script
# Inference and Evaluate
bash evaluate.sh
# Decoding Matters Inference (Our Methods) and Evaluate
bash evaluate2.sh # You need to specify your logits file in the script
```

### Results and Model
The results and the parameters of Qwen2-0.5B trained on five Amazon datasets are presented in the following table:


|Dataset|NDCG@10|HR@10|Link|
|----------------|----------------|----------------|----------------|
|CDs_and_Vinyl|0.077|0.109|[link](https://huggingface.co/USTCbaokq/BIGRec_CDs_and_Vinyl_0.5B)|
|Video_Games|0.052|0.085|[link](https://huggingface.co/USTCbaokq/BIGRec_Video_Games_0.5B)|
|Toys_and_Games|0.053|0.096|[link](https://huggingface.co/USTCbaokq/BIGRec_Toys_and_Games_0.5B)|
|Sports_and_Outdoors|0.099|0.120|[link](https://huggingface.co/USTCbaokq/BIGRec_Sports_and_Outdoors_0.5B)|
|Book|0.018|0.027|[link](https://huggingface.co/USTCbaokq/BIGRec_Books_0.5B)|


If you're using this code in your research or applications, please cite our paper using this BibTeX:
```bibtex
@article{bao2024decoding,
  title={Decoding Matters: Addressing Amplification Bias and Homogeneity Issue for LLM-based Recommendation},
  author={Bao, Keqin and Zhang, Jizhi and Zhang, Yang and Huo, Xinyue and Chen, Chong and Feng, Fuli},
  journal={arXiv preprint arXiv:2406.14900},
  year={2024}
}
```
and
```bibtex
@article{bao2023bi,
  title={A bi-step grounding paradigm for large language models in recommendation systems},
  author={Bao, Keqin and Zhang, Jizhi and Wang, Wenjie and Zhang, Yang and Yang, Zhengyi and Luo, Yancheng and Chen, Chong and Feng, Fuli and Tian, Qi},
  journal={arXiv preprint arXiv:2308.08434},
  year={2023}
}
```


