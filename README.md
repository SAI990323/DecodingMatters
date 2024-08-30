# DecodingMatters

This is the raw implementation of our paper **[Decoding Matters: Addressing Amplification Bias and Homogeneity Issue for LLM-based Recommendation](https://arxiv.org/abs/2406.14900)**

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

If you're using this code in your research or applications, please cite our paper using this BibTeX:
```bibtex
@article{bao2024decoding,
  title={Decoding Matters: Addressing Amplification Bias and Homogeneity Issue for LLM-based Recommendation},
  author={Bao, Keqin and Zhang, Jizhi and Zhang, Yang and Huo, Xinyue and Chen, Chong and Feng, Fuli},
  journal={arXiv preprint arXiv:2406.14900},
  year={2024}
}
```


