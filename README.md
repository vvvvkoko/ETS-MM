# ETS-MM: A Multi-Modal Social Bot Detection Model Based on Enhanced Textual Semantic Representation

Our paper has been accepted to WWW 2025 (Web conf 2025). The code and data can be visited at this [link](https://github.com/vvvvkoko/ETS-MM). 

This is the repository for the paper: [ETS-MM: A Multi-Modal Social Bot Detection Model Based on Enhanced Textual Semantic Representation](https://dl.acm.org/doi/abs/10.1145/3696410.3714551#core-collateral-purchase-access).

## Abstract 
Social bots are becoming increasingly common in social networks, and their activities affect the security and authenticity of social media platforms. Current state-of-the-art social bot detection methods leverage multimodal approaches that analyze various modalities, such as user metadata, text, and social network relationships. However, these methods may not always extract additional dimensions of semantic feature information that could offer a deeper understanding of users' social patterns. To address this issue, we propose ETS-MM, a multimodal detection framework designed to augment multidimensional information from text and extract the semantic feature representation of user text information. We first analyze the user's tweeting behavior based on topic preference and emotion tendency, integrating them into the textual data. Then, we try to extract enhanced semantic representations that reveal the latent relationship between tweeting behavior and tweet content while identifying potential contextual associations and emotional changes. Additionally, to capture the complex interaction between users, we integrate the user's multimodal information, including metadata, textual features, enhanced semantic features, and social network relationships to propagate and aggregate information across various modalities. Experimental results demonstrate that ETS-MM significantly outperforms existing methods across two widely used social bot detection benchmark datasets, validating its effectiveness and superiority.


## Dataset
We use Cresci15 and Twibot-20 as our datasets. The user tweet topics and emotions extracted by ChatGPT have been uploaded to [Google Drive](https://drive.google.com/file/d/1vZWUYsE77zvkMTqC595aPEMyY2KVaSfo/view?usp=drive_link). 
The correspondence between topics, emotions content and ID is as follows

```python
dict_topic = {"none": 0,
              "arts & culture": 1,
              "business & finance": 2,
              "careers": 3,
              "entertainment": 4,
              "fashion & beauty": 5,
              "food": 6,
              "gaming": 7,
              "hobbies & interests": 8,
              "movies & tv": 9,
              "music": 10,
              "news": 11,
              "outdoors": 12,
              "science": 13,
              "sports": 14,
              "technology": 15,
              "travel": 16
              }
dict_emotion = {"positive": 1,
                "negative": 2,
                "neutral": 3
                }
```

## Code
Now moved to [link](https://github.com/vvvvkoko/ETS-MM).


## Citation
Please consider citing this paper if you find this repository useful:

```text
@inproceedings{10.1145/3696410.3714551,
    author = {Li, Wei and Deng, Jiawen and You, Jiali and He, Yuanyuan and Zhuang, Yan and Ren, Fuji},
    title = {ETS-MM: A Multi-Modal Social Bot Detection Model Based on Enhanced Textual Semantic Representation}, 
    year = {2025},
    url = {https://doi.org/10.1145/3696410.3714551}, 
    doi = {10.1145/3696410.3714551}, 
    booktitle = {Proceedings of the ACM on Web Conference 2025}, 
    pages = {4160–4170} 
}
```





