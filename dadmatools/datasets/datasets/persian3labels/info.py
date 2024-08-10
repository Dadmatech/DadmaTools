dataset_info = {
    "name": "Persian3Labels",
    "version": "1.0.0",
    "task": "Sentiment Classification",
    "splits": ["train", "test", "dev"],
    "description": "The Persian3Labels dataset contains Persian text data labeled with sentiment categories: negative, positive, and neutral. This dataset is used for training and evaluating sentiment classification models.",
    "filenames": ["train.txt", "test.txt", "dev.txt"],
    "size": {
        "train": 3232 + 2780 + 2490 - 1536,  
        "test": 2490,  
        "dev": 1536    
    },
    "label_distribution": {
        "negative": 3232,
        "positive": 2780,
        "neutral": 2490
    }
}
