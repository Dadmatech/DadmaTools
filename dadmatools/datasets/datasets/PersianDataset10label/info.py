dataset_info = {
    "name": "PersianSentimentEmotionDataset",
    "version": "1.0.0",
    "task": ["Sentiment Classification", "Emotion Classification"],
    "splits": ["train", "test", "dev"],
    "description": (
        "This dataset contains Persian text data with sentiment and emotion labels. "
        "Each entry is labeled with a sentiment category (positive, negative, neutral) "
        "and one of ten emotion categories (sadness, praise, happiness, request, hope, fear, "
        "anger, wonder, hate, other). The dataset can be used for training and evaluating "
        "models for both sentiment and emotion classification in Persian."
    ),
    "filenames": {
        "persian": {
            "threeLabeled": ["train.txt", "test.txt", "dev.txt"],
            "tenLabeled": ["train.txt", "test.txt", "dev.txt"]
        }
    },
    "size": {
        "train": {
            "threeLabeled": len(train_df),
            "tenLabeled": len(train_df)
        },
        "test": {
            "threeLabeled": len(test_df),
            "tenLabeled": len(test_df)
        },
        "dev": {
            "threeLabeled": len(dev_df),
            "tenLabeled": len(dev_df)
        }
    },
    "label_distribution": {
        "Sentiment": {
            "Positive": df['sentiment'].value_counts().get('مثبت', 0),
            "Negative": df['sentiment'].value_counts().get('منفی', 0),
            "Neutral": df['sentiment'].value_counts().get('خنثی', 0)  # if applicable
        },
        "Emotion": {
            "sadness": df['emotion'].value_counts().get('sadness', 0),
            "praise": df['emotion'].value_counts().get('praise', 0),
            "happiness": df['emotion'].value_counts().get('happiness', 0),
            "request": df['emotion'].value_counts().get('request', 0),
            "hope": df['emotion'].value_counts().get('hope', 0),
            "fear": df['emotion'].value_counts().get('fear', 0),
            "anger": df['emotion'].value_counts().get('anger', 0),
            "wonder": df['emotion'].value_counts().get('wonder', 0),
            "hate": df['emotion'].value_counts().get('hate', 0),
            "other": df['emotion'].value_counts().get('other', 0)
        }
    },
    "languages": ["Persian"]
}
