dataset_info = {
    "name": "TranslatedSentimentEmotionDataset",
    "version": "1.0.0",
    "task": ["Sentiment Classification", "Emotion Classification"],
    "splits": ["train", "test", "dev"],
    "description": (
        "This dataset contains reviews of the Redmi 12C smartphone, translated into both Persian and English. "
        "Each review is labeled with both sentiment (positive, negative, neutral) and emotion (Happy, Love, "
        "Sadness, Fear, Anger) categories. The dataset can be used for training and evaluating models for both "
        "sentiment and emotion classification in multiple languages."
    ),
    "filenames": {
        "persian": {
            "threeLabeled": ["train.txt", "test.txt", "dev.txt"],
            "fifthLabeled": ["train.txt", "test.txt", "dev.txt"]
        },
        "english": {
            "threeLabeled": ["train.txt", "test.txt", "dev.txt"],
            "fifthLabeled": ["train.txt", "test.txt", "dev.txt"]
        }
    },
    "size": {
        "train": {
            "threeLabeled": len(train_df),
            "fifthLabeled": len(train_df)
        },
        "test": {
            "threeLabeled": len(test_df),
            "fifthLabeled": len(test_df)
        },
        "dev": {
            "threeLabeled": len(dev_df),
            "fifthLabeled": len(dev_df)
        }
    },
    "label_distribution": {
        "Sentiment": {
            "Positive": df['Sentiment'].value_counts().get('Positive', 0),
            "Negative": df['Sentiment'].value_counts().get('Negative', 0),
            "Neutral": df['Sentiment'].value_counts().get('Neutral', 0)
        },
        "Emotion": {
            "Happy": 44293,
            "Love": 20000,
            "Sadness": 6494,
            "Anger": 3134,
            "Fear": 1153
        }
    },
    "languages": ["Persian", "English"]
}
