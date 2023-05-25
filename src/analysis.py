print("[INFO]: Importing packages...")
import os
import pandas as pd
import numpy as np
from transformers import pipeline
import matplotlib.pyplot as plt


def main():
    
    # load data
    datapath = os.path.join("..","data","fake_or_real_news.csv")
    
    data = pd.read_csv(datapath, usecols=["title", "label"])
    
    # initialize text classification model
    classifier = pipeline(task="text-classification", 
                            model="j-hartmann/emotion-english-distilroberta-base")

    # apply classifier to news headlines
    print("[INFO]: classifying news headlines...")
    emotions = []
    for headline in data['title']:
        # classify headline
        output = classifier(headline)
        # extract emotion label
        emotion = output[0]['label']
        emotions.append(emotion)

    # add emotion labels to data
    data['sentiment'] = emotions

    # save data frame
    outpath = os.path.join("..", "out", "news_w_sentiment.csv")
    data.to_csv(outpath, index=False)

    # count ocurrences of emotions
    print("[INFO]: Creating plots...")
    counts = data.groupby(['sentiment']).size()
    x_labels = counts.index.tolist()
    y_labels = counts.values.tolist()

    # plotting count distribution
    fig = plt.figure()
    plt.bar(x_labels, y_labels)

    ## make horizontal grid lines appear behind bars
    plt.rc('axes', axisbelow=True)
    plt.grid(which='major', axis='y', alpha=0.7)

    ## adding titles
    plt.xlabel("Emotions", fontweight ='bold', fontsize = 10)
    plt.ylabel("Count", fontweight ='bold', fontsize = 10)
    plt.title("Distribution of emotions across all news headlines")

    # save plot
    outpath = os.path.join("..","out","overall_emotion_distribution.png")
    plt.savefig(outpath)

    # count ocurrences of emotions within news types
    counts = data.groupby(['label', 'sentiment']).size()
    fake_counts = counts.values[:7]
    real_counts = counts.values[7:]

    # plotting multiple bar plot
    ## set width of bar
    bar_width = 0.25
    fig = plt.subplots()
    
    ## set position of bar on x axis
    real_bars = np.arange(len(real_counts))
    fake_bars = [x + bar_width for x in real_bars]
    
    ## make the plot
    plt.bar(real_bars, real_counts, color ="#22a7f0", width = bar_width, label ='real')
    plt.bar(fake_bars, fake_counts, color ="#a7d5ed", width = bar_width, label ='fake')

    ## make horizontal grid lines appear behind bars
    plt.rc('axes', axisbelow=True)
    plt.grid(which='major', axis='y', alpha=0.7)

    ## adding titles and x ticks
    plt.xlabel('Emotions', fontweight ='bold', fontsize = 10)
    plt.ylabel('Count', fontweight ='bold', fontsize = 10)
    plt.title("Distribution of emotions within news type")
    plt.xticks([r + bar_width/2 for r in range(len(real_counts))],
           ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'])
 
    plt.legend()

    # save plot
    outpath = os.path.join("..","out","real_fake_emotion_distribution.png")
    plt.savefig(outpath)

if __name__ == "__main__":
    main()
