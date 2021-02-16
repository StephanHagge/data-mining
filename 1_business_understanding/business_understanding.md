# Business Understanding

## What is the process?

YouTube is one of the biggest video sharing websites. It maintains a list of the top trending videos on the platform.
For our project we chose the [Trending YouTube Video Statistics](https://www.kaggle.com/datasnaek/youtube-new) from Kaggle.

The dataset describes trending YouTube videos over several months, but the data is separated into multiple subsets due to differences in
their category_id attribute based on the given region. Due to the high number of possible data, and the resulting complexity we decided
to use the data related to the region DE (Germany) only.

The chosen dataset for [Germany](https://www.kaggle.com/datasnaek/youtube-new?select=DEvideos.csv) consists of 16 columns and 40840 rows.

## What is the task?

Our task is the prediction of the time it takes for a video to reach the state of trending based on the time it was uploaded.

## Can the problem definition be focused?

A video reaching the state of trending seems random often.
Therefore, we want to use the attributes we have to analyze whether it is possible to predict how long it
takes for a video to get the state trending and which values are involved.

## Does it relate to other problem definitions?

As described it relates to the problem why a video is trending. Which values are involved?
Are there any correlations between the values?