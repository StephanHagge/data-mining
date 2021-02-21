from pandas import DataFrame
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from itertools import combinations
import pandas as pd
import numpy as np

from datetime import datetime
from datetime import timedelta


# Data preperation

df = pd.read_csv('../0_data/DEvideos.csv',
    low_memory=False)

df['trending_date'] = df.apply(lambda row: datetime.strptime(row['trending_date'], '%y.%d.%m'), axis=1)
df['publish_time'] = df.apply(lambda row: datetime.strptime(row['publish_time'], '%Y-%m-%dT%H:%M:%S.000Z'), axis=1)
df['days_until_trending'] = df.apply(lambda row: ((row['trending_date'] - row['publish_time']).days + 1), axis=1)

df['tags_count'] = df.apply(lambda row: len(row['tags'].split('|')), axis=1)
df['publish_hour'] = df['publish_time'].map(lambda x: x.hour)
df['publish_month'] = df['publish_time'].map(lambda x: x.month)
df['publish_year'] = df['publish_time'].map(lambda x: x.year)
df['publish_day_of_month'] = df['publish_time'].map(lambda x: x.day)
df['publish_weekday'] = df['publish_time'].map(lambda x: x.weekday()) # 0: Monday, 6: Sunday

df['like_dislike_ratio'] = df.apply(lambda row: row['likes'] / (row['dislikes'] + 1), axis=1)
df['like_view_ratio'] = df.apply(lambda row: row['likes'] / (row['views'] + 1), axis=1)

df['ratings'] = df['likes'] + df['dislikes']
df['likes_per_rating'] = df.apply(lambda row: 0 if row['ratings'] == 0 else row['likes'] / row['ratings'], axis=1)
df['ratings_per_view'] = df['ratings'] / df['views']
df['comments_per_view'] = df['comment_count'] / df['views']

# Using int instead of cat
def assign_target_category(row):
    if row['days_until_trending'] == 0: 
        return 0
    elif row['days_until_trending'] == 1:
        return 1
    elif row['days_until_trending'] == 2:
        return 2
    elif row['days_until_trending'] <= 5:
        return 3
    else:
        return 6

df['target_category'] = df.apply(assign_target_category, axis=1)
df['channel_title'] = df['channel_title'].astype('category')


# Map tag factor

tag_df = pd.read_csv('../0_data/DEtags.csv')
tag_df = tag_df.set_index('tag')
def calculate_tag_factor(tag_string, tag_data):
    tag_list = pd.Series(list(set(map(lambda x: x.strip('\"').lower(), tag_string.split('|')))))
    return tag_list.apply(lambda tag: tag_data['factor'].get(tag, np.nan)).mean(skipna=True)
    
df['tag_factors'] = df['tags'].apply(lambda x: calculate_tag_factor(x, tag_df))
df['tag_factors'] = df.apply(lambda row: 0 if np.isnan(row['tag_factors']) else row['tag_factors'], axis=1)


# Remove unneccessary columns

N = len(df)
dropColumns = ['video_id', 'title', 'tags', 'thumbnail_link', 'description']
for column in df.columns:
    numberOfUniqueValues = df[column].nunique()
    if numberOfUniqueValues < 2:
        dropColumns.append(column)
    elif df[column].dtype == 'object' and numberOfUniqueValues > N * 0.9:
        dropColumns.append(column)
    elif df[column].isna().sum() / N > 0.95:
        dropColumns.append(column)
        
df.drop(columns=dropColumns, inplace=True)


# Target encoding

target = df['target_category'].astype(str)
y_label_encoder = preprocessing.LabelEncoder()
y_label_encoder.fit(target)
y = y_label_encoder.transform(target)


# Preparation of Featuresets
best_featureset = []
best_accuracy = 0

for i in range(1, 5):
    featuresets = combinations(df.columns, i)
    print('--------------------')
    print('Training with ' + str(i) + ' features')
    print('\n')

    for featureset in featuresets:
        # Continue if featureset includes "forbidden" feature
        if any(item in ['trending_date', 'days_until_trending', 'target_category'] for item in featureset) is True:
            continue 
        
        print('Testing featureset:')
        print(featureset)

        # Feature encoding

        x_df = DataFrame(index=df.index)
        for feature in featureset:
            feature_data = df[feature]
            if df[feature].dtype.name == 'category':
                x_label_encoder = preprocessing.LabelEncoder()
                x_label_encoder.fit(feature_data.astype(str))
                x_df[feature] = x_label_encoder.transform(feature_data)
            elif df[feature].dtype.name == 'datetime64[ns]':
                x_df[feature] = feature_data.apply(lambda x: x.timestamp())
            elif df[feature].dtype.name == 'bool':
                x_df[feature] = feature_data.apply(lambda x: int(x))
            else:
                x_df[feature] = feature_data

        x = np.reshape(x_df, (-1, len(x_df.columns)))


        # Create train and test datasubset

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.40, random_state=0)

        
        # Choose classifier

        # Fast random forest
        # classifier = RandomForestClassifier(n_jobs=-1, n_estimators=10, max_depth=2, min_samples_split=5)

        # Optimized random forest (slow)
        # classifier = RandomForestClassifier(n_jobs=-1, class_weight=None, criterion='gini', max_depth=9, max_features='log2', max_leaf_nodes=None, min_impurity_decrease=0.0, min_samples_leaf=1, min_samples_split=4, min_weight_fraction_leaf=0.0, n_estimators=200)
        
        # Optimized decision tree
        classifier = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None, max_features=None, max_leaf_nodes=50, min_impurity_decrease=0.0, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, splitter='best')

        # Train classifier

        print('Training ...')
        classifier.fit(x_train, y_train)


        # Predict test data

        print('Predicting ...')
        prediction = classifier.predict(x_test)


        # Calculate accuracy score

        accuracy = accuracy_score(y_test, prediction)
        print('Accuracy: ' + str(accuracy))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_featureset = featureset
            print('New top accuracy score!')

        print('\n')

print('----------------------------------------')
print('Best accuracy was: ' + str(best_accuracy))
print('Best featureset was:')
print(best_featureset)