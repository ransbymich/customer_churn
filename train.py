import pickle

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sb
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline

train_news = pd.read_csv('test.csv')
test_news = pd.read_csv('train.csv')
valid_news = pd.read_csv('valid.csv')


def create_distribution(file_name):
    create_distribution(train_news)
    create_distribution(test_news)
    create_distribution(valid_news)
    return sb.countplot(x='Label', data=file_name, palette='hls')


eng_stemmer = SnowballStemmer('english')
stopwords = set(nltk.corpus.stopwords.words('english'))

countV = CountVectorizer()
train_count = countV.fit_transform(train_news['Statement'].values)

print(countV)
print(train_count)

tfidfV = TfidfTransformer()
train_tfidf = tfidfV.fit_transform(train_count)


def get_tfidf_stats():
    print(train_tfidf.A[:10])


tfidf_ngram = TfidfVectorizer(stop_words='english', ngram_range=(1, 4), use_idf=True, smooth_idf=True)

tagged_sentences = nltk.corpus.treebank.tagged_sents()

cutoff = int(0.75 * len(tagged_sentences))
training_sentences = train_news['Statement']

print(training_sentences)


def features(sentence, index):
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }


logistic_pipeline = Pipeline([
    ('LogRCV', countV),
    ('LogR_clf', LogisticRegression())
])

logistic_pipeline.fit(train_news['Statement'], train_news['Label'])
predicted_LogR = logistic_pipeline.predict(test_news['Statement'])


def output_confusion(classifier):
    k_fold = KFold(n_splits=5)
    scores = []
    confusion = np.array([[0, 0], [0, 0]])

    for train_ind, test_ind in k_fold.split(train_news):
        train_text = train_news.iloc[train_ind]['Statement']
        train_y = train_news.iloc[train_ind]['Label']

        test_text = train_news.iloc[test_ind]['Statement']
        test_y = train_news.iloc[test_ind]['Label']

        classifier.fit(train_text, train_y)
        predictions = classifier.predict(test_text)

        confusion += confusion_matrix(test_y, predictions)
        score = f1_score(test_y, predictions)
        scores.append(score)

    print('Total statements classified:', len(train_news))
    print('Score:', sum(scores) / len(scores))
    print('score length', len(scores))
    print('Confusion matrix:')
    print(confusion)


output_confusion(logistic_pipeline)

logistic_ngram = Pipeline([('LogR_tfidf', tfidf_ngram), ('LogR_clf', LogisticRegression(penalty="l2", C=1))])

logistic_ngram.fit(train_news['Statement'], train_news['Label'])
pred_logistic_ngram = logistic_ngram.predict(test_news['Statement'])
np.mean(pred_logistic_ngram == test_news['Label'])

output_confusion(logistic_ngram)
print(classification_report(test_news['Label'], pred_logistic_ngram))

params = {'LogR_tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)],
          'LogR_tfidf__use_idf': (True, False),
          'LogR_tfidf__smooth_idf': (True, False)
          }

logR_pipeline_final = Pipeline([
    ('LogR_tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 5), use_idf=True, smooth_idf=False)),
    ('LogR_clf', LogisticRegression(penalty="l2", C=1))
])

logR_pipeline_final.fit(train_news['Statement'], train_news['Label'])
predicted_LogR_final = logR_pipeline_final.predict(test_news['Statement'])
np.mean(predicted_LogR_final == test_news['Label'])

print(classification_report(test_news['Label'], predicted_LogR_final))

model_file = 'final_model.sav'
pickle.dump(logistic_ngram, open(model_file, 'wb'))


def plot_loss(pipeline, title):
    size = 159
    cv = KFold(size, shuffle=True)

    X = train_news["Statement"]
    y = train_news["Label"]

    pl = pipeline
    pl.fit(X, y)

    train_sizes, train_scores, test_scores = learning_curve(pl, X, y, n_jobs=-1, cv=cv,
                                                            train_sizes=np.linspace(.1, 1.0, 5), verbose=0)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title(title)
    plt.legend(loc="best")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.gca().invert_yaxis()

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.ylim(-.1, 1.1)
    plt.show()


plot_loss(logistic_ngram, "Naive-bayes Classifier")
