from sklearn.naive_bayes import MultinomialNB
from nltk import RegexpTokenizer
import os
import numpy as np

tokenizer = RegexpTokenizer(r"\w+")

def encode_caption(word2token, caption):
    n_words = len(word2token)
    encoding = np.zeros(n_words)
    processed_caption = tokenizer.tokenize(caption)
    processed_caption = [word.lower() for word in processed_caption]
    for word in processed_caption:
        encoding[word2token[word]] += 1
    return encoding


def create_training_data(experiment_dir, macro_dict, train_tasks, train_names, test_tasks):

    X = []
    y = []

    all_words = set()
    task_captions = {}
    encoded_captions = {}

    # add training example words
    for task_fn, task_name in zip(train_tasks, train_names):
        _, caption = task_fn()
        processed_caption = tokenizer.tokenize(caption)
        processed_caption = [word.lower() for word in processed_caption]
        task_captions[task_name] = caption
        for word in processed_caption:
            all_words.add(word)

    # add test set words
    for task_fn in test_tasks:
        _, caption = task_fn()
        processed_caption = tokenizer.tokenize(caption)
        processed_caption = [word.lower() for word in processed_caption]
        for word in processed_caption:
            all_words.add(word)

    all_words = list(all_words)
    word2token = {word:i for i,word in enumerate(all_words)}
    token2word = {i:word for i,word in enumerate(all_words)}

    for task_name in train_names:
        encoding = encode_caption(word2token, task_captions[task_name])
        action_file = os.path.join(experiment_dir, "action_sequences", f"{task_name}_actions.npy")
        action_sequences = list(np.load(action_file, allow_pickle=True))
        for macro_label, macro in macro_dict.items():
            for sequence in action_sequences:
                for i in range(len(sequence)-len(macro)):
                    if tuple(sequence[i:i+len(macro)]) == macro:
                        X.append(encoding)
                        y.append(macro_label)

    X = np.stack(X)
    y = np.array(y)
    print(y)

    return X,y,word2token,token2word


def train_naive_bayes(X,y):
    model = MultinomialNB()
    model.fit(X,y)
    print(model.classes_)
    return model


def predict_macros(model, caption, macro_dict,word2token):
    encoding = encode_caption(word2token, caption)
    results = model.predict_proba(np.array([encoding]))[0]
    macro_ordering = np.flip(np.argsort(results)) + 10
    macro_ordering = list(macro_ordering)
    best_macros = [macro_dict[x] for x in macro_ordering]
    return best_macros, macro_ordering