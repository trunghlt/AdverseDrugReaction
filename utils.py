import re
import numpy as np
import nltk
import theano
from theano import tensor as T
from keras.models import Sequential, Graph
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import MaxPooling1D, Convolution1D
from keras.layers.core import Dense, Dropout, Flatten, Lambda
from keras.callbacks import EarlyStopping
from keras.layers.recurrent import GRU, LSTM, SimpleRNN
from keras.constraints import maxnorm
from keras.regularizers import l2, activity_l2


class FixedEmbedding(Embedding):

    def __init__(self, *args, **kwargs):
        super(FixedEmbedding, self).__init__(*args, **kwargs)
        self.params = []


class MyEarlyStopping(EarlyStopping):

    def __init__(self, **kwargs):
        super(MyEarlyStopping, self).__init__(**kwargs)
        self.mode = kwargs.get('mode', 'auto')

    def on_train_begin(self, logs={}):
        self.wait = 0
        self.best = -np.inf if self.mode=='max' else np.inf
        self.best_weights = None

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % (self.monitor), RuntimeWarning)

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print("Epoch %05d: early stopping" % (epoch))
                self.model.stop_training = True

    def on_train_end(self, logs={}):
        super(MyEarlyStopping, self).on_train_end(logs)
        self.model.set_weights(self.best_weights)


def yc_tokenize(string, TREC=False):
    """
    Yoon Kim's tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().split(' ') if TREC\
           else string.strip().lower().split(' ')


class Twitter:
    url_regex = re.compile(r"\b(((https?|ftp|file)?://)|(www\.))[-A-Z0-9+&@#/%?=~_|!:,.;]*[-A-Z0-9+&@#/%=~_|]", re.I)
    mention_regex = re.compile(r"@\S+", re.I)
    norm = re.compile(r"(.)\1{3,}", re.I)
    MENTION = 'MENTIONMENTION'
    URL = 'URLURL'
    TOKENIZER = nltk.RegexpTokenizer(
        r"""(?x)      # set flag to allow verbose regexps "
             http://[^ ]+       #urls
           | \@[^ ]+            # Twitter usernames
           | \#[^ ]+            # Twitter hashtags
           | [A-Z]([A-Z]|\.|&)+        # abbreviations, e.g. U.S.A., AT&T
           | \w+(-\w+)*        # words with optional internal hyphens
           | \$?\d+(\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
           | \.\.\.            # ellipsis
           | \'s                # various things
           | \'t
           | n\'t
           | [][.,;"'?():-_`]  # these are separate tokens
        """,
        False, True, re.UNICODE | re.MULTILINE | re.DOTALL
    )

    @staticmethod
    def underscorize(matchobj):
        return "_" * len(matchobj.group(0))

    @staticmethod
    def clean(text):
        s = text
        s = re.sub(Twitter.mention_regex, Twitter.MENTION, s)
        s = re.sub(Twitter.url_regex, Twitter.URL, s)
        return s

    @staticmethod
    def normalize(text):
        s = re.sub(Twitter.norm, '\g<1>\g<1>', text)
        return s

    @staticmethod
    def tokenize(text, tokenize=None):
        token_f = tokenize or Twitter.TOKENIZER.tokenize
        return token_f(Twitter.normalize(Twitter.clean(text.lower())))


def emotion_tokenizer(string, TREC=False):
    """
    emotion-icon aware tokenizers
    """
    string = re.sub(r"[^\w(),|!?\'\`\:\-\.;\$%#]", " ", string)
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"n\'t", " n't", string)
    string = re.sub(r"\'re", " 're", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r"\'ll", " 'll", string)
    string = re.sub(r"(?<=\w)\.\.\.", " ... ", string)
    string = re.sub(r"(?<=\w)\.", " . ", string)
    string = re.sub(r"(?<=\w),", " , ", string)
    string = re.sub(r"(?<=\w);", " ; ", string)
    string = re.sub(r"(?<=\w)!", " ! ", string)
    string = re.sub(r"\((?=\w)", " ( ", string)
    string = re.sub(r"(?<=\w)\)", " ) ", string)
    string = re.sub(r"(?<=\w)\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.strip() if TREC else string.strip().lower()
    return string.split(' ')


def vectorize(docs, V, max_len=None):
    if max_len is None:
        max_len = max([len(d) for d in docs])
    oov, pad = len(V), len(V) + 1
    X = []
    for doc in docs:
        padding = [pad]*(max_len - len(doc))
        X.append([V.get(w, oov) for w in doc[:max_len]] + padding)
    result = np.asarray(X)
    assert result.ndim==2 and result.shape[0]==len(docs) and result.shape[1]==max_len
    return result


def mk_yk_model_f(max_len, embeddings, embedding_fixed=False, subspace=None, n_filters=300,
        filter_height=5, W_constraint=9, optimizer='adadelta', pool_length=None):

    def yk_model():
        cnn = Sequential()
        cnn.add((FixedEmbedding if embedding_fixed else Embedding)\
                    (*embeddings.shape, input_length=max_len, weights=[embeddings]))
        if subspace is not None:
            cnn.add(Convolution1D(subspace, 1, border_mode='same'))
        cnn.add(Convolution1D(n_filters, filter_height, border_mode='valid',
            activation='relu', W_constraint=maxnorm(W_constraint)))
        cnn.add(MaxPooling1D(pool_length=pool_length or (max_len - filter_height + 1)))
        cnn.add(Flatten())
        cnn.add(Dropout(.5))
        cnn.add(Dense(1, activation='sigmoid', W_constraint=maxnorm(W_constraint)))
        cnn.compile(loss='binary_crossentropy', optimizer=optimizer, class_mode='binary')
        return cnn

    return yk_model


def mk_cgru_model_f(max_len, embeddings, filter_length=3, nb_filter=64, pool_length=2,
        rnn_output=70, W_constraint=9, embedding_fixed=False):

    def cgru_model():
        m = Sequential()
        layers = [
            (FixedEmbedding if embedding_fixed else Embedding)\
                (*embeddings.shape, input_length=max_len, weights=[embeddings]),
            Dropout(0.25),
            Convolution1D(
                nb_filter,
                filter_length,
                border_mode="valid",
                activation="relu"),
            MaxPooling1D(pool_length=pool_length),
            GRU(rnn_output),
            # Lambda(lambda X: X.mean(axis=-2), output_shape=(rnn_output,)),
            Dense(1, activation='sigmoid', W_constraint=maxnorm(W_constraint))
        ]
        for l in layers:
            m.add(l)
        m.compile(loss='binary_crossentropy', optimizer="adadelta", class_mode='binary')
        return m

    return cgru_model


def mk_rcnn_model_f(max_len, embeddings, filter_length=5, nb_filter=300,
        rnn_output=300, W_constraint=9, embedding_fixed=False):

    def cgru_model():
        m = Sequential()
        layers = [
            (FixedEmbedding if embedding_fixed else Embedding)\
                (*embeddings.shape, input_length=max_len, weights=[embeddings]),
            SimpleRNN(rnn_output, activation="relu", return_sequences=True),
            Convolution1D(
                nb_filter,
                filter_length,
                border_mode="valid",
                activation="relu"),
            MaxPooling1D(pool_length=max_len - filter_length + 1),
            Flatten(),
            Dropout(0.5),
            Dense(1, activation='sigmoid', W_constraint=maxnorm(W_constraint))
        ]
        for l in layers:
            m.add(l)
        m.compile(loss='binary_crossentropy', optimizer="adadelta", class_mode='binary')
        return m

    return cgru_model


def add_yk_node(graph, input_name, max_len, embeddings, embedding_fixed=False):
    graph.add_node((FixedEmbedding if embedding_fixed else Embedding)\
                        (*embeddings.shape, input_length=max_len, weights=[embeddings]),
                        name='embedding',
                        input=input_name)
    graph.add_node(Convolution1D(300, 5, border_mode='same', activation='relu', W_constraint=maxnorm(9)),
                name='conv1', input='embedding')
    graph.add_node(MaxPooling1D(pool_length=max_len), name='maxpooling1', input='conv1')
    graph.add_node(Flatten(), name='flatten1', input='maxpooling1')
    graph.add_node(Dropout(.5), name='yk', input='flatten1')


def mk_gru_model_f(max_len, embeddings, embedding_fixed=False, rnn_output=300):

    def gru_model():
        m = Sequential()
        m.add((FixedEmbedding if embedding_fixed else Embedding)\
                (*embeddings.shape, input_length=max_len, weights=[embeddings]))
        m.add(GRU(rnn_output))
        m.add(Dropout(0.5))
        m.add(Dense(1, activation='sigmoid'))
        m.compile(loss='binary_crossentropy', optimizer="adadelta", class_mode='binary')
        return m

    return gru_model


def mk_attention_based_model_f(max_len, embeddings, filter_length=5,
        nb_filter=300, attention_filter_length=5, W_constraint=9, embedding_fixed=False,
        attention_l2=0.01):

    def attention_based_model():
        m = Graph()
        m.add_input(name='tokens', input_shape=(max_len, ), dtype='int')
        m.add_node((FixedEmbedding if embedding_fixed else Embedding)\
                        (*embeddings.shape, input_length=max_len, weights=[embeddings]),
                        name='embedding',
                        input='tokens')
        m.add_node(Convolution1D(nb_filter, filter_length, border_mode='same', activation='relu',
            W_constraint=maxnorm(9)), name='features', input='embedding')
        m.add_node(Convolution1D(1, attention_filter_length, border_mode='same', activation='relu',
            W_regularizer=l2(attention_l2)), name='attn_weights',
            input='features')
        m.add_node(Lambda(lambda X: T.addbroadcast(
                                        T.exp(X)/T.addbroadcast(T.exp(X).sum(axis=1, keepdims=True), 1),
                                        2
                                    )*theano.shared(np.ones((1, 1, nb_filter), dtype=theano.config.floatX),
                                                    broadcastable=(True, True, False)),
                         output_shape=(max_len, nb_filter)
                     ),
                   input='attn_weights', name='norm_attn_weights')
        # m.add_node(MaxPooling1D(pool_length=max_len), name='maxpooling1',
            # inputs=['features', 'norm_attn_weights'], merge_mode="mul")
        # m.add_node(Flatten(), name='flatten1', input='maxpooling1')
        m.add_node(Lambda(lambda X: X.sum(axis=1), output_shape=(nb_filter, )),
            name='sum_weighted_features', inputs=['features', 'norm_attn_weights'],
            merge_mode="mul")
        m.add_node(Dropout(.5), name='dropout', input='sum_weighted_features')
        m.add_node(Dense(1, activation='sigmoid'),
            input="dropout", create_output=True, name="output")
        m.compile(optimizer='adadelta', loss={'output': 'binary_crossentropy'})
        return m

    return attention_based_model


def seq_cross_validate(model_f, X, y, cv_folds, fit_params=None, verbose=True, eval_f=None):
    results = []
    if fit_params is None:
        fit_params = {}
    for i, (train, test) in enumerate(cv_folds, 1):
        model = model_f()
        model.stop_training = False
        model.fit(X[train], y[train], **fit_params)
        model.last_fit_X = X[train]
        model.last_fit_y = y[train]
        model.last_fit_params = fit_params
        if eval_f is None:
            score = model.evaluate(X[test], y[test], show_accuracy=True)[1]
        else:
            score = eval_f(model, X[test], y[test])

        if verbose:
            print 'Fold #{}: {:.3f}'.format(i, score)
        results.append(score)
    return results


def graph_cross_validate(model_f, data, cv_folds, eval_f=None, fit_params=None, verbose=True):
    results = []
    if fit_params is None:
        fit_params = {}
    for i, (train, test) in enumerate(cv_folds, 1):
        model = model_f()
        model.stop_training = False
        model.fit({k:data[k][train]for k in data}, **fit_params)
        test_data = {k:data[k][test] for k in data}
        model.last_fit_data = data
        model.last_fit_params = fit_params
        if eval_f is None:
            score = model.evaluate(test_data, show_accuracy=True)[1]
        else:
            score = eval_f(model, test_data)

        if verbose:
            print 'Fold #{}: {:.3f}'.format(i, score)
        results.append(score)
    return results
