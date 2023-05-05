# ======================================= #
# ----------- Data4BertModel ------------ #
# ======================================= #
import pprint
import os

pp = pprint.PrettyPrinter(indent=4)
from dataset.utils import SentenceProcessor


class IMDB(SentenceProcessor):
    NAME = 'IMDB'
    NUM_CLASSES = 10

    def __init__(self, data_dir='corpus'):
        self.d_train = self._read_file(os.path.join(data_dir, 'imdb', 'imdb.train.txt.ss'))
        self.d_dev = self._read_file(os.path.join(data_dir, 'imdb', 'imdb.dev.txt.ss'))
        self.d_test = self._read_file(os.path.join(data_dir, 'imdb', 'imdb.test.txt.ss'))

    def get_documents(self):
        train = self._create_examples(self.d_train, 'train')
        dev = self._create_examples(self.d_dev, 'dev')
        test = self._create_examples(self.d_test, 'test')
        return tuple([train, dev, test])

    def get_sent_doc(self):
        train = self._creat_sent_doc(self.d_train)
        dev = self._creat_sent_doc(self.d_dev)
        test = self._creat_sent_doc(self.d_test)
        return tuple([train, dev, test])

    def get_attributes(self):
        return self._get_attributes(self.d_train, self.d_dev, self.d_test)  # tuple(attributes) rather tuple(users, products)


class YELP_13(SentenceProcessor):
    NAME = 'YELP_13'
    NUM_CLASSES = 5

    def __init__(self, data_dir='corpus'):
        super().__init__()
        self.d_train = self._read_file(os.path.join(data_dir, 'yelp_13', 'yelp-2013-seg-20-20.train.ss'))
        self.d_dev = self._read_file(os.path.join(data_dir, 'yelp_13', 'yelp-2013-seg-20-20.dev.ss'))
        self.d_test = self._read_file(os.path.join(data_dir, 'yelp_13', 'yelp-2013-seg-20-20.test.ss'))

    def get_documents(self):
        train = self._create_examples(self.d_train, 'train')
        dev = self._create_examples(self.d_dev, 'dev')
        test = self._create_examples(self.d_test, 'test')
        return tuple([train, dev, test])

    def get_sent_doc(self):
        train = self._creat_sent_doc(self.d_train)
        dev = self._creat_sent_doc(self.d_dev)
        test = self._creat_sent_doc(self.d_test)
        return tuple([train, dev, test])

    def get_attributes(self):
        return self._get_attributes(self.d_train, self.d_dev, self.d_test)


class YELP_14(SentenceProcessor):
    NAME = 'YELP_14'
    NUM_CLASSES = 5

    def __init__(self, data_dir='corpus'):
        self.d_train = self._read_file(os.path.join(data_dir, 'yelp_14', 'yelp-2014-seg-20-20.train.ss'))
        self.d_dev = self._read_file(os.path.join(data_dir, 'yelp_14', 'yelp-2014-seg-20-20.dev.ss'))
        self.d_test = self._read_file(os.path.join(data_dir, 'yelp_14', 'yelp-2014-seg-20-20.test.ss'))

    def get_documents(self):
        train = self._create_examples(self.d_train, 'train')
        dev = self._create_examples(self.d_dev, 'dev')
        test = self._create_examples(self.d_test, 'test')
        return tuple([train, dev, test])

    def get_sent_doc(self):
        train = self._creat_sent_doc(self.d_train)
        dev = self._creat_sent_doc(self.d_dev)
        test = self._creat_sent_doc(self.d_test)
        return tuple([train, dev, test])

    def get_attributes(self):
        return self._get_attributes(self.d_train, self.d_dev,
                                    self.d_test)  # tuple(attributes) rather tuple(users, products)
