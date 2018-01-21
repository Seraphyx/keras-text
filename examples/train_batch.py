import sys
import os
from math import ceil
import en_core_web_sm

import urllib.request
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

from keras.utils import to_categorical

from keras_text.processing import WordTokenizer
from keras_text.data import Dataset
from keras_text.models.token_model import TokenModelFactory
from keras_text.models import YoonKimCNN, AttentionRNN, StackedRNN
# from keras_text.utils import dump, load


# Overcome pickle recurssion limit
sys.setrecursionlimit(10000)

'''
See: https://raghakot.github.io/keras-text/
* You may need to fix the imports in keras_text to have a period in front in the model/__init__.py file
* For Python 3 you must make the dictionary.values() to be wrapped in list()
* If you cannot download the SpaCy en model in Windows 10 then run as admin
* If you hit a recussion limit when using utils.dump then set a higher limit: sys.setrecursionlimit(10000)
'''

# tokenizer = WordTokenizer()
# tokenizer.build_vocab(texts)


data_path = '../data'



def doc_to_sentence(doc_token_list, max_sents, max_tokens):

	# Count how many sentences
	n_doc = len(doc_token_list)

	# Initialize
	ds_embedding = np.zeros((n_doc, max_sents, max_tokens))

	# Convert words to vector embeddings
	for i, doc in enumerate(doc_token_list):
		if i + 1 > n_doc:
			break
		print(doc)
		for j, sentence in enumerate(doc):
			if j + 1 > max_sents:
				break
			print(sentence)
			for k, word in enumerate(sentence):
				if k + 1 > max_tokens:
					break
				ds_embedding[i, j, k] = word

	return ds_embedding


def doc_to_token(doc_token_list, max_tokens):

	# Count how many sentences
	n_doc = len(doc_token_list)

	# Initialize
	ds_embedding = np.zeros((n_doc, max_tokens))

	# Convert words to vector embeddings
	for doc_i, doc in enumerate(doc_token_list):
		token_i = 0
		for j, sentence in enumerate(doc):
			if token_i + 1 > max_tokens:
				break;
			for token in sentence:
				if token_i + 1 > max_tokens:
					break;
				ds_embedding[doc_i, token_i] = token
				token_i += 1

	return ds_embedding


def sentence_to_token(doc_token_list, max_tokens):

	# Count how many sentences
	n_doc = len(doc_token_list)

	# Initialize
	ds_embedding = np.zeros((n_doc, max_tokens))

	# Convert words to vector embeddings
	for doc_i, doc in enumerate(doc_token_list):
		token_i = 0
		for j, token in enumerate(doc):
			if token_i + 1 > max_tokens:
				break;
			ds_embedding[doc_i, token_i] = token
			token_i += 1

	return ds_embedding


class CreateH5(object):

	def __init__(self, data_path, h5_filename, download_example=False):
		self.data_path = data_path
		self.h5_filename = h5_filename
		self.download_example = download_example
		self.key = 'all'

		if download_example:
			self.example()


	# H5 functions
	def store_h5(self, key, df):

		path = Path(self.data_path)
		path.mkdir(parents=True, exist_ok=True)
		path_h5 = os.path.join(self.data_path, self.h5_filename)

		print(path_h5)

		with pd.HDFStore(path_h5) as store:
			store[key] = df


	def describe_h5(self, key):
		path_h5 = os.path.join(self.data_path, self.h5_filename)

		with pd.HDFStore(path_h5) as store:
			print('=== Getting description of h5: %s' % path_h5)
			print('\t', store[key].shape)
			return store[key].shape


	def get_h5(self, key, index=None):
		path_h5 = os.path.join(self.data_path, self.h5_filename)
		with pd.HDFStore(path_h5) as store:
			if isinstance(index, np.ndarray):
				return store[key].iloc[index.tolist()]
			else:
				return store[key]


	def example(self):

		url = 'https://github.com/lesley2958/twilio-sent-analysis/raw/master'

		def download_tweet_sentiment(url, label):
			file_name, headers = urllib.request.urlretrieve(url)

			text = []
			with open(file_name, encoding="utf8") as f:
				for i in f:
					text.append(i)

			df = pd.DataFrame({'text': text})
			df['label'] = label

			return df

		df_neg = download_tweet_sentiment(url + '/neg_tweets.txt', 0)
		df_pos = download_tweet_sentiment(url + '/pos_tweets.txt', 1)

		df = df_neg.append(df_pos)

		# Create a folder within
		self.data_path = os.path.join(self.data_path, 'twitter_example')
		self.h5_filename = 'example.h5'
		
		# Save locally
		self.store_h5(self.key, df)

		# Start Batch
		self.split_dataset()


	def split_dataset(self, seed=1337, partition=[0.8, 0.1, 0.1]):
		n, _ = self.describe_h5(self.key)
		index = np.arange(n)

		if sum(partition) != 1:
			raise Exception("'partition' must be a list with 3 floating values that add up to 1")


		# Shuffle
		np.random.seed(seed)
		np.random.shuffle(index)

		# Divide dataset
		splits = [ceil(float(i * n)) for i in partition]
		splits = np.cumsum(splits)

		self.i_train, self.i_valid, self.i_test = np.split(index, splits[:2])

	def batch(self, dataset='train', batch_size=100):

		if dataset == 'all':
			print(type(self.i_train))
			index = np.append(self.i_train, [self.i_valid, self.i_test])
		else:
			index = getattr(self, 'i_' + dataset)

		n = len(index)
		batches = ceil(n/batch_size)

		for batch_i in range(batches):
			i_start = batch_i * batch_size
			i_end   = min((batch_i + 1) * batch_size, n)
			i_batch = index[i_start:i_end]

			yield self.get_h5(self.key, index=i_batch)



def save_folder(folder_path):
	'''
	Create folder and subfolders to save results
	'''
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)
		os.makedirs(folder_path + '/models')
		os.makedirs(folder_path + '/embeddings')
		os.makedirs(folder_path + '/data')
		os.makedirs(folder_path + '/diagnostics')
		os.makedirs(folder_path + '/results')


def build_tokenizer(data):
	print("=== Building Tokenizer")
	tokenizer = WordTokenizer()
	tokenizer.build_vocab(data.data['x'])
	return tokenizer

def build_dataset(data, tokenizer):
	print("=== Building Dataset")
	ds = Dataset(data.data['x'], data.data['y'], tokenizer=tokenizer)
	ds.update_test_indices(test_size=0.1)
	return ds

# Trail a sentence level model
def train_han(tokenizer):

	# Pad sentences to 500 and words to 200.
	factory = SentenceModelFactory(
		num_classes=2, 
		token_index=tokenizer.token_index, 
		max_sents=500, 
		max_tokens=200, 
		embedding_type='glove.6B.100d')

	# Hieararchy
	word_encoder_model = AttentionRNN()
	sentence_encoder_model = AttentionRNN()

	# Allows you to compose arbitrary word encoders followed by sentence encoder.
	model = factory.build_model(word_encoder_model, sentence_encoder_model)
	model.compile(optimizer='adam', loss='categorical_crossentropy')
	model.summary()

	return model

def main():

	# Save all results into a directory
	save_directory = '../data/train_han'
	save_folder(save_directory)


	# Steps to perform
	BUILD_TOKENIZER = False
	BUILD_DATASET   = False
	TRAIN_MODEL     = False
	INFERENCE       = True

	# Read data
	print("=== Loading Data")
	data = load_data.data(dataset='acllmdb')

	# Build a token. Be default it uses 'en' model from SpaCy
	if BUILD_TOKENIZER:
		tokenizer = build_tokenizer(data)

	# Build a Dataset
	if BUILD_DATASET:
		ds = build_dataset(data, tokenizer)
		ds.save(os.path.join(save_directory, 'dataset', 'dataset_example'))
	else:
		print("=== Loading Saved Dataset")
		ds = Dataset(data.data['x'], data.data['y']).load(os.path.join(save_directory, 'dataset', 'dataset_example'))
		print(ds)
		print(type(ds))
		print(vars(ds).keys())
		tokenizer = ds.tokenizer

	# Will automagically handle padding for models that require padding (Ex: Yoon Kim CNN)
	if TRAIN_MODEL:
		print("=== Tokenizing Dataset and Padding")
		factory = TokenModelFactory(1, tokenizer.token_index, max_tokens=100, embedding_type='glove.6B.100d')
		word_encoder_model = YoonKimCNN()

		# Train Model
		print("=== Training Model")
		model = factory.build_model(token_encoder_model=word_encoder_model)
		model.compile(optimizer='adam', loss='categorical_crossentropy')
		model.summary()

		# Save
		dump(factory, file_name='../data/embeddings/factory_example')
		dump(model, file_name='../data/models/YoonKimCNN_example')
	else:
		print("=== Loading Embeddings")
		factory = load(file_name='../data/embeddings/factory_example')
		print("=== Loading Model")
		model   = load(file_name='../data/models/YoonKimCNN_example')


	# Make predictions
	if INFERENCE:
		print("=== Making Inference")
		data_infer = ['I thought the movie was terrible', 'Very fun and exciting']

		print("\t--- Tokenizing raw inference dataset")		
		print(vars(tokenizer).keys())
		ds_infer = tokenizer.encode_texts(data_infer)
		ds_decode = tokenizer.decode_texts(ds_infer)
		print(ds_infer)
		print(ds_decode)
		print(type(factory))

		# Convert words to vector embeddings
		ds_embedding = []
		for i, sentence in enumerate(ds_decode):
			print('\t\tWorking on sentence %d' % (i + 1))
			sentence_embedding = []
			for j, word in enumerate(sentence):
				word_embedding = factory.embeddings_index[word.encode()]
				if word_embedding is not None:
					sentence_embedding.append(word_embedding)

			ds_embedding.append(sentence_embedding)

		print("\t--- Feeding tokenized text to model")
		pred = model.fit(x=ds_embedding)
		print(pred)

def train_process():


	MAX_TOKENS = 100
	MAX_SENTS  = 100

	# Call data
	data = CreateH5(os.path.join(data_path, 'example'), 'example.h5', True)


	# Build tokenizer with entire vocabulary
	df_all = data.get_h5('all')
	tokenizer = WordTokenizer(lang='en_core_web_sm')
	tokenizer.build_vocab(list(df_all.text))


	# Make dataset
	ds = Dataset(inputs=list(df_all.text), labels=list(df_all.label), tokenizer=tokenizer)
	ds.update_test_indices(test_size=0.1)
	# ds.save('dataset')

	# Tokenizer
	# tokenizer = build_tokenizer(data)

	# RNN models can use `max_tokens=None` to indicate variable length words per mini-batch.
	factory = TokenModelFactory(1, ds.tokenizer.token_index, max_tokens=MAX_TOKENS, embedding_type='glove.6B.100d')
	word_encoder_model = YoonKimCNN()
	model = factory.build_model(token_encoder_model=word_encoder_model)
	model.compile(optimizer='adam', loss='categorical_crossentropy')
	model.summary()

	# Train

	x_train = ds.tokenizer.encode_texts(ds.X.tolist())
	# x_train = np.array([np.array(x) for x in x_train])
	# x_train = tokenizer.decode_texts(x_train)

	print(x_train)
	print(type(x_train))

	x_train = sentence_to_token(x_train, MAX_TOKENS)
	# x_train = doc_to_token(x_train, factory, MAX_TOKENS)

	# Convert from array to categorical
	y_train = to_categorical(ds.y)

	print(x_train)
	print(y_train)
	print(type(x_train))
	print(type(y_train))

	# Fit
	model.fit(x=x_train, y=y_train,
		epochs=25,
		batch_size=256)

	# model.fit(x=list(df_all.text), y=)


if __name__ == '__main__':

	train_process()
