
# load the Corpus
from flair.data import Corpus
columns = {0 : 'test', 1 : 'ner'}
data_folder = '/path_to_data'
from pathlib import Path
from flair.data import Corpus
from flair.datasets import ColumnCorpus
corpus : Corpus = ColumnCorpus(data_folder, columns, train_file = 'train.tsv', test_file = 'test.tsv', dev_file = 'devel.tsv')

# making tag dictionary from Corpus
tag_type = 'ner'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary)
print(tag_dictionary.idx2item)

#initialize embeddings
from flair.embeddings import WordEmbeddings
from flair.embeddings import FlairEmbeddings
pubmed_forward_embedding = FlairEmbeddings('pubmed-forward')
pubmed_backward_embedding = FlairEmbeddings('pubmed-backward')
word_embedding = WordEmbeddings('glove')
embedding_types = [
    WordEmbeddings('glove'),
    FlairEmbeddings('pubmed-forward'),
    FlairEmbeddings('pubmed-backward')
]
from flair.embeddings import StackedEmbeddings
embeddings : StackedEmbeddings = StackedEmbeddings(embeddings = embedding_types)

#initialize sequence tagger
from flair.models import SequenceTagger
tagger : SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

# training
from flair.trainers import ModelTrainer
from flair.training_utils import EvaluationMetric
trainer : ModelTrainer = ModelTrainer(tagger,corpus)
trainer.train('/path_to_save_files',
             learning_rate=0.1,
             mini_batch_size=32,
             max_epochs=20,
             embeddings_storage_mode='gpu',
             patience=3)
