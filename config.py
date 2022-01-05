# -*- coding: utf-8 -*-
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn
# 默认参数配置: 尽量不要出现同名参数

import argparse

from copy import deepcopy

class BaseConfig:
	"""基础的全局配置"""
	parser = argparse.ArgumentParser('--')
	parser.add_argument('--filter_stopword', default=True, type=bool, help='是否过滤停用词')
	parser.add_argument('--retrieval_model_name', default='tfidf', type=str, help='最终使用的文档检索模型')
	parser.add_argument('--num_top_subject', default=3, type=int, help='每道题目对应的候选法律门类数量')
	parser.add_argument('--use_reference', default=True, type=bool, help='是否使用参考书目文档')

	parser.add_argument('--train_batch_size', default=32, type=int, help='训练集批训练的批大小')
	parser.add_argument('--valid_batch_size', default=32, type=int, help='验证集批训练的批大小')
	parser.add_argument('--test_batch_size', default=32, type=int, help='测试集批训练的批大小')
	
	parser.add_argument('--max_reference_length', default=512, type=int, help='参考书目段落分词序列最大长度, 512超过参考书目文档段落分词长度的0.998分位数')
	parser.add_argument('--max_statement_length', default=256, type=int, help='题目题干分词序列最大长度')
	parser.add_argument('--max_option_length', default=128, type=int, help='题目选项分词序列最大长度')
	
	parser.add_argument('--word_embedding', default=None, type=str, help='使用的词嵌入, 默认值None表示只用token2id的顺序编码值, 目前可用的值包括word2vec, fasttext, bert')
	parser.add_argument('--document_embedding', default=None, type=str, help='使用的文档嵌入, 2021/12/27 13:55:42新增, 因为我发现调用BERT模型的话是不需要分词, 直接输入句子即可, 可以视为对句子(或文档)进行的嵌入, 而且Gensim里也有doc2vec的模型')

	# 模型训练的配置
	parser.add_argument('--num_epoch', default=32, type=int, help='训练轮数')
	parser.add_argument('--lr_multiplier', default=.95, type=float, help='lr_scheduler的gamma参数值, 即学习率的衰减')
	parser.add_argument('--learning_rate', default=.01, type=float, help='学习率或步长')
	parser.add_argument('--weight_decay', default=.0, type=float, help='权重衰减')
	
	parser.add_argument('--do_valid', default=True, type=bool, help='是否在训练过程中进行模型验证')
	parser.add_argument('--do_valid_plot', default=True, type=bool, help='是否在验证过程中绘制图像, 目前指对于判断题模型进行ROC曲线和PR曲线的绘制')


	parser.add_argument('--num_best', default=32, type=int, help='Similarity的num_best参数值, 是目前引用的参考文献数目')


class DatasetConfig:
	"""数据集相关配置"""
	parser = deepcopy(BaseConfig.parser)

	parser.add_argument('--num_workers', default=0, type=int, help='DataLoader的num_workers参数值')
	

class RetrievalModelConfig:
	"""文档检索模型相关配置"""
	parser = deepcopy(BaseConfig.parser)
	
	parser.add_argument('--smartirs_tfidf', default=None, type=int, help='从{btnaldL}{xnftp}{xncub}的组合中挑选, 默认值为nfc, 详见https://radimrehurek.com/gensim/models/tfidfmodel.html')
	parser.add_argument('--pivot_tfidf', default=None, type=float, help='针对长文档进行的枢轴修正参数, 见https://radimrehurek.com/gensim/models/tfidfmodel.html')
	parser.add_argument('--slope_tfidf', default=.25, type=float, help='针对长文档进行的枢轴修正参数, 见https://radimrehurek.com/gensim/models/tfidfmodel.html')
		
	parser.add_argument('--num_topics_lsi', default=200, type=int, help='LSI模型的num_topics参数值')
	parser.add_argument('--decay_lsi', default=1., type=float, help='LSI模型的decay参数值')
	parser.add_argument('--power_iters_lsi', default=2, type=int, help='LSI模型的power_iters参数值')
	parser.add_argument('--extra_samples_lsi', default=100, type=int, help='LSI模型的extra_samples参数值')
	
	parser.add_argument('--num_topics_lda', default=100, type=int, help='LDA模型的num_topics参数值')
	parser.add_argument('--decay_lda', default=.5, type=float, help='LDA模型的decay参数值')
	parser.add_argument('--iterations_lda', default=50, type=float, help='LDA模型的iterations参数值')
	parser.add_argument('--gamma_threshold_lda', default=.001, type=float, help='LDA模型的gamma_threshold参数值')
	parser.add_argument('--minimum_probability_lda', default=.01, type=float, help='LDA模型的minimum_probability参数值')
			
	parser.add_argument('--kappa_hdp', default=1., type=float, help='HDP模型的kappa参数值')
	parser.add_argument('--tau_hdp', default=64., type=float, help='HDP模型的tau参数值')
	parser.add_argument('--K_hdp', default=15, type=int, help='HDP模型的K参数值')
	parser.add_argument('--T_hdp', default=150, type=int, help='HDP模型的T参数值')


class EmbeddingModelConfig:
	"""词嵌入模型相关配置"""
	parser = deepcopy(BaseConfig.parser)
	parser.add_argument('--size_word2vec', default=256, type=int, help='gensim嵌入模型Word2Vec的嵌入维数, 即Word2Vec模型的size参数')
	parser.add_argument('--min_count_word2vec', default=5, type=int, help='Word2Vec模型的min_count参数')
	parser.add_argument('--window_word2vec', default=5, type=int, help='Word2Vec模型的window参数')
	parser.add_argument('--workers_word2vec', default=3, type=int, help='Word2Vec模型的workers参数')

	parser.add_argument('--size_fasttext', default=256, type=int, help='gensim嵌入模型FastText的嵌入维数, 即FastText模型的size参数')
	parser.add_argument('--min_count_fasttext', default=5, type=int, help='FastText模型的min_count参数')
	parser.add_argument('--window_fasttext', default=5, type=int, help='FastText模型的window参数')
	parser.add_argument('--workers_fasttext', default=3, type=int, help='FastText模型的workers参数')
	
	parser.add_argument('--size_doc2vec', default=512, type=int, help='gensim嵌入模型Doc2Vec的嵌入维数, 即FastText模型的vector_size参数(size参数即将被弃用)')
	parser.add_argument('--min_count_doc2vec', default=5, type=int, help='Doc2Vec模型的min_count参数')
	parser.add_argument('--window_doc2vec', default=5, type=int, help='Doc2Vec模型的window参数')
	parser.add_argument('--workers_doc2vec', default=3, type=int, help='Doc2Vec模型的workers参数')
	
	
	parser.add_argument('--bert_output', default='pooler_output', type=str, help='BERT模型使用的输出, 默认pooler_output即池化后的输出结果, 也可以使用last_hidden_output, 会比pooler多一个维度')

class QAModelConfig:
	"""问答模型相关配置"""
	parser = deepcopy(BaseConfig.parser)
	
	parser.add_argument('--test_thresholds', default=[.4, .5, .6], type=list, help='判断题测试的阈值')




if __name__ == '__main__':
	config = BaseConfig()
	parser = config.parser
	args = parser.parse_args()
	print('num_best' in args)
