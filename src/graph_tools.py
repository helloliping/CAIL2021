# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn
# 图模型相关工具函数

if __name__ == '__main__':
	import sys
	sys.path.append('../')	


import os
import time
import json

from setting import *

from src.graph_tools import *


from src.data_tools import load_stopwords, filter_stopwords
from src.utils import load_args, timer


# 2022/01/02 20:14:05 绘制不同法律门类的词云, 去除停用词, 保存到TEMP_DIR下
@timer
def plot_reference_wordcloud():
	reference_dataframe = pandas.read_csv(REFERENCE_PATH, sep='\t', header=0)
	stopwords = load_stopwords(stopword_names=None)
	for law, group_dataframe in reference_dataframe.groupby(['law']):
		group_dataframe = group_dataframe.reset_index(drop=True)
		contents = []
		for i in range(group_dataframe.shape[0]):
			content = ' '.join(filter_stopwords(tokens=eval(group_dataframe.loc[i, 'content']), stopwords=stopwords))
			contents.append(content)
		text = '\n'.join(contents)
		wordcloud = WordCloud().generate(text=text)
		wordcloud.to_file(os.path.join(TEMP_DIR, f'{law}.png'))
