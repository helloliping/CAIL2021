# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn
# 模型训练

import os
import time
import json
import torch
import numpy
import pandas
import logging

from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, BCELoss
from torch.optim import Adam, lr_scheduler
from sklearn.metrics import auc, confusion_matrix, accuracy_score, roc_auc_score

from setting import *
from config import QAModelConfig, DatasetConfig

from src.dataset import generate_dataloader
from src.evaluation_tools import evaluate_qa_model_choice, evaluate_qa_model_judgment, evaluate_classifier
from src.plot_tools import plot_roc_curve, plot_pr_curve
from src.qa_model import BaseChoiceModel, BaseJudgmentModel, ReferenceChoiceModel, ReferenceJudgmentModel
from src.torch_tools import save_checkpoint
from src.utils import initialize_logger, terminate_logger, load_args, save_args



# 选择题模型训练
def train_choice_model(mode, model_name, **kwargs):
	assert mode in ['train', 'train_kd', 'train_ca']
	logger = initialize_logger(filename=os.path.join(LOGGING_DIR, f'{time.strftime("%Y%m%d")}_{model_name}_{mode}'), filemode='w')
	logger.info(f'args: {kwargs}')
	
	# 配置模型
	args_1 = load_args(Config=QAModelConfig)							# 问答模型配置
	# 根据kwargs调整问答模型配置的参数
	for key, value in kwargs.items():
		args_1.__setattr__(key, value)
	model = eval(model_name)(args=args_1).to(DEVICE)					# 构建模型
	
	
	# 配置训练集
	args_2 = load_args(Config=DatasetConfig)
	# 根据kwargs调整问答模型配置的参数
	for key, value in kwargs.items():
		args_2.__setattr__(key, value)																																
	train_dataloader = generate_dataloader(args=args_2, mode=mode, do_export=False, pipeline='choice', for_test=False)	

	# 提取训练参数值
	num_epoch = args_1.num_epoch																				
	learning_rate = args_1.learning_rate
	lr_multiplier = args_1.lr_multiplier
	weight_decay = args_1.weight_decay

	# 配置验证集: do_valid为True时生效
	if args_2.do_valid:
		valid_dataloader = generate_dataloader(args=args_2, mode=mode.replace('train', 'valid'), do_export=False, pipeline='choice', for_test=False)	
		valid_logging = {
			'epoch'			: [],
			'accuracy'		: [],
			'strict_score'	: [],
			'loose_score'	: [],
		}

	loss_function = CrossEntropyLoss()													# 构建损失函数
	optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)	# 构建优化器
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_multiplier)	# 构建学习率规划期

	# 模型训练
	train_logging = {
		'epoch'			: [],
		'iteration'		: [],
		'loss'			: [],
		'accuracy'		: [],
		'strict_score'	: [],
		'loose_score'	: [],
	}
	for epoch in range(num_epoch):
		model.train()
		for iteration, data in enumerate(train_dataloader):
			for key in data.keys():
				if isinstance(data[key], torch.Tensor):
					data[key] = Variable(data[key]).to(DEVICE)
			optimizer.zero_grad()
			output = model(data)
			evaluation_summary = evaluate_qa_model_choice(input=data, output=output, mode='train')
			strict_score = evaluation_summary['strict_score']
			loose_score = evaluation_summary['loose_score']
			accuracy = evaluation_summary['accuracy']
			loss = loss_function(output, data['label_choice'])
			loss.backward()
			optimizer.step()
			
			logging.info(f'train | epoch: {epoch} - iteration - {iteration} - loss: {loss.item()} - accuracy: {accuracy} - score: {strict_score, loose_score}')
		
			train_logging['epoch'].append(epoch)
			train_logging['iteration'].append(iteration)
			train_logging['loss'].append(loss.item())
			train_logging['accuracy'].append(accuracy)
			train_logging['strict_score'].append(strict_score)
			train_logging['loose_score'].append(loose_score)
			
		exp_lr_scheduler.step()
		
		# 验证集评估
		if args_1.do_valid:
			model.eval()
			accuracys = []
			strict_scores = []
			loose_scores = []
			total_size = 0
			with torch.no_grad():
				for data in valid_dataloader:
					_batch_size = len(data['id'])
					total_size += _batch_size
					for key in data.keys():
						if isinstance(data[key], torch.Tensor):
							data[key] = data[key].to(DEVICE)
					output = model(data)
					evaluation_summary = evaluate_qa_model_choice(input=data, output=output, mode='valid')
					strict_score = evaluation_summary['strict_score']
					loose_score = evaluation_summary['loose_score']
					accuracy = evaluation_summary['accuracy']
					accuracys.append(accuracy * _batch_size)
					strict_scores.append(strict_score * _batch_size)
					loose_scores.append(loose_score * _batch_size)
			mean_accuracy = numpy.sum(accuracys) / total_size
			mean_strict_score = numpy.sum(strict_scores) / total_size
			mean_loose_score = numpy.sum(loose_scores) / total_size
			valid_logging['epoch'].append(epoch)
			valid_logging['accuracy'].append(mean_accuracy)
			valid_logging['strict_score'].append(mean_strict_score)
			valid_logging['loose_score'].append(mean_loose_score)
			logging.info(f'valid | epoch: {epoch} - accuracy: {mean_accuracy} - score: {mean_strict_score, mean_loose_score}')
			
		# 保存模型
		save_checkpoint(model=model, save_path=os.path.join(CHECKPOINT_DIR, f'{model_name}_{mode}_{epoch}.h5'), optimizer=optimizer, scheduler=scheduler)
	
		# 2021/12/20 22:20:40 每个epoch结束都记录一下结果
		train_logging_dataframe = pandas.DataFrame(train_logging, columns=list(train_logging.keys()))
		train_logging_dataframe.to_csv(os.path.join(TEMP_DIR, f'{model_name}_{mode}.csv'), header=True, index=False, sep='\t')
		if args_1.do_valid:
			valid_logging_dataframe = pandas.DataFrame(valid_logging, columns=list(valid_logging.keys()))
			valid_logging_dataframe.to_csv(os.path.join(TEMP_DIR, f'{model_name}_{mode.replace("train", "valid")}.csv'), header=True, index=False, sep='\t')
	
	terminate_logger(logger=logger)
	
	train_logging_dataframe = pandas.DataFrame(train_logging, columns=list(train_logging.keys()))
	train_logging_dataframe.to_csv(os.path.join(TEMP_DIR, f'{model_name}_{mode}.csv'), header=True, index=False, sep='\t')
	if args_1.do_valid:
		valid_logging_dataframe = pandas.DataFrame(valid_logging, columns=list(valid_logging.keys()))
		valid_logging_dataframe.to_csv(os.path.join(TEMP_DIR, f'{model_name}_{mode.replace("train", "valid")}.csv'), header=True, index=False, sep='\t')
		return train_logging_dataframe, valid_logging_dataframe
	return train_logging_dataframe


# 判断题模型训练
def train_judgment_model(mode, model_name, **kwargs):
	assert mode in ['train', 'train_kd', 'train_ca']
	logger = initialize_logger(filename=os.path.join(LOGGING_DIR, f'{time.strftime("%Y%m%d")}_{model_name}_{mode}'), filemode='w')
	logger.info(f'args: {kwargs}')
	
	# 配置模型
	args_1 = load_args(Config=QAModelConfig)							# 问答模型配置
	# 根据kwargs调整问答模型配置的参数
	for key, value in kwargs.items():
		args_1.__setattr__(key, value)
	model = eval(model_name)(args=args_1).to(DEVICE)					# 构建模型
	
	# 配置训练集
	args_2 = load_args(Config=DatasetConfig)
	# 根据kwargs调整问答模型配置的参数
	for key, value in kwargs.items():
		args_2.__setattr__(key, value)																																
	train_dataloader = generate_dataloader(args=args_2, mode=mode, do_export=False, pipeline='judgment', for_test=False)	
	
	# 提取训练参数值
	num_epoch = args_1.num_epoch																				
	learning_rate = args_1.learning_rate
	lr_multiplier = args_1.lr_multiplier
	weight_decay = args_1.weight_decay
	test_thresholds = args_1.test_thresholds
	do_valid_plot = args_1.do_valid_plot

	# 配置验证集: do_valid为True时生效
	if args_2.do_valid:
		valid_dataloader = generate_dataloader(args=args_2, mode=mode.replace('train', 'valid'), do_export=False, pipeline='judgment', for_test=False)	
		valid_logging = {
			'epoch'	: [],
			'auc'	: [],
		}
		# 记录每个测试阈值的精确度情况
		for threshold in test_thresholds:
			valid_logging[f'accuracy{threshold}'] = []
		valid_target = valid_dataloader.dataset.data['label_judgment'].values			# 验证集标签全集, 用于进行AUC等评估

	loss_function = BCELoss()															# 构建损失函数: 二分类交叉熵
	optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)	# 构建优化器
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_multiplier)	# 构建学习率规划期

	# 模型训练
	train_logging = {
		'epoch'			: [],
		'iteration'		: [],
		'loss'			: [],
	}
	for threshold in test_thresholds:
		train_logging[f'accuracy{threshold}'] = []
	
	
	# for iteration, data in enumerate(train_dataloader):
		# print(data['reference'].shape)
	
	
	for epoch in range(num_epoch):
		model.train()
		for iteration, data in enumerate(train_dataloader):
			for key in data.keys():
				if isinstance(data[key], torch.Tensor):
					data[key] = Variable(data[key]).to(DEVICE)
			optimizer.zero_grad()
			output = model(data)
			evaluation_summary = evaluate_qa_model_judgment(input=data, output=output, mode='train', thresholds=test_thresholds)
			accuracy = evaluation_summary['accuracy']
			loss = loss_function(output, data['label_judgment'].float())# BCELoss或MSELoss要求两个输入都是浮点数
			loss.backward()
			optimizer.step()
			
			logging.info(f'train | epoch: {epoch} - iteration - {iteration} - loss: {loss.item()} - accuracy: {accuracy}')
			
			# 记录模型训练情况
			train_logging['epoch'].append(epoch)
			train_logging['iteration'].append(iteration)
			train_logging['loss'].append(loss.item())
			for threshold in test_thresholds:
				train_logging[f'accuracy{threshold}'].append(accuracy[threshold])

		exp_lr_scheduler.step()
		
		# 验证集评估
		if args_1.do_valid:
			model.eval()
			accuracys = {threshold: [] for threshold in test_thresholds}
			total_size = 0
			valid_predict_probas = []									# 存放模型预测输出的验证集概率值
			with torch.no_grad():
				for data in valid_dataloader:
					_batch_size = len(data['id'])
					total_size += _batch_size
					for key in data.keys():
						if isinstance(data[key], torch.Tensor):
							data[key] = data[key].to(DEVICE)
					output = model(data)
					valid_predict_probas.append(output)
					evaluation_summary = evaluate_qa_model_judgment(input=data, output=output, mode='valid', thresholds=test_thresholds)
					accuracy = evaluation_summary['accuracy']
					for threshold in test_thresholds:
						accuracys[threshold].append(accuracy[threshold] * _batch_size)

			# 计算AUC值
			valid_predict_proba = torch.cat(valid_predict_probas).cpu().numpy()		# 必须先转到CPU上才能转换为numpy数组
			auc = roc_auc_score(valid_target, valid_predict_proba)
			valid_logging['auc'].append(auc)			
			
			# 计算每个测试阈值下的精确度
			valid_logging['epoch'].append(epoch)
			threshold2accuracy = {}
			for threshold in test_thresholds:
				mean_accuracy = numpy.sum(accuracys[threshold]) / total_size
				threshold2accuracy[threshold] = mean_accuracy
				valid_logging[f'accuracy{threshold}'].append(mean_accuracy)

			logging.info(f'valid | epoch: {epoch} - accuracy: {threshold2accuracy} - AUC: {auc}')
			
			# 绘制ROC曲线与PR曲线
			if do_valid_plot:
				plot_roc_curve(target=valid_target, 
							   predict_proba=valid_predict_proba,
							   title=f'ROC Curve of {model_name} in Epoch {epoch} and Mode {mode}', 
							   export_path=os.path.join(IMAGE_DIR, f'roc_{model_name}_{mode}_{epoch}.png'))
				plot_pr_curve(target=valid_target, 
							  predict_proba=valid_predict_proba, 
							  title=f'PR Curve of {model_name} in Epoch {epoch} and Mode {mode}', 
							  export_path=os.path.join(IMAGE_DIR, f'pr_{model_name}_{mode}_{epoch}.png'))
		
		# 保存模型
		save_checkpoint(model=model, save_path=os.path.join(CHECKPOINT_DIR, f'{model_name}_{mode}_{epoch}.h5'), optimizer=optimizer, scheduler=scheduler)
		
		# 2021/12/20 22:20:40 每个epoch结束都记录一下结果
		train_logging_dataframe = pandas.DataFrame(train_logging, columns=list(train_logging.keys()))
		train_logging_dataframe.to_csv(os.path.join(TEMP_DIR, f'{model_name}_{mode}.csv'), header=True, index=False, sep='\t')
		if args_1.do_valid:
			valid_logging_dataframe = pandas.DataFrame(valid_logging, columns=list(valid_logging.keys()))
			valid_logging_dataframe.to_csv(os.path.join(TEMP_DIR, f'{model_name}_{mode.replace("train", "valid")}.csv'), header=True, index=False, sep='\t')
	
	terminate_logger(logger=logger)	# 终止日志
	
	train_logging_dataframe = pandas.DataFrame(train_logging, columns=list(train_logging.keys()))
	train_logging_dataframe.to_csv(os.path.join(TEMP_DIR, f'{model_name}_{mode}.csv'), header=True, index=False, sep='\t')
	if args_1.do_valid:
		valid_logging_dataframe = pandas.DataFrame(valid_logging, columns=list(valid_logging.keys()))
		valid_logging_dataframe.to_csv(os.path.join(TEMP_DIR, f'{model_name}_{mode.replace("train", "valid")}.csv'), header=True, index=False, sep='\t')
		return train_logging_dataframe, valid_logging_dataframe
		
	return train_logging_dataframe


# 选择题模型训练脚本
def run_choice():
	# kwargs = {
		# 'num_epoch'			: 32, 
		# 'train_batch_size'	: 32, 
		# 'valid_batch_size'	: 32, 
		# 'use_reference'		: False,
		# 'word_embedding'	: None,
		# 'document_embedding': None,
		# 'num_best'			: 32,
	# }
	# train_choice_model(mode='train_kd', model_name='BaseChoiceModel', **kwargs)
	# train_choice_model(mode='train_ca', model_name='BaseChoiceModel', **kwargs)
	
	kwargs = {
		'num_epoch'			: 32, 
		'train_batch_size'	: 32, 
		'valid_batch_size'	: 32, 
		'use_reference'		: True,
		'word_embedding'	: None,
		'document_embedding': None,
		'num_best'			: 32,
	}
	train_choice_model(mode='train_kd', model_name='ReferenceChoiceModel', **kwargs)
	train_choice_model(mode='train_ca', model_name='ReferenceChoiceModel', **kwargs)

# 判断题模型训练脚本
def run_judgment():
	# kwargs = {
		# 'num_epoch'			: 32, 
		# 'train_batch_size'	: 128, 
		# 'valid_batch_size'	: 128, 
		# 'use_reference'		: False,
		# 'word_embedding'		: None,
		# 'document_embedding'	: None,
		# 'num_best'			: 32,
	# }
	
	# train_judgment_model(mode='train_kd', model_name='BaseJudgmentModel', **kwargs)
	# train_judgment_model(mode='train_ca', model_name='BaseJudgmentModel', **kwargs)
	
	kwargs = {
		'num_epoch'			: 32, 
		'train_batch_size'	: 32, 
		'valid_batch_size'	: 128, 
		'use_reference'		: True,
		'word_embedding'	: None,
		'document_embedding': None,
		'num_best'			: 32,
	}
	# train_judgment_model(mode='train_kd', model_name='ReferenceJudgmentModel', **kwargs)
	train_judgment_model(mode='train_ca', model_name='ReferenceJudgmentModel', **kwargs)
	

if __name__ == '__main__':
	# run_choice()
	run_judgment()
	
