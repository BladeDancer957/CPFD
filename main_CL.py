import torch
import numpy as np
from tqdm import tqdm
import random
import scipy
from copy import deepcopy

from src.utils import *
from src.dataloader import *
from src.trainer import *
from src.model import *
from src.config import *

import time


def main_cl(params):
    # ===========================================================================
    # Using Fixed Random Seed
    if params.seed: # 固定随机种子 
        random.seed(params.seed)
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)
        torch.cuda.manual_seed(params.seed)
        torch.backends.cudnn.deterministic = True
        
    # Initialize Experiment
    logger = init_experiment(params, logger_filename=params.logger_filename) # 保存日志
    logger.info(params.__dict__) # 打印当前配置

    # Set domain name
    domain_name = os.path.basename(params.data_path[0]) # 数据集名字
    if domain_name=='':
        # Remove the final char '\' in the path
        domain_name = os.path.basename(params.data_path[0][:-1])

    # Generate Dataloader 
    ner_dataloader = NER_dataloader(data_path=params.data_path,
                                    domain_name=domain_name,
                                    batch_size=params.batch_size, 
                                    entity_list=params.entity_list,
                                    n_samples=params.n_samples,
                                    is_filter_O=params.is_filter_O,
                                    schema=params.schema,
                                    is_load_disjoin_train=params.is_load_disjoin_train)

    label_list = ner_dataloader.label_list # 标签列表
    entity_list = ner_dataloader.entity_list # 实体列表
    num_classes_all = len(ner_dataloader.entity_list) # 实体/类别 数量
    pad_token_id = ner_dataloader.auto_tokenizer.pad_token_id # 填充token 对应的id = 0
    class_per_entity = len(params.schema)-1 # 2  每个实体对应的label 数量 B- I-
    
    # Initialize the model for the first group of classes
    if params.model_name in ['bert-base-cased','roberta-base','bert-base-chinese']:
        # BERT-based NER Tagger
        model = BertTagger(output_dim=(1+class_per_entity*params.nb_class_fg), params=params)
    else:
        raise Exception('model name %s is invalid'%params.model_name)

    model.cuda()
    trainer = BaseTrainer(params, model, label_list)
    trainer.pad_token_id = pad_token_id # 填充token 的id=0

    # ===========================================================================
    # Start training
    total_iter = int((num_classes_all-params.nb_class_fg)/params.nb_class_pg)+1 # 任务或step的数量
    assert (num_classes_all-params.nb_class_fg)%params.nb_class_pg==0, "Invalid class number!"

    trainer.inital_nb_classes = 1+class_per_entity*params.nb_class_fg
    trainer.nb_classes = num_classes_all*class_per_entity + 1
    trainer.classes = []

    for iteration in range(total_iter): # 遍历每个任务

        logger.info("=========================================================")   
        logger.info("Beggin training the %d-th iter (total %d iters)"%(iteration+1, 
                                                                        total_iter))     
        logger.info("=========================================================")
        
        best_model_ckpt_name = "best_finetune_domain_%s_iteration_%d.pth"%(
                                domain_name, 
                                iteration) # 当前数据集 各个任务上 最好的模型名字 (基于当前任务的开发集进行选择)
        best_model_ckpt_path = os.path.join(
            params.dump_path, 
            best_model_ckpt_name
        ) # ./experiments/exp_name/exp_id/best_model_ckpt_name

        if params.is_load_common_first_model: # True
            # 相同setting下 base model 只需要训练一次，其余直接加载 
            common_first_model_ckpt_name = "best_finetune_domain_%s_iteration_%d_fg_%d.pth"%(
                                    domain_name, 
                                    iteration,
                                    params.nb_class_fg)
            common_first_model_ckpth_path = os.path.join(
                os.path.dirname(os.path.dirname(params.dump_path)),
                common_first_model_ckpt_name
            ) 

        # Initialize a new model
        if params.is_from_scratch or iteration == 0:
            # Initialize the model for the first group of classes
            if params.model_name in ['bert-base-cased','roberta-base','bert-base-chinese']:
                # BERT-based NER Tagger 输出层动态变化
                model = BertTagger(output_dim=(1+class_per_entity*(params.nb_class_fg+iteration*params.nb_class_pg)), params=params)
            else:
                raise Exception('model name %s is invalid'%params.model_name)
            trainer.model = model
            trainer.model.cuda()

            trainer.refer_model = None
            hidden_dim = trainer.model.classifier.hidden_dim # 768
            output_dim = trainer.model.classifier.output_dim # curr_entity*2 + 1 动态变化
            logger.info("hidden_dim=%d, output_dim=%d"%(hidden_dim,output_dim))

        # Update the architecture of the classifier
        elif iteration == 1:
            trainer.refer_model = deepcopy(trainer.model) # old model
            trainer.refer_model.eval()
            # Change model classifier
            hidden_dim = trainer.model.classifier.hidden_dim
            output_dim = trainer.model.classifier.output_dim
            logger.info("hidden_dim=%d, old_output_dim=%d, new_output_dim=%d"%(
                                        hidden_dim,
                                        output_dim,
                                        class_per_entity*params.nb_class_pg))
            new_fc = SplitCosineLinear(hidden_dim, output_dim, class_per_entity*params.nb_class_pg)

            new_fc.fc0.weight.data = trainer.model.classifier.weight.data[:1] # for O class
            new_fc.fc1.weight.data = trainer.model.classifier.weight.data[1:] # for old class
            new_fc.sigma.data = trainer.model.classifier.sigma.data

            trainer.model.classifier = new_fc
            trainer.model.cuda()

        else:
            trainer.refer_model = deepcopy(trainer.model) # old model
            trainer.refer_model.eval()
            # Change model classifier
            hidden_dim = trainer.model.classifier.hidden_dim
            output_dim1 = trainer.model.classifier.fc1.output_dim
            output_dim2 = trainer.model.classifier.fc2.output_dim
            logger.info("hidden_dim=%d, old_output_dim=%d, new_output_dim=%d"%(
                                                            hidden_dim,
                                                            1+output_dim1+output_dim2,
                                                            class_per_entity*params.nb_class_pg))                                                
            new_fc = SplitCosineLinear(hidden_dim, 1+output_dim1+output_dim2, class_per_entity*params.nb_class_pg)

            new_fc.fc0.weight.data = trainer.model.classifier.fc0.weight.data # for O classes
            new_fc.fc1.weight.data[:output_dim1] = trainer.model.classifier.fc1.weight.data
            new_fc.fc1.weight.data[output_dim1:] = trainer.model.classifier.fc2.weight.data
            new_fc.sigma.data = trainer.model.classifier.sigma.data

            trainer.model.classifier = new_fc
            trainer.model.cuda()

        # Update entity list and label list
        if iteration==0:
            new_entity_list = ner_dataloader.entity_list[:params.nb_class_fg] # 当前任务的实体集合
            all_seen_entity_list = ner_dataloader.entity_list[:params.nb_class_fg] # 截至当前任务的实体集合
        else:
            new_entity_list = ner_dataloader.entity_list[\
                                params.nb_class_fg+(iteration-1)*params.nb_class_pg
                                :params.nb_class_fg+iteration*params.nb_class_pg] # 当前任务的实体集合
            all_seen_entity_list = ner_dataloader.entity_list[\
                                :params.nb_class_fg+iteration*params.nb_class_pg] # 截至当前任务的实体集合

        num_classes_new = 1+class_per_entity*len(all_seen_entity_list) # 截至当前任务的标签数量

        if iteration>0:
            num_classes_old = num_classes_new - class_per_entity*len(new_entity_list) #旧任务的标签数量
            trainer.old_classes = num_classes_old
            trainer.nb_new_classes = class_per_entity*len(new_entity_list) 
            trainer.nb_current_classes = num_classes_new
            trainer.classes.append(trainer.nb_new_classes)
        else:
            num_classes_old = 0 #第一个任务 旧标签数量为0
            trainer.old_classes = num_classes_old
            trainer.nb_new_classes = class_per_entity*len(new_entity_list) + 1
            trainer.nb_current_classes = num_classes_new
            trainer.classes.append(trainer.nb_new_classes)


        new_classes_list = list(range(num_classes_old,num_classes_new)) # 当前任务的标签集合
        logger.info("All seen entity types = %s"%str(all_seen_entity_list))
        logger.info("New entity types = %s"%str(new_entity_list))
        
        # Prepare data 当前任务下的训练集 和 开发集
        dataloader_train, dataloader_dev = ner_dataloader.get_dataloader(
                                                            first_N_classes=-1,
                                                            select_entity_list=new_entity_list,
                                                            phase=['train','dev'],
                                                            is_filter_O=params.is_filter_O,
                                                            reserved_ratio=params.reserved_ratio)
        # for debug 截至当前任务的开发集和测试集
        dataloader_dev_cumul, dataloader_test_cumul = ner_dataloader.get_dataloader(
                                                            first_N_classes=len(all_seen_entity_list),
                                                            select_entity_list=[],
                                                            phase=['dev','test'],
                                                            is_filter_O=False)


        if iteration==0: # 第一个任务
            # build scheduler and optimizer
            trainer.optimizer = torch.optim.SGD(trainer.model.parameters(),
                                            lr=trainer.lr,
                                            momentum=trainer.mu,
                                            weight_decay=trainer.weight_decay)

            trainer.scheduler = torch.optim.lr_scheduler.MultiStepLR(trainer.optimizer,
                                                                milestones=eval(params.schedule),
                                                                gamma=params.gamma)  

        else:
            # iteration>0
            # Update optimizer and scheduler: Fix the embedding of old classes
            if params.is_fix_trained_classifier: # 固定已经训练好的分类器
                # if fix the O classifier
                if params.is_unfix_O_classifier: # False
                    ignored_params = list(map(id, trainer.model.classifier.fc1.parameters()))
                    base_params = filter(lambda p: id(p) not in ignored_params, \
                                trainer.model.parameters())
                    tg_params =[{'params': base_params, 'lr': float(params.stable_lr),
                                'weight_decay': float(params.weight_decay)}, \
                                {'params': trainer.model.classifier.fc1.parameters(), 'lr': 0., 
                                'weight_decay': 0.}]
                else:
                    ignored_params = list(map(id, trainer.model.classifier.fc1.parameters())) + \
                                    list(map(id, trainer.model.classifier.fc0.parameters()))
                    base_params = filter(lambda p: id(p) not in ignored_params, \
                                trainer.model.parameters())
                    tg_params =[{'params': base_params, 'lr': float(params.stable_lr),
                                'weight_decay': float(params.weight_decay)}, \
                                {'params': trainer.model.classifier.fc0.parameters(), 'lr': 0., 
                                'weight_decay': 0.}, \
                                {'params': trainer.model.classifier.fc1.parameters(), 'lr': 0., 
                                'weight_decay': 0.}]
            else:
                tg_params = [{'params': trainer.model.parameters(), 'lr': float(params.stable_lr), 
                            'weight_decay': float(params.weight_decay)}]
            trainer.optimizer = torch.optim.SGD(tg_params, 
                                                momentum=params.mu)
            # last_epoch_or_step = last_global_step if params.is_train_by_steps \
            #                                     else last_global_epoch
            trainer.scheduler = None

        # Scaling the weights in the new classifier(imprint)
        if iteration>0 and params.is_rescale_new_weight and (not params.is_from_scratch):   # True
            # (1) compute the average norm of old embdding
            old_embedding_norm = trainer.model.classifier.fc1.weight.data.norm(dim=1, keepdim=True)
            average_old_embedding_norm = torch.mean(old_embedding_norm, dim=0).cpu().type(torch.DoubleTensor)
            # (2) compute class centers for each new classes (B-/I-)
            class_center_matrix = compute_class_feature_center(dataloader_dev, 
                                        feature_model=trainer.model.encoder, 
                                        select_class_indexes=new_classes_list, 
                                        is_normalize=True,
                                        is_return_flatten_feat_and_Y=False)
            # (3) rescale the norm for each classes (each row) 
            rescale_weight_matrix = F.normalize(class_center_matrix, p=2, dim=-1) * average_old_embedding_norm
            nan_pos_list = torch.where(torch.isnan(rescale_weight_matrix[:,0]))[0]
            for nan_pos in nan_pos_list:
                assert nan_pos%2==1, "Entity not appear in dataloader!!!"
                # replace the weight of I- with B-
                rescale_weight_matrix[nan_pos] = rescale_weight_matrix[nan_pos-1].clone()
            trainer.model.classifier.fc2.weight.data = rescale_weight_matrix.type(torch.FloatTensor).cuda()

        # Evaluation before training the target model
        # logger.info('Before training evaluation')
        # f1_dev, ma_f1_dev, f1_dev_each_class = trainer.evaluate(dataloader_dev, 
        #                                             each_class=True,
        #                                             entity_order=new_entity_list)
        # logger.info("New data: Dev_f1=%.3f, Dev_ma_f1=%.3f, Dev_f1_each_class=%s" % (
        #      f1_dev, ma_f1_dev, str(f1_dev_each_class)
        # ))
        # f1_dev_cuml, ma_f1_dev_cuml, f1_dev_each_class_cuml = trainer.evaluate(dataloader_dev_cumul, 
        #                                             each_class=True,
        #                                             entity_order=all_seen_entity_list)
        # logger.info("Accumulation: Dev_f1=%.3f, Dev_ma_f1=%.3f, Dev_f1_each_class=%s" % (
        #     f1_dev_cuml, ma_f1_dev_cuml, str(f1_dev_each_class_cuml)
        # ))

        # Init training variables
        if iteration==0 and params.first_training_epochs>0:
            training_epochs = params.first_training_epochs # 20
        else:
            training_epochs = params.training_epochs # 20

        no_improvement_num = 0
        best_f1 = -1
        step = 0
        is_finish = False

        # Reset the training epoch if train by steps
        if params.is_train_by_steps: # False
            steps_per_epoch = int(len(dataloader_train.dataset)/params.batch_size)
            if iteration==0 and params.first_training_steps>0:
                training_epochs = int(params.first_training_steps/steps_per_epoch)+1
            else:
                training_epochs = int(params.training_steps/steps_per_epoch)+1

        # Check if checkpoint exists and continal training on that checkpoint
        if params.is_load_ckpt_if_exists: # True
            # 跳过第一个任务/base的训练
            if iteration==0 and params.is_load_common_first_model and os.path.isfile(common_first_model_ckpth_path):
                logger.info("Skip training %d-th iter checkpoint %s exists"%\
                                (iteration+1, common_first_model_ckpth_path))
                training_epochs = 0
            elif os.path.isfile(best_model_ckpt_path): # 之前跑过的其他任务也可以跳过，此时没有训练 只有测试 (之前跑过的setting 相同条件下重复运行 相当于只进行测试）
                logger.info("Skip training %d-th iter checkpoint %s exists"%\
                                (iteration+1, best_model_ckpt_path))
                training_epochs = 0

                
        # Start training the target model
        if trainer.scheduler!=None:
            logger.info("Initial lr is %s"%( str(trainer.scheduler.get_last_lr())))

    
        if iteration>0:
            trainer.before(train_loader=dataloader_train)
       
        for e in range(1, training_epochs+1):
            if is_finish:
                break
            logger.info("============== epoch %d ==============" % e)
            # loss list 总loss 蒸馏loss 交叉熵loss
            loss_list, distill_list, ce_list = [], [], []
            # average loss
            mean_loss = 0.0
            # training acc
            total_cnt, correct_cnt = 0, 0

            for X, y in dataloader_train:
                if is_finish:
                    break
                # Update the step count   累积batch 计数
                step += 1

                X, y = X.cuda(), y.cuda()
        
                # Forward
                trainer.batch_forward(X)

                # Record training accuracy
                mask_O = torch.not_equal(y, ner_dataloader.O_index) # mask 0
                mask_pad = torch.not_equal(y, pad_token_label_id) # mask -100
                eval_mask = torch.logical_and(mask_O, mask_pad)
                predictions = torch.max(trainer.logits,dim=2)[1]
                correct_cnt += int(torch.sum(torch.eq(predictions,y)[eval_mask].float()).item())
                total_cnt += int(torch.sum(eval_mask.float()).item())
                # Compute loss
                if iteration>0:
                    ce_loss, distill_loss = trainer.batch_loss_cpfd(y)
                    ce_list.append(ce_loss)
                    distill_list.append(distill_loss)
                else: #  第一个任务 只有ce loss
                    ce_loss = trainer.batch_loss(y)
                    ce_list.append(ce_loss) # 每个batch的loss

                total_loss = trainer.batch_backward() # 总loss
                loss_list.append(total_loss) # 追加每个batch的总loss
                mean_loss = np.mean(loss_list) # 平均每个batch的总loss
                mean_distill_loss = np.mean(distill_list) if len(distill_list)>0 else 0 # 平均每个batch的distill loss
                mean_ce_loss = np.mean(ce_list) if len(ce_list)>0 else 0 # 平均每个batch的ce loss
           

                # Print training information
                if params.info_per_steps>0 and step%params.info_per_steps==0: # params.info_per_steps=0
                    logger.info("Epoch %d, Step %d: Total_loss=%.3f, CE_loss=%.3f, Distill_loss=%.3f, Training_exact_match=%.2f%%"%(
                            e, step, mean_loss, \
                            mean_ce_loss, mean_distill_loss, correct_cnt/total_cnt*100
                    ))
                    # reset the loss lst
                    loss_list = []
                    distill_list = []
                    ce_list = []
                # Update lr + save skpt + do evaluation
                if params.is_train_by_steps: # False
                    if step>=params.training_steps:
                        is_finish = True
                    # Update learning rate
                    if trainer.scheduler != None:
                        old_lr = trainer.scheduler.get_last_lr()
                        trainer.scheduler.step()
                        new_lr = trainer.scheduler.get_last_lr()
                        if old_lr != new_lr:
                            logger.info("Epoch %d, Step %d: lr is %s"%(
                                e, step, str(new_lr)
                            ))
                    # Save checkpoint 
                    if params.save_per_steps>0 and step%params.save_per_steps==0:
                        trainer.save_model("checkpoint_domain_%s_iteration_%d_steps_%d.pth"%(
                                                domain_name, 
                                                iteration,
                                                step), 
                                            path=params.dump_path)
                    # For evaluation
                    if not params.debug and step%params.evaluate_interval==0:
                        f1_dev, ma_f1_dev, f1_dev_each_class = trainer.evaluate(dataloader_dev, 
                                                                    each_class=True,
                                                                    entity_order=new_entity_list)
                        logger.info("New data: Epoch %d, Step %d: Dev_f1=%.3f, Dev_ma_f1=%.3f, Dev_f1_each_class=%s" % (
                            e, step, f1_dev, ma_f1_dev, str(f1_dev_each_class)
                        ))
                  
                        if f1_dev > best_f1:
                            logger.info("Find better model!!")
                            best_f1 = f1_dev
                            no_improvement_num = 0
                            if iteration==0 and params.is_load_common_first_model:
                                trainer.save_model(common_first_model_ckpt_name, 
                                                    path=os.path.dirname(os.path.dirname(params.dump_path)))
                            else:
                                trainer.save_model(best_model_ckpt_name, path=params.dump_path)
                        else:
                            no_improvement_num += 1
                            logger.info("No better model is found (%d/%d)" % (no_improvement_num, params.early_stop))
                        if no_improvement_num >= params.early_stop:
                            logger.info("Stop training because no better model is found!!!")
                            is_finish = True

    

            # Print training information
            if params.info_per_epochs>0 and e%params.info_per_epochs==0: # params.info_per_epochs=1    每隔一个epoch 输出信息s
                logger.info("Epoch %d, Step %d: Total_loss=%.3f, CE_loss=%.3f, Distill_loss=%.3f, Training_exact_match=%.2f%%"%(
                            e, step, mean_loss, \
                            mean_ce_loss, mean_distill_loss, correct_cnt/total_cnt*100
                    ))
            # Update lr + save skpt + do evaluation
            # Update learning rate
            if trainer.scheduler != None:
                old_lr = trainer.scheduler.get_last_lr()
                trainer.scheduler.step() # 学习率衰减
                new_lr = trainer.scheduler.get_last_lr()
                if old_lr != new_lr:
                    logger.info("Epoch %d, Step %d: lr is %s"%(
                        e, step, str(new_lr)
                    ))
            # Save checkpoint 
            if params.save_per_epochs>0 and e%params.save_per_epochs==0: # params.save_per_epochs=0
                trainer.save_model("checkpoint_domain_%s_iteration_%d_epoch_%d.pth"%(
                                        domain_name, 
                                        iteration,
                                        e), 
                                    path=params.dump_path)
            # For evaluation
            if not params.debug and e%params.evaluate_interval==0: # params.debug=False  params.evaluate_interval=1 每隔一个epoch 评估一次
                # 当前任务的开发集 当前任务的实体集合.
                f1_dev, ma_f1_dev, f1_dev_each_class = trainer.evaluate(dataloader_dev, 
                                                            each_class=True,
                                                            entity_order=new_entity_list)
                logger.info("New data: Epoch %d, Step %d: Dev_f1=%.3f, Dev_ma_f1=%.3f, Dev_f1_each_class=%s" % (
                    e, step, f1_dev, ma_f1_dev, str(f1_dev_each_class)
                ))
                
                #  选择在当前任务开发集上表现最好的模型
                if f1_dev > best_f1: # 默认是micro平均，这个是首选指标
                    logger.info("Find better model!!")
                    best_f1 = f1_dev
                    no_improvement_num = 0
                    if iteration==0 and params.is_load_common_first_model:
                        # base model
                        trainer.save_model(common_first_model_ckpt_name, 
                                            path=os.path.dirname(os.path.dirname(params.dump_path)))
                    else: # 其他任务 最好的模型
                        trainer.save_model(best_model_ckpt_name, path=params.dump_path)
                else:
                    no_improvement_num += 1
                    logger.info("No better model is found (%d/%d)" % (no_improvement_num, params.early_stop))
                if no_improvement_num >= params.early_stop:
                    logger.info("Stop training because no better model is found!!!")
                    is_finish = True

        logger.info("Finish training ...")

        # ===========================================================================
        # testing
        if params.debug: # False
            logger.info("Skip testing for debug...")
            continue

        # 加载当前任务 最好的model
        if iteration==0 and params.is_load_common_first_model:
            trainer.load_model(common_first_model_ckpt_name, 
                                path=os.path.dirname(os.path.dirname(params.dump_path)))
        else:
            trainer.load_model(best_model_ckpt_name, path=params.dump_path)
        trainer.model.cuda()


        # testing  截至当前任务的测试集 截至当前任务的实体集合
        logger.info("Testing...")


        f1_test_cumul, ma_f1_test_cumul, f1_test_each_class_cumul = trainer.evaluate(dataloader_test_cumul, 
                                                    each_class=True,
                                                    entity_order=all_seen_entity_list,
                                                    is_plot_hist=False)   
        logger.info("Accumulation: Test_f1=%.3f, Test_ma_f1=%.3f, Test_f1_each_class=%s"%(
                    f1_test_cumul, ma_f1_test_cumul, str(f1_test_each_class_cumul)))
        logger.info("Finish testing the %d-th iter!"%(iteration+1))

        

if __name__ == "__main__":
    params = get_params() # 获取配置
    main_cl(params)

    
