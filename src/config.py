import yaml
import argparse

# ===========================================================================
# LOAD CONFIGURATIONS
def get_params():
    parser = argparse.ArgumentParser(description="Continual Learning for NER")


    # =========================================================================================
    # Experimental Settings
    # =========================================================================================
    # Debug
    parser.add_argument("--debug", default=False, action="store_true", help="if skipping the test on training and validation set")
    
    # Config
    parser.add_argument("--cfg", default="./config/default.yaml", help="Hyper-parameters") # 超参数配置文件

    # Path
    parser.add_argument("--exp_name", type=str, default="default", help="Experiment name")
    parser.add_argument("--logger_filename", type=str, default="train.log")
    parser.add_argument("--dump_path", type=str, default="experiments", help="Experiment saved root path")
    parser.add_argument("--exp_id", type=str, default="1", help="Experiment id")
    parser.add_argument("--seed", type=int, default=None, help="Random Seed")

    # Model
    parser.add_argument("--model_name", type=str, default="bert-base-cased", help="model name (e.g., bert-base-cased, roberta-base or wide_resnet)")
    parser.add_argument("--is_load_ckpt_if_exists", default=False, action='store_true', help="Loading the ckpt if best finetuned ckpt exists")
    parser.add_argument("--is_load_common_first_model", default=False, action='store_true', help="Loading the common first ckpt if best finetuned ckpt exists")
    parser.add_argument("--ckpt", type=str, default=None, help="the pretrained lauguage model")
    parser.add_argument("--dropout", type=float, default=0, help="dropout rate")
    parser.add_argument("--hidden_dim", type=int, default=768, help="Hidden layer dimension")
    parser.add_argument("--alpha", type=float, default=0, help="Trade-off parameter")
    parser.add_argument("--none_idx", type=int, default=103, help="None token index(103=[mask])")

    # Data
    parser.add_argument("--data_path", type=str, default="./datasets/NER_data/ai/", help="source domain")
    parser.add_argument("--n_samples", type=int, default=-1, help="conduct few-shot learning (10, 25, 40, 55, 70, 85, 100)")
    parser.add_argument("--entity_list", type=str, default="", help="entity list")
    parser.add_argument("--schema", type=str, default="BIO", choices=['IO','BIO','BIOES'], help="Lable schema")
    parser.add_argument("--is_filter_O", default=False, action='store_true', help="If filter out samples contains only O labels")
    parser.add_argument("--is_load_disjoin_train", default=False, action='store_true', help="If loading the join ckpt for training dat (only for CL)")

    # =========================================================================================
    # Training Settings
    # =========================================================================================
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size") 
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max length for each sentence") 
    
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate") 
    parser.add_argument("--is_train_by_steps", default=False, action='store_true', help="If the scheduer and evaluation is meausured by steps")
    parser.add_argument("--training_epochs", type=int, default=0, help="Number of training epochs")
    parser.add_argument("--first_training_epochs", type=int, default=0, help="Number of training epochs in first iteration (will be set as training_epochs by default)")
    parser.add_argument("--training_steps", type=int, default=0, help="Number of training steps")
    parser.add_argument("--first_training_steps", type=int, default=0, help="Number of training steps in first iteration (will be set as training_steps by default)")
    
    parser.add_argument("--schedule", type=str, default='(20, 40)', help="Multistep scheduler")
    parser.add_argument("--stable_lr", type=float, default=4e-4, help="Stable learning rate")
    parser.add_argument("--gamma", type=float, default=0.2, help="Factor of the learning rate decay")

    parser.add_argument("--mu", type=float, default=0.9, help="Momentum")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")

    parser.add_argument("--info_per_epochs", type=int, default=1, help="Print information every how many epochs")
    parser.add_argument("--info_per_steps", type=int, default=0, help="Print information every how many steps")
    parser.add_argument("--save_per_epochs", type=int, default=0, help="Save checkpoints every how many epochs")
    parser.add_argument("--save_per_steps", type=int, default=0, help="Save checkpoints every how many steps")
    parser.add_argument("--evaluate_interval", type=int, default=3, help="Evaluation interval")
    parser.add_argument("--early_stop", type=int, default=5, help="No improvement after several epoch, we stop training")


    # =========================================================================================
    # Incremental Learning Settings
    # =========================================================================================
    parser.add_argument("--nb_class_pg", type=int, default=2, help="number of classes in each group")
    parser.add_argument("--nb_class_fg", type=int, default=4, help="number of classes in the first group")
    
    parser.add_argument("--is_rescale_new_weight", default=False, action='store_true', help="If rescale the new weight matrix")
    parser.add_argument("--is_fix_trained_classifier", default=False, action='store_true', help="If fix the trained classifer")
    parser.add_argument("--is_unfix_O_classifier", default=False, action='store_true', help="If not fix the O classifer")
    
    parser.add_argument("--is_MTL", default=False, action='store_true', help="If using multi-task learning")
    parser.add_argument("--extra_annotate_type", type=str, default='none', choices=['none','current','all'] , help="Simulate mannual annotation in each data split")
    parser.add_argument("--is_from_scratch", default=False, action='store_true', help="If training from scratch for multi-task learning")

    parser.add_argument("--reserved_ratio", type=float, default=0, help="the ratio of reserved samples")

    # =========================================================================================
    # Baseline Settings
    # =========================================================================================
    # CPFD
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--classif_adaptive_factor", default=True, action='store_true')
    parser.add_argument("--classif_adaptive_min_factor", default=0.0, type=float)
    parser.add_argument("--distill_weight", type=float, default=2, help="distillation weight for loss")
    parser.add_argument("--adaptive_distill_weight", default=True, action='store_true', help="If using adaptive weight")
    parser.add_argument("--adaptive_schedule", type=str, default='root', choices=['root','linear','square'], help="The schedule for adaptive weight")
    parser.add_argument("--temperature", type=int, default=1, help="temperature of the student model")
    parser.add_argument("--ref_temperature", type=int, default=1, help="temperature of the teacher model")
    
    params = parser.parse_args() # 默认配置

    with open(params.cfg) as f: # 读取yaml文件 进行配置覆盖
        config = yaml.safe_load(f)
        for k, v in config.items():
            # for parameters set in the args
            if k in ['none_idx']:
                continue
            params.__setattr__(k,v)

    return params