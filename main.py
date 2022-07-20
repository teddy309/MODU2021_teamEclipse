'''
main part to train/inference model for task1~4.
'''
import argparse
from pprint import pprint
from scripts import pipeline_COLA, pipeline_COPA, pipeline_WiC

import setproctitle

from utilities.config_constructor import Config

from scripts import pipeline_BoolQ
from scripts.pipeline_COLA import train_cola,eval_cola,inference_cola        #task1: 
from scripts.pipeline_WiC import train_wic,eval_wic           #task2: 
#from scripts.train_booleanQA import train_boolQ            #task4:
#from scripts.train_booleanQA import train_boolQ            #task4:

from scripts.trainer import train_model

from utilities.utils import compute_metrics,  MCC, get_label, set_seed
from utilities.utils import MODEL_CLASSES, MODEL_PATH_MAP
from utilities.utils import TOKEN_MAX_LENGTH #SPECIAL_TOKENS
from utilities.utils import getParentPath, save_model, load_model, save_json, DATASET_PATHS

#from scripts.utils import init_logger, load_tokenizer, set_seed, MODEL_CLASSES, MODEL_PATH_MAP


def main(cfg):
    if cfg.task == 'cola': #task1
        train_model()
    elif cfg.task == 'wic': #task2
        train_model()
    elif cfg.task == 'copa': #task3
        train_model()
    elif cfg.task == 'boolQ': #task4
        if cfg.procedure == 'train':
            train_boolQ(cfg)
        elif cfg.procedure == 'eval':
            eval_boolQ(cfg)
    else:
        raise NotImplementedError
    '''
    if cfg.task == 'cola': #task1
        if cfg.procedure == 'train':
            train_cola(cfg)
        elif cfg.procedure == 'eval':
            eval_cola(cfg)
    elif cfg.task == 'wic': #task2
        if cfg.procedure == 'train':
            train_wic(cfg)
        elif cfg.procedure == 'eval':
            eval_wic(cfg)
    elif cfg.task == 'copa': #task3
        if cfg.procedure == 'train':
            train_copa(cfg)
        elif cfg.procedure == 'eval':
            eval_copa(cfg)
    elif cfg.task == 'boolQ': #task4
        if cfg.procedure == 'train':
            train_boolQ(cfg)
        elif cfg.procedure == 'eval':
            eval_boolQ(cfg)
    else:
        raise NotImplementedError
    '''

'''
args config style from AVSD/main.py
-   out: parser(argparse.ArgumentParser)
'''
def get_parser():
    parser = argparse.ArgumentParser(description='Run experiment')

    ## DATA
    # paths to the precalculated train meta files
    parser.add_argument('--dataset_path', type=str, default='./dataset/') #'/home/nlplab/hdd3/2021_AIlangCpt/dataset/'
    parser.add_argument('--train_tsv_path', type=str, default='task1_grammar/NIKL_CoLA_train.tsv')
    parser.add_argument('--test_cola_path', type=str, default='task1_grammar/NIKL_CoLA_test.tsv')
    parser.add_argument('--dev_cola_path', type=str, default='task1_grammar/NIKL_CoLA_dev.tsv')
    parser.add_argument('--train_wic_path', type=str, default='task2_homonym/NIKL_SKT_WiC_train.tsv')
    parser.add_argument('--test_wic_path', type=str, default='task2_homonym/NIKL_SKT_WiC_test.tsv')
    parser.add_argument('--dev_wic_path', type=str, default='task2_homonym/NIKL_SKT_WiC_dev.tsv')
    parser.add_argument('--train_copa_path', type=str, default='task2_homonym/NIKL_SKT_WiC_train.tsv') #바꺼야댐
    parser.add_argument('--test_copa_path', type=str, default='task2_homonym/NIKL_SKT_WiC_test.tsv')
    parser.add_argument('--dev_copa_path', type=str, default='task2_homonym/NIKL_SKT_WiC_dev.tsv')
    parser.add_argument('--train_boolq_path', type=str, default='task2_homonym/NIKL_SKT_WiC_train.tsv')
    parser.add_argument('--test_boolq_path', type=str, default='task2_homonym/NIKL_SKT_WiC_test.tsv')
    parser.add_argument('--dev_boolq_path', type=str, default='task2_homonym/NIKL_SKT_WiC_dev.tsv')
    parser.add_argument('--task', type=str, default='cola',
                        choices=['cola', 'wic', 'copa','boolq'],
                        help='select task 1~4')

    ## TRAINING
    #parser.add_argument('--procedure', type=str, required=True, 
    #                    choices=['train','eval']) #choices=['train_cola','eval_cola', 'train_wic','eval_wic', 'train_copa', 'eval_copa', 'train_boolQ', 'eval_boolQ']
    parser.add_argument('--device_ids', type=int, nargs='+', default=[0], help='separated by a whitespace')
    parser.add_argument('--start_token', type=str, default='<s>', help='starting token')
    parser.add_argument('--end_token', type=str, default='</s>', help='ending token')
    parser.add_argument('--pad_token', type=str, default='<blank>', help='padding token')

    parser.add_argument('--logging_steps', type=int, default=2000, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=2000, help="Save checkpoint every X updates steps.")

    ## EVALUATION
    
    ## MODEL
    parser.add_argument("--model_type", default="kobert", type=str, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

    parser.add_argument('--dout_p', type=float, default=0.1, help='dropout probability: in [0, 1]')
    parser.add_argument('--N', type=int, default=2, help='number of layers in a model')
    parser.add_argument(
        '--d_model', type=int, default=1024,
        help='the internal space in the multi-headed attention (when input dims of Q, K, V differ)')
    parser.add_argument(
        '--d_model_video', type=int,
        help='If use_linear_embedder is true, this is going to be the d_model size for video model'
    )
    parser.add_argument(
        '--d_model_audio', type=int,
        help='If use_linear_embedder is true, this is going to be the d_model size for audio model'
    )
    parser.add_argument(
        '--d_model_caps', type=int, default=300,
        help='hidden size of the crossmodal decoder (caption tokens are mapped into this dim)'
    )
    parser.add_argument(
        '--use_linear_embedder', dest='use_linear_embedder', action='store_true', default=False,
        help='Whether to include a dense layer between the raw features and input to the model'
    )
    parser.add_argument('--H', type=int, default=4, help='number of heads in multiheaded attention')
    parser.add_argument(
        '--d_ff_video', type=int, help='size of the internal layer of PositionwiseFeedForward')
    parser.add_argument(
        '--d_ff_audio', type=int, help='size of the internal layer of PositionwiseFeedForward')
    parser.add_argument(
        '--d_ff_caps', type=int, help='size of the internal layer of PositionwiseFeedForward')

    ## DEBUGGING
    parser.add_argument('--debug', dest='debug', action='store_true', default=False,
                        help='runs test() instead of main()')
    parser.add_argument('--dont_log', dest='to_log', action='store_false',
                        help='Prevent logging in the experiment.')

    ## Process name
    parser.add_argument("--process_title", type=str, default='modu2021_run_task',
                        help='show the process title in the nvidia-smi')

    parser.set_defaults(to_log=True)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    ##args setting
    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    args.task = 'cola'

    setproctitle.setproctitle(args.process_title)
    pprint(vars(args))
    cfg = Config(args)

    if args.debug:
        # load your test to debug something using the same config as main() would
        # from tests import test_features_max_length
        # test_features_max_length(cfg)
        pass
    else:
        main(cfg)
