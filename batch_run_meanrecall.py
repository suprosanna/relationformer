'''
Run mean recall across all checkpoint sirted by time wise
'''

import glob
import os
import numpy as np
import yaml
import traceback
import torch.distributed as dist
import ignite.distributed as igdist
from train import dict2obj,main,parse_args


dir_name = 'output/Data/transformer_graph_gen/runs/'
#'visulagenome_200_obj_resn101xt_fulltrain_matcher323_drfrq_10',
#'visulagenome_300_edgeloss_6_rel_drnorm_fulltrain_lrbkbn_logsoftmax_norelnorn_detr_lr1e4_10'
#'visulagenome_300_edgeloss_6_fulltrain_lrbackbone_logsoftmaxtrn_relemb_detr_10'
# Get list of all files only in the given directory

list_of_files = ['visulagenome_200_edgeloss_6_fulltrain_drop_final_wdoutlogsoftmax_b16_10']
# Iterate over sorted list of files and print file path

if __name__ == '__main__':
    args = parse_args()
    if args.cuda_visible_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.cuda_visible_device))
    for file_path in list_of_files:
        print("Run to be evaluated : ", file_path)
        try:
            if os.path.isfile(dir_name+file_path + '/config.yaml'):
                print('True')
                args.exp_name = os.path.basename(file_path).rsplit('_',1)[0]
                args.config = dir_name+file_path + '/config.yaml'
                ckpt_to_run = glob.glob(os.path.join(dir_name+file_path, 'models/*.pt'))
                if len(ckpt_to_run) > 0:
                    ckpt_to_run = np.sort(ckpt_to_run)[-1]
                args.resume = ckpt_to_run
                args.mR = True
                args.eval = True
                print('Running for experiemnt : ',args.exp_name,' :: for checkpoint : ',os.path.basename(args.resume))
                with igdist.Parallel(backend='gloo', nproc_per_node=args.nproc_per_node) as parallel:
                    parallel.run(main, args)
        except BaseException as err:
            print('Error in run : ', os.path.basename(file_path))
            print(f"Unexpected {err=}, {type(err)=}")
            traceback.print_exc()

