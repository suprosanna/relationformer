import json
import os
import gc
import torch
import yaml
import numpy as np
import time

from monai.engines import SupervisedEvaluator
from monai.handlers import StatsHandler, CheckpointSaver, TensorBoardStatsHandler

# from datasets.coco_eval import CocoEvaluator
from metrics.boxap import MeanBoxAP
from argparse import ArgumentParser
import json

from multiprocessing import Pool
import pdb
from inference import graph_infer
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from monai.utils import ForwardMode, min_version, optional_import
from monai.config import IgniteInfo
from monai.engines.utils import default_metric_cmp_fn, default_prepare_batch
from monai.inferers import Inferer, SimpleInferer
from monai.transforms import Transform
from monai.utils import ForwardMode, min_version, optional_import
from ignite.engine import Events
from util.sg_recall import BasicSceneGraphEvaluator,calculate_mR_from_evaluator_list

if TYPE_CHECKING:
    from ignite.engine import Engine, EventEnum
    from ignite.metrics import Metric
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")
    Metric, _ = optional_import("ignite.metrics", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Metric")
    EventEnum, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "EventEnum")

parser = ArgumentParser()
parser.add_argument('--config',
                    default=None,
                    help='config file (.yml) containing the hyper-parameters for training. '
                         'If None, use the nnU-Net config. See /config for examples.')
parser.add_argument('--resume', default=None, help='checkpoint of the last epoch of the model')
parser.add_argument('--device', default='cuda',
                        help='device to use for training')
parser.add_argument('--cuda_visible_device', nargs='*', type=int, default=[0,1],
                        help='list of index where skip conn will be made')


#mock cocoDetection classes
class CocoMock(COCO):
    def __init__(self,imgs,categories):
        super().__init__(annotation_file=None)
        self.dataset['categories'] = [categories]
        self.dataset['images'] = []
        for img in imgs:
            self.imgs[img['id']]={'file_name':img['image'],'id':img['id']}
            self.dataset['images'].append({'file_name':img['image'],'id':img['id']})
# Define customized evaluator
class RelationformerEvaluator(SupervisedEvaluator):
    def __init__(
            self,
            device: torch.device,
            val_data_loader: Union[Iterable, DataLoader],
            network: torch.nn.Module,
            epoch_length: Optional[int] = None,
            non_blocking: bool = False,
            prepare_batch: Callable = default_prepare_batch,
            iteration_update: Optional[Callable] = None,
            inferer: Optional[Inferer] = None,
            postprocessing: Optional[Transform] = None,
            key_val_metric: Optional[Dict[str, Metric]] = None,
            additional_metrics: Optional[Dict[str, Metric]] = None,
            metric_cmp_fn: Callable = default_metric_cmp_fn,
            val_handlers: Optional[Sequence] = None,
            amp: bool = False,
            mode: Union[ForwardMode, str] = ForwardMode.EVAL,
            event_names: Optional[List[Union[str, EventEnum]]] = None,
            event_to_attr: Optional[dict] = None,
            decollate: bool = True,
            **kwargs,
    ) -> None:
        super().__init__(
            device=device,
            val_data_loader=val_data_loader,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            postprocessing=postprocessing,
            key_val_metric=key_val_metric,
            additional_metrics=additional_metrics,
            metric_cmp_fn=metric_cmp_fn,
            val_handlers=val_handlers,
            amp=amp,
            mode=mode,
            event_names=event_names,
            event_to_attr=event_to_attr,
            decollate=decollate,
            network=network,
            inferer=SimpleInferer() if inferer is None else inferer,
        )

        self.config = kwargs.pop('config')
        # self.coco_evaluator=kwargs['coco_evaluator']
        self.sg_evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=False,config=self.config)#todo replace wd param
        self.run_mode = 'sgdet'  # TODO: hard coded for SGDET evaluation
        self.debug = kwargs['debug']
        self.distributed = kwargs.pop('distributed')
        if 'freq_baseline' in kwargs.keys():
            self.freq_baseline = kwargs['freq_baseline']
        self.writer = kwargs.pop('writer')
        self._accumulate()
        self.add_emd_rel = self.config.MODEL.DECODER.ADD_EMB_REL
        self.mean_recall = hasattr(self.config.DATA,'MEAN_RECALL') and self.config.DATA.MEAN_RECALL
        self.fps = []
        if self.mean_recall:
            self.evaluator_list = []
            ind_2_pred = json.load(open(self.config.DATA.LABEL_DATA_DIR))['idx_to_predicate'] #todo only fo vg replace
            for idx in enumerate(ind_2_pred.keys()):
                self.evaluator_list.append(
                    (int(idx[1]), ind_2_pred[idx[1]], BasicSceneGraphEvaluator.all_modes(config=self.config)))
    def _iteration(self, engine, batchdata):
        start = time.time()
        images, gt_datas = batchdata[0], batchdata[1]
        # # inputs, targets = self.get_batch(batchdata, image_keys=IMAGE_KEYS, label_keys="label")
        #images = torch.stack(images)
        images = [image.to(engine.state.device,  non_blocking=False) for image in images]
        boxes = [data['boxes'].cpu().numpy() for data in gt_datas]
        boxes_class = [data['labels'].cpu().numpy()-1.0 for data in gt_datas]
        boxes_score = [np.ones(data['labels'].shape[0]) for data in gt_datas]
        edges = [data['edges'].cpu().numpy() for data in gt_datas]

        self.network.eval()
        h, out = self.network(images)  # todo output logit and edge are same value
        if self.distributed:
            relation_embed = self.network.module.relation_embed
        else:
            relation_embed = self.network.relation_embed
        out = graph_infer(h, out, relation_embed, freq = self.freq_baseline,emb = self.add_emd_rel)

        pred_edges = [{'node_pair': pred_rels, 'edge_score': edge_score} for pred_rels, edge_score in zip(out['all_node_pairs'], out['all_relation'])]
        pred_classes = [{'labels': pred_class+1, 'scores': pred_score, 'boxes': torch.tensor(pred_box)} for pred_class, pred_score, pred_box in zip(out['pred_boxes_class'], out['pred_boxes_score'], out['pred_boxes'])]

        for i, (gt_data, pred_class, pred_edge) in enumerate(zip(gt_datas, pred_classes, pred_edges)):
            # prepare scene graph evaluation
            self.sg_evaluator[self.run_mode].evaluate_scene_graph_entry(gt_data,[pred_class,pred_edge])
        if self.mean_recall: # for mean recall
            for (pred_id, _, evaluator_rel) in self.evaluator_list:
                gt_data_copy = gt_data.copy()
                gt_rel = gt_data_copy['edges']
                mask = np.in1d(gt_rel[:, -1], pred_id)
                gt_data_copy['edges'] = gt_rel[mask, :]
                if gt_data_copy['edges'].shape[0]> 0:
                    evaluator_rel[self.run_mode].evaluate_scene_graph_entry(gt_data_copy, [pred_class, pred_edge])

        gc.collect()
        torch.cuda.empty_cache()

        return {**{"images": images, "boxes": boxes, "boxes_class": boxes_class, "boxes_score":boxes_score, "edges": edges}, **out}

    def _accumulate(self):

        @self.on(Events.EPOCH_COMPLETED)
        def update_cls_sg_metrices(engine: Engine) -> None:
            file_path=None
            if self.mean_recall:
                save_path = os.path.dirname(self.config.MODEL.RESUME)
                ckpt_name = os.path.basename(self.config.MODEL.RESUME).split('.')[0]
                file_path = os.path.join(save_path, ckpt_name + '_Recall.txt')
                calculate_mR_from_evaluator_list(self.evaluator_list, self.run_mode, file_path, save_file=True)
            self.sg_evaluator[self.run_mode].print_stats(epoch_num=self.state.epoch, writer=self.writer,file_path=file_path)

        
        @self.on(Events.EPOCH_STARTED)
        def empty_buffers(engine: Engine) -> None:
            self.sg_evaluator[self.run_mode].reset()
                

def build_evaluator(val_loader, net, optimizer, scheduler, writer, config, device, distributed=False, local_rank=0, **kwargs):
    """[summary]

    Args:
        val_loader ([type]): [description]
        net ([type]): [description]
        device ([type]): [description]

    Returns:
        [type]: [description]
    """
    val_handlers = [
        TensorBoardStatsHandler(
            writer,
            tag_name="val_smd",
            output_transform=lambda x: None,
            global_epoch_transform=lambda x: scheduler.last_epoch
        ),
    ]
    if local_rank==0:
        val_handlers.extend(
            [
                StatsHandler(output_transform=lambda x: None),
                CheckpointSaver(
                    save_dir=os.path.join(config.TRAIN.SAVE_PATH, "runs", '%s_%d' % (config.log.exp_name, config.DATA.SEED),
                                        'models'),
                    save_dict={"net": net, "optimizer": optimizer, "scheduler": scheduler},
                    save_key_metric=False,
                    key_metric_n_saved=5,
                    save_interval=1
                ),
            ]
        )


    #val_post_transform = RelationformerEvaluator.update_cls_sg_metrices

    evaluator = RelationformerEvaluator(
        config=config,
        device=device,
        val_data_loader=val_loader,
        network=net,
        inferer=SimpleInferer(),
        #post_transform=val_post_transform,
        key_val_metric={
            "val_AP": MeanBoxAP(
                output_transform=lambda x: (x["pred_boxes"], x["pred_boxes_class"], x["pred_boxes_score"], x["boxes"], x["boxes_class"]),
                max_detections=100, writer=writer,
            )
        },
        val_handlers=val_handlers,
        amp=False,
        distributed=distributed,
        writer=writer,
        **kwargs,
    )

    return evaluator