import os
from torch.nn.functional import interpolate
from monai.engines import SupervisedTrainer
from monai.inferers import SimpleInferer
from monai.handlers import LrScheduleHandler, ValidationHandler, StatsHandler, TensorBoardStatsHandler, CheckpointSaver, MeanDice
from monai.transforms import (
    Compose,
    AsDiscreted,
)
import torch.nn.functional as F
import torch
from torch.nn.utils import clip_grad_norm
from inference import relation_infer
import gc

from utils import get_total_grad_norm


# define customized trainer
class RelationformerTrainer(SupervisedTrainer):

    def _iteration(self, engine, batchdata):
        images, seg, nodes, edges = batchdata[0], batchdata[1], batchdata[2], batchdata[3]
        # inputs, targets = self.get_batch(batchdata, image_keys=IMAGE_KEYS, label_keys="label")
        # inputs = torch.cat(inputs, 1)

        images = images.to(engine.state.device,  non_blocking=False)
        seg = seg.to(engine.state.device,  non_blocking=False)
        nodes = [node.to(engine.state.device,  non_blocking=False) for node in nodes]
        edges = [edge.to(engine.state.device,  non_blocking=False) for edge in edges]
        target = {'nodes': nodes, 'edges': edges}

        self.network[0].train()
        self.network[1].eval()
        self.optimizer.zero_grad()
        
        h, out, srcs = self.network[0](images, seg=False)

        # _, _, seg_srcs = self.network[1](seg, seg=True)
        
        # # valid_token = torch.argmax(out['pred_logits'], -1)
        # # valid_token = torch.sigmoid(nodes_prob[...,3])>0.5
        # # print('valid_token number', valid_token.sum(1))

        # # pred_nodes, pred_edges = relation_infer(h, out, self.network.relation_embed)

        losses = self.loss_function(h, out, target)
        
        # loss_feat = 0
        # for i in range(len(seg_srcs)):
        #     loss_feat += F.l1_loss(srcs[i], seg_srcs[i].detach())

        # losses.update({'feature':loss_feat})
        # losses['total'] = losses['total']+loss_feat
        # Clip the gradient
        # clip_grad_norm_(
        #     self.network.parameters(),
        #     max_norm=GRADIENT_CLIP_L2_NORM,
        #     norm_type=2,
        # )
        losses['total'].backward()

        # if 0.1 > 0:
        #     _ = torch.nn.utils.clip_grad_norm_(self.network[0].parameters(), 0.1)
        # else:
        #     _ = get_total_grad_norm(self.network[0].parameters(), 0.1)
    
        self.optimizer.step()
        
        gc.collect()
        torch.cuda.empty_cache()

        return {"images": images, "points": nodes, "edges": edges, "loss": losses}


def build_trainer(train_loader, net, seg_net, loss, optimizer, scheduler, writer,
                  evaluator, config, device, fp16=False):
    """[summary]

    Args:
        train_loader ([type]): [description]
        net ([type]): [description]
        loss ([type]): [description]
        optimizer ([type]): [description]
        evaluator ([type]): [description]
        scheduler ([type]): [description]
        max_epochs ([type]): [description]
        device ([type]): [description]

    Returns:
        [type]: [description]
    """
    train_handlers = [
        LrScheduleHandler(
            lr_scheduler=scheduler,
            print_lr=True,
            epoch_level=True,
        ),
        ValidationHandler(
            validator=evaluator,
            interval=config.TRAIN.VAL_INTERVAL,
            epoch_level=True
        ),
        StatsHandler(
            tag_name="train_loss",
            output_transform=lambda x: x["loss"]["total"]
        ),
        CheckpointSaver(
            save_dir=os.path.join(config.TRAIN.SAVE_PATH, "runs", '%s_%d' % (config.log.exp_name, config.DATA.SEED), 'models'),
            save_dict={"net": net, "optimizer": optimizer, "scheduler": scheduler},
            save_interval=1,
            n_saved=1
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="classification_loss",
            output_transform=lambda x: x["loss"]["class"],
            global_epoch_transform=lambda x: scheduler.last_epoch,
            iteration_interval=10,
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="node_loss",
            output_transform=lambda x: x["loss"]["nodes"],
            global_epoch_transform=lambda x: scheduler.last_epoch,
            iteration_interval=10,
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="edge_loss",
            output_transform=lambda x: x["loss"]["edges"],
            global_epoch_transform=lambda x: scheduler.last_epoch,
            iteration_interval=10,
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="box_loss",
            output_transform=lambda x: x["loss"]["boxes"],
            global_epoch_transform=lambda x: scheduler.last_epoch,
            iteration_interval=10,
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="card_loss",
            output_transform=lambda x: x["loss"]["cards"],
            global_epoch_transform=lambda x: scheduler.last_epoch,
            iteration_interval=10,
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="total_loss",
            output_transform=lambda x: x["loss"]["total"],
            global_epoch_transform=lambda x: scheduler.last_epoch,
            iteration_interval=10,
        )
    ]
    # train_post_transform = Compose(
    #     [AsDiscreted(keys=("pred", "label"),
    #     argmax=(True, False),
    #     to_onehot=True,
    #     n_classes=N_CLASS)]
    # )

    trainer = RelationformerTrainer(
        device=device,
        max_epochs=config.TRAIN.EPOCHS,
        train_data_loader=train_loader,
        network=[net, seg_net],
        optimizer=optimizer,
        loss_function=loss,
        inferer=SimpleInferer(),
        # post_transform=train_post_transform,
        # key_train_metric={
        #     "train_mean_dice": MeanDice(
        #         include_background=False,
        #         output_transform=lambda x: (x["pred"], x["label"]),
        #     )
        # },
        train_handlers=train_handlers,
        # amp=fp16,
    )

    return trainer
