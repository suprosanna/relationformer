import os
from torch.nn.functional import interpolate
from monai.engines import SupervisedTrainer
from monai.inferers import SimpleInferer
from monai.handlers import LrScheduleHandler, ValidationHandler, StatsHandler, TensorBoardStatsHandler, CheckpointSaver, MeanDice
from monai.transforms import (
    Compose,
    AsDiscreted,
)
import torch
from torch.nn.utils import clip_grad_norm
from inference import graph_infer
import gc
from utils import get_total_grad_norm
import pdb

# define customized trainer
class RelationformerTrainer(SupervisedTrainer):

    def __init__(self, **kwargs):
        self.distributed = kwargs.pop('distributed')
        # Initialize superclass things
        super().__init__(**kwargs)
        self.tl = ['boxes','labels','edges','iscrowd' ]

    def _iteration(self, engine, batchdata):
        images, target = batchdata[0], batchdata[1]

        images = [image.to(engine.state.device,  non_blocking=False) for image in images]
        target = [{k: v.to(engine.state.device, non_blocking=True) if k in self.tl else v for k, v in t.items()} for t in target]


        net_wo_ddp = self.network
        if self.distributed:
            net_wo_ddp = self.network.module
            
        self.network.train()
        self.optimizer.zero_grad()
        # self.network.encoder.eval()
        h, out = self.network(images)

        losses = self.loss_function(h, out, target)
        # Clip the gradient
        # clip_grad_norm_(
        #     self.network.parameters(),
        #     max_norm=GRADIENT_CLIP_L2_NORM,
        #     norm_type=2,
        # )
        losses['total'].backward()
        
        if 0.1 > 0: #todo replace
            _ = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.1)
        else:
            _ = get_total_grad_norm(self.networm.parameters(), 0.1)


        self.optimizer.step()
        
        gc.collect()
        torch.cuda.empty_cache()

        return {"images": images, "points": None, "boxes": None, "loss": losses}


def build_trainer(train_loader, net, loss, optimizer, scheduler, writer,
                  evaluator, config, device, fp16=False, distributed=False, local_rank=0):
    """[summary]

    Args:
        train_loader ([type]): [description]
        net ([type]): [description]
        loss ([type]): [description]
        optimizer ([type]): [description]
        evaluator ([type]): [description]
        scheduler ([type]): [description]
        max_epochs ([type]): [description]

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
        TensorBoardStatsHandler(
            writer,
            tag_name="classification_loss",
            output_transform=lambda x: x["loss"]["class"],
            global_epoch_transform=lambda x: scheduler.last_epoch
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="node_loss",
            output_transform=lambda x: x["loss"]["nodes"],
            global_epoch_transform=lambda x: scheduler.last_epoch
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="edge_loss",
            output_transform=lambda x: x["loss"]["edges"],
            global_epoch_transform=lambda x: scheduler.last_epoch
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="box_loss",
            output_transform=lambda x: x["loss"]["boxes"],
            global_epoch_transform=lambda x: scheduler.last_epoch
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="card_loss",
            output_transform=lambda x: x["loss"]["cards"],
            global_epoch_transform=lambda x: scheduler.last_epoch
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="total_loss",
            output_transform=lambda x: x["loss"]["total"],
            global_epoch_transform=lambda x: scheduler.last_epoch
        )
    ]
    if local_rank==0:
        train_handlers.extend(
            [
                # StatsHandler(
                #     tag_name="train_loss",
                #     output_transform=lambda x: x["loss"]["total"]
                # ),
                CheckpointSaver(
                    save_dir=os.path.join(config.TRAIN.SAVE_PATH, "runs", '%s_%d' % (config.log.exp_name, config.DATA.SEED), 'models'),
                    save_dict={"net": net, "optimizer": optimizer, "scheduler": scheduler},
                    save_interval=1,
                    n_saved=1
                ),
            ]
        )
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
        network=net,
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
        distributed=distributed,
    )

    return trainer
