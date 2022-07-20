import fiftyone as fo
import numpy as np
import os
from datasets.coco import CocoDetection
from util.misc import get_local_rank, get_local_size
from datasets import get_coco_api_from_dataset
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator


def convert_to_coco(dataloader, config,val=False):
    dataset = config.DATA.DATASET + '_val' if val else config.DATA.DATASET + '_train'
    export_dir = os.path.join("data/coco-detection-dataset", dataset )
    if not config.DATA.DATASET in fo.list_datasets():
        sg_dataset = fo.load_dataset(dataset)
        assert len(dataloader.dataset) == len(sg_dataset)
        coco_dataset=CocoDetection(os.path.join(export_dir, 'data'), os.path.join(export_dir, 'labels.json'),
                                                    transforms=dataloader.dataset.transforms, return_masks=False,
                                                    cache_mode=False, local_rank=get_local_rank(), local_size=get_local_size())
    else:
        sg_dataset = fo.Dataset(name=dataset)
        # Persist the dataset on disk in order to
        # be able to load it in one line in the future
        sg_dataset.persistent = True
        for each_data in dataloader.dataset.data_dicts:
            sample = fo.Sample(filepath=each_data['image'])
            # Convert detections to FiftyOne format
            detections = []
            boxes = each_data['boxes']
            boxes = boxes / config.DATA.BOX_SCALE
            # crop boxes that are too large. This seems to be only a problem for image heights, but whatevs
            boxes = boxes.clip(0, 1)
            for (box, lbl) in zip(boxes, each_data['label']):
                label = str(lbl)
                iscrowd = str(0)
                # Bounding box coordinates should be relative values
                # in [0, 1] in the following format:
                # [top-left-x, top-left-y, width, height]
                bbox = (box[0], box[1], box[2] - box[0], box[3] - box[1])
                detections.append(
                    fo.Detection(label=str(label), bounding_box=list(bbox), iscrowd=iscrowd)
                )
            # Store detections in a field name of your choice
            sample["ground_truth"] = fo.Detections(detections=detections)

            sg_dataset.add_sample(sample)
        #save dataset
        sg_dataset.save()

        label_field = "ground_truth"  # for example
        # Export the dataset
        sg_dataset.export(
            export_dir=export_dir,
            dataset_type=fo.types.COCODetectionDataset,
            label_field=label_field,
        )

        coco_dataset = CocoDetection(os.path.join(export_dir, 'data'), os.path.join(export_dir, 'labels.json'),
                                 transforms=dataloader.dataset.transforms, return_masks=False,
                                 cache_mode=False, local_rank=get_local_rank(), local_size=get_local_size())
    #     if args.run_mode == "sg_panoptic":
    #         coco_val = datasets.coco.build("val", args)
    #         base_ds = get_coco_api_from_dataset(coco_val)
    #     else:
    base_ds = get_coco_api_from_dataset(coco_dataset)
    coco_evaluator = CocoEvaluator(base_ds, ["bbox", ])

    return coco_evaluator