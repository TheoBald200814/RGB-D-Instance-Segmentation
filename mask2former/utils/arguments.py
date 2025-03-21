from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


@dataclass
class Arguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify
    them on the command line.
    """

    model_name_or_path: str = field(
        default="facebook/instance_seg-swin-tiny-coco-instance",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    dataset_name: str = field(
        default="qubvel-hf/ade20k-mini",
        metadata={
            "help": "Name of a dataset from the hub (could be your own, possibly private dataset hosted on the hub)."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    image_height: Optional[int] = field(default=512, metadata={"help": "Image height after resizing."})
    image_width: Optional[int] = field(default=512, metadata={"help": "Image width after resizing."})
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    do_reduce_labels: bool = field(
        default=False,
        metadata={
            "help": (
                "If background class is labeled as 0 and you want to remove it from the labels, set this flag to True."
            )
        },
    )
    label2id_path: str = field(
        default='',
        metadata={
            "help": (
                "the path of label2id.json"
            )
        }
    )
    ignore_index: int = field(
        default=0,
        metadata={
            "help": (
                "ignore index in the dataset"
            )
        }
    )
    root_path: str = field(
        default='',
        metadata={
            "help": (
                "the path of root"
            )
        }
    )
    train_json_path: str = field(
        default='',
        metadata={
            "help": (
                "the path of train json file"
            )
        }
    )
    valid_json_path: str = field(
        default='',
        metadata={
            "help": (
                "the path of valid json file"
            )
        }
    )
    rgb_d: bool = field(
        default=False,
        metadata={
            "help": (
                "input data contain rgb and depth"
            )
        }
    )