from dataclasses import dataclass, field


@dataclass
class DataArguments:
    """
    Arguments relating to data.
    """

    train_data_path: str = field(
        default="./data",
        metadata={"help": "path to data, splited by comma"},
    )
    val_data_path: str = field(
        default="./data",
        metadata={"help": "path to data, splited by comma"},
    )
    fold: int = field(default=0, metadata={"help": "fold number"})
    image_size: int = field(default=256, metadata={"help": "image size"})
    negative_ratio: float = field(default=5.0, metadata={"help": "negative ratio"})
