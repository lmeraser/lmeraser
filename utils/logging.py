# Description: Logging utilities
# Modified from [VPT](https://github.com/KMnP/vpt). Thanks to the authors.

import builtins
import decimal
import functools
import logging
import simplejson
import sys
import os
from termcolor import colored

from .file_io import PathManager

# Show filename and line number in logs
_FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"


def _suppress_print():
    """Suppresses printing from the current process."""

    def print_pass(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        pass

    builtins.print = print_pass


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    return PathManager.open(filename, "a")


@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers  # noqa
def setup_logging(
    num_gpu, output_dir="", name="visual_prompt", color=True, rank=None):
    """Sets up the logging."""

    rank_str = f"[GPU {rank}]" if rank is not None else ""
    _FORMAT = f"{rank_str}[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"

    # Clear the root logger to prevent any existing logging config
    # (e.g. set by another module) from messing with our setup
    logging.root.handlers = []
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format=_FORMAT, stream=sys.stdout
    )

    if name is None:
        name = __name__
    logger = logging.getLogger(name)
    # remove any lingering handler
    logger.handlers.clear()

    logger.setLevel(logging.INFO)
    logger.propagate = False

    plain_formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(name)s: %(lineno)4d: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )
    if color:
        formatter = RankFormatter(
            colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
            datefmt="%m/%d %H:%M:%S",
            root_name=name,
            abbrev_name=str(name),
            rank=rank,
        )
    else:
        formatter = plain_formatter

    ch = logging.StreamHandler(stream=sys.stdout)
    configure_and_add_handler(ch, logging, formatter, logger)

    if len(output_dir) > 0: # if save log to file
        filename = os.path.join(output_dir, f"rank_{rank}.log") if rank is not None else os.path.join(output_dir, "logs.txt")
        PathManager.mkdirs(os.path.dirname(filename))
        fh = logging.StreamHandler(_cached_log_stream(filename))
        configure_and_add_handler(fh, logging, plain_formatter, logger)

    return logger


def setup_single_logging(name, output=""):
    """Sets up the logging."""
    # Enable logging only for the master process
    # Clear the root logger to prevent any existing logging config
    # (e.g. set by another module) from messing with our setup
    logging.root.handlers = []
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format=_FORMAT, stream=sys.stdout
    )

    if len(name) == 0:
        name = __name__
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    plain_formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(name)s: %(lineno)4d: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )
    formatter = _ColorfulFormatter(
        colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
        datefmt="%m/%d %H:%M:%S",
        root_name=name,
        abbrev_name=str(name),
    )

    ch = logging.StreamHandler(stream=sys.stdout)
    configure_and_add_handler(ch, logging, formatter, logger)
    if len(output) > 0:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "logs.txt")

        PathManager.mkdirs(os.path.dirname(filename))

        fh = logging.StreamHandler(_cached_log_stream(filename))
        configure_and_add_handler(fh, logging, plain_formatter, logger)
    return logger


def configure_and_add_handler(handler, logging, formatter, logger):
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_logger(name):
    """Retrieves the logger."""
    return logging.getLogger(name)


def log_json_stats(stats, sort_keys=True):
    """Logs json stats."""
    # It seems that in Python >= 3.6 json.encoder.FLOAT_REPR has no effect
    # Use decimal+string as a workaround for having fixed length values in logs
    logger = get_logger(__name__)
    stats = {
        k: decimal.Decimal("{:.6f}".format(v)) if isinstance(v, float) else v
        for k, v in stats.items()
    }
    json_stats = simplejson.dumps(stats, sort_keys=True, use_decimal=True)
    if stats["_type"] in ["test_epoch", "train_epoch"]:
        logger.info("json_stats: {:s}".format(json_stats))
    else:
        logger.info("{:s}".format(json_stats))


class _ColorfulFormatter(logging.Formatter):
    # from detectron2
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = f"{self._abbrev_name}."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def get_log(self, record: logging.LogRecord) -> str:
        return super(_ColorfulFormatter, self).formatMessage(record)
    
    def get_prefix(self, record: logging.LogRecord) -> str:
        if record.levelno == logging.WARNING:
            return colored("WARNING", "red", attrs=["blink"])
        elif record.levelno in [logging.ERROR, logging.CRITICAL]:
            return colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return ""

    def formatMessage(self, record: logging.LogRecord) -> str:
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = self.get_log(record)
        prefix = self.get_prefix(record)
        return " ".join([prefix, log]) if prefix else log

class RankFormatter(_ColorfulFormatter):
    def __init__(self, *args, **kwargs):
        self._rank = None if "rank" not in kwargs else kwargs.pop("rank")
        super().__init__(*args, **kwargs)

    def formatMessage(self, record: logging.LogRecord) -> str:
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        prefix = self.get_prefix(record)
        rank_str = colored(f"[GPU {self._rank}]", "cyan") if self._rank is not None else ""
        log = " ".join([rank_str, self.get_log(record)]) if self._rank is not None else self.get_log(record)
        return " ".join([prefix, log]) if prefix else log
        
