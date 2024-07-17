from logging import DEBUG, Formatter, Logger, StreamHandler, getLogger


def get_logger(name: str) -> Logger:
    """Get a logger object."""
    logger = getLogger(name)
    logger.setLevel(DEBUG)
    handler = StreamHandler()
    handler.setLevel(DEBUG)
    formatter = Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# Instantiate the logger
logger = get_logger(__name__)
