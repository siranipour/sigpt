import logging


class ColorFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.DEBUG: "\033[94m",
        logging.INFO: "\033[92m",
        logging.WARNING: "\033[93m",
        logging.ERROR: "\033[91m",
        logging.CRITICAL: "\033[95m",
    }
    RESET = "\033[0m"

    def format(self, record):
        log_color = self.LEVEL_COLORS.get(record.levelno, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger():
    logger = logging.getLogger("sigpt")
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = ColorFormatter(
        "[%(levelname)s] %(asctime)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S %Z"
    )
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    return logger


log = setup_logger()
