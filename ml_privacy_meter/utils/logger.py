import logging
import os


def get_logger(root_dir, name, nature, filename, time_stamp, level=logging.INFO,
               formatter='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    """
    Logger function to log and flag various training and test
    time checkpoints. All log files are placed in:
    `${root}/logs/`.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    log_directory = os.path.join(root_dir, "logs",
                                 "latest",
                                 nature,
                                 "log_files")
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    log_file = os.path.join(log_directory, filename+".log")

    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    formatter = logging.Formatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def log_accuracies(logger, best_train_acc, best_test_acc):
    """
    Logs best training and testing accuracies
    """
    logger.info("Train accuracy is : %.2f%%", 100 * best_train_acc)
    logger.info("Test accuracy is : %.2f%%", 100 * best_test_acc)
