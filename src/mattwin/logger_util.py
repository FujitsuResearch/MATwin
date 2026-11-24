import logging
import os

def create_logger(name, log_level=logging.INFO, folder_path=".", filename="training_info.log"):
    """
    Creates a logger object that can be used to log the details across MaTTwin operations like data preprcessing, training, validation or intervention.

    Args:
        name (str): The name of the logger.  This should be descriptive or use __name__ to know the source of your log statements
        log_level (int): The logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING). Defaults to logging.INFO.
        folder_path (str): The folder path where the log file will be created. Defaults to the current directory.
        filename (str): The name of the log file. Defaults to "training_info.log".

    Returns:
        logging.Logger: A configured logger object.
    """

    # Ensure the output directory exists
    output_path = os.path.join(folder_path, "log")
    os.makedirs(output_path, exist_ok=True)
    print("Created folder ",output_path)
    # Create the logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Check if handlers already exist to prevent duplicate logging
    if not logger.hasHandlers():

        # Create a file handler
        log_file_path = os.path.join(output_path, filename)
        print("Log file path:",log_file_path)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(log_level)

        ## Create a console handler. console handler is optional and can be removed once all wrks fine!!
        # console_handler = logging.StreamHandler()
        #console_handler.setLevel(log_level)

        # Create a formatter and set it for both handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        # console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(file_handler)
        # logger.addHandler(console_handler)  #console handler is optional and can be removed once all wrks fine!!

    return logger