import sys

def error_message_detail(err, err_detail: sys) -> str:
    """This function is used to get the error message and the details of the error.

    Args:
        err (_type_): Error message
        err_detail (sys): The error details

    Returns:
        str: The error message with the details
    """
    _, _, exc_tb = err_detail.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    err_message = "Error: [{0}] in [{1}] at line [{2}]".format(str(err), filename, exc_tb.tb_lineno)
    return err_message
    
class CustomException(Exception):
    def __init__(self, err_message, err_detail: sys):
        super().__init__(err_message)
        self.err_message = error_message_detail(err_message, err_detail)
        
    def __str__(self):
        return self.err_message