import sys
import logging

def error_message(error, detail:sys):
    _,_,tb = sys.exc_info()
    file_name = tb.tb_frame.f_code.co_filename
    error_message = "error occured in file {} on line {} with error message {}".format(file_name, tb.tb_lineno, error)

    return error_message

class CustomException(Exception):
    def __init__(self, error_ms, detail:sys):
        super().__init__(error_ms)
        self.error = error_message(error_ms, error_detail = detail)

    def __str__(self):
        return self.error