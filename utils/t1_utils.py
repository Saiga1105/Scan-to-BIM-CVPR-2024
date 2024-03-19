
import time

def t1_time_funtion(func, *args):
    """Measures how long the functions takes to run and returns the result 

    Args:
        func (function): The funtion to measure, write without ()
        *args (Tuple) : The arguments for the funtion, pass as a tuple with a * in front to pass the arguments seperatly

    Returns:
        object: The result of the function
    """

    start = time.time()
    result = func(*args)
    end = time.time()
    print("Completed function `" + func.__name__ + "()` in", np.round(end - start,3), "seconds")
    return result 