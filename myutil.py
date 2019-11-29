"""General utility functions"""
import time

# Like the Matlab tic toc
def tic():
    return time.time()

def toc(t, string='Elapsed time'):
    print(string + ' = ', time.time() - t, 's')
    return time.time()