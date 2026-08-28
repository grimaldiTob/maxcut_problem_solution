import numpy as np

class Wrapper:
    """Early-stopping wrapper around a scipy objective: keeps a sliding
    window of the last `window` objective values and raises StopIteration
    from the callback when cond_fun(window) <= threshold."""
    
    def __init__(self, obj_fun, cond_fun, threshold):
        self.obj_fun = obj_fun # objective function to evaluate
        self.cond_fun = cond_fun # conditional function to evaluate
        self.threshold = threshold # threshold
        self.last_vals = np.full(5, np.nan) # initialize as nan
        self.iters = 0

    def callback(self, xk=None, *_):
        if self.iters > 4 and self.cond_fun(self.last_vals) <= self.threshold:
            raise StopIteration

    def objective(self, x):
        # evaluate the obj_fun, update the last four objective values
        # and return the current objective value
        np.roll(self.last_vals, 1)
        self.last_vals[1:] = self.last_vals[:-1] # take the first 4 elements of the last vals array
        self.last_vals[0] = self.obj_fun(x)
        self.iters += 1
        return self.last_vals[0]
    
