class BlackBoxFunc:
    '''
    Class representing a black box function, which
    internal workings are unknown
    '''
    def __init__(self, func, optimal_value):
        '''Initialize with a callable function'''
        self.func = func
        self.optimal_value = optimal_value

    def evaluate(self, x):
        '''Evaluate the black box function at x'''
        return self.func(x)
    
    def get_optimal_value(self):
        return self.optimal_value