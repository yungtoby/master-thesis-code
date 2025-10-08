class BlackBoxFunc:
    '''
    Class representing a black box function, which
    internal workings are unknown
    '''
    def __init__(self, func):
        '''Initialize with a callable function'''
        self.func = func

    def evaluate(self, x):
        '''Evaluate the black box function at x'''
        return self.func(x)