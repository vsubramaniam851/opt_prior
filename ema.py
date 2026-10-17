class EMAMeter(object):
    def __init__(self, beta = 0.98):
        self.beta = beta
        self.val = None
        self.best = float('inf')
    def update(self, x):
        if self.val is None:
            self.val = x
        else:
            self.val = self.beta * self.val + (1 - self.beta) * x
        if self.val < self.best: 
            self.best = self.val
        return self.val