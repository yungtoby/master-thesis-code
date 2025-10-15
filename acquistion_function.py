from abc import abstractmethod, ABC
import torch as t

class AcquisitionFunction(ABC):
    
    @abstractmethod
    def compute(self, surrogate_model, candidate_points):
        '''
        Compute method to compute the next query point based
        on surrogate model and corresponding candidate points
        '''
        raise NotImplementedError
    

class EI(AcquisitionFunction):
    def __init__(self, epsilon = 0.01):
        '''Initialize EI with corresponding epsilon (exploration coeff)'''
        self.epsilon = epsilon    
    
    def compute(self, surrogate_model, candidate_points):
        posterior = surrogate_model.predict(candidate_points)
        mean = posterior.mean
        std = posterior.stddev

        # NOTE: Might be slow, consider precomputing best_y and saving in surrogate model.
        best_y = surrogate_model.train_y.max()

        # Compute EI
        with t.no_grad():
            # To avoid division by zero error
            std_safe = std + 1e-9
            z = (mean - best_y - self.epsilon) / std_safe

            normal = t.distributions.Normal(0, 1)
            pdf = normal.log_prob(z).exp()
            cdf = normal.cdf(z)

            expected_imp = (mean - best_y - self.epsilon) * cdf + std * pdf
        
        return expected_imp

class UCB(AcquisitionFunction):
    def compute():
        pass