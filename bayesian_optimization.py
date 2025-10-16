import torch as t
import matplotlib.pyplot as plt


class BO_Pipeline:
    def __init__(self, device, dtype, cost_budget, surrogate_model, acq_function, candidate_set, blackbox_func):
        # Initialize device and dtype
        self.device = t.device(device)
        self.dtype = dtype

        # Initialize BO params
        self.cost_budget = cost_budget
        self.surrogate_model = surrogate_model
        self.acq_function = acq_function
        self.candidate_set = candidate_set
        self.blackbox_func = blackbox_func


    def run_BO(self, num_steps):
        init_best_ind = t.argmax(self.surrogate_model.train_y)
        best_found_y = self.surrogate_model.train_y[init_best_ind]
        best_found_x = self.surrogate_model.train_x[init_best_ind]

        grid = self.candidate_set.get_grid()
        self.surrogate_model.train()

        for _ in range(num_steps):
            # Retrieve scores
            scores = self.acq_function.compute(self.surrogate_model, grid)

            # Set already visited x_s to -inf
            scores = scores.masked_fill(self.candidate_set.get_mask(), -t.inf)
            
            # pick new x_next and evaluate y_next
            x_next_ind = t.argmax(scores)
            x_next = grid[x_next_ind]
            y_next = self.blackbox_func.evaluate(x_next)

            # Mark chosen candidate as used
            self.candidate_set.mark_as_visited(x_next_ind)

            # NOTE: For safe comparison, in case of shape mismatch
            if (y_next > best_found_y).any():
                best_found_y = y_next
                best_found_x = x_next

            self.surrogate_model.add_data(x_next, y_next)
            self.surrogate_model.train()

        return best_found_x, best_found_y
    


















    ##########
    # UNUSED #
    ##########
    def _visualize_BO_1D(self, x_s, iter_count):
        with t.no_grad():
            f, ax = plt.subplots(1, 1, figsize=(14, 7))
            
            x_s = x_s.squeeze()
            observed_pred = self.surrogate_model.predict(x_s)
            lower, upper = observed_pred.confidence_region()


            ax.set_title(f"BO Loop - Iteration {iter_count}")

            ax.plot(self.surrogate_model.train_x.numpy(), self.surrogate_model.train_y.numpy(), 'k.')
            ax.plot(x_s.numpy(), observed_pred.mean.numpy(), 'b')
            ax.fill_between(x_s.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
            ax.plot(x_s.numpy(), self.blackbox_func.evaluate(x_s), 'r--')
            
            ax.set_ylim([-10, 10])
            ax.legend(['Observed Data', 'Mean', 'Confidence', 'True function'])

            plt.show()