import gpytorch


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, covar, mean):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean
        self.covar_module = covar

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
