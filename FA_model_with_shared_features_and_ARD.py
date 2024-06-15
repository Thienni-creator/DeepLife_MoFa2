import pyro
from pyro.nn import PyroSample, PyroModule
from pyro.infer import SVI, Trace_ELBO, autoguide
import torch
from torch.nn.functional import softplus
from sklearn.metrics import mean_squared_error
import random
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ann
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
import numpy as np
import tempfile
#from numpyro.distributions import TransformedDistribution, transforms
from torch.distributions.transforms import SoftmaxTransform
from scipy.stats import spearmanr, pearsonr
import scanpy as sc
import anndata as ad
import pandas as pd
import scipy.sparse
import os, time, math
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


## set modelname for saving
modelname = "FA_model_sharedWeights_ARD_10factors"
if os.path.exists("/scratch/deeplife/projekt/data/neurips2021.h5ad"):
    data_dir = "/scratch/deeplife/projekt/data/"
    model_dir = "/scratch/deeplife/projekt/models/"
else:
    data_dir = "/mnt/storage/anna/data/"
    model_dir = "/mnt/storage/anna/"

print(f"{data_dir=}")

# simulate data
n_obs = 100
n_features1 = 20
n_features2 = 30
n_factors = 20

random.seed(2024)
torch.manual_seed(2024)
np.random.seed(2024)

device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else 'cpu'

torch.set_default_device(device)

# percentage = 0.3 # percentage of values set to zero

# Z_in = torch.randn(n_obs, n_factors)
# W1_in = torch.randn(n_features1, n_factors)
# W2_in = torch.randn(n_features2, n_factors)

# # create observated values from the simulated factor and weight matrix with some random noise\n,
# Y1 = torch.matmul(Z_in, W1_in.t()) #+ 0.2 * torch.randn(n_obs, n_features1)
# Y2 = torch.matmul(Z_in, W2_in.t()) #+ 0.2 * torch.randn(n_obs, n_features2)

# #try creating shared factors which are correlating
# Y1[:,:9] = torch.sort(Y1[:,:9]).values
# Y2[:,:9] = torch.sort(Y2[:,:9]).values

# #introducing NANs/0 into dataframe
# Y1_indices = np.random.choice(Y1.shape[1]*Y1.shape[0], replace=False, size=int(Y1.shape[1]*Y1.shape[0]*percentage))
# Y1[np.unravel_index(Y1_indices, Y1.shape)] = 0 

# Y2_indices = np.random.choice(Y2.shape[1]*Y2.shape[0], replace=False, size=int(Y2.shape[1]*Y2.shape[0]*percentage))
# Y2[np.unravel_index(Y2_indices, Y2.shape)] = 0 

# Y = torch.cat((Y1.T, Y2.T)).T
# Y.to(device)

def to_torch_sparse(matrix: scipy.sparse.csr_matrix, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    return torch.sparse_csr_tensor(matrix.indptr, matrix.indices, matrix.data, size=matrix.shape, device=device)
def to_device(t): return torch.tensor(t).to(device)

class FA(PyroModule):
    def __init__(self, dat, n_features1, n_features2, K, batch_size = 1024):
        """
        Args:
            Y: Tensor (Samples x Features)
            K: Number of Latent Factors
        """
        super().__init__()
        pyro.clear_param_store()

        # data
        self.Y1 = dat[:,:n_features1]
        self.Y2 = dat[:,n_features1:]
        self.Y = dat
        self.K = K
        self.batch_size = batch_size
        

        ### reorder Y1 and Y2 such that shared genes/prots are at the end
        shared_names = [name  for name in list(self.Y1.var_names) if name in list(self.Y2.var_names)]
        # Y1
        Y1_idx_shared = [list(self.Y1.var_names).index(name) for name in shared_names]
        self.Y1 = ad.concat([self.Y1, self.Y1[:,Y1_idx_shared]], axis = 1, merge="same") #add to bottom
        Y1_mask = np.ones(self.Y1.n_vars, dtype=bool) # drop from within
        Y1_mask[Y1_idx_shared] = False
        self.Y1 = self.Y1[:,Y1_mask]
        # Y2
        Y2_idx_shared = [list(self.Y2.var_names).index(name) for name in shared_names]
        self.Y2 = ad.concat([self.Y2, self.Y2[:,Y2_idx_shared]], axis = 1, merge="same") #add to bottom
        Y2_mask = np.ones(self.Y2.n_vars, dtype=bool) # drop from within
        Y2_mask[Y2_idx_shared] = False
        self.Y2 = self.Y2[:,Y2_mask]

        # self.Y1 = torch.tensor(self.Y1.X.A).to(device)## extract only matrices 
        # self.Y2 = torch.tensor(self.Y2.X.A).to(device)

        self.num_shared = len(shared_names)
        self.num_features1 = n_features1-self.num_shared
        self.num_features2 = n_features2-self.num_shared


        self.num_samples = self.Y1.shape[0]
        #self.batch_size = batch_size
        self.feature_plate1 = pyro.plate("feature1", self.num_features1)
        self.feature_plate2 = pyro.plate("feature2", self.num_features2)
        self.shared_plate = pyro.plate("shared_features", self.num_shared)
        self.latent_factor_plate = pyro.plate("latent factors", self.K)
        self.to(device)
        self.subsample_ix = None
        self.num_samples = self.Y1.shape[0]

    def model(self, subsample_ix):
        """
        how to generate a matrix
        """
        with self.latent_factor_plate:
            theta1 = pyro.sample("theta1", pyro.distributions.Beta(to_device(1.), 1.))
            alpha1 = pyro.sample("alpha1", pyro.distributions.Gamma(to_device(10.), to_device(10.)))
            with self.feature_plate1:
                # sample weight matrix with Normal prior distribution
                #W1_hat = pyro.sample("W1", pyro.distributions.Normal(0., 1.)).to(device)
                W1_hat = pyro.sample("W1", pyro.distributions.Normal(to_device(0.), 1./alpha1))*pyro.sample("s1", pyro.distributions.Bernoulli(theta1))
            theta2 = pyro.sample("theta2", pyro.distributions.Beta(to_device(1.), 1.))
            alpha2 = pyro.sample("alpha2", pyro.distributions.Gamma(to_device(10.), to_device(10.)))
            with self.feature_plate2:
                # sample weight matrix with Normal prior distribution
                #W2_hat = pyro.sample("W2", pyro.distributions.Normal(0., 1.)).to(device)              
                W2_hat = pyro.sample("W2",pyro.distributions.Normal(to_device(0.), 1./alpha2))*pyro.sample("s2",pyro.distributions.Bernoulli(theta2))
            theta_shared = pyro.sample("theta_shared", pyro.distributions.Beta(to_device(1.), 1.))
            alpha_shared = pyro.sample("alpha_shared", pyro.distributions.Gamma(to_device(10.), to_device(10.)))
            with self.shared_plate:
                # sample weight matrix with Normal prior distribution
                #W_shared_hat = pyro.sample("W_shared", pyro.distributions.Normal(0., 1.)).to(device)  
                W_shared_hat = pyro.sample("W_shared",pyro.distributions.Normal(to_device(0.), 1./alpha_shared))*pyro.sample("s_shared",pyro.distributions.Bernoulli(theta_shared))
                
            with pyro.plate("sample", self.num_samples, subsample=subsample_ix, subsample_size=len(subsample_ix)):
                # sample factor matrix with Normal prior distribution
                Z = pyro.sample("Z", pyro.distributions.Normal(to_device(0.), 1.))
                print(Z.shape, len(subsample_ix))
        
        ## concat shared to W1_hat and W2_hat
        W1_hat = torch.cat((W1_hat, W_shared_hat), dim =0)
        W2_hat = torch.cat((W2_hat, W_shared_hat), dim = 0)                
        # estimate for Y
        Y1_hat = torch.matmul(Z, W1_hat.t())
        Y2_hat = torch.matmul(Z, W2_hat.t())
        print(f"{Y1_hat.shape=} {Y2_hat.shape=} {len(subsample_ix)}")
        
        with pyro.plate("feature1_", self.Y1.shape[1]): 
            precision1 = pyro.sample("precision1", pyro.distributions.Gamma(torch.tensor(20.), torch.tensor(20.))).to(device)
                    #precision1 = 0.6
                    #print(precision1)
            scale = pyro.sample("scale1", pyro.distributions.LogNormal(0., 1./(precision1))).to(device)
            print(len(subsample_ix))
            with pyro.plate("sample_", self.num_samples, subsample=subsample_ix):
                # sample scale parameter for each feature-sample pair with LogNormal prior (has to be positive)
                #print(scale)
                # compare sampled estimation to the true observation Y
                Y = to_torch_sparse(self.Y1.X[subsample_ix], device=device).to_dense()
                pyro.sample("obs1", pyro.distributions.Normal(Y1_hat, scale), obs=Y).to(device)
                #Y1_sampled = pyro.distributions.Normal(Y1_hat, scale)
                #pyro.sample("obs1", pyro.distributions.TransformedDistribution(Y1_sampled, SoftmaxTransform()), obs=self.Y1)


        with pyro.plate("feature2_", self.Y2.shape[1]):
            precision2 = pyro.sample("precision2", pyro.distributions.Gamma(torch.tensor(20.), torch.tensor(20.))).to(device)
                    #precision2 = 0.6
            scale = pyro.sample("scale2", pyro.distributions.LogNormal(0., 1./(precision2))).to(device)
            with pyro.plate("sample2_", self.num_samples, subsample=subsample_ix):
                # masking the NA values such that they are not considered in the distributions
                # sample scale parameter for each feature-sample pair with LogNormal prior (has to be positive)
                # compare sampled estimation to the true observation Y
                Y = to_torch_sparse(self.Y2.X[subsample_ix], device=device).to_dense()
                pyro.sample("obs2", pyro.distributions.Normal(Y2_hat, scale), obs=Y).to(device)
                #Y2_sampled = pyro.distributions.Normal(Y2_hat, scale)
                #pyro.sample("obs2", pyro.distributions.TransformedDistribution(Y2_sampled, SoftmaxTransform()), obs=self.Y2)


    def train(self):
        # set training parameters
        optimizer = pyro.optim.Adam({"lr": 0.02})
        elbo = Trace_ELBO()
        #data_loader = torch.utils.data.DataLoader(self.Y, batch_size=self.batch_size, shuffle=True, generator=torch.Generator(device='cuda'), drop_last = True)
        # guide = autoguide.AutoNormal(self.model)
        guide = autoguide.AutoGuideList(self.model).to(device)
        guide.append(autoguide.AutoNormal(pyro.poutine.block(self.model, hide=['s1', "s2", "s_shared"])))
        guide.append(autoguide.AutoDiscreteParallel(pyro.poutine.block(self.model, expose=["s1", "s2", "s_shared"])))
        
        # initialize stochastic variational inference
        svi = SVI(
            model = self.model,
            guide = guide,
            optim = optimizer,
            loss = elbo
        )

        t0 = time.time()
        
        num_epochs = 100
        train_loss = []
        train_loss = []
        for epoch_i in range(num_epochs):
            indices = np.arange(self.num_samples)
            np.random.shuffle(indices)
            for batch_i in range(math.ceil(len(indices) / self.batch_size)):
                subsample_ix = indices[batch_i * self.batch_size: (batch_i + 1) * self.batch_size]
                # calculate the loss and take a gradient step (loss should be already scaled down by the pyro.places)
                loss = svi.step(subsample_ix)

                train_loss.append(loss/self.num_samples)
            print("[%02d:%02.1d iteration %02d:%05d] loss: %.4f" % (
                round((time.time() - t0)/60),
                round(time.time() - t0, ndigits=1) % 60,
                epoch_i+1,
                ((epoch_i+1)*math.ceil(len(indices) / self.batch_size)),
                loss / self.num_samples))

        torch.save({"model": self.state_dict(), "guide" : guide}, model_dir+f"{modelname}.pt")
        pyro.get_param_store().save(model_dir+f"{modelname}_params.pt")
            
        return train_loss

if __name__ == "__main__":
    Cite_GEX = sc.read_h5ad(data_dir+"Cite_GEX_preprocessed.h5ad")
    Cite_ADT = sc.read_h5ad(data_dir+"Cite_ADT_preprocessed.h5ad")

    Cite_GEX.var_names = [name[:-2] if "-1" in name else name for name in Cite_GEX.var_names] ## make var names not unique to find
    print(f"matrix density Y_GEX={np.sum(Cite_GEX.X != 0) / np.product(Cite_GEX.X.shape)} Y_ADT={np.sum(Cite_ADT.X != 0) / np.product(Cite_ADT.X.shape)}")

    Y = ad.concat([Cite_ADT, Cite_GEX], axis = 1, merge="same")
    #Y = torch.tensor(Y.X.A).to(device)
    n_factors = 10
    FA_model = FA(Y,Cite_ADT.shape[1], Cite_GEX.shape[1], n_factors)
    losses = FA_model.train()
    pd.DataFrame(losses).to_csv(model_dir+f"{modelname}.csv")

    #FA_model = FA(Y,20, 30, n_factors)
    #losses, estimates = FA_model.train()


    # Y1_hat = torch.matmul(torch.Tensor(estimates["Z"].detach().cpu().numpy()),torch.Tensor(estimates["W1"].detach().cpu().numpy()).t())
    # correlations_pearson = []
    # correlations_spearman = []
    # for observation in range(n_obs):
    #     correlations_pearson_temp = []
    #     correlations_spearman_temp = []
    #     for observation2 in range(n_obs):
    #         correlations_pearson_temp.append(pearsonr(Y1[observation,:], Y1_hat[observation2,:]))
    #         correlations_spearman_temp.append(spearmanr(Y1[observation,:], Y1_hat[observation2,:], axis=0))
    #     correlations_pearson.append(max(correlations_pearson_temp))
    #     correlations_spearman.append(max(correlations_spearman_temp))

    # correlations_pearson
    # correlations_spearman



    #load data
    #Cite_GEX = sc.read_h5ad(data_dir+"Cite_GEX_preprocessed.h5ad")
    #Cite_ADT = sc.read_h5ad(data_dir+"data/Cite_ADT_preprocessed.h5ad")

    #Y = ad.concat([Cite_ADT, Cite_GEX], axis = 1, merge="same")
    #Y = torch.tensor(Y.X.A)
