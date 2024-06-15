import numpy as np

def compute_r2(prot_hat, prot, gex_hat, gex):
    """ Shape samples* features """
    def r2(y, yhat):
        SST = np.sum((y - np.mean(y))**2)
        SSR = np.sum((y - yhat)**2)
        return 1 - SSR/SST
    
    return dict(
        protein=r2(prot, prot_hat), 
        gex=r2(gex, gex_hat), 
        total=r2(np.hstack([prot, gex]), np.hstack([prot_hat, gex_hat])),
    )
