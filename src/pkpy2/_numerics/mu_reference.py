"""Explicit conditional log-mean regression compiled from the PK effect plan."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import lsq_linear


@dataclass
class MuReference:
    coordinates: np.ndarray
    design: np.ndarray
    prior_center: np.ndarray
    prior_precision: np.ndarray
    eta_indices: np.ndarray

    def solve(self,x,decode,target,bounds):
        p,omega,_,_=decode(x)
        mu=np.log(np.asarray(p)[:,self.eta_indices])
        offset=mu.ravel()-self.design@x[self.coordinates]
        scale=np.broadcast_to(np.sqrt(omega),mu.shape).ravel()
        a=self.design/scale[:,None];b=(target.ravel()-offset)/scale
        active=self.prior_precision>0
        if np.any(active):
            prior=np.diag(np.sqrt(self.prior_precision))[active]
            a=np.vstack((a,prior));b=np.r_[b,prior@self.prior_center]
        lower=np.full(len(self.coordinates),-np.inf);upper=-lower
        if bounds is not None:
            for k,j in enumerate(self.coordinates):
                lo,hi=bounds[j]
                if lo is not None:lower[k]=lo
                if hi is not None:upper[k]=hi
        if np.all(np.isneginf(lower)) and np.all(np.isposinf(upper)):
            beta=np.linalg.lstsq(a,b,rcond=1e-10)[0]
        else:
            fit=lsq_linear(a,b,bounds=(lower,upper),tol=1e-10,max_iter=100)
            if not fit.success:return x.copy(),False
            beta=fit.x
        result=x.copy();result[self.coordinates]=beta
        return result,bool(np.isfinite(result).all())


def compile_mu_reference(effects,eta_names,free_theta,coefficient_start,prior_terms,
                         excluded_theta=(),excluded_coefficients=()):
    """Use declared power/exponential designs; never infer linearity by probes.

    Nonlinear log(1+beta*z) terms and nonquadratic hint penalties remain in the
    general optimizer. Their contributions are conditional offsets in this block.
    """
    eta_indices=np.array([effects.parameter_names.index(n) for n in eta_names],dtype=int)
    n=effects.design.shape[0];q=len(eta_names);columns=[];coordinates=[];centers=[];precisions=[]
    for j,name in enumerate(free_theta):
        if name not in eta_names or name in excluded_theta:continue
        column=np.zeros((n,q));column[:,eta_names.index(name)]=1.
        columns.append(column.ravel());coordinates.append(j)
        prior=prior_terms.get(name)
        centers.append(prior['log_center'] if prior else 0.)
        precisions.append(1./prior['log_sd']**2 if prior else 0.)
    for c,original in enumerate(effects.order):
        target=int(effects.targets[c])
        if effects.linear[c] or original in excluded_coefficients or target not in eta_indices:continue
        column=np.zeros((n,q));column[:,list(eta_indices).index(target)]=effects.design[:,c]
        columns.append(column.ravel());coordinates.append(coefficient_start+int(original))
        centers.append(0.);precisions.append(0.)
    if not columns:return None
    return MuReference(np.array(coordinates,dtype=int),np.column_stack(columns),
                       np.array(centers),np.array(precisions),eta_indices)
