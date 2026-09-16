"""Fixed balanced Gaussian mixture covering two population endpoints."""
import numpy as np
from scipy.special import logsumexp
from .importance_objective import PhysicalImportance,PhysicalSnapshot
from .packed_importance import normal_logpdf


def endpoint_bridge(study,indices,left,right,*,power=16,seed=0,work=None):
    """Use half the total particle budget at each endpoint, full mixture weights.

    Endpoint tuples are (typical, omega, proportional sigma, additive sigma).
    No component-specific self-density approximation or target-dependent weights.
    """
    if type(power) is not int or not 3<=power<=24:raise ValueError('bridge power must be in [3,24]')
    if not len(indices):return PhysicalImportance(study,indices,*left,power=power,seed=seed,work=work)
    streams=np.random.SeedSequence(seed).spawn(2)
    banks=[PhysicalImportance(study,indices,*point,power=power-1,
        seed=int(stream.generate_state(1,dtype=np.uint64)[0]),work=work)
        for point,stream in zip((left,right),streams)]
    samples=[];densities=[];components=[]
    for i in range(len(study.subject_ids)):
        phi=np.ascontiguousarray(np.concatenate([b.phi[i] for b in banks]))
        means=[];covs=[]
        for bank,point in zip(banks,(left,right)):
            proposal=bank.boundary.proposals[i]
            if len(proposal.centers)!=proposal.components:raise ValueError('missing proposal density components')
            means.extend(c+np.log(np.asarray(point[0])[i,list(indices)]) for c in proposal.centers)
            covs.extend(proposal.covariances)
        if banks[0].boundary.proposals[i].components!=banks[1].boundary.proposals[i].components:
            raise ValueError('bridge requires equally stratified endpoint proposals')
        logq=logsumexp(np.array([normal_logpdf(phi,m,c) for m,c in zip(means,covs)]),axis=0)-np.log(len(means))
        samples.append(phi);densities.append(logq);components.append(len(means))
    # The normal evaluator will compute predictions once for these fixed particles.
    # A zero-column cache with an impossible key forces the normal invalidation path.
    snapshot=PhysicalSnapshot(study,tuple(indices),tuple(samples),tuple(densities),
        tuple(np.empty((len(a),0)) for a in samples),tuple(None for _ in samples),tuple(components))
    return PhysicalImportance.from_snapshot(snapshot,study,indices,work)
