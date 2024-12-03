from gp_models import get_gpmodel

def PES_Energy(qcoord):
    model = get_gpmodel()
    return model.mean(qcoord)

def PES_Hessian(qcoord):
    model = get_gpmodel()
    return model.hess(qcoord)

def PES_Force(qcoord):
    model = get_gpmodel()
    return -model.grad(qcoord)