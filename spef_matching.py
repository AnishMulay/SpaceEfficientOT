import numpy as np
import torch

def spef_matching_torch(
    xA,
    xB,
    C,
    delta,
    device,
    seed=1
):
    dtyp = torch.int64
    n = xA.shape[0]
    m = xB.shape[0]

    xA = xA.to(device=device, dtype=dtyp)
    xB = xB.to(device=device, dtype=dtyp)

    yB = torch.ones(m, device=device, dtype=dtyp, requires_grad=False)
    yA = torch.zeros(n, device=device, dtype=dtyp, requires_grad=False)
    Mb = torch.ones(m, device=device, dtype=dtyp, requires_grad=False) * -1
    Ma = torch.ones(n, device=device, dtype=dtyp, requires_grad=False) * -1

    n = xA.shape[0]
    zero = torch.tensor([0], device=device, dtype=dtyp, requires_grad=False)[0]
    one = torch.tensor([1], device=device, dtype=dtyp, requires_grad=False)[0]
    minus_one = torch.tensor([-1], device=device, dtype=dtyp, requires_grad=False)[0]

    f_threshold = n*delta/C
    torch.manual_seed(seed)

    f = n
    iteration = 0

    return Mb, yA, yB, None, iteration