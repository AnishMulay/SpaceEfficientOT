import torch
import triton
import triton.language as tl

# -----------------------------
# 4D spherical embedding utils
# -----------------------------
@torch.compile(mode="reduce-overhead", dynamic=True)
def spherical_embed(lat_deg: torch.Tensor, lon_deg: torch.Tensor) -> torch.Tensor:
    """Return (N,4) embedding on the same device/dtype as inputs.
    lat_deg, lon_deg: shape (N,), degrees.
    """
    pi = torch.pi
    lat = lat_deg * (pi/180.0)
    lon = lon_deg * (pi/180.0)
    sφ, cφ = torch.sin(lat), torch.cos(lat)
    sλ, cλ = torch.sin(lon), torch.cos(lon)
    return torch.stack((cφ*cλ, cφ*sλ, sφ*cλ, sφ*sλ), dim=1)

# --------------------------------------------
# Triton kernel: fused slack for one B-tile
# --------------------------------------------
@triton.jit
def _fused_slack_kernel(
    E_A_ptr, E_B_ptr, yA_ptr, yB_ptr, out_ptr,
    NA: tl.constexpr, KB: tl.constexpr,
    stride_EA, stride_EB, stride_out,
    R: tl.constexpr, USE_METERS: tl.constexpr,
    delta: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Program ids
    pid_m = tl.program_id(0)  # rows over B tile
    pid_n = tl.program_id(1)  # cols over A

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows [0..KB)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols [0..NA)

    mask_m = offs_m < KB
    mask_n = offs_n < NA

    # Load yB(rows) and yA(cols)
    yB = tl.load(yB_ptr + offs_m, mask=mask_m, other=0.0)
    yA = tl.load(yA_ptr + offs_n, mask=mask_n, other=0.0)

    # Compute M = E_B @ E_A^T via explicit reduction over 4 dims
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduce over k=0..3 (embedding dim)
    for k in range(4):
        # Pointers for E_B[:, k]
        EB_k = E_B_ptr + offs_m * stride_EB + k
        # Pointers for E_A[:, k]
        EA_k = E_A_ptr + offs_n * stride_EA + k
        # Load vectors
        b = tl.load(EB_k, mask=mask_m, other=0.0)[:, None]          # (BLOCK_M,1)
        a = tl.load(EA_k, mask=mask_n, other=0.0)[None, :]          # (1,BLOCK_N)
        acc += b * a

    # Now acc = M (dot product of unit vectors)
    if USE_METERS:
        # Use small-angle approximation: distance ≈ 2*R*sqrt(0.5*(1-M))
        h = 0.5 * (1.0 - acc)
        h = tl.maximum(h, 0.0)  # clamp for safety
        cost = 2.0 * R * tl.sqrt(h)
    else:
        # squared-chord cost = 4R^2 h
        h = 0.5 * (1.0 - acc)
        h = tl.maximum(tl.minimum(h, 1.0), 0.0)  # clamp for safety
        cost = 4.0 * (R*R) * h

    # slack = cost - yA[col] - yB[row] - delta
    slack = cost - yB[:, None] - yA[None, :] - delta

    # Write out
    out_row_ptrs = out_ptr + offs_m[:, None] * stride_out + offs_n[None, :]
    tl.store(out_row_ptrs, slack, mask=(mask_m[:, None] & mask_n[None, :]))


def fused_slack_haversine(
    latA_deg: torch.Tensor,
    lonA_deg: torch.Tensor,
    latB_deg: torch.Tensor,
    lonB_deg: torch.Tensor,
    yA: torch.Tensor,
    yB_tile: torch.Tensor,
    out: torch.Tensor,
    *,
    use_meters: bool = True,
    R: float = 6371000.0,
    delta: float = 0.0,
    block_m: int = 64,
    block_n: int = 128,
):
    """
    Compute slack for one B-tile against all A in a single Triton kernel.
    """
    device = latA_deg.device
    assert device.type == "cuda"

    # Embeddings (1D elementwise only, cheap)
    EA = spherical_embed(latA_deg, lonA_deg).to(dtype=torch.float32)
    EB = spherical_embed(latB_deg, lonB_deg).to(dtype=torch.float32)

    NA = EA.shape[0]
    KB = EB.shape[0]

    # Ensure contiguous row-major layout
    EA = EA.contiguous()
    EB = EB.contiguous()
    yA = yA.contiguous()
    yB_tile = yB_tile.contiguous()

    assert out.shape == (KB, NA)
    assert out.is_cuda and out.dtype == torch.float32

    grid = (triton.cdiv(KB, block_m), triton.cdiv(NA, block_n))

    _fused_slack_kernel[grid](
        EA, EB, yA, yB_tile, out,
        NA, KB,
        EA.stride(0), EB.stride(0), out.stride(0),
        R, int(use_meters), float(delta),
        block_m, block_n,
        num_warps=4,
        num_stages=2,
    )
    return out

# Fallback CPU implementation for Haversine distance
def haversine_distance_cpu(lat1, lon1, lat2, lon2, R=6371000.0):
    """
    Calculate Haversine distance on CPU.
    All inputs in degrees, returns distance in meters.
    """
    # Convert to radians
    lat1_rad = torch.deg2rad(lat1)
    lon1_rad = torch.deg2rad(lon1)
    lat2_rad = torch.deg2rad(lat2)
    lon2_rad = torch.deg2rad(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = torch.sin(dlat/2)**2 + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon/2)**2
    c = 2 * torch.asin(torch.sqrt(torch.clamp(a, 0, 1)))
    
    return R * c