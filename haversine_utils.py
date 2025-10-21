import torch


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
