def assert_cuda(t):
    assert isinstance(t, torch.Tensor) and t.is_cuda

def test_basic_push_pop():
    dq = CudaDeque(max_size=10, device=torch.device('cuda'))
    x = torch.arange(0, 5, device='cuda', dtype=torch.int64)
    dq.push_back(x)
    assert int(dq.current_size().item()) == 5
    y = dq.pop_front(5)
    assert_cuda(y)
    assert y.device.type == 'cuda'
    assert torch.equal(y, x)
    assert int(dq.current_size().item()) == 0

def test_push_front_back_mix():
    dq = CudaDeque(8, device=torch.device('cuda'))
    a = torch.tensor([1,2,3], device='cuda', dtype=torch.int64)
    b = torch.tensor([4,5], device='cuda', dtype=torch.int64)
    dq.push_back(a)      # [1,2,3]
    dq.push_front(b)     # [4,5,1,2,3]
    out = dq.pop_front(2)# -> [4,5]
    assert torch.equal(out, b)
    out2 = dq.pop_back(2)# -> [2,3]
    assert torch.equal(out2, torch.tensor([2,3], device='cuda', dtype=torch.int64))
    rem = dq.pop_front(1)# -> [1]
    assert torch.equal(rem, torch.tensor([1], device='cuda', dtype=torch.int64))
    assert int(dq.current_size().item()) == 0

def test_wraparound_and_peek():
    dq = CudaDeque(5, device=torch.device('cuda'))
    dq.push_back(torch.tensor([10,11,12], device='cuda'))
    _ = dq.pop_front(2)  # leaves [21]
    dq.push_back(torch.tensor([13,14,15,16], device='cuda'))  # wrap
    # queue content now [12,13,14,15,16]
    peek = dq.peek_front(3)
    assert torch.equal(peek, torch.tensor([12,13,14], device='cuda'))
    peek_b = dq.peek_back(2)
    assert torch.equal(peek_b, torch.tensor([15,16], device='cuda'))
    out = dq.pop_front(5)
    assert torch.equal(out, torch.tensor([12,13,14,15,16], device='cuda'))
    assert int(dq.current_size().item()) == 0

def test_overflow_underflow():
    dq = CudaDeque(3, device=torch.device('cuda'))
    dq.push_back(torch.tensor([1,2], device='cuda'))
    try:
        dq.pop_front(3)
        assert False, "expected underflow"
    except RuntimeError:
        pass
    dq.push_back(torch.tensor([22], device='cuda'))
    try:
        dq.push_back(torch.tensor([20], device='cuda'))  # overflow
        assert False, "expected overflow"
    except RuntimeError:
        pass
    out = dq.pop_back(2)
    assert torch.equal(out, torch.tensor([2,3], device='cuda'))

def test_batched_ops_large():
    N=10000
    dq = CudaDeque(N, device=torch.device('cuda'))
    data = torch.arange(N, device='cuda', dtype=torch.int64)
    dq.push_back(data)
    assert int(dq.current_size().item()) == N
    # peek does not change size
    p = dq.peek_front(5)
    assert torch.equal(p, torch.arange(5, device='cuda', dtype=torch.int64))
    # pop in chunks
    out1 = dq.pop_front(4096)
    assert torch.equal(out1, torch.arange(0,4096, device='cuda', dtype=torch.int64))
    out2 = dq.pop_front(N-4096)
    assert torch.equal(out2, torch.arange(4096,N, device='cuda', dtype=torch.int64))
    assert int(dq.current_size().item()) == 0

if __name__ == "__main__":
    assert torch.cuda.is_available()
    test_basic_push_pop()
    test_push_front_back_mix()
    test_wraparound_and_peek()
    test_overflow_underflow()
    test_batched_ops_large()
    print("All CUDA queue tests passed")