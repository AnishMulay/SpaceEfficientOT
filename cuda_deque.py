import torch

class CudaDeque:
    def __init__(self, max_size: int, device=None, dtype=torch.int64):
        assert max_size > 0, "max_size must be positive"
        self.device = torch.device('cuda') if device is None else device
        self.dtype = dtype
        # storage and counters on GPU
        self.buf = torch.empty(max_size, device=self.device, dtype=dtype)
        self.capacity = torch.tensor(max_size, device=self.device, dtype=torch.int64)
        self.head = torch.zeros((), device=self.device, dtype=torch.int64)  # index of first element
        self.size = torch.zeros((), device=self.device, dtype=torch.int64)  # number of elements

    @property
    def tail(self):
        # tail is (head + size) % capacity
        return (self.head + self.size).remainder(self.capacity)

    def _ensure_cuda(self, t):
        assert isinstance(t, torch.Tensor) and t.is_cuda, "input must be a CUDA tensor"  # [9]
        assert t.dtype == self.dtype, f"dtype mismatch: expected {self.dtype}, got {t.dtype}"

    def __len__(self):
        # Caution: converting to Python int synchronizes; avoid in hot path.
        return int(self.size.item())

    def current_size(self):
        # return CUDA scalar (no host sync if not converted)
        return self.size

    def remaining_capacity(self):
        return (self.capacity - self.size)

    def _write_range(self, start, data):
        n = torch.tensor(data.shape[0], device=self.device, dtype=torch.int64)
        cap = self.capacity
        end = (start + n)
        # two-slice wrap write
        first = torch.minimum(n, cap - start)
        second = n - first
        if int(first.item()) > 0:
            self.buf[start : start + first] = data[:first]
        if int(second.item()) > 0:
            self.buf[0 : second] = data[first:]

    def _read_range(self, start, n):
        n = torch.tensor(n, device=self.device, dtype=torch.int64)
        cap = self.capacity
        end = (start + n)
        first = torch.minimum(n, cap - start)
        second = n - first
        if int(second.item()) == 0:
            return self.buf[start : start + first].clone()
        else:
            part1 = self.buf[start : start + first]
            part2 = self.buf[0 : second]
            return torch.cat((part1, part2), dim=0).clone()

    def push_back(self, x_batch: torch.Tensor):
        # append batch at tail
        self._ensure_cuda(x_batch)
        k = torch.tensor(x_batch.shape[0], device=self.device, dtype=torch.int64)
        if int((self.size + k > self.capacity).item()):
            raise RuntimeError("CudaDeque overflow on push_back")
        t = self.tail
        self._write_range(t, x_batch)
        self.size += k

    def push_front(self, x_batch: torch.Tensor):
        # prepend batch at head
        self._ensure_cuda(x_batch)
        k = torch.tensor(x_batch.shape[0], device=self.device, dtype=torch.int64)
        if int((self.size + k > self.capacity).item()):
            raise RuntimeError("CudaDeque overflow on push_front")
        cap = self.capacity
        new_head = (self.head - k).remainder(cap)
        self._write_range(new_head, x_batch)
        self.head = new_head
        self.size += k

    def pop_front(self, k: int):
        if k < 0:
            raise ValueError("k must be non-negative")
        k_t = torch.tensor(k, device=self.device, dtype=torch.int64)
        if int((k_t > self.size).item()):
            raise RuntimeError("CudaDeque underflow on pop_front")
        out = self._read_range(self.head, k_t)
        self.head = (self.head + k_t).remainder(self.capacity)
        self.size -= k_t
        return out  # CUDA tensor

    def pop_back(self, k: int):
        if k < 0:
            raise ValueError("k must be non-negative")
        k_t = torch.tensor(k, device=self.device, dtype=torch.int64)
        if int((k_t > self.size).item()):
            raise RuntimeError("CudaDeque underflow on pop_back")
        new_tail = (self.tail - k_t).remainder(self.capacity)
        out = self._read_range(new_tail, k_t)
        self.size -= k_t
        return out  # CUDA tensor

    def peek_front(self, k: int):
        k_t = torch.tensor(k, device=self.device, dtype=torch.int64)
        if int((k_t > self.size).item()):
            raise RuntimeError("CudaDeque insufficient elements on peek_front")
        return self._read_range(self.head, k_t)

    def peek_back(self, k: int):
        k_t = torch.tensor(k, device=self.device, dtype=torch.int64)
        if int((k_t > self.size).item()):
            raise RuntimeError("CudaDeque insufficient elements on peek_back")
        start = (self.tail - k_t).remainder(self.capacity)
        return self._read_range(start, k_t)

    def clear(self):
        self.head.zero_()
        self.size.zero_()
