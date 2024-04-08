import numpy as np

def _process_llm_index(indices, array_shape):
    """
    processing function for returned voxel maps (which are to be manipulated by LLMs)
    handles non-integer indexing
    handles negative indexing with manually designed special cases
    """
    if isinstance(indices, int) or isinstance(indices, np.int64) or isinstance(indices, np.int32) or isinstance(indices, np.int16) or isinstance(indices, np.int8):
        processed = indices if indices >= 0 or indices == -1 else 0
        assert len(array_shape) == 1, "1D array expected"
        processed = min(processed, array_shape[0] - 1)
    elif isinstance(indices, float) or isinstance(indices, np.float64) or isinstance(indices, np.float32) or isinstance(indices, np.float16):
        processed = np.round(indices).astype(int) if indices >= 0 or indices == -1 else 0
        assert len(array_shape) == 1, "1D array expected"
        processed = min(processed, array_shape[0] - 1)
    elif isinstance(indices, slice):
        start, stop, step = indices.start, indices.stop, indices.step
        if start is not None:
            start = np.round(start).astype(int)
        if stop is not None:
            stop = np.round(stop).astype(int)
        if step is not None:
            step = np.round(step).astype(int)
        # only convert the case where the start is negative and the stop is positive/negative
        if (start is not None and start < 0) and (stop is not None):
            if stop >= 0:
                processed = slice(0, stop, step)
            else:
                processed = slice(0, 0, step)
        else:
            processed = slice(start, stop, step)
    elif isinstance(indices, tuple) or isinstance(indices, list):
        processed = tuple(
            _process_llm_index(idx, (array_shape[i],)) for i, idx in enumerate(indices)
        )
    elif isinstance(indices, np.ndarray):
        print("[IndexingWrapper] Warning: numpy array indexing was converted to list")
        processed = _process_llm_index(indices.tolist(), array_shape)
    else:
        print(f"[IndexingWrapper] {indices} (type: {type(indices)}) not supported")
        raise TypeError("Indexing type not supported")
    # give warning if index was negative
    if processed != indices:
        print(f"[IndexingWrapper] Warning: index was changed from {indices} to {processed}")
    # print(f"[IndexingWrapper] {idx} -> {processed}")
    return processed


class VoxelIndexingWrapper:
    """
    LLM indexing wrapper that uses _process_llm_index to process indexing
    behaves like a numpy array
    """
    def __init__(self, array):
        self.array = array

    def __getitem__(self, idx):
        return self.array[_process_llm_index(idx, tuple(self.array.shape))]
    
    def __setitem__(self, idx, value):
        self.array[_process_llm_index(idx, tuple(self.array.shape))] = value
    
    def __repr__(self) -> str:
        return self.array.__repr__()
    
    def __str__(self) -> str:
        return self.array.__str__()
    
    def __eq__(self, other):
        return self.array == other
    
    def __ne__(self, other):
        return self.array != other
    
    def __lt__(self, other):
        return self.array < other
    
    def __le__(self, other):
        return self.array <= other
    
    def __gt__(self, other):
        return self.array > other
    
    def __ge__(self, other):
        return self.array >= other
    
    def __add__(self, other):
        return self.array + other
    
    def __sub__(self, other):
        return self.array - other
    
    def __mul__(self, other):
        return self.array * other
    
    def __truediv__(self, other):
        return self.array / other
    
    def __floordiv__(self, other):
        return self.array // other
    
    def __mod__(self, other):
        return self.array % other
    
    def __divmod__(self, other):
        return self.array.__divmod__(other)
    
    def __pow__(self, other):
        return self.array ** other
    
    def __lshift__(self, other):
        return self.array << other
    
    def __rshift__(self, other):
        return self.array >> other
    
    def __and__(self, other):
        return self.array & other
    
    def __xor__(self, other):
        return self.array ^ other
    
    def __or__(self, other):
        return self.array | other
    
    def __radd__(self, other):
        return other + self.array
    
    def __rsub__(self, other):
        return other - self.array
    
    def __rmul__(self, other):
        return other * self.array
    
    def __rtruediv__(self, other):
        return other / self.array
    
    def __rfloordiv__(self, other):
        return other // self.array
    
    def __rmod__(self, other):
        return other % self.array
    
    def __rdivmod__(self, other):
        return other.__divmod__(self.array)
    
    def __rpow__(self, other):
        return other ** self.array
    
    def __rlshift__(self, other):
        return other << self.array
    
    def __rrshift__(self, other):
        return other >> self.array
    
    def __rand__(self, other):
        return other & self.array
    
    def __rxor__(self, other):
        return other ^ self.array
    
    def __ror__(self, other):
        return other | self.array
    
    def __getattribute__(self, name):
        if name == "array":
            return super().__getattribute__(name)
        elif name == "__getitem__":
            return super().__getitem__
        elif name == "__setitem__":
            return super().__setitem__
        else:
            # print(name)
            return super().array.__getattribute__(name)
    
    def __getattr__(self, name):
        return self.array.__getattribute__(name)
