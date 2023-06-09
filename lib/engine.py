import numpy as np
from pycuda.gpuarray import GPUArray
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from pycuda.driver import PointerHolderBase
import torch


class Holder(PointerHolderBase):

    def __init__(self, tensor):
        super().__init__()
        self.tensor = tensor
        self.gpudata = tensor.data_ptr()

    def get_pointer(self):
        return self.tensor.data_ptr()

    # without an __index__ method, arithmetic calls to the GPUArray backed by this pointer fail
    # not sure why, this needs to return some integer, apparently
    def __index__(self):
        return self.gpudata

    def __int__(self):
        return self.gpudata


# dict to map between torch and numpy dtypes
dtype_map = {
    # signed integers
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.short: np.int16,
    torch.int32: np.int32,
    torch.int: np.int32,
    torch.int64: np.int64,
    torch.long: np.int64,

    # unsinged inters
    torch.uint8: np.uint8,

    # floating point
    torch.float: np.float32,
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.half: np.float16,
    torch.float64: np.float64,
    torch.double: np.float64
}


def torch_dtype_to_numpy(dtype):
    '''Convert a torch ``dtype`` to an equivalent numpy ``dtype``, if it is also available in pycuda.
    Parameters
    ----------
    dtype   :   np.dtype
    Returns
    -------
    torch.dtype
    Raises
    ------
    ValueError
        If there is not PyTorch equivalent, or the equivalent would not work with pycuda
    '''

    from pycuda.compyte.dtypes import dtype_to_ctype
    if dtype not in dtype_map:
        raise ValueError(f'{dtype} has no PyTorch equivalent')
    else:
        candidate = dtype_map[dtype]
        # we can raise exception early by checking of the type can be used with pycuda. Otherwise
        # we realize it only later when using the array
        try:
            _ = dtype_to_ctype(candidate)
        except ValueError:
            raise ValueError(f'{dtype} cannot be used in pycuda')
        else:
            return candidate


def numpy_dtype_to_torch(dtype):
    '''Convert numpy ``dtype`` to torch ``dtype``. The first matching one will be returned, if there
    are synonyms.
    Parameters
    ----------
    dtype   :   torch.dtype
    Returns
    -------
    np.dtype
    '''
    for dtype_t, dtype_n in dtype_map.items():
        if dtype_n == dtype_t:
            return dtype_t


def tensor_to_gpuarray(tensor):
    '''Convert a :class:`torch.Tensor` to a :class:`pycuda.gpuarray.GPUArray`. The underlying
    storage will be shared, so that modifications to the array will reflect in the tensor object.
    Parameters
    ----------
    tensor  :   torch.Tensor
    Returns
    -------
    pycuda.gpuarray.GPUArray
    Raises
    ------
    ValueError
        If the ``tensor`` does not live on the gpu
    '''
    if not tensor.is_cuda:
        raise ValueError('Cannot convert CPU tensor to GPUArray (call `cuda()` on it)')
    else:
        array = GPUArray(tensor.shape, dtype=torch_dtype_to_numpy(tensor.dtype),
                         gpudata=Holder(tensor))
        return array


def gpuarray_to_tensor(gpuarray):
    '''Convert a :class:`pycuda.gpuarray.GPUArray` to a :class:`torch.Tensor`. The underlying
    storage will NOT be shared, since a new copy must be allocated.
    Parameters
    ----------
    gpuarray  :   pycuda.gpuarray.GPUArray
    Returns
    -------
    torch.Tensor
    '''
    shape = gpuarray.shape
    dtype = gpuarray.dtype
    out_dtype = numpy_dtype_to_torch(dtype)
    out = torch.zeros(shape, dtype=out_dtype).cuda()
    gpuarray_copy = tensor_to_gpuarray(out)
    byte_size = gpuarray.itemsize * gpuarray.size
    pycuda.driver.memcpy_dtod(gpuarray_copy.gpudata, gpuarray.gpudata, byte_size)
    return out


class Engine(object):
    def _load_engine(self, model_path):
        with open(model_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self):
        outputs = []
        bindings = []
        out_ptr = 0
        for binding in self.engine:

            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                # NOTE init input binding with dummy int
                bindings.append(1000)
            else:
                out = self.out_ptrs[out_ptr]
                out_ptr += 1
                device_mem = GPUArray(out.shape, dtype=torch_dtype_to_numpy(out.dtype),
                                      gpudata=Holder(out))

                outputs.append(device_mem)
                bindings.append(int(device_mem.gpudata))
        return outputs, bindings

    def __init__(self, load_model, batch=1):
        self.threshold = 0.4
        self.batch_size = batch
        self.img_h_new, self.img_w_new = 384, 1280
        self.striped_h, self.striped_w = int(self.img_h_new / 4), int(self.img_w_new / 4)

        self.shape_of_output = [(batch, 8, self.striped_h, self.striped_w),  # hm
                                (batch, 18, self.striped_h, self.striped_w),  # hps
                                (batch, 8, self.striped_h, self.striped_w),  # rot
                                (batch, 3, self.striped_h, self.striped_w),  # dim
                                (batch, 1, self.striped_h, self.striped_w)  # prob
                                ]

        self.trt_logger = trt.Logger(trt.Logger.WARNING)

        mean = np.ones((384, 1280, 3), dtype=np.float32) * np.array([0.485 * 255, 0.456 * 255, 0.406 * 255],
                                                                    dtype=np.float32).reshape(1, 1, 3)
        mean = np.transpose(mean, [2, 0, 1])
        mean = np.ascontiguousarray(mean).ravel()
        self.mean_gpu = gpuarray.to_gpu(mean)

        std = np.ones((384, 1280, 3), dtype=np.float32) * np.array([0.229 * 255, 0.224 * 255, 0.225 * 255],
                                                                   dtype=np.float32).reshape(1, 1, 3)
        std = np.transpose(std, [2, 0, 1])
        std = np.ascontiguousarray(std).ravel()
        self.std_gpu = gpuarray.to_gpu(std).astype(np.float32)

        del mean, std
        self.hm = torch.zeros((batch * 8 * self.striped_h * self.striped_w), dtype=torch.float32).cuda()
        self.hps = torch.zeros((batch * 18 * self.striped_h * self.striped_w), dtype=torch.float32).cuda()
        self.rot = torch.zeros((batch * 8 * self.striped_h * self.striped_w), dtype=torch.float32).cuda()
        self.dim = torch.zeros((batch * 3 * self.striped_h * self.striped_w), dtype=torch.float32).cuda()
        self.prob = torch.zeros((batch * 1 * self.striped_h * self.striped_w), dtype=torch.float32).cuda()
        # self.features = torch.zeros((batch * 32 * self.striped_h * self.striped_w), dtype=torch.float32).cuda()

        # NOTE INPUT GPU MEM
        self.img_gpu = gpuarray.empty(3 * 384 * 1280, dtype=np.float32)
        # NOTE INPUT HOST MEM
        self.host_mem = cuda.pagelocked_empty((3, 384, 1280), dtype=np.float32)

        # NOTE OUT GPU TENSOR
        self.out_ptrs = [self.hm, self.hps, self.rot, self.dim, self.prob]
        # self.out_ptrs = [self.hm, self.features]
        self.engine = self._load_engine(load_model)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.outputs, self.bindings = self._allocate_buffers()

    def __call__(self, img):

        # NOTE preprocess img
        self.host_mem = img
        self.img_gpu.set_async(self.host_mem, stream=self.stream)
        self.img_gpu = ((self.img_gpu - self.mean_gpu) / self.std_gpu)

        # NOTE modify input binding
        self.bindings[0] = self.img_gpu.gpudata

        # NOTE execute engine
        self.context.execute_async(
            bindings=self.bindings,
            stream_handle=self.stream.handle)

        #self.stream.synchronize()

        hm, hps, rot, dim, prob = [output.reshape(shape) for output, shape in zip(self.out_ptrs, self.shape_of_output)]
        return hm, hps, rot, dim, prob
