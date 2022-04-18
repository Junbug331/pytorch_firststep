from numpy import dtype
import torch
import numpy as np

def make_tensor():
    # int16
    a = torch.tensor([[1, 2], [3, 4]], dtype=torch.int16)

    # float
    b = torch.tensor([2], dtype=torch.float32)
    
    # double
    c = torch.tensor([3], dtype=torch.float64)

    print(a, b, c)

    tensor_list = [a, b, c]

    for t in tensor_list:
        print("shape of tensor {}".format(t.shape))
        print("datatype of tensor {}".format(t.dtype))
        print("device tensor is stored on {}".format(t.device))
        print()

def sumsub_tensor():
    a = torch.tensor([3, 2]) 
    b = torch.tensor([5, 3])

    print("input {}, {} ".format(a, b))
    
    # sum
    sum = a + b
    print("sum : {}".format(sum))

    # sub
    sub = a - b
    print("sub : {}".format(sub))

    # sum of elements
    sum_element_a = a.sum()
    print(sum_element_a)


def muldiv_tensor():
    a = torch.arange(0, 9).view(3, 3) #[0, 9)
    b = torch.arange(0, 9).view(3, 3) #[0, 9)
    print("input tensor \n {}, \n {}".format(a, b))

    # mat_mul
    c = torch.matmul(a, b)
    print(c)
    print(a @ b)
    print()

    # element mul
    c = torch.mul(a, b)
    print(c)

def reshape_tensor():
    a = torch.tensor([2, 4, 5, 6 ,7, 8]) 
    print("input tensor : \n {}".format(a))

    b = a.view(2, 3)
    print("view \n {}".format(b))

    # transpose
    bt = b.t()
    print("transpose \n {}".format(bt))

def access_tensor():
    a = torch.arange(1, 13).view(4, 3)

    print("input : \n {}".format(a))

    # first col(slicing)
    print(a[:, 0])

    # first row
    print(a[0, :])

    print(a[1, 1].item())

def transform_numpy():
    a = torch.arange(1, 13).view(4, 3)
    print("input : \n {}".format(a))

    a_np = a.numpy()
    print("numpy : \n {}".format(a_np))

    b = np.array([1, 2, 3])
    bt = torch.from_numpy(b)
    print(b)

def concat_tensor():
    a = torch.arange(1, 10).view(3, 3)
    b = torch.arange(10, 19).view(3, 3)
    c = torch.arange(19, 28).view(3, 3)

    abc = torch.cat([a, b, c], dim=0)

    print("input tensor : \n {} \n {} \n {}".format(a, b, c))
    print("concat : \n {}".format(abc))
    print(abc.shape)

def stack_tensor():
    a = torch.arange(1, 10).view(3, 3)
    b = torch.arange(10, 19).view(3, 3)
    c = torch.arange(19, 28).view(3, 3)
    
    abc = torch.stack([a, b, c], dim=-1)     
    print("input tensor : \n {} \n {} \n {}".format(a, b, c))
    print("concat : \n {}".format(abc))
    print(abc.shape)


def transpose_tensor():
    a = torch.arange(1, 10).view(3, 3)
    print("input tensor : \n {}".format(a))

    # transpose
    at = torch.transpose(a, 0, 1)
    print("transpose: \n {}".format(at))

    b = torch.arange(1, 25).view(4, 3, 2)
    print("input tensor : \n {}".format(b))
    print(b.shape)

    bt = torch.transpose(b, 0, 2)
    print("transpose: \n {}".format(bt))
    print(bt.shape)

    bp = b.permute(2, 0, 1) # 0, 1, 2
    print("permute : \n {}".format(bp))
    print(bp.shape)


if __name__ == '__main__':
    #make_tensor()
    #sumsub_tensor()
    #muldiv_tensor()
    #reshape_tensor() 
    #access_tensor()
    #transform_numpy()
    #concat_tensor()
    #stack_tensor()
    transpose_tensor()