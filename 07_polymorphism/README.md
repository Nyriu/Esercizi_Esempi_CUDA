# Polymorphism and virtual methods
Example to better understand this:

> It is not allowed to pass as an argument to a __global__ function an object of a class with virtual functions.


## Resources
* <https://migocpp.wordpress.com/2018/03/26/virtual-cuda/>
* <https://stackoverflow.com/questions/26812913/how-to-implement-device-side-cuda-virtual-function://stackoverflow.com/questions/26812913/how-to-implement-device-side-cuda-virtual-functions>
* <http://stackoverflow.com/questions/22988244/polymorphism-and-derived-classes-in-cuda-cuda-thrust/23476510?s=2|1.7036#23476510>

> A key observation I would make is that it is not allowed to pass as an argument to a __global__ function an object of a class with virtual functions.
> This means that polymorphic objects created on the host cannot be passed to the device (via thrust, or in ordinary CUDA C++)

* <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#classe://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#classes>

> If an object is created in host code, invoking a virtual function for that object in device code has undefined behavior.




