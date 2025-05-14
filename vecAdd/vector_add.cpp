#include <torch/extension.h>

torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x "must be a CUDA Tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x "must be contiguous")

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b)
{
	CHECK_CUDA(a);
	CHECK_CUDA(b);
	CHECK_CONTIGUOUS(a);
	CHECK_CONTIGUOUS(b);

	return vector_add_cuda(a,b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME , m)
{
	m.def("vector_add" , &vector_add, "Vector addition (CUDA)");
}