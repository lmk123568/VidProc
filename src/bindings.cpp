#include <torch/extension.h>

#include "Decoder.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<Decoder>(m, "Decoder")
        .def(py::init<const std::string&, bool, int, int>(),
             py::arg("filename"),
             py::arg("enable_frame_skip") = false,
             py::arg("output_width") = 0,
             py::arg("output_height") = 0)
        .def("next_frame", &Decoder::next_frame)
        .def("get_width", &Decoder::get_width)
        .def("get_height", &Decoder::get_height)
        .def("get_fps", &Decoder::get_fps);
}
