add_library(model STATIC onnx_model.cc speech_encoder.cc ctc_model.cc projector.cc llm.cc touch_asr.cc)
target_link_libraries(model PUBLIC onnxruntime llama frontend)
