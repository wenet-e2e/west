add_library(model STATIC
  hifi_gan.cc
  onnx_model.cc
  s3tokenizer.cc
  speaker_model.cc
  touch_tts_flow.cc
  touch_tts_llm.cc)
target_link_libraries(model PUBLIC onnxruntime llama)
