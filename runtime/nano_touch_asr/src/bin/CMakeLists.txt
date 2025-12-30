add_executable(compute_feature compute_feature.cc)
target_link_libraries(compute_feature PUBLIC frontend glog gflags)

add_executable(decode_ctc_main decode_ctc_main.cc)
target_link_libraries(decode_ctc_main PUBLIC frontend model glog gflags)

add_executable(projector_main projector_main.cc)
target_link_libraries(projector_main PUBLIC model glog gflags)

add_executable(touch_asr_main touch_asr_main.cc)
target_link_libraries(touch_asr_main PUBLIC model glog gflags)

add_executable(llm_main llm_main.cc)
target_link_libraries(llm_main PUBLIC model glog gflags)
