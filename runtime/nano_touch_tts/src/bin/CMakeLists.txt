add_executable(compute_feature compute_feature.cc)
target_link_libraries(compute_feature PUBLIC frontend glog gflags)

add_executable(compute_speech_token compute_speech_token.cc)
target_link_libraries(compute_speech_token PUBLIC frontend model glog gflags)

add_executable(compute_speaker_embedding compute_speaker_embedding.cc)
target_link_libraries(compute_speaker_embedding PUBLIC frontend model glog gflags)

add_executable(touch_tts_llm_gen touch_tts_llm_gen.cc)
target_link_libraries(touch_tts_llm_gen PUBLIC frontend model glog gflags)

add_executable(touch_tts_flow_gen touch_tts_flow_gen.cc)
target_link_libraries(touch_tts_flow_gen PUBLIC frontend model glog gflags)

add_executable(mel2wav mel2wav.cc)
target_link_libraries(mel2wav PUBLIC frontend model glog gflags)

add_executable(touch_tts_gen touch_tts_gen.cc)
target_link_libraries(touch_tts_gen PUBLIC frontend model glog gflags)
