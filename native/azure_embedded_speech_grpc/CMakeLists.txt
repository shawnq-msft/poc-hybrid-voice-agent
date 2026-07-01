cmake_minimum_required(VERSION 3.26)
project(azure_embedded_speech_grpc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()

set(SPEECHSDK_NUGET_VERSION "1.47.0" CACHE STRING "Microsoft Cognitive Services Speech NuGet package version")
set(SPEECHSDK_API_VERSION "1.47" CACHE STRING "Speech SDK version label compiled into the native sidecar")

function(set_speechsdk_package_root variable package version suffix description)
  if(NOT DEFINED ENV{USERPROFILE})
    return()
  endif()

  set(default_path "$ENV{USERPROFILE}/.nuget/packages/${package}/${version}/${suffix}")
  set(use_default_path FALSE)
  if(NOT DEFINED ${variable})
    set(use_default_path TRUE)
  else()
    file(TO_CMAKE_PATH "${${variable}}" current_path)
    string(TOLOWER "${current_path}" current_path_lower)
    string(FIND "${current_path_lower}" "/.nuget/packages/${package}/" package_index)
    string(FIND "${current_path_lower}" "/${version}/" version_index)
    if(NOT package_index EQUAL -1 AND version_index EQUAL -1)
      set(use_default_path TRUE)
    endif()
  endif()

  if(use_default_path)
    set(${variable} "${default_path}" CACHE PATH "${description}" FORCE)
  endif()
endfunction()

find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

set_speechsdk_package_root(SPEECHSDK_ROOT microsoft.cognitiveservices.speech ${SPEECHSDK_NUGET_VERSION} build/native "Microsoft Cognitive Services Speech native build root")
set_speechsdk_package_root(SPEECHSDK_EMBEDDED_SR_ROOT microsoft.cognitiveservices.speech.extension.embedded.sr ${SPEECHSDK_NUGET_VERSION} "" "Microsoft Cognitive Services Speech embedded SR package root")
set_speechsdk_package_root(SPEECHSDK_EMBEDDED_TTS_ROOT microsoft.cognitiveservices.speech.extension.embedded.tts ${SPEECHSDK_NUGET_VERSION} "" "Microsoft Cognitive Services Speech embedded TTS package root")
set_speechsdk_package_root(SPEECHSDK_ONNX_RUNTIME_ROOT microsoft.cognitiveservices.speech.extension.onnx.runtime ${SPEECHSDK_NUGET_VERSION} "" "Microsoft Cognitive Services Speech ONNX runtime extension package root")

set(PROTO_FILE ${CMAKE_CURRENT_LIST_DIR}/../../protos/azure_embedded_speech.proto)
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GENERATED_DIR})

add_custom_command(
  OUTPUT
    ${GENERATED_DIR}/azure_embedded_speech.pb.cc
    ${GENERATED_DIR}/azure_embedded_speech.pb.h
    ${GENERATED_DIR}/azure_embedded_speech.grpc.pb.cc
    ${GENERATED_DIR}/azure_embedded_speech.grpc.pb.h
  COMMAND protobuf::protoc
  ARGS --grpc_out=${GENERATED_DIR}
       --cpp_out=${GENERATED_DIR}
       --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
       -I ${CMAKE_CURRENT_LIST_DIR}/../../protos
       ${PROTO_FILE}
  DEPENDS ${PROTO_FILE}
)

add_executable(azure_embedded_speech_grpc
  src/main.cpp
  ${GENERATED_DIR}/azure_embedded_speech.pb.cc
  ${GENERATED_DIR}/azure_embedded_speech.grpc.pb.cc
)

add_executable(azure_embedded_tts_grpc
  src/tts_main.cpp
  ${GENERATED_DIR}/azure_embedded_speech.pb.cc
  ${GENERATED_DIR}/azure_embedded_speech.grpc.pb.cc
)

set_target_properties(azure_embedded_tts_grpc PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tts
)

target_include_directories(azure_embedded_speech_grpc PRIVATE ${GENERATED_DIR})
target_link_libraries(azure_embedded_speech_grpc PRIVATE gRPC::grpc++ protobuf::libprotobuf)
target_include_directories(azure_embedded_tts_grpc PRIVATE ${GENERATED_DIR})
target_link_libraries(azure_embedded_tts_grpc PRIVATE gRPC::grpc++ protobuf::libprotobuf)

if(DEFINED SPEECHSDK_ROOT)
  target_include_directories(azure_embedded_speech_grpc PRIVATE
    ${SPEECHSDK_ROOT}/include/c_api
    ${SPEECHSDK_ROOT}/include/cxx_api
  )
  if(WIN32)
    target_link_directories(azure_embedded_speech_grpc PRIVATE ${SPEECHSDK_ROOT}/x64/Release)
    target_link_libraries(azure_embedded_speech_grpc PRIVATE Microsoft.CognitiveServices.Speech.core)
    if(DEFINED ENV{USERPROFILE})
      add_custom_command(TARGET azure_embedded_speech_grpc POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "$ENV{USERPROFILE}/.nuget/packages/microsoft.cognitiveservices.speech/${SPEECHSDK_NUGET_VERSION}/runtimes/win-x64/native/Microsoft.CognitiveServices.Speech.core.dll"
          $<TARGET_FILE_DIR:azure_embedded_speech_grpc>
      )
    endif()
  endif()
endif()

if(DEFINED SPEECHSDK_ROOT)
  target_include_directories(azure_embedded_tts_grpc PRIVATE
    ${SPEECHSDK_ROOT}/include/c_api
    ${SPEECHSDK_ROOT}/include/cxx_api
  )
  if(WIN32)
    target_link_directories(azure_embedded_tts_grpc PRIVATE ${SPEECHSDK_ROOT}/x64/Release)
    target_link_libraries(azure_embedded_tts_grpc PRIVATE Microsoft.CognitiveServices.Speech.core)
    if(DEFINED ENV{USERPROFILE})
      add_custom_command(TARGET azure_embedded_tts_grpc POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "$ENV{USERPROFILE}/.nuget/packages/microsoft.cognitiveservices.speech/${SPEECHSDK_NUGET_VERSION}/runtimes/win-x64/native/Microsoft.CognitiveServices.Speech.core.dll"
          $<TARGET_FILE_DIR:azure_embedded_tts_grpc>
      )
    endif()
  endif()
endif()

if(WIN32)
  set(SPEECHSDK_EXTENSION_LIB_DIRS "")
  set(SPEECHSDK_EXTENSION_DLLS "")

  if(DEFINED SPEECHSDK_EMBEDDED_SR_ROOT)
    list(APPEND SPEECHSDK_EXTENSION_LIB_DIRS ${SPEECHSDK_EMBEDDED_SR_ROOT}/build/native/x64/Release)
    list(APPEND SPEECHSDK_EXTENSION_DLLS
      ${SPEECHSDK_EMBEDDED_SR_ROOT}/runtimes/win-x64/native/Microsoft.CognitiveServices.Speech.extension.embedded.sr.dll
      ${SPEECHSDK_EMBEDDED_SR_ROOT}/runtimes/win-x64/native/Microsoft.CognitiveServices.Speech.extension.embedded.sr.runtime.dll
    )
  endif()

  if(DEFINED SPEECHSDK_EMBEDDED_TTS_ROOT)
    list(APPEND SPEECHSDK_EXTENSION_LIB_DIRS ${SPEECHSDK_EMBEDDED_TTS_ROOT}/build/native/x64/Release)
    list(APPEND SPEECHSDK_EXTENSION_DLLS
      ${SPEECHSDK_EMBEDDED_TTS_ROOT}/runtimes/win-x64/native/Microsoft.CognitiveServices.Speech.extension.embedded.tts.dll
    )
  endif()

  if(DEFINED SPEECHSDK_ONNX_RUNTIME_ROOT)
    list(APPEND SPEECHSDK_EXTENSION_LIB_DIRS ${SPEECHSDK_ONNX_RUNTIME_ROOT}/build/native/x64/Release)
    list(APPEND SPEECHSDK_EXTENSION_DLLS
      ${SPEECHSDK_ONNX_RUNTIME_ROOT}/runtimes/win-x64/native/Microsoft.CognitiveServices.Speech.extension.onnxruntime.dll
    )
  endif()

  target_link_directories(azure_embedded_speech_grpc PRIVATE ${SPEECHSDK_EXTENSION_LIB_DIRS})
  target_link_libraries(azure_embedded_speech_grpc PRIVATE
    Microsoft.CognitiveServices.Speech.extension.embedded.sr
    Microsoft.CognitiveServices.Speech.extension.embedded.tts
    Microsoft.CognitiveServices.Speech.extension.onnxruntime
  )

  add_custom_command(TARGET azure_embedded_speech_grpc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SPEECHSDK_EXTENSION_DLLS} $<TARGET_FILE_DIR:azure_embedded_speech_grpc>
  )

  target_link_directories(azure_embedded_tts_grpc PRIVATE ${SPEECHSDK_EXTENSION_LIB_DIRS})
  target_link_libraries(azure_embedded_tts_grpc PRIVATE
    Microsoft.CognitiveServices.Speech.extension.embedded.tts
    Microsoft.CognitiveServices.Speech.extension.onnxruntime
  )

  add_custom_command(TARGET azure_embedded_tts_grpc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SPEECHSDK_EXTENSION_DLLS} $<TARGET_FILE_DIR:azure_embedded_tts_grpc>
  )
endif()

target_compile_definitions(azure_embedded_speech_grpc PRIVATE SPEECHSDK_VERSION="${SPEECHSDK_API_VERSION}")
target_compile_definitions(azure_embedded_tts_grpc PRIVATE SPEECHSDK_VERSION="${SPEECHSDK_API_VERSION}")

add_executable(azure_embedded_speech_grpc_preload_registry_test
  tests/preload_registry_test.cpp
  ${GENERATED_DIR}/azure_embedded_speech.pb.cc
  ${GENERATED_DIR}/azure_embedded_speech.grpc.pb.cc
)

target_include_directories(azure_embedded_speech_grpc_preload_registry_test PRIVATE ${GENERATED_DIR})
target_compile_definitions(azure_embedded_speech_grpc_preload_registry_test PRIVATE SPEECHSDK_VERSION="${SPEECHSDK_API_VERSION}")
target_link_libraries(azure_embedded_speech_grpc_preload_registry_test PRIVATE gRPC::grpc++ protobuf::libprotobuf)

if(DEFINED SPEECHSDK_ROOT)
  target_include_directories(azure_embedded_speech_grpc_preload_registry_test PRIVATE
    ${SPEECHSDK_ROOT}/include/c_api
    ${SPEECHSDK_ROOT}/include/cxx_api
  )
  if(WIN32)
    target_link_directories(azure_embedded_speech_grpc_preload_registry_test PRIVATE ${SPEECHSDK_ROOT}/x64/Release)
    target_link_libraries(azure_embedded_speech_grpc_preload_registry_test PRIVATE Microsoft.CognitiveServices.Speech.core)
  endif()
endif()

if(WIN32)
  target_link_directories(azure_embedded_speech_grpc_preload_registry_test PRIVATE ${SPEECHSDK_EXTENSION_LIB_DIRS})
  target_link_libraries(azure_embedded_speech_grpc_preload_registry_test PRIVATE
    Microsoft.CognitiveServices.Speech.extension.embedded.sr
    Microsoft.CognitiveServices.Speech.extension.embedded.tts
    Microsoft.CognitiveServices.Speech.extension.onnxruntime
  )
  add_custom_command(TARGET azure_embedded_speech_grpc_preload_registry_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "$ENV{USERPROFILE}/.nuget/packages/microsoft.cognitiveservices.speech/${SPEECHSDK_NUGET_VERSION}/runtimes/win-x64/native/Microsoft.CognitiveServices.Speech.core.dll"
      ${SPEECHSDK_EXTENSION_DLLS}
      $<TARGET_FILE_DIR:azure_embedded_speech_grpc_preload_registry_test>
  )
endif()

add_test(NAME azure_embedded_speech_grpc_preload_registry COMMAND azure_embedded_speech_grpc_preload_registry_test)
