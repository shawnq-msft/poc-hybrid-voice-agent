cmake_minimum_required(VERSION 3.26)
project(azure_embedded_speech_grpc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

if(NOT DEFINED SPEECHSDK_ROOT AND DEFINED ENV{USERPROFILE})
  set(SPEECHSDK_ROOT "$ENV{USERPROFILE}/.nuget/packages/microsoft.cognitiveservices.speech/1.47.0/build/native")
endif()

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

target_include_directories(azure_embedded_speech_grpc PRIVATE ${GENERATED_DIR})
target_link_libraries(azure_embedded_speech_grpc PRIVATE gRPC::grpc++ protobuf::libprotobuf)

if(DEFINED SPEECHSDK_ROOT)
  target_include_directories(azure_embedded_speech_grpc PRIVATE ${SPEECHSDK_ROOT}/include/cxx_api)
  if(WIN32)
    target_link_directories(azure_embedded_speech_grpc PRIVATE ${SPEECHSDK_ROOT}/x64/Release)
    target_link_libraries(azure_embedded_speech_grpc PRIVATE Microsoft.CognitiveServices.Speech.core)
  endif()
endif()

target_compile_definitions(azure_embedded_speech_grpc PRIVATE SPEECHSDK_VERSION="1.47")
