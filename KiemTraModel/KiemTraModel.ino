#include <TensorFlowLite_ESP32.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

//load model
#include "hand_model.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
 TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;

// Create an area of memory to use for input, output, and other TensorFlow
  // arrays. You'll need to adjust this by combiling, running, and looking
  // for errors.
  constexpr int kTensorArenaSize = 32 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
} // namespace

void setup() {
  Serial.begin(115200);
  // Set up logging (will report to Serial, even within TFLite functions)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  // Map the model into a usable data structure
  model = tflite::GetModel(hand_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddRelu();

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Assign model input and output buffers (tensors) to pointers
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

  // Kiểm tra kiểu dữ liệu của tensor đầu vào và đầu ra
  if (model_input->type != kTfLiteFloat32) {
    TF_LITE_REPORT_ERROR(error_reporter, "Model input is not kTfLiteFloat32");
    return;
  }
  if (model_output->type != kTfLiteFloat32) {
    TF_LITE_REPORT_ERROR(error_reporter, "Model output is not kTfLiteFloat32");
    return;
  }
}

void loop() {
  int x[42] = {277,303,269,230,205,149,126,125,86,172,117,174,63,156,129,174,146,182,105,229,63,218,148,227,153,229,106,283,64,277,147,278,158,280,107,341,80,332,139,322,155,320};

  for (int i = 0; i < 42; i++) {
    // Chuyển đổi giá trị đầu vào từ int sang float
    model_input->data.f[i] = static_cast<float>(x[i]);
  }

  // In ra giá trị đầu vào để kiểm tra
  Serial.print("Input: ");
  for (int i = 0; i < 42; i++) {
    Serial.print(model_input->data.f[i]);
    Serial.print(" ");
  }
  Serial.println("");

  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Error: %s\n", invoke_status);
  }

  // In ra giá trị đầu ra
  Serial.print("Output: ");
  for (int i = 0; i < 10; i++) {
    Serial.print(model_output->data.f[i]);
    Serial.print(" ");
  }
  Serial.println("");

  delay(1000);
}

