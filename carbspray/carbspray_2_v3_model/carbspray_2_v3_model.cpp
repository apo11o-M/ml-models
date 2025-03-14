/* COPYRIGHT HEADER GOES HERE: No CopyRight Header String Passed During Model Conversion */

/* Command Line used:
qnn-onnx-converter; act_bitwidth=8; act_quantizer=tf; act_quantizer_calibration=min-max; act_quantizer_schema=asymmetric; adjust_nms_features_dims=True; algorithms=[]; align_matmul_ranks=True; apply_masked_softmax=uncompressed; arch_checker=False; batch=None; bias_bitwidth=8; converter_op_package_lib=; copyright_file=None; custom_io=; custom_op_config_paths=None; debug=-1; define_symbol=None; disable_batchnorm_folding=False; disable_node_validation=False; disable_qnn_op_config_validation=False; disable_relu_squashing=False; dry_run=None; dumpIR=False; dump_custom_io_config_template=; dump_encoding_json=False; dump_inferred_model=False; dump_qairt_io_config_yaml=; dump_qairt_quantizer_command=None; dump_value_info=False; enable_framework_trace=False; enable_match_gathernd=False; exclude_named_tensors=False; expand_gru_op_structure=True; expand_lstm_op_structure=False; expand_sparse_op_structure=False; export_format=cpp; extract_color_transform=True; float_bias_bitwidth=0; float_bias_bw=0; float_bitwidth=32; float_bw=32; float_fallback=False; force_prune_cast_ops=False; handle_gather_negative_indices=True; ignore_encodings=False; include_data_invariant_ops=False; inject_cast_for_gather=True; input_dim=None; input_dtype=[]; input_encoding=[]; input_layout=[]; input_list=None; input_type=[]; keep_disconnected_nodes=False; keep_int64_inputs=False; keep_quant_nodes=False; keep_weights_quantized=False; match_caffe_ssd_to_tf=True; model_version=None; multi_time_steps_gru=False; multi_time_steps_lstm=False; no_simplification=False; op_package_lib=; out_names=['1362', '1340', '1341']; overwrite_model_prefix=False; pack_4_bit_weights=False; package_name=None; packed_masked_softmax_inputs=[]; packed_max_seq=1; param_quantizer=None; param_quantizer_calibration=min-max; param_quantizer_schema=asymmetric; percentile_calibration_value=99.99; perform_axes_to_spatial_first_order=True; perform_layout_transformation=False; prepare_inputs_as_params=False; preprocess_roi_pool_inputs=True; preserve_io=[]; quantization_overrides=; restrict_quantization_steps=[]; squash_box_decoder=True; unroll_gru_time_steps=True; unroll_lstm_time_steps=True; use_aimet_quantizer=False; use_convert_quantization_nodes=False; use_dynamic_16_bit_weights=False; use_native_dtype=False; use_native_input_files=False; use_native_output_files=False; use_per_channel_quantization=False; use_per_row_quantization=False; validate_models=False; weights_bitwidth=8
*/

#include "QnnOpDef.h"
#include "QnnModel.hpp"

// Flag to determine if Backend should do node validation for each opNode added
#define DO_GRAPH_NODE_VALIDATIONS 1

using namespace qnn_wrapper_api;
const __attribute__((visibility("default"))) char* QNN_SDK_VERSION = "qaisw-v2.27.0.240926142112_100894";
extern "C" {
static ModelError_t addTensor_images(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_images[] = {1, 540, 720, 3};
  VALIDATE(model.addTensor("images", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "images",
                                 .type= QNN_TENSOR_TYPE_APP_WRITE,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_images,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=nullptr,
                                                .dataSize=0}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode_images_nchw(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR images_nchw */
  uint32_t dimensions_images_nchw_perm[] = {4};
  uint32_t images_nchw_perm[] = {0, 3, 1, 2};
  Qnn_Param_t params_images_nchw[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "images_nchw_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_images_nchw_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)images_nchw_perm,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs_images_nchw[] = {
    "images"
  };
  uint32_t dimensions_images_nchw[] = {1, 3, 540, 720};
  Qnn_Tensor_t outputs_images_nchw[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "images_nchw",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions_images_nchw,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "images_nchw", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params_images_nchw, // Node Params
                         1, // Num Node Params
                         inputs_images_nchw, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_images_nchw, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Squeeze(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Squeeze */
  const char*  inputs__Squeeze[] = {
    "images_nchw"
  };
  uint32_t dimensions__Squeeze_output_0[] = {3, 540, 720};
  Qnn_Tensor_t outputs__Squeeze[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Squeeze_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__Squeeze_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Squeeze", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__Squeeze, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Squeeze, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor__transform_Constant_output_0(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions__transform_Constant_output_0[] = {3, 1, 1};
  VALIDATE(model.addTensor("_transform_Constant_output_0", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "_transform_Constant_output_0",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 3,
                                 .dimensions=dimensions__transform_Constant_output_0,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(_transform_Constant_output_0),
                                                .dataSize=BINLEN(_transform_Constant_output_0)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__transform_Sub(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _transform_Sub */
  Qnn_Param_t params__transform_Sub[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 18}}}}
  };
  const char*  inputs__transform_Sub[] = {
    "_Squeeze_output_0",
    "_transform_Constant_output_0"
  };
  uint32_t dimensions__transform_Sub_output_0[] = {3, 540, 720};
  Qnn_Tensor_t outputs__transform_Sub[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_transform_Sub_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__transform_Sub_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_transform_Sub", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__transform_Sub, // Node Params
                         1, // Num Node Params
                         inputs__transform_Sub, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__transform_Sub, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor__transform_Constant_1_output_0(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions__transform_Constant_1_output_0[] = {3, 1, 1};
  VALIDATE(model.addTensor("_transform_Constant_1_output_0", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "_transform_Constant_1_output_0",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 3,
                                 .dimensions=dimensions__transform_Constant_1_output_0,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(_transform_Constant_1_output_0),
                                                .dataSize=BINLEN(_transform_Constant_1_output_0)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__transform_Div(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _transform_Div */
  Qnn_Param_t params__transform_Div[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 2}}}}
  };
  const char*  inputs__transform_Div[] = {
    "_transform_Sub_output_0",
    "_transform_Constant_1_output_0"
  };
  uint32_t dimensions__transform_Div_output_0[] = {3, 540, 720};
  Qnn_Tensor_t outputs__transform_Div[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_transform_Div_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__transform_Div_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_transform_Div", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__transform_Div, // Node Params
                         1, // Num Node Params
                         inputs__transform_Div, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__transform_Div, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__transform_Unsqueeze(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _transform_Unsqueeze */
  const char*  inputs__transform_Unsqueeze[] = {
    "_transform_Div_output_0"
  };
  uint32_t dimensions__transform_Unsqueeze_output_0[] = {1, 3, 540, 720};
  Qnn_Tensor_t outputs__transform_Unsqueeze[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_transform_Unsqueeze_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__transform_Unsqueeze_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_transform_Unsqueeze", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__transform_Unsqueeze, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__transform_Unsqueeze, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__transform_Unsqueeze_output_0_nhwc(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _transform_Unsqueeze_output_0_nhwc */
  uint32_t dimensions___transform_Unsqueeze_output_0_nhwc_perm[] = {4};
  uint32_t __transform_Unsqueeze_output_0_nhwc_perm[] = {0, 2, 3, 1};
  Qnn_Param_t params__transform_Unsqueeze_output_0_nhwc[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__transform_Unsqueeze_output_0_nhwc_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___transform_Unsqueeze_output_0_nhwc_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__transform_Unsqueeze_output_0_nhwc_perm,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__transform_Unsqueeze_output_0_nhwc[] = {
    "_transform_Unsqueeze_output_0"
  };
  uint32_t dimensions__transform_Unsqueeze_output_0_nhwc[] = {1, 540, 720, 3};
  Qnn_Tensor_t outputs__transform_Unsqueeze_output_0_nhwc[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_transform_Unsqueeze_output_0_nhwc",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__transform_Unsqueeze_output_0_nhwc,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_transform_Unsqueeze_output_0_nhwc", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__transform_Unsqueeze_output_0_nhwc, // Node Params
                         1, // Num Node Params
                         inputs__transform_Unsqueeze_output_0_nhwc, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__transform_Unsqueeze_output_0_nhwc, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__transform_Resize(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _transform_Resize */
  Qnn_Param_t params__transform_Resize[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="align_corners",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="antialias",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="half_pixel_centers",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 1}}}}
  };
  const char*  inputs__transform_Resize[] = {
    "_transform_Unsqueeze_output_0_nhwc"
  };
  uint32_t dimensions__transform_Gather_output_0_pre_reshape[] = {1, 540, 720, 3};
  Qnn_Tensor_t outputs__transform_Resize[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_transform_Gather_output_0_pre_reshape",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__transform_Gather_output_0_pre_reshape,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_transform_Resize", // Node Name
                         "qti.aisw", // Package Name
                         "ResizeBilinear", // Qnn Node Type
                         params__transform_Resize, // Node Params
                         3, // Num Node Params
                         inputs__transform_Resize, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__transform_Resize, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1364(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1364[] = {7, 7, 3, 64};
  VALIDATE(model.addTensor("onnx__Conv_1364", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1364",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_onnx__Conv_1364,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1364),
                                                .dataSize=BINLEN(onnx__Conv_1364)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1365(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1365[] = {64};
  VALIDATE(model.addTensor("onnx__Conv_1365", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1365",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_onnx__Conv_1365,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1365),
                                                .dataSize=BINLEN(onnx__Conv_1365)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_conv1_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_conv1_Conv */
  uint32_t dimensions___backbone_base_conv1_Conv_dilation[] = {2};
  uint32_t __backbone_base_conv1_Conv_dilation[] = {1, 1};
  uint32_t dimensions___backbone_base_conv1_Conv_pad_amount[] = {2, 2};
  uint32_t __backbone_base_conv1_Conv_pad_amount[] = {3, 3, 3, 3};
  uint32_t dimensions___backbone_base_conv1_Conv_stride[] = {2};
  uint32_t __backbone_base_conv1_Conv_stride[] = {2, 2};
  Qnn_Param_t params__backbone_base_conv1_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_conv1_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_conv1_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_conv1_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_conv1_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___backbone_base_conv1_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_conv1_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_conv1_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_conv1_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_conv1_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__backbone_base_conv1_Conv[] = {
    "_transform_Gather_output_0_pre_reshape",
    "onnx__Conv_1364",
    "onnx__Conv_1365"
  };
  uint32_t dimensions__backbone_base_conv1_Conv_output_0[] = {1, 270, 360, 64};
  Qnn_Tensor_t outputs__backbone_base_conv1_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_conv1_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_conv1_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_conv1_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__backbone_base_conv1_Conv, // Node Params
                         4, // Num Node Params
                         inputs__backbone_base_conv1_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__backbone_base_conv1_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_relu_Relu(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_relu_Relu */
  Qnn_Param_t params__backbone_base_relu_Relu[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 4}}}}
  };
  const char*  inputs__backbone_base_relu_Relu[] = {
    "_backbone_base_conv1_Conv_output_0"
  };
  uint32_t dimensions__backbone_base_relu_Relu_output_0[] = {1, 270, 360, 64};
  Qnn_Tensor_t outputs__backbone_base_relu_Relu[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_relu_Relu_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_relu_Relu_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_relu_Relu", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params__backbone_base_relu_Relu, // Node Params
                         1, // Num Node Params
                         inputs__backbone_base_relu_Relu, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__backbone_base_relu_Relu, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_maxpool_MaxPool(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_maxpool_MaxPool */
  uint32_t dimensions___backbone_base_maxpool_MaxPool_filter_size[] = {2};
  uint32_t __backbone_base_maxpool_MaxPool_filter_size[] = {3, 3};
  uint32_t dimensions___backbone_base_maxpool_MaxPool_pad_amount[] = {2, 2};
  uint32_t __backbone_base_maxpool_MaxPool_pad_amount[] = {1, 0, 1, 0};
  uint32_t dimensions___backbone_base_maxpool_MaxPool_stride[] = {2};
  uint32_t __backbone_base_maxpool_MaxPool_stride[] = {2, 2};
  Qnn_Param_t params__backbone_base_maxpool_MaxPool[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="filter_size",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_maxpool_MaxPool_filter_size",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_maxpool_MaxPool_filter_size,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_maxpool_MaxPool_filter_size,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_maxpool_MaxPool_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___backbone_base_maxpool_MaxPool_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_maxpool_MaxPool_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_maxpool_MaxPool_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_maxpool_MaxPool_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_maxpool_MaxPool_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__backbone_base_maxpool_MaxPool[] = {
    "_backbone_base_relu_Relu_output_0"
  };
  uint32_t dimensions__backbone_base_maxpool_MaxPool_output_0[] = {1, 135, 180, 64};
  Qnn_Tensor_t outputs__backbone_base_maxpool_MaxPool[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_maxpool_MaxPool_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_maxpool_MaxPool_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_maxpool_MaxPool", // Node Name
                         "qti.aisw", // Package Name
                         "PoolMax2d", // Qnn Node Type
                         params__backbone_base_maxpool_MaxPool, // Node Params
                         3, // Num Node Params
                         inputs__backbone_base_maxpool_MaxPool, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__backbone_base_maxpool_MaxPool, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1367(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1367[] = {3, 3, 64, 64};
  VALIDATE(model.addTensor("onnx__Conv_1367", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1367",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_onnx__Conv_1367,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1367),
                                                .dataSize=BINLEN(onnx__Conv_1367)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1368(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1368[] = {64};
  VALIDATE(model.addTensor("onnx__Conv_1368", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1368",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_onnx__Conv_1368,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1368),
                                                .dataSize=BINLEN(onnx__Conv_1368)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer1_layer1_0_conv1_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer1_layer1_0_conv1_Conv */
  uint32_t dimensions___backbone_base_layer1_layer1_0_conv1_Conv_dilation[] = {2};
  uint32_t __backbone_base_layer1_layer1_0_conv1_Conv_dilation[] = {1, 1};
  uint32_t dimensions___backbone_base_layer1_layer1_0_conv1_Conv_pad_amount[] = {2, 2};
  uint32_t __backbone_base_layer1_layer1_0_conv1_Conv_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions___backbone_base_layer1_layer1_0_conv1_Conv_stride[] = {2};
  uint32_t __backbone_base_layer1_layer1_0_conv1_Conv_stride[] = {1, 1};
  Qnn_Param_t params__backbone_base_layer1_layer1_0_conv1_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer1_layer1_0_conv1_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer1_layer1_0_conv1_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer1_layer1_0_conv1_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer1_layer1_0_conv1_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___backbone_base_layer1_layer1_0_conv1_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer1_layer1_0_conv1_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer1_layer1_0_conv1_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer1_layer1_0_conv1_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer1_layer1_0_conv1_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__backbone_base_layer1_layer1_0_conv1_Conv[] = {
    "_backbone_base_maxpool_MaxPool_output_0",
    "onnx__Conv_1367",
    "onnx__Conv_1368"
  };
  uint32_t dimensions__backbone_base_layer1_layer1_0_conv1_Conv_output_0[] = {1, 135, 180, 64};
  Qnn_Tensor_t outputs__backbone_base_layer1_layer1_0_conv1_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer1_layer1_0_conv1_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer1_layer1_0_conv1_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer1_layer1_0_conv1_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__backbone_base_layer1_layer1_0_conv1_Conv, // Node Params
                         4, // Num Node Params
                         inputs__backbone_base_layer1_layer1_0_conv1_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__backbone_base_layer1_layer1_0_conv1_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer1_layer1_0_relu_Relu(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer1_layer1_0_relu_Relu */
  Qnn_Param_t params__backbone_base_layer1_layer1_0_relu_Relu[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 4}}}}
  };
  const char*  inputs__backbone_base_layer1_layer1_0_relu_Relu[] = {
    "_backbone_base_layer1_layer1_0_conv1_Conv_output_0"
  };
  uint32_t dimensions__backbone_base_layer1_layer1_0_relu_Relu_output_0[] = {1, 135, 180, 64};
  Qnn_Tensor_t outputs__backbone_base_layer1_layer1_0_relu_Relu[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer1_layer1_0_relu_Relu_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer1_layer1_0_relu_Relu_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer1_layer1_0_relu_Relu", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params__backbone_base_layer1_layer1_0_relu_Relu, // Node Params
                         1, // Num Node Params
                         inputs__backbone_base_layer1_layer1_0_relu_Relu, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__backbone_base_layer1_layer1_0_relu_Relu, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1370(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1370[] = {3, 3, 64, 64};
  VALIDATE(model.addTensor("onnx__Conv_1370", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1370",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_onnx__Conv_1370,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1370),
                                                .dataSize=BINLEN(onnx__Conv_1370)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1371(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1371[] = {64};
  VALIDATE(model.addTensor("onnx__Conv_1371", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1371",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_onnx__Conv_1371,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1371),
                                                .dataSize=BINLEN(onnx__Conv_1371)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer1_layer1_0_conv2_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer1_layer1_0_conv2_Conv */
  uint32_t dimensions___backbone_base_layer1_layer1_0_conv2_Conv_dilation[] = {2};
  uint32_t __backbone_base_layer1_layer1_0_conv2_Conv_dilation[] = {1, 1};
  uint32_t dimensions___backbone_base_layer1_layer1_0_conv2_Conv_pad_amount[] = {2, 2};
  uint32_t __backbone_base_layer1_layer1_0_conv2_Conv_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions___backbone_base_layer1_layer1_0_conv2_Conv_stride[] = {2};
  uint32_t __backbone_base_layer1_layer1_0_conv2_Conv_stride[] = {1, 1};
  Qnn_Param_t params__backbone_base_layer1_layer1_0_conv2_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer1_layer1_0_conv2_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer1_layer1_0_conv2_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer1_layer1_0_conv2_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer1_layer1_0_conv2_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___backbone_base_layer1_layer1_0_conv2_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer1_layer1_0_conv2_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer1_layer1_0_conv2_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer1_layer1_0_conv2_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer1_layer1_0_conv2_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__backbone_base_layer1_layer1_0_conv2_Conv[] = {
    "_backbone_base_layer1_layer1_0_relu_Relu_output_0",
    "onnx__Conv_1370",
    "onnx__Conv_1371"
  };
  uint32_t dimensions__backbone_base_layer1_layer1_0_conv2_Conv_output_0[] = {1, 135, 180, 64};
  Qnn_Tensor_t outputs__backbone_base_layer1_layer1_0_conv2_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer1_layer1_0_conv2_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer1_layer1_0_conv2_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer1_layer1_0_conv2_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__backbone_base_layer1_layer1_0_conv2_Conv, // Node Params
                         4, // Num Node Params
                         inputs__backbone_base_layer1_layer1_0_conv2_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__backbone_base_layer1_layer1_0_conv2_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer1_layer1_0_Add(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer1_layer1_0_Add */
  Qnn_Param_t params__backbone_base_layer1_layer1_0_Add[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__backbone_base_layer1_layer1_0_Add[] = {
    "_backbone_base_layer1_layer1_0_conv2_Conv_output_0",
    "_backbone_base_maxpool_MaxPool_output_0"
  };
  uint32_t dimensions__backbone_base_layer1_layer1_0_Add_output_0[] = {1, 135, 180, 64};
  Qnn_Tensor_t outputs__backbone_base_layer1_layer1_0_Add[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer1_layer1_0_Add_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer1_layer1_0_Add_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer1_layer1_0_Add", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__backbone_base_layer1_layer1_0_Add, // Node Params
                         1, // Num Node Params
                         inputs__backbone_base_layer1_layer1_0_Add, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__backbone_base_layer1_layer1_0_Add, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer1_layer1_0_relu_1_Relu(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer1_layer1_0_relu_1_Relu */
  Qnn_Param_t params__backbone_base_layer1_layer1_0_relu_1_Relu[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 4}}}}
  };
  const char*  inputs__backbone_base_layer1_layer1_0_relu_1_Relu[] = {
    "_backbone_base_layer1_layer1_0_Add_output_0"
  };
  uint32_t dimensions__backbone_base_layer1_layer1_0_relu_1_Relu_output_0[] = {1, 135, 180, 64};
  Qnn_Tensor_t outputs__backbone_base_layer1_layer1_0_relu_1_Relu[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer1_layer1_0_relu_1_Relu_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer1_layer1_0_relu_1_Relu_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer1_layer1_0_relu_1_Relu", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params__backbone_base_layer1_layer1_0_relu_1_Relu, // Node Params
                         1, // Num Node Params
                         inputs__backbone_base_layer1_layer1_0_relu_1_Relu, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__backbone_base_layer1_layer1_0_relu_1_Relu, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1373(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1373[] = {3, 3, 64, 64};
  VALIDATE(model.addTensor("onnx__Conv_1373", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1373",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_onnx__Conv_1373,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1373),
                                                .dataSize=BINLEN(onnx__Conv_1373)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1374(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1374[] = {64};
  VALIDATE(model.addTensor("onnx__Conv_1374", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1374",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_onnx__Conv_1374,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1374),
                                                .dataSize=BINLEN(onnx__Conv_1374)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer1_layer1_1_conv1_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer1_layer1_1_conv1_Conv */
  uint32_t dimensions___backbone_base_layer1_layer1_1_conv1_Conv_dilation[] = {2};
  uint32_t __backbone_base_layer1_layer1_1_conv1_Conv_dilation[] = {1, 1};
  uint32_t dimensions___backbone_base_layer1_layer1_1_conv1_Conv_pad_amount[] = {2, 2};
  uint32_t __backbone_base_layer1_layer1_1_conv1_Conv_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions___backbone_base_layer1_layer1_1_conv1_Conv_stride[] = {2};
  uint32_t __backbone_base_layer1_layer1_1_conv1_Conv_stride[] = {1, 1};
  Qnn_Param_t params__backbone_base_layer1_layer1_1_conv1_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer1_layer1_1_conv1_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer1_layer1_1_conv1_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer1_layer1_1_conv1_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer1_layer1_1_conv1_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___backbone_base_layer1_layer1_1_conv1_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer1_layer1_1_conv1_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer1_layer1_1_conv1_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer1_layer1_1_conv1_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer1_layer1_1_conv1_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__backbone_base_layer1_layer1_1_conv1_Conv[] = {
    "_backbone_base_layer1_layer1_0_relu_1_Relu_output_0",
    "onnx__Conv_1373",
    "onnx__Conv_1374"
  };
  uint32_t dimensions__backbone_base_layer1_layer1_1_conv1_Conv_output_0[] = {1, 135, 180, 64};
  Qnn_Tensor_t outputs__backbone_base_layer1_layer1_1_conv1_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer1_layer1_1_conv1_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer1_layer1_1_conv1_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer1_layer1_1_conv1_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__backbone_base_layer1_layer1_1_conv1_Conv, // Node Params
                         4, // Num Node Params
                         inputs__backbone_base_layer1_layer1_1_conv1_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__backbone_base_layer1_layer1_1_conv1_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer1_layer1_1_relu_Relu(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer1_layer1_1_relu_Relu */
  Qnn_Param_t params__backbone_base_layer1_layer1_1_relu_Relu[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 4}}}}
  };
  const char*  inputs__backbone_base_layer1_layer1_1_relu_Relu[] = {
    "_backbone_base_layer1_layer1_1_conv1_Conv_output_0"
  };
  uint32_t dimensions__backbone_base_layer1_layer1_1_relu_Relu_output_0[] = {1, 135, 180, 64};
  Qnn_Tensor_t outputs__backbone_base_layer1_layer1_1_relu_Relu[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer1_layer1_1_relu_Relu_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer1_layer1_1_relu_Relu_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer1_layer1_1_relu_Relu", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params__backbone_base_layer1_layer1_1_relu_Relu, // Node Params
                         1, // Num Node Params
                         inputs__backbone_base_layer1_layer1_1_relu_Relu, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__backbone_base_layer1_layer1_1_relu_Relu, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1376(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1376[] = {3, 3, 64, 64};
  VALIDATE(model.addTensor("onnx__Conv_1376", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1376",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_onnx__Conv_1376,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1376),
                                                .dataSize=BINLEN(onnx__Conv_1376)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1377(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1377[] = {64};
  VALIDATE(model.addTensor("onnx__Conv_1377", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1377",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_onnx__Conv_1377,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1377),
                                                .dataSize=BINLEN(onnx__Conv_1377)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer1_layer1_1_conv2_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer1_layer1_1_conv2_Conv */
  uint32_t dimensions___backbone_base_layer1_layer1_1_conv2_Conv_dilation[] = {2};
  uint32_t __backbone_base_layer1_layer1_1_conv2_Conv_dilation[] = {1, 1};
  uint32_t dimensions___backbone_base_layer1_layer1_1_conv2_Conv_pad_amount[] = {2, 2};
  uint32_t __backbone_base_layer1_layer1_1_conv2_Conv_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions___backbone_base_layer1_layer1_1_conv2_Conv_stride[] = {2};
  uint32_t __backbone_base_layer1_layer1_1_conv2_Conv_stride[] = {1, 1};
  Qnn_Param_t params__backbone_base_layer1_layer1_1_conv2_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer1_layer1_1_conv2_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer1_layer1_1_conv2_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer1_layer1_1_conv2_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer1_layer1_1_conv2_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___backbone_base_layer1_layer1_1_conv2_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer1_layer1_1_conv2_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer1_layer1_1_conv2_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer1_layer1_1_conv2_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer1_layer1_1_conv2_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__backbone_base_layer1_layer1_1_conv2_Conv[] = {
    "_backbone_base_layer1_layer1_1_relu_Relu_output_0",
    "onnx__Conv_1376",
    "onnx__Conv_1377"
  };
  uint32_t dimensions__backbone_base_layer1_layer1_1_conv2_Conv_output_0[] = {1, 135, 180, 64};
  Qnn_Tensor_t outputs__backbone_base_layer1_layer1_1_conv2_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer1_layer1_1_conv2_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer1_layer1_1_conv2_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer1_layer1_1_conv2_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__backbone_base_layer1_layer1_1_conv2_Conv, // Node Params
                         4, // Num Node Params
                         inputs__backbone_base_layer1_layer1_1_conv2_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__backbone_base_layer1_layer1_1_conv2_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer1_layer1_1_Add(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer1_layer1_1_Add */
  Qnn_Param_t params__backbone_base_layer1_layer1_1_Add[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__backbone_base_layer1_layer1_1_Add[] = {
    "_backbone_base_layer1_layer1_1_conv2_Conv_output_0",
    "_backbone_base_layer1_layer1_0_relu_1_Relu_output_0"
  };
  uint32_t dimensions__backbone_base_layer1_layer1_1_Add_output_0[] = {1, 135, 180, 64};
  Qnn_Tensor_t outputs__backbone_base_layer1_layer1_1_Add[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer1_layer1_1_Add_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer1_layer1_1_Add_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer1_layer1_1_Add", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__backbone_base_layer1_layer1_1_Add, // Node Params
                         1, // Num Node Params
                         inputs__backbone_base_layer1_layer1_1_Add, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__backbone_base_layer1_layer1_1_Add, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer1_layer1_1_relu_1_Relu(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer1_layer1_1_relu_1_Relu */
  Qnn_Param_t params__backbone_base_layer1_layer1_1_relu_1_Relu[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 4}}}}
  };
  const char*  inputs__backbone_base_layer1_layer1_1_relu_1_Relu[] = {
    "_backbone_base_layer1_layer1_1_Add_output_0"
  };
  uint32_t dimensions__backbone_base_layer1_layer1_1_relu_1_Relu_output_0[] = {1, 135, 180, 64};
  Qnn_Tensor_t outputs__backbone_base_layer1_layer1_1_relu_1_Relu[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer1_layer1_1_relu_1_Relu_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer1_layer1_1_relu_1_Relu_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer1_layer1_1_relu_1_Relu", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params__backbone_base_layer1_layer1_1_relu_1_Relu, // Node Params
                         1, // Num Node Params
                         inputs__backbone_base_layer1_layer1_1_relu_1_Relu, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__backbone_base_layer1_layer1_1_relu_1_Relu, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1379(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1379[] = {3, 3, 64, 128};
  VALIDATE(model.addTensor("onnx__Conv_1379", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1379",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_onnx__Conv_1379,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1379),
                                                .dataSize=BINLEN(onnx__Conv_1379)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1380(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1380[] = {128};
  VALIDATE(model.addTensor("onnx__Conv_1380", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1380",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_onnx__Conv_1380,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1380),
                                                .dataSize=BINLEN(onnx__Conv_1380)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer2_layer2_0_conv1_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer2_layer2_0_conv1_Conv */
  uint32_t dimensions___backbone_base_layer2_layer2_0_conv1_Conv_dilation[] = {2};
  uint32_t __backbone_base_layer2_layer2_0_conv1_Conv_dilation[] = {1, 1};
  uint32_t dimensions___backbone_base_layer2_layer2_0_conv1_Conv_pad_amount[] = {2, 2};
  uint32_t __backbone_base_layer2_layer2_0_conv1_Conv_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions___backbone_base_layer2_layer2_0_conv1_Conv_stride[] = {2};
  uint32_t __backbone_base_layer2_layer2_0_conv1_Conv_stride[] = {2, 2};
  Qnn_Param_t params__backbone_base_layer2_layer2_0_conv1_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer2_layer2_0_conv1_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer2_layer2_0_conv1_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer2_layer2_0_conv1_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer2_layer2_0_conv1_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___backbone_base_layer2_layer2_0_conv1_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer2_layer2_0_conv1_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer2_layer2_0_conv1_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer2_layer2_0_conv1_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer2_layer2_0_conv1_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__backbone_base_layer2_layer2_0_conv1_Conv[] = {
    "_backbone_base_layer1_layer1_1_relu_1_Relu_output_0",
    "onnx__Conv_1379",
    "onnx__Conv_1380"
  };
  uint32_t dimensions__backbone_base_layer2_layer2_0_conv1_Conv_output_0[] = {1, 68, 90, 128};
  Qnn_Tensor_t outputs__backbone_base_layer2_layer2_0_conv1_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer2_layer2_0_conv1_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer2_layer2_0_conv1_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer2_layer2_0_conv1_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__backbone_base_layer2_layer2_0_conv1_Conv, // Node Params
                         4, // Num Node Params
                         inputs__backbone_base_layer2_layer2_0_conv1_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__backbone_base_layer2_layer2_0_conv1_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer2_layer2_0_relu_Relu(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer2_layer2_0_relu_Relu */
  Qnn_Param_t params__backbone_base_layer2_layer2_0_relu_Relu[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 4}}}}
  };
  const char*  inputs__backbone_base_layer2_layer2_0_relu_Relu[] = {
    "_backbone_base_layer2_layer2_0_conv1_Conv_output_0"
  };
  uint32_t dimensions__backbone_base_layer2_layer2_0_relu_Relu_output_0[] = {1, 68, 90, 128};
  Qnn_Tensor_t outputs__backbone_base_layer2_layer2_0_relu_Relu[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer2_layer2_0_relu_Relu_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer2_layer2_0_relu_Relu_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer2_layer2_0_relu_Relu", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params__backbone_base_layer2_layer2_0_relu_Relu, // Node Params
                         1, // Num Node Params
                         inputs__backbone_base_layer2_layer2_0_relu_Relu, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__backbone_base_layer2_layer2_0_relu_Relu, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1382(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1382[] = {3, 3, 128, 128};
  VALIDATE(model.addTensor("onnx__Conv_1382", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1382",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_onnx__Conv_1382,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1382),
                                                .dataSize=BINLEN(onnx__Conv_1382)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1383(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1383[] = {128};
  VALIDATE(model.addTensor("onnx__Conv_1383", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1383",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_onnx__Conv_1383,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1383),
                                                .dataSize=BINLEN(onnx__Conv_1383)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer2_layer2_0_conv2_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer2_layer2_0_conv2_Conv */
  uint32_t dimensions___backbone_base_layer2_layer2_0_conv2_Conv_dilation[] = {2};
  uint32_t __backbone_base_layer2_layer2_0_conv2_Conv_dilation[] = {1, 1};
  uint32_t dimensions___backbone_base_layer2_layer2_0_conv2_Conv_pad_amount[] = {2, 2};
  uint32_t __backbone_base_layer2_layer2_0_conv2_Conv_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions___backbone_base_layer2_layer2_0_conv2_Conv_stride[] = {2};
  uint32_t __backbone_base_layer2_layer2_0_conv2_Conv_stride[] = {1, 1};
  Qnn_Param_t params__backbone_base_layer2_layer2_0_conv2_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer2_layer2_0_conv2_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer2_layer2_0_conv2_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer2_layer2_0_conv2_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer2_layer2_0_conv2_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___backbone_base_layer2_layer2_0_conv2_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer2_layer2_0_conv2_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer2_layer2_0_conv2_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer2_layer2_0_conv2_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer2_layer2_0_conv2_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__backbone_base_layer2_layer2_0_conv2_Conv[] = {
    "_backbone_base_layer2_layer2_0_relu_Relu_output_0",
    "onnx__Conv_1382",
    "onnx__Conv_1383"
  };
  uint32_t dimensions__backbone_base_layer2_layer2_0_conv2_Conv_output_0[] = {1, 68, 90, 128};
  Qnn_Tensor_t outputs__backbone_base_layer2_layer2_0_conv2_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer2_layer2_0_conv2_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer2_layer2_0_conv2_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer2_layer2_0_conv2_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__backbone_base_layer2_layer2_0_conv2_Conv, // Node Params
                         4, // Num Node Params
                         inputs__backbone_base_layer2_layer2_0_conv2_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__backbone_base_layer2_layer2_0_conv2_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1385(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1385[] = {1, 1, 64, 128};
  VALIDATE(model.addTensor("onnx__Conv_1385", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1385",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_onnx__Conv_1385,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1385),
                                                .dataSize=BINLEN(onnx__Conv_1385)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1386(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1386[] = {128};
  VALIDATE(model.addTensor("onnx__Conv_1386", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1386",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_onnx__Conv_1386,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1386),
                                                .dataSize=BINLEN(onnx__Conv_1386)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer2_layer2_0_downsample_downsample_0_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer2_layer2_0_downsample_downsample_0_Conv */
  uint32_t dimensions___backbone_base_layer2_layer2_0_downsample_downsample_0_Conv_dilation[] = {2};
  uint32_t __backbone_base_layer2_layer2_0_downsample_downsample_0_Conv_dilation[] = {1, 1};
  uint32_t dimensions___backbone_base_layer2_layer2_0_downsample_downsample_0_Conv_pad_amount[] = {2, 2};
  uint32_t __backbone_base_layer2_layer2_0_downsample_downsample_0_Conv_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions___backbone_base_layer2_layer2_0_downsample_downsample_0_Conv_stride[] = {2};
  uint32_t __backbone_base_layer2_layer2_0_downsample_downsample_0_Conv_stride[] = {2, 2};
  Qnn_Param_t params__backbone_base_layer2_layer2_0_downsample_downsample_0_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer2_layer2_0_downsample_downsample_0_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer2_layer2_0_downsample_downsample_0_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer2_layer2_0_downsample_downsample_0_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer2_layer2_0_downsample_downsample_0_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___backbone_base_layer2_layer2_0_downsample_downsample_0_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer2_layer2_0_downsample_downsample_0_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer2_layer2_0_downsample_downsample_0_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer2_layer2_0_downsample_downsample_0_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer2_layer2_0_downsample_downsample_0_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__backbone_base_layer2_layer2_0_downsample_downsample_0_Conv[] = {
    "_backbone_base_layer1_layer1_1_relu_1_Relu_output_0",
    "onnx__Conv_1385",
    "onnx__Conv_1386"
  };
  uint32_t dimensions__backbone_base_layer2_layer2_0_downsample_downsample_0_Conv_output_0[] = {1, 68, 90, 128};
  Qnn_Tensor_t outputs__backbone_base_layer2_layer2_0_downsample_downsample_0_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer2_layer2_0_downsample_downsample_0_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer2_layer2_0_downsample_downsample_0_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer2_layer2_0_downsample_downsample_0_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__backbone_base_layer2_layer2_0_downsample_downsample_0_Conv, // Node Params
                         4, // Num Node Params
                         inputs__backbone_base_layer2_layer2_0_downsample_downsample_0_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__backbone_base_layer2_layer2_0_downsample_downsample_0_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer2_layer2_0_Add(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer2_layer2_0_Add */
  Qnn_Param_t params__backbone_base_layer2_layer2_0_Add[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__backbone_base_layer2_layer2_0_Add[] = {
    "_backbone_base_layer2_layer2_0_conv2_Conv_output_0",
    "_backbone_base_layer2_layer2_0_downsample_downsample_0_Conv_output_0"
  };
  uint32_t dimensions__backbone_base_layer2_layer2_0_Add_output_0[] = {1, 68, 90, 128};
  Qnn_Tensor_t outputs__backbone_base_layer2_layer2_0_Add[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer2_layer2_0_Add_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer2_layer2_0_Add_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer2_layer2_0_Add", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__backbone_base_layer2_layer2_0_Add, // Node Params
                         1, // Num Node Params
                         inputs__backbone_base_layer2_layer2_0_Add, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__backbone_base_layer2_layer2_0_Add, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer2_layer2_0_relu_1_Relu(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer2_layer2_0_relu_1_Relu */
  Qnn_Param_t params__backbone_base_layer2_layer2_0_relu_1_Relu[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 4}}}}
  };
  const char*  inputs__backbone_base_layer2_layer2_0_relu_1_Relu[] = {
    "_backbone_base_layer2_layer2_0_Add_output_0"
  };
  uint32_t dimensions__backbone_base_layer2_layer2_0_relu_1_Relu_output_0[] = {1, 68, 90, 128};
  Qnn_Tensor_t outputs__backbone_base_layer2_layer2_0_relu_1_Relu[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer2_layer2_0_relu_1_Relu_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer2_layer2_0_relu_1_Relu_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer2_layer2_0_relu_1_Relu", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params__backbone_base_layer2_layer2_0_relu_1_Relu, // Node Params
                         1, // Num Node Params
                         inputs__backbone_base_layer2_layer2_0_relu_1_Relu, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__backbone_base_layer2_layer2_0_relu_1_Relu, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1388(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1388[] = {3, 3, 128, 128};
  VALIDATE(model.addTensor("onnx__Conv_1388", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1388",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_onnx__Conv_1388,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1388),
                                                .dataSize=BINLEN(onnx__Conv_1388)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1389(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1389[] = {128};
  VALIDATE(model.addTensor("onnx__Conv_1389", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1389",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_onnx__Conv_1389,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1389),
                                                .dataSize=BINLEN(onnx__Conv_1389)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer2_layer2_1_conv1_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer2_layer2_1_conv1_Conv */
  uint32_t dimensions___backbone_base_layer2_layer2_1_conv1_Conv_dilation[] = {2};
  uint32_t __backbone_base_layer2_layer2_1_conv1_Conv_dilation[] = {1, 1};
  uint32_t dimensions___backbone_base_layer2_layer2_1_conv1_Conv_pad_amount[] = {2, 2};
  uint32_t __backbone_base_layer2_layer2_1_conv1_Conv_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions___backbone_base_layer2_layer2_1_conv1_Conv_stride[] = {2};
  uint32_t __backbone_base_layer2_layer2_1_conv1_Conv_stride[] = {1, 1};
  Qnn_Param_t params__backbone_base_layer2_layer2_1_conv1_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer2_layer2_1_conv1_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer2_layer2_1_conv1_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer2_layer2_1_conv1_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer2_layer2_1_conv1_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___backbone_base_layer2_layer2_1_conv1_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer2_layer2_1_conv1_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer2_layer2_1_conv1_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer2_layer2_1_conv1_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer2_layer2_1_conv1_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__backbone_base_layer2_layer2_1_conv1_Conv[] = {
    "_backbone_base_layer2_layer2_0_relu_1_Relu_output_0",
    "onnx__Conv_1388",
    "onnx__Conv_1389"
  };
  uint32_t dimensions__backbone_base_layer2_layer2_1_conv1_Conv_output_0[] = {1, 68, 90, 128};
  Qnn_Tensor_t outputs__backbone_base_layer2_layer2_1_conv1_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer2_layer2_1_conv1_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer2_layer2_1_conv1_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer2_layer2_1_conv1_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__backbone_base_layer2_layer2_1_conv1_Conv, // Node Params
                         4, // Num Node Params
                         inputs__backbone_base_layer2_layer2_1_conv1_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__backbone_base_layer2_layer2_1_conv1_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer2_layer2_1_relu_Relu(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer2_layer2_1_relu_Relu */
  Qnn_Param_t params__backbone_base_layer2_layer2_1_relu_Relu[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 4}}}}
  };
  const char*  inputs__backbone_base_layer2_layer2_1_relu_Relu[] = {
    "_backbone_base_layer2_layer2_1_conv1_Conv_output_0"
  };
  uint32_t dimensions__backbone_base_layer2_layer2_1_relu_Relu_output_0[] = {1, 68, 90, 128};
  Qnn_Tensor_t outputs__backbone_base_layer2_layer2_1_relu_Relu[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer2_layer2_1_relu_Relu_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer2_layer2_1_relu_Relu_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer2_layer2_1_relu_Relu", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params__backbone_base_layer2_layer2_1_relu_Relu, // Node Params
                         1, // Num Node Params
                         inputs__backbone_base_layer2_layer2_1_relu_Relu, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__backbone_base_layer2_layer2_1_relu_Relu, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1391(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1391[] = {3, 3, 128, 128};
  VALIDATE(model.addTensor("onnx__Conv_1391", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1391",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_onnx__Conv_1391,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1391),
                                                .dataSize=BINLEN(onnx__Conv_1391)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1392(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1392[] = {128};
  VALIDATE(model.addTensor("onnx__Conv_1392", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1392",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_onnx__Conv_1392,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1392),
                                                .dataSize=BINLEN(onnx__Conv_1392)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer2_layer2_1_conv2_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer2_layer2_1_conv2_Conv */
  uint32_t dimensions___backbone_base_layer2_layer2_1_conv2_Conv_dilation[] = {2};
  uint32_t __backbone_base_layer2_layer2_1_conv2_Conv_dilation[] = {1, 1};
  uint32_t dimensions___backbone_base_layer2_layer2_1_conv2_Conv_pad_amount[] = {2, 2};
  uint32_t __backbone_base_layer2_layer2_1_conv2_Conv_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions___backbone_base_layer2_layer2_1_conv2_Conv_stride[] = {2};
  uint32_t __backbone_base_layer2_layer2_1_conv2_Conv_stride[] = {1, 1};
  Qnn_Param_t params__backbone_base_layer2_layer2_1_conv2_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer2_layer2_1_conv2_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer2_layer2_1_conv2_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer2_layer2_1_conv2_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer2_layer2_1_conv2_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___backbone_base_layer2_layer2_1_conv2_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer2_layer2_1_conv2_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer2_layer2_1_conv2_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer2_layer2_1_conv2_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer2_layer2_1_conv2_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__backbone_base_layer2_layer2_1_conv2_Conv[] = {
    "_backbone_base_layer2_layer2_1_relu_Relu_output_0",
    "onnx__Conv_1391",
    "onnx__Conv_1392"
  };
  uint32_t dimensions__backbone_base_layer2_layer2_1_conv2_Conv_output_0[] = {1, 68, 90, 128};
  Qnn_Tensor_t outputs__backbone_base_layer2_layer2_1_conv2_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer2_layer2_1_conv2_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer2_layer2_1_conv2_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer2_layer2_1_conv2_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__backbone_base_layer2_layer2_1_conv2_Conv, // Node Params
                         4, // Num Node Params
                         inputs__backbone_base_layer2_layer2_1_conv2_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__backbone_base_layer2_layer2_1_conv2_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer2_layer2_1_Add(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer2_layer2_1_Add */
  Qnn_Param_t params__backbone_base_layer2_layer2_1_Add[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__backbone_base_layer2_layer2_1_Add[] = {
    "_backbone_base_layer2_layer2_1_conv2_Conv_output_0",
    "_backbone_base_layer2_layer2_0_relu_1_Relu_output_0"
  };
  uint32_t dimensions__backbone_base_layer2_layer2_1_Add_output_0[] = {1, 68, 90, 128};
  Qnn_Tensor_t outputs__backbone_base_layer2_layer2_1_Add[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer2_layer2_1_Add_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer2_layer2_1_Add_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer2_layer2_1_Add", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__backbone_base_layer2_layer2_1_Add, // Node Params
                         1, // Num Node Params
                         inputs__backbone_base_layer2_layer2_1_Add, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__backbone_base_layer2_layer2_1_Add, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer2_layer2_1_relu_1_Relu(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer2_layer2_1_relu_1_Relu */
  Qnn_Param_t params__backbone_base_layer2_layer2_1_relu_1_Relu[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 4}}}}
  };
  const char*  inputs__backbone_base_layer2_layer2_1_relu_1_Relu[] = {
    "_backbone_base_layer2_layer2_1_Add_output_0"
  };
  uint32_t dimensions__backbone_base_layer2_layer2_1_relu_1_Relu_output_0[] = {1, 68, 90, 128};
  Qnn_Tensor_t outputs__backbone_base_layer2_layer2_1_relu_1_Relu[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer2_layer2_1_relu_1_Relu_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer2_layer2_1_relu_1_Relu_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer2_layer2_1_relu_1_Relu", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params__backbone_base_layer2_layer2_1_relu_1_Relu, // Node Params
                         1, // Num Node Params
                         inputs__backbone_base_layer2_layer2_1_relu_1_Relu, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__backbone_base_layer2_layer2_1_relu_1_Relu, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1394(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1394[] = {3, 3, 128, 256};
  VALIDATE(model.addTensor("onnx__Conv_1394", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1394",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_onnx__Conv_1394,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1394),
                                                .dataSize=BINLEN(onnx__Conv_1394)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1395(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1395[] = {256};
  VALIDATE(model.addTensor("onnx__Conv_1395", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1395",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_onnx__Conv_1395,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1395),
                                                .dataSize=BINLEN(onnx__Conv_1395)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer3_layer3_0_conv1_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer3_layer3_0_conv1_Conv */
  uint32_t dimensions___backbone_base_layer3_layer3_0_conv1_Conv_dilation[] = {2};
  uint32_t __backbone_base_layer3_layer3_0_conv1_Conv_dilation[] = {1, 1};
  uint32_t dimensions___backbone_base_layer3_layer3_0_conv1_Conv_pad_amount[] = {2, 2};
  uint32_t __backbone_base_layer3_layer3_0_conv1_Conv_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions___backbone_base_layer3_layer3_0_conv1_Conv_stride[] = {2};
  uint32_t __backbone_base_layer3_layer3_0_conv1_Conv_stride[] = {2, 2};
  Qnn_Param_t params__backbone_base_layer3_layer3_0_conv1_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer3_layer3_0_conv1_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer3_layer3_0_conv1_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer3_layer3_0_conv1_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer3_layer3_0_conv1_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___backbone_base_layer3_layer3_0_conv1_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer3_layer3_0_conv1_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer3_layer3_0_conv1_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer3_layer3_0_conv1_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer3_layer3_0_conv1_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__backbone_base_layer3_layer3_0_conv1_Conv[] = {
    "_backbone_base_layer2_layer2_1_relu_1_Relu_output_0",
    "onnx__Conv_1394",
    "onnx__Conv_1395"
  };
  uint32_t dimensions__backbone_base_layer3_layer3_0_conv1_Conv_output_0[] = {1, 34, 45, 256};
  Qnn_Tensor_t outputs__backbone_base_layer3_layer3_0_conv1_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer3_layer3_0_conv1_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer3_layer3_0_conv1_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer3_layer3_0_conv1_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__backbone_base_layer3_layer3_0_conv1_Conv, // Node Params
                         4, // Num Node Params
                         inputs__backbone_base_layer3_layer3_0_conv1_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__backbone_base_layer3_layer3_0_conv1_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer3_layer3_0_relu_Relu(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer3_layer3_0_relu_Relu */
  Qnn_Param_t params__backbone_base_layer3_layer3_0_relu_Relu[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 4}}}}
  };
  const char*  inputs__backbone_base_layer3_layer3_0_relu_Relu[] = {
    "_backbone_base_layer3_layer3_0_conv1_Conv_output_0"
  };
  uint32_t dimensions__backbone_base_layer3_layer3_0_relu_Relu_output_0[] = {1, 34, 45, 256};
  Qnn_Tensor_t outputs__backbone_base_layer3_layer3_0_relu_Relu[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer3_layer3_0_relu_Relu_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer3_layer3_0_relu_Relu_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer3_layer3_0_relu_Relu", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params__backbone_base_layer3_layer3_0_relu_Relu, // Node Params
                         1, // Num Node Params
                         inputs__backbone_base_layer3_layer3_0_relu_Relu, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__backbone_base_layer3_layer3_0_relu_Relu, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1397(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1397[] = {3, 3, 256, 256};
  VALIDATE(model.addTensor("onnx__Conv_1397", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1397",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_onnx__Conv_1397,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1397),
                                                .dataSize=BINLEN(onnx__Conv_1397)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1398(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1398[] = {256};
  VALIDATE(model.addTensor("onnx__Conv_1398", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1398",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_onnx__Conv_1398,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1398),
                                                .dataSize=BINLEN(onnx__Conv_1398)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer3_layer3_0_conv2_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer3_layer3_0_conv2_Conv */
  uint32_t dimensions___backbone_base_layer3_layer3_0_conv2_Conv_dilation[] = {2};
  uint32_t __backbone_base_layer3_layer3_0_conv2_Conv_dilation[] = {1, 1};
  uint32_t dimensions___backbone_base_layer3_layer3_0_conv2_Conv_pad_amount[] = {2, 2};
  uint32_t __backbone_base_layer3_layer3_0_conv2_Conv_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions___backbone_base_layer3_layer3_0_conv2_Conv_stride[] = {2};
  uint32_t __backbone_base_layer3_layer3_0_conv2_Conv_stride[] = {1, 1};
  Qnn_Param_t params__backbone_base_layer3_layer3_0_conv2_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer3_layer3_0_conv2_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer3_layer3_0_conv2_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer3_layer3_0_conv2_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer3_layer3_0_conv2_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___backbone_base_layer3_layer3_0_conv2_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer3_layer3_0_conv2_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer3_layer3_0_conv2_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer3_layer3_0_conv2_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer3_layer3_0_conv2_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__backbone_base_layer3_layer3_0_conv2_Conv[] = {
    "_backbone_base_layer3_layer3_0_relu_Relu_output_0",
    "onnx__Conv_1397",
    "onnx__Conv_1398"
  };
  uint32_t dimensions__backbone_base_layer3_layer3_0_conv2_Conv_output_0[] = {1, 34, 45, 256};
  Qnn_Tensor_t outputs__backbone_base_layer3_layer3_0_conv2_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer3_layer3_0_conv2_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer3_layer3_0_conv2_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer3_layer3_0_conv2_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__backbone_base_layer3_layer3_0_conv2_Conv, // Node Params
                         4, // Num Node Params
                         inputs__backbone_base_layer3_layer3_0_conv2_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__backbone_base_layer3_layer3_0_conv2_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1400(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1400[] = {1, 1, 128, 256};
  VALIDATE(model.addTensor("onnx__Conv_1400", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1400",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_onnx__Conv_1400,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1400),
                                                .dataSize=BINLEN(onnx__Conv_1400)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1401(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1401[] = {256};
  VALIDATE(model.addTensor("onnx__Conv_1401", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1401",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_onnx__Conv_1401,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1401),
                                                .dataSize=BINLEN(onnx__Conv_1401)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer3_layer3_0_downsample_downsample_0_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer3_layer3_0_downsample_downsample_0_Conv */
  uint32_t dimensions___backbone_base_layer3_layer3_0_downsample_downsample_0_Conv_dilation[] = {2};
  uint32_t __backbone_base_layer3_layer3_0_downsample_downsample_0_Conv_dilation[] = {1, 1};
  uint32_t dimensions___backbone_base_layer3_layer3_0_downsample_downsample_0_Conv_pad_amount[] = {2, 2};
  uint32_t __backbone_base_layer3_layer3_0_downsample_downsample_0_Conv_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions___backbone_base_layer3_layer3_0_downsample_downsample_0_Conv_stride[] = {2};
  uint32_t __backbone_base_layer3_layer3_0_downsample_downsample_0_Conv_stride[] = {2, 2};
  Qnn_Param_t params__backbone_base_layer3_layer3_0_downsample_downsample_0_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer3_layer3_0_downsample_downsample_0_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer3_layer3_0_downsample_downsample_0_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer3_layer3_0_downsample_downsample_0_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer3_layer3_0_downsample_downsample_0_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___backbone_base_layer3_layer3_0_downsample_downsample_0_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer3_layer3_0_downsample_downsample_0_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer3_layer3_0_downsample_downsample_0_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer3_layer3_0_downsample_downsample_0_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer3_layer3_0_downsample_downsample_0_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__backbone_base_layer3_layer3_0_downsample_downsample_0_Conv[] = {
    "_backbone_base_layer2_layer2_1_relu_1_Relu_output_0",
    "onnx__Conv_1400",
    "onnx__Conv_1401"
  };
  uint32_t dimensions__backbone_base_layer3_layer3_0_downsample_downsample_0_Conv_output_0[] = {1, 34, 45, 256};
  Qnn_Tensor_t outputs__backbone_base_layer3_layer3_0_downsample_downsample_0_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer3_layer3_0_downsample_downsample_0_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer3_layer3_0_downsample_downsample_0_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer3_layer3_0_downsample_downsample_0_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__backbone_base_layer3_layer3_0_downsample_downsample_0_Conv, // Node Params
                         4, // Num Node Params
                         inputs__backbone_base_layer3_layer3_0_downsample_downsample_0_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__backbone_base_layer3_layer3_0_downsample_downsample_0_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer3_layer3_0_Add(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer3_layer3_0_Add */
  Qnn_Param_t params__backbone_base_layer3_layer3_0_Add[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__backbone_base_layer3_layer3_0_Add[] = {
    "_backbone_base_layer3_layer3_0_conv2_Conv_output_0",
    "_backbone_base_layer3_layer3_0_downsample_downsample_0_Conv_output_0"
  };
  uint32_t dimensions__backbone_base_layer3_layer3_0_Add_output_0[] = {1, 34, 45, 256};
  Qnn_Tensor_t outputs__backbone_base_layer3_layer3_0_Add[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer3_layer3_0_Add_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer3_layer3_0_Add_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer3_layer3_0_Add", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__backbone_base_layer3_layer3_0_Add, // Node Params
                         1, // Num Node Params
                         inputs__backbone_base_layer3_layer3_0_Add, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__backbone_base_layer3_layer3_0_Add, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer3_layer3_0_relu_1_Relu(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer3_layer3_0_relu_1_Relu */
  Qnn_Param_t params__backbone_base_layer3_layer3_0_relu_1_Relu[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 4}}}}
  };
  const char*  inputs__backbone_base_layer3_layer3_0_relu_1_Relu[] = {
    "_backbone_base_layer3_layer3_0_Add_output_0"
  };
  uint32_t dimensions__backbone_base_layer3_layer3_0_relu_1_Relu_output_0[] = {1, 34, 45, 256};
  Qnn_Tensor_t outputs__backbone_base_layer3_layer3_0_relu_1_Relu[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer3_layer3_0_relu_1_Relu_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer3_layer3_0_relu_1_Relu_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer3_layer3_0_relu_1_Relu", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params__backbone_base_layer3_layer3_0_relu_1_Relu, // Node Params
                         1, // Num Node Params
                         inputs__backbone_base_layer3_layer3_0_relu_1_Relu, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__backbone_base_layer3_layer3_0_relu_1_Relu, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1403(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1403[] = {3, 3, 256, 256};
  VALIDATE(model.addTensor("onnx__Conv_1403", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1403",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_onnx__Conv_1403,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1403),
                                                .dataSize=BINLEN(onnx__Conv_1403)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1404(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1404[] = {256};
  VALIDATE(model.addTensor("onnx__Conv_1404", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1404",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_onnx__Conv_1404,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1404),
                                                .dataSize=BINLEN(onnx__Conv_1404)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer3_layer3_1_conv1_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer3_layer3_1_conv1_Conv */
  uint32_t dimensions___backbone_base_layer3_layer3_1_conv1_Conv_dilation[] = {2};
  uint32_t __backbone_base_layer3_layer3_1_conv1_Conv_dilation[] = {1, 1};
  uint32_t dimensions___backbone_base_layer3_layer3_1_conv1_Conv_pad_amount[] = {2, 2};
  uint32_t __backbone_base_layer3_layer3_1_conv1_Conv_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions___backbone_base_layer3_layer3_1_conv1_Conv_stride[] = {2};
  uint32_t __backbone_base_layer3_layer3_1_conv1_Conv_stride[] = {1, 1};
  Qnn_Param_t params__backbone_base_layer3_layer3_1_conv1_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer3_layer3_1_conv1_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer3_layer3_1_conv1_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer3_layer3_1_conv1_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer3_layer3_1_conv1_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___backbone_base_layer3_layer3_1_conv1_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer3_layer3_1_conv1_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer3_layer3_1_conv1_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer3_layer3_1_conv1_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer3_layer3_1_conv1_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__backbone_base_layer3_layer3_1_conv1_Conv[] = {
    "_backbone_base_layer3_layer3_0_relu_1_Relu_output_0",
    "onnx__Conv_1403",
    "onnx__Conv_1404"
  };
  uint32_t dimensions__backbone_base_layer3_layer3_1_conv1_Conv_output_0[] = {1, 34, 45, 256};
  Qnn_Tensor_t outputs__backbone_base_layer3_layer3_1_conv1_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer3_layer3_1_conv1_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer3_layer3_1_conv1_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer3_layer3_1_conv1_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__backbone_base_layer3_layer3_1_conv1_Conv, // Node Params
                         4, // Num Node Params
                         inputs__backbone_base_layer3_layer3_1_conv1_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__backbone_base_layer3_layer3_1_conv1_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer3_layer3_1_relu_Relu(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer3_layer3_1_relu_Relu */
  Qnn_Param_t params__backbone_base_layer3_layer3_1_relu_Relu[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 4}}}}
  };
  const char*  inputs__backbone_base_layer3_layer3_1_relu_Relu[] = {
    "_backbone_base_layer3_layer3_1_conv1_Conv_output_0"
  };
  uint32_t dimensions__backbone_base_layer3_layer3_1_relu_Relu_output_0[] = {1, 34, 45, 256};
  Qnn_Tensor_t outputs__backbone_base_layer3_layer3_1_relu_Relu[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer3_layer3_1_relu_Relu_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer3_layer3_1_relu_Relu_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer3_layer3_1_relu_Relu", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params__backbone_base_layer3_layer3_1_relu_Relu, // Node Params
                         1, // Num Node Params
                         inputs__backbone_base_layer3_layer3_1_relu_Relu, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__backbone_base_layer3_layer3_1_relu_Relu, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1406(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1406[] = {3, 3, 256, 256};
  VALIDATE(model.addTensor("onnx__Conv_1406", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1406",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_onnx__Conv_1406,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1406),
                                                .dataSize=BINLEN(onnx__Conv_1406)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1407(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1407[] = {256};
  VALIDATE(model.addTensor("onnx__Conv_1407", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1407",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_onnx__Conv_1407,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1407),
                                                .dataSize=BINLEN(onnx__Conv_1407)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer3_layer3_1_conv2_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer3_layer3_1_conv2_Conv */
  uint32_t dimensions___backbone_base_layer3_layer3_1_conv2_Conv_dilation[] = {2};
  uint32_t __backbone_base_layer3_layer3_1_conv2_Conv_dilation[] = {1, 1};
  uint32_t dimensions___backbone_base_layer3_layer3_1_conv2_Conv_pad_amount[] = {2, 2};
  uint32_t __backbone_base_layer3_layer3_1_conv2_Conv_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions___backbone_base_layer3_layer3_1_conv2_Conv_stride[] = {2};
  uint32_t __backbone_base_layer3_layer3_1_conv2_Conv_stride[] = {1, 1};
  Qnn_Param_t params__backbone_base_layer3_layer3_1_conv2_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer3_layer3_1_conv2_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer3_layer3_1_conv2_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer3_layer3_1_conv2_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer3_layer3_1_conv2_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___backbone_base_layer3_layer3_1_conv2_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer3_layer3_1_conv2_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer3_layer3_1_conv2_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer3_layer3_1_conv2_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer3_layer3_1_conv2_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__backbone_base_layer3_layer3_1_conv2_Conv[] = {
    "_backbone_base_layer3_layer3_1_relu_Relu_output_0",
    "onnx__Conv_1406",
    "onnx__Conv_1407"
  };
  uint32_t dimensions__backbone_base_layer3_layer3_1_conv2_Conv_output_0[] = {1, 34, 45, 256};
  Qnn_Tensor_t outputs__backbone_base_layer3_layer3_1_conv2_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer3_layer3_1_conv2_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer3_layer3_1_conv2_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer3_layer3_1_conv2_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__backbone_base_layer3_layer3_1_conv2_Conv, // Node Params
                         4, // Num Node Params
                         inputs__backbone_base_layer3_layer3_1_conv2_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__backbone_base_layer3_layer3_1_conv2_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer3_layer3_1_Add(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer3_layer3_1_Add */
  Qnn_Param_t params__backbone_base_layer3_layer3_1_Add[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__backbone_base_layer3_layer3_1_Add[] = {
    "_backbone_base_layer3_layer3_1_conv2_Conv_output_0",
    "_backbone_base_layer3_layer3_0_relu_1_Relu_output_0"
  };
  uint32_t dimensions__backbone_base_layer3_layer3_1_Add_output_0[] = {1, 34, 45, 256};
  Qnn_Tensor_t outputs__backbone_base_layer3_layer3_1_Add[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer3_layer3_1_Add_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer3_layer3_1_Add_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer3_layer3_1_Add", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__backbone_base_layer3_layer3_1_Add, // Node Params
                         1, // Num Node Params
                         inputs__backbone_base_layer3_layer3_1_Add, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__backbone_base_layer3_layer3_1_Add, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer3_layer3_1_relu_1_Relu(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer3_layer3_1_relu_1_Relu */
  Qnn_Param_t params__backbone_base_layer3_layer3_1_relu_1_Relu[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 4}}}}
  };
  const char*  inputs__backbone_base_layer3_layer3_1_relu_1_Relu[] = {
    "_backbone_base_layer3_layer3_1_Add_output_0"
  };
  uint32_t dimensions__backbone_base_layer3_layer3_1_relu_1_Relu_output_0[] = {1, 34, 45, 256};
  Qnn_Tensor_t outputs__backbone_base_layer3_layer3_1_relu_1_Relu[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer3_layer3_1_relu_1_Relu_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer3_layer3_1_relu_1_Relu_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer3_layer3_1_relu_1_Relu", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params__backbone_base_layer3_layer3_1_relu_1_Relu, // Node Params
                         1, // Num Node Params
                         inputs__backbone_base_layer3_layer3_1_relu_1_Relu, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__backbone_base_layer3_layer3_1_relu_1_Relu, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1409(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1409[] = {3, 3, 256, 512};
  VALIDATE(model.addTensor("onnx__Conv_1409", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1409",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_onnx__Conv_1409,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1409),
                                                .dataSize=BINLEN(onnx__Conv_1409)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1410(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1410[] = {512};
  VALIDATE(model.addTensor("onnx__Conv_1410", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1410",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_onnx__Conv_1410,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1410),
                                                .dataSize=BINLEN(onnx__Conv_1410)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer4_layer4_0_conv1_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer4_layer4_0_conv1_Conv */
  uint32_t dimensions___backbone_base_layer4_layer4_0_conv1_Conv_dilation[] = {2};
  uint32_t __backbone_base_layer4_layer4_0_conv1_Conv_dilation[] = {1, 1};
  uint32_t dimensions___backbone_base_layer4_layer4_0_conv1_Conv_pad_amount[] = {2, 2};
  uint32_t __backbone_base_layer4_layer4_0_conv1_Conv_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions___backbone_base_layer4_layer4_0_conv1_Conv_stride[] = {2};
  uint32_t __backbone_base_layer4_layer4_0_conv1_Conv_stride[] = {2, 2};
  Qnn_Param_t params__backbone_base_layer4_layer4_0_conv1_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer4_layer4_0_conv1_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer4_layer4_0_conv1_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer4_layer4_0_conv1_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer4_layer4_0_conv1_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___backbone_base_layer4_layer4_0_conv1_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer4_layer4_0_conv1_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer4_layer4_0_conv1_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer4_layer4_0_conv1_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer4_layer4_0_conv1_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__backbone_base_layer4_layer4_0_conv1_Conv[] = {
    "_backbone_base_layer3_layer3_1_relu_1_Relu_output_0",
    "onnx__Conv_1409",
    "onnx__Conv_1410"
  };
  uint32_t dimensions__backbone_base_layer4_layer4_0_conv1_Conv_output_0[] = {1, 17, 23, 512};
  Qnn_Tensor_t outputs__backbone_base_layer4_layer4_0_conv1_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer4_layer4_0_conv1_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer4_layer4_0_conv1_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer4_layer4_0_conv1_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__backbone_base_layer4_layer4_0_conv1_Conv, // Node Params
                         4, // Num Node Params
                         inputs__backbone_base_layer4_layer4_0_conv1_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__backbone_base_layer4_layer4_0_conv1_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer4_layer4_0_relu_Relu(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer4_layer4_0_relu_Relu */
  Qnn_Param_t params__backbone_base_layer4_layer4_0_relu_Relu[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 4}}}}
  };
  const char*  inputs__backbone_base_layer4_layer4_0_relu_Relu[] = {
    "_backbone_base_layer4_layer4_0_conv1_Conv_output_0"
  };
  uint32_t dimensions__backbone_base_layer4_layer4_0_relu_Relu_output_0[] = {1, 17, 23, 512};
  Qnn_Tensor_t outputs__backbone_base_layer4_layer4_0_relu_Relu[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer4_layer4_0_relu_Relu_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer4_layer4_0_relu_Relu_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer4_layer4_0_relu_Relu", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params__backbone_base_layer4_layer4_0_relu_Relu, // Node Params
                         1, // Num Node Params
                         inputs__backbone_base_layer4_layer4_0_relu_Relu, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__backbone_base_layer4_layer4_0_relu_Relu, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1412(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1412[] = {3, 3, 512, 512};
  VALIDATE(model.addTensor("onnx__Conv_1412", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1412",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_onnx__Conv_1412,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1412),
                                                .dataSize=BINLEN(onnx__Conv_1412)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1413(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1413[] = {512};
  VALIDATE(model.addTensor("onnx__Conv_1413", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1413",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_onnx__Conv_1413,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1413),
                                                .dataSize=BINLEN(onnx__Conv_1413)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer4_layer4_0_conv2_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer4_layer4_0_conv2_Conv */
  uint32_t dimensions___backbone_base_layer4_layer4_0_conv2_Conv_dilation[] = {2};
  uint32_t __backbone_base_layer4_layer4_0_conv2_Conv_dilation[] = {1, 1};
  uint32_t dimensions___backbone_base_layer4_layer4_0_conv2_Conv_pad_amount[] = {2, 2};
  uint32_t __backbone_base_layer4_layer4_0_conv2_Conv_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions___backbone_base_layer4_layer4_0_conv2_Conv_stride[] = {2};
  uint32_t __backbone_base_layer4_layer4_0_conv2_Conv_stride[] = {1, 1};
  Qnn_Param_t params__backbone_base_layer4_layer4_0_conv2_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer4_layer4_0_conv2_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer4_layer4_0_conv2_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer4_layer4_0_conv2_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer4_layer4_0_conv2_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___backbone_base_layer4_layer4_0_conv2_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer4_layer4_0_conv2_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer4_layer4_0_conv2_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer4_layer4_0_conv2_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer4_layer4_0_conv2_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__backbone_base_layer4_layer4_0_conv2_Conv[] = {
    "_backbone_base_layer4_layer4_0_relu_Relu_output_0",
    "onnx__Conv_1412",
    "onnx__Conv_1413"
  };
  uint32_t dimensions__backbone_base_layer4_layer4_0_conv2_Conv_output_0[] = {1, 17, 23, 512};
  Qnn_Tensor_t outputs__backbone_base_layer4_layer4_0_conv2_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer4_layer4_0_conv2_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer4_layer4_0_conv2_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer4_layer4_0_conv2_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__backbone_base_layer4_layer4_0_conv2_Conv, // Node Params
                         4, // Num Node Params
                         inputs__backbone_base_layer4_layer4_0_conv2_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__backbone_base_layer4_layer4_0_conv2_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1415(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1415[] = {1, 1, 256, 512};
  VALIDATE(model.addTensor("onnx__Conv_1415", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1415",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_onnx__Conv_1415,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1415),
                                                .dataSize=BINLEN(onnx__Conv_1415)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1416(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1416[] = {512};
  VALIDATE(model.addTensor("onnx__Conv_1416", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1416",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_onnx__Conv_1416,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1416),
                                                .dataSize=BINLEN(onnx__Conv_1416)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer4_layer4_0_downsample_downsample_0_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer4_layer4_0_downsample_downsample_0_Conv */
  uint32_t dimensions___backbone_base_layer4_layer4_0_downsample_downsample_0_Conv_dilation[] = {2};
  uint32_t __backbone_base_layer4_layer4_0_downsample_downsample_0_Conv_dilation[] = {1, 1};
  uint32_t dimensions___backbone_base_layer4_layer4_0_downsample_downsample_0_Conv_pad_amount[] = {2, 2};
  uint32_t __backbone_base_layer4_layer4_0_downsample_downsample_0_Conv_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions___backbone_base_layer4_layer4_0_downsample_downsample_0_Conv_stride[] = {2};
  uint32_t __backbone_base_layer4_layer4_0_downsample_downsample_0_Conv_stride[] = {2, 2};
  Qnn_Param_t params__backbone_base_layer4_layer4_0_downsample_downsample_0_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer4_layer4_0_downsample_downsample_0_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer4_layer4_0_downsample_downsample_0_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer4_layer4_0_downsample_downsample_0_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer4_layer4_0_downsample_downsample_0_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___backbone_base_layer4_layer4_0_downsample_downsample_0_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer4_layer4_0_downsample_downsample_0_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer4_layer4_0_downsample_downsample_0_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer4_layer4_0_downsample_downsample_0_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer4_layer4_0_downsample_downsample_0_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__backbone_base_layer4_layer4_0_downsample_downsample_0_Conv[] = {
    "_backbone_base_layer3_layer3_1_relu_1_Relu_output_0",
    "onnx__Conv_1415",
    "onnx__Conv_1416"
  };
  uint32_t dimensions__backbone_base_layer4_layer4_0_downsample_downsample_0_Conv_output_0[] = {1, 17, 23, 512};
  Qnn_Tensor_t outputs__backbone_base_layer4_layer4_0_downsample_downsample_0_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer4_layer4_0_downsample_downsample_0_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer4_layer4_0_downsample_downsample_0_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer4_layer4_0_downsample_downsample_0_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__backbone_base_layer4_layer4_0_downsample_downsample_0_Conv, // Node Params
                         4, // Num Node Params
                         inputs__backbone_base_layer4_layer4_0_downsample_downsample_0_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__backbone_base_layer4_layer4_0_downsample_downsample_0_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer4_layer4_0_Add(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer4_layer4_0_Add */
  Qnn_Param_t params__backbone_base_layer4_layer4_0_Add[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__backbone_base_layer4_layer4_0_Add[] = {
    "_backbone_base_layer4_layer4_0_conv2_Conv_output_0",
    "_backbone_base_layer4_layer4_0_downsample_downsample_0_Conv_output_0"
  };
  uint32_t dimensions__backbone_base_layer4_layer4_0_Add_output_0[] = {1, 17, 23, 512};
  Qnn_Tensor_t outputs__backbone_base_layer4_layer4_0_Add[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer4_layer4_0_Add_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer4_layer4_0_Add_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer4_layer4_0_Add", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__backbone_base_layer4_layer4_0_Add, // Node Params
                         1, // Num Node Params
                         inputs__backbone_base_layer4_layer4_0_Add, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__backbone_base_layer4_layer4_0_Add, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer4_layer4_0_relu_1_Relu(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer4_layer4_0_relu_1_Relu */
  Qnn_Param_t params__backbone_base_layer4_layer4_0_relu_1_Relu[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 4}}}}
  };
  const char*  inputs__backbone_base_layer4_layer4_0_relu_1_Relu[] = {
    "_backbone_base_layer4_layer4_0_Add_output_0"
  };
  uint32_t dimensions__backbone_base_layer4_layer4_0_relu_1_Relu_output_0[] = {1, 17, 23, 512};
  Qnn_Tensor_t outputs__backbone_base_layer4_layer4_0_relu_1_Relu[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer4_layer4_0_relu_1_Relu_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer4_layer4_0_relu_1_Relu_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer4_layer4_0_relu_1_Relu", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params__backbone_base_layer4_layer4_0_relu_1_Relu, // Node Params
                         1, // Num Node Params
                         inputs__backbone_base_layer4_layer4_0_relu_1_Relu, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__backbone_base_layer4_layer4_0_relu_1_Relu, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1418(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1418[] = {3, 3, 512, 512};
  VALIDATE(model.addTensor("onnx__Conv_1418", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1418",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_onnx__Conv_1418,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1418),
                                                .dataSize=BINLEN(onnx__Conv_1418)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1419(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1419[] = {512};
  VALIDATE(model.addTensor("onnx__Conv_1419", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1419",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_onnx__Conv_1419,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1419),
                                                .dataSize=BINLEN(onnx__Conv_1419)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer4_layer4_1_conv1_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer4_layer4_1_conv1_Conv */
  uint32_t dimensions___backbone_base_layer4_layer4_1_conv1_Conv_dilation[] = {2};
  uint32_t __backbone_base_layer4_layer4_1_conv1_Conv_dilation[] = {1, 1};
  uint32_t dimensions___backbone_base_layer4_layer4_1_conv1_Conv_pad_amount[] = {2, 2};
  uint32_t __backbone_base_layer4_layer4_1_conv1_Conv_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions___backbone_base_layer4_layer4_1_conv1_Conv_stride[] = {2};
  uint32_t __backbone_base_layer4_layer4_1_conv1_Conv_stride[] = {1, 1};
  Qnn_Param_t params__backbone_base_layer4_layer4_1_conv1_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer4_layer4_1_conv1_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer4_layer4_1_conv1_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer4_layer4_1_conv1_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer4_layer4_1_conv1_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___backbone_base_layer4_layer4_1_conv1_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer4_layer4_1_conv1_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer4_layer4_1_conv1_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer4_layer4_1_conv1_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer4_layer4_1_conv1_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__backbone_base_layer4_layer4_1_conv1_Conv[] = {
    "_backbone_base_layer4_layer4_0_relu_1_Relu_output_0",
    "onnx__Conv_1418",
    "onnx__Conv_1419"
  };
  uint32_t dimensions__backbone_base_layer4_layer4_1_conv1_Conv_output_0[] = {1, 17, 23, 512};
  Qnn_Tensor_t outputs__backbone_base_layer4_layer4_1_conv1_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer4_layer4_1_conv1_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer4_layer4_1_conv1_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer4_layer4_1_conv1_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__backbone_base_layer4_layer4_1_conv1_Conv, // Node Params
                         4, // Num Node Params
                         inputs__backbone_base_layer4_layer4_1_conv1_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__backbone_base_layer4_layer4_1_conv1_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer4_layer4_1_relu_Relu(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer4_layer4_1_relu_Relu */
  Qnn_Param_t params__backbone_base_layer4_layer4_1_relu_Relu[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 4}}}}
  };
  const char*  inputs__backbone_base_layer4_layer4_1_relu_Relu[] = {
    "_backbone_base_layer4_layer4_1_conv1_Conv_output_0"
  };
  uint32_t dimensions__backbone_base_layer4_layer4_1_relu_Relu_output_0[] = {1, 17, 23, 512};
  Qnn_Tensor_t outputs__backbone_base_layer4_layer4_1_relu_Relu[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer4_layer4_1_relu_Relu_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer4_layer4_1_relu_Relu_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer4_layer4_1_relu_Relu", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params__backbone_base_layer4_layer4_1_relu_Relu, // Node Params
                         1, // Num Node Params
                         inputs__backbone_base_layer4_layer4_1_relu_Relu, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__backbone_base_layer4_layer4_1_relu_Relu, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1421(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1421[] = {3, 3, 512, 512};
  VALIDATE(model.addTensor("onnx__Conv_1421", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1421",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_onnx__Conv_1421,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1421),
                                                .dataSize=BINLEN(onnx__Conv_1421)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1422(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1422[] = {512};
  VALIDATE(model.addTensor("onnx__Conv_1422", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1422",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_onnx__Conv_1422,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1422),
                                                .dataSize=BINLEN(onnx__Conv_1422)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer4_layer4_1_conv2_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer4_layer4_1_conv2_Conv */
  uint32_t dimensions___backbone_base_layer4_layer4_1_conv2_Conv_dilation[] = {2};
  uint32_t __backbone_base_layer4_layer4_1_conv2_Conv_dilation[] = {1, 1};
  uint32_t dimensions___backbone_base_layer4_layer4_1_conv2_Conv_pad_amount[] = {2, 2};
  uint32_t __backbone_base_layer4_layer4_1_conv2_Conv_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions___backbone_base_layer4_layer4_1_conv2_Conv_stride[] = {2};
  uint32_t __backbone_base_layer4_layer4_1_conv2_Conv_stride[] = {1, 1};
  Qnn_Param_t params__backbone_base_layer4_layer4_1_conv2_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer4_layer4_1_conv2_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer4_layer4_1_conv2_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer4_layer4_1_conv2_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer4_layer4_1_conv2_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___backbone_base_layer4_layer4_1_conv2_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer4_layer4_1_conv2_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_base_layer4_layer4_1_conv2_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_base_layer4_layer4_1_conv2_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_base_layer4_layer4_1_conv2_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__backbone_base_layer4_layer4_1_conv2_Conv[] = {
    "_backbone_base_layer4_layer4_1_relu_Relu_output_0",
    "onnx__Conv_1421",
    "onnx__Conv_1422"
  };
  uint32_t dimensions__backbone_base_layer4_layer4_1_conv2_Conv_output_0[] = {1, 17, 23, 512};
  Qnn_Tensor_t outputs__backbone_base_layer4_layer4_1_conv2_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer4_layer4_1_conv2_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer4_layer4_1_conv2_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer4_layer4_1_conv2_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__backbone_base_layer4_layer4_1_conv2_Conv, // Node Params
                         4, // Num Node Params
                         inputs__backbone_base_layer4_layer4_1_conv2_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__backbone_base_layer4_layer4_1_conv2_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer4_layer4_1_Add(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer4_layer4_1_Add */
  Qnn_Param_t params__backbone_base_layer4_layer4_1_Add[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__backbone_base_layer4_layer4_1_Add[] = {
    "_backbone_base_layer4_layer4_1_conv2_Conv_output_0",
    "_backbone_base_layer4_layer4_0_relu_1_Relu_output_0"
  };
  uint32_t dimensions__backbone_base_layer4_layer4_1_Add_output_0[] = {1, 17, 23, 512};
  Qnn_Tensor_t outputs__backbone_base_layer4_layer4_1_Add[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer4_layer4_1_Add_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer4_layer4_1_Add_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer4_layer4_1_Add", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__backbone_base_layer4_layer4_1_Add, // Node Params
                         1, // Num Node Params
                         inputs__backbone_base_layer4_layer4_1_Add, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__backbone_base_layer4_layer4_1_Add, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__backbone_base_layer4_layer4_1_relu_1_Relu(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_base_layer4_layer4_1_relu_1_Relu */
  Qnn_Param_t params__backbone_base_layer4_layer4_1_relu_1_Relu[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 4}}}}
  };
  const char*  inputs__backbone_base_layer4_layer4_1_relu_1_Relu[] = {
    "_backbone_base_layer4_layer4_1_Add_output_0"
  };
  uint32_t dimensions__backbone_base_layer4_layer4_1_relu_1_Relu_output_0[] = {1, 17, 23, 512};
  Qnn_Tensor_t outputs__backbone_base_layer4_layer4_1_relu_1_Relu[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_base_layer4_layer4_1_relu_1_Relu_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_base_layer4_layer4_1_relu_1_Relu_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_base_layer4_layer4_1_relu_1_Relu", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params__backbone_base_layer4_layer4_1_relu_1_Relu, // Node Params
                         1, // Num Node Params
                         inputs__backbone_base_layer4_layer4_1_relu_1_Relu, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__backbone_base_layer4_layer4_1_relu_1_Relu, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1424(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1424[] = {1, 1, 512, 256};
  VALIDATE(model.addTensor("onnx__Conv_1424", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1424",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_onnx__Conv_1424,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1424),
                                                .dataSize=BINLEN(onnx__Conv_1424)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1425(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1425[] = {256};
  VALIDATE(model.addTensor("onnx__Conv_1425", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1425",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_onnx__Conv_1425,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1425),
                                                .dataSize=BINLEN(onnx__Conv_1425)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__backbone_fpn_inner_blocks_2_inner_blocks_2_0_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_fpn_inner_blocks_2_inner_blocks_2_0_Conv */
  uint32_t dimensions___backbone_fpn_inner_blocks_2_inner_blocks_2_0_Conv_dilation[] = {2};
  uint32_t __backbone_fpn_inner_blocks_2_inner_blocks_2_0_Conv_dilation[] = {1, 1};
  uint32_t dimensions___backbone_fpn_inner_blocks_2_inner_blocks_2_0_Conv_pad_amount[] = {2, 2};
  uint32_t __backbone_fpn_inner_blocks_2_inner_blocks_2_0_Conv_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions___backbone_fpn_inner_blocks_2_inner_blocks_2_0_Conv_stride[] = {2};
  uint32_t __backbone_fpn_inner_blocks_2_inner_blocks_2_0_Conv_stride[] = {1, 1};
  Qnn_Param_t params__backbone_fpn_inner_blocks_2_inner_blocks_2_0_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_fpn_inner_blocks_2_inner_blocks_2_0_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_fpn_inner_blocks_2_inner_blocks_2_0_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_fpn_inner_blocks_2_inner_blocks_2_0_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_fpn_inner_blocks_2_inner_blocks_2_0_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___backbone_fpn_inner_blocks_2_inner_blocks_2_0_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_fpn_inner_blocks_2_inner_blocks_2_0_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_fpn_inner_blocks_2_inner_blocks_2_0_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_fpn_inner_blocks_2_inner_blocks_2_0_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_fpn_inner_blocks_2_inner_blocks_2_0_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__backbone_fpn_inner_blocks_2_inner_blocks_2_0_Conv[] = {
    "_backbone_base_layer4_layer4_1_relu_1_Relu_output_0",
    "onnx__Conv_1424",
    "onnx__Conv_1425"
  };
  uint32_t dimensions__backbone_fpn_inner_blocks_2_inner_blocks_2_0_Conv_output_0[] = {1, 17, 23, 256};
  Qnn_Tensor_t outputs__backbone_fpn_inner_blocks_2_inner_blocks_2_0_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_fpn_inner_blocks_2_inner_blocks_2_0_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_fpn_inner_blocks_2_inner_blocks_2_0_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_fpn_inner_blocks_2_inner_blocks_2_0_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__backbone_fpn_inner_blocks_2_inner_blocks_2_0_Conv, // Node Params
                         4, // Num Node Params
                         inputs__backbone_fpn_inner_blocks_2_inner_blocks_2_0_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__backbone_fpn_inner_blocks_2_inner_blocks_2_0_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1427(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1427[] = {3, 3, 256, 256};
  VALIDATE(model.addTensor("onnx__Conv_1427", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1427",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_onnx__Conv_1427,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1427),
                                                .dataSize=BINLEN(onnx__Conv_1427)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1428(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1428[] = {256};
  VALIDATE(model.addTensor("onnx__Conv_1428", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1428",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_onnx__Conv_1428,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1428),
                                                .dataSize=BINLEN(onnx__Conv_1428)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__backbone_fpn_layer_blocks_2_layer_blocks_2_0_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_fpn_layer_blocks_2_layer_blocks_2_0_Conv */
  uint32_t dimensions___backbone_fpn_layer_blocks_2_layer_blocks_2_0_Conv_dilation[] = {2};
  uint32_t __backbone_fpn_layer_blocks_2_layer_blocks_2_0_Conv_dilation[] = {1, 1};
  uint32_t dimensions___backbone_fpn_layer_blocks_2_layer_blocks_2_0_Conv_pad_amount[] = {2, 2};
  uint32_t __backbone_fpn_layer_blocks_2_layer_blocks_2_0_Conv_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions___backbone_fpn_layer_blocks_2_layer_blocks_2_0_Conv_stride[] = {2};
  uint32_t __backbone_fpn_layer_blocks_2_layer_blocks_2_0_Conv_stride[] = {1, 1};
  Qnn_Param_t params__backbone_fpn_layer_blocks_2_layer_blocks_2_0_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_fpn_layer_blocks_2_layer_blocks_2_0_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_fpn_layer_blocks_2_layer_blocks_2_0_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_fpn_layer_blocks_2_layer_blocks_2_0_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_fpn_layer_blocks_2_layer_blocks_2_0_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___backbone_fpn_layer_blocks_2_layer_blocks_2_0_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_fpn_layer_blocks_2_layer_blocks_2_0_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_fpn_layer_blocks_2_layer_blocks_2_0_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_fpn_layer_blocks_2_layer_blocks_2_0_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_fpn_layer_blocks_2_layer_blocks_2_0_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__backbone_fpn_layer_blocks_2_layer_blocks_2_0_Conv[] = {
    "_backbone_fpn_inner_blocks_2_inner_blocks_2_0_Conv_output_0",
    "onnx__Conv_1427",
    "onnx__Conv_1428"
  };
  uint32_t dimensions__backbone_fpn_layer_blocks_2_layer_blocks_2_0_Conv_output_0[] = {1, 17, 23, 256};
  Qnn_Tensor_t outputs__backbone_fpn_layer_blocks_2_layer_blocks_2_0_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_fpn_layer_blocks_2_layer_blocks_2_0_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_fpn_layer_blocks_2_layer_blocks_2_0_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_fpn_layer_blocks_2_layer_blocks_2_0_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__backbone_fpn_layer_blocks_2_layer_blocks_2_0_Conv, // Node Params
                         4, // Num Node Params
                         inputs__backbone_fpn_layer_blocks_2_layer_blocks_2_0_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__backbone_fpn_layer_blocks_2_layer_blocks_2_0_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1430(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1430[] = {1, 1, 256, 256};
  VALIDATE(model.addTensor("onnx__Conv_1430", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1430",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_onnx__Conv_1430,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1430),
                                                .dataSize=BINLEN(onnx__Conv_1430)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1431(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1431[] = {256};
  VALIDATE(model.addTensor("onnx__Conv_1431", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1431",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_onnx__Conv_1431,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1431),
                                                .dataSize=BINLEN(onnx__Conv_1431)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__backbone_fpn_inner_blocks_1_inner_blocks_1_0_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_fpn_inner_blocks_1_inner_blocks_1_0_Conv */
  uint32_t dimensions___backbone_fpn_inner_blocks_1_inner_blocks_1_0_Conv_dilation[] = {2};
  uint32_t __backbone_fpn_inner_blocks_1_inner_blocks_1_0_Conv_dilation[] = {1, 1};
  uint32_t dimensions___backbone_fpn_inner_blocks_1_inner_blocks_1_0_Conv_pad_amount[] = {2, 2};
  uint32_t __backbone_fpn_inner_blocks_1_inner_blocks_1_0_Conv_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions___backbone_fpn_inner_blocks_1_inner_blocks_1_0_Conv_stride[] = {2};
  uint32_t __backbone_fpn_inner_blocks_1_inner_blocks_1_0_Conv_stride[] = {1, 1};
  Qnn_Param_t params__backbone_fpn_inner_blocks_1_inner_blocks_1_0_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_fpn_inner_blocks_1_inner_blocks_1_0_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_fpn_inner_blocks_1_inner_blocks_1_0_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_fpn_inner_blocks_1_inner_blocks_1_0_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_fpn_inner_blocks_1_inner_blocks_1_0_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___backbone_fpn_inner_blocks_1_inner_blocks_1_0_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_fpn_inner_blocks_1_inner_blocks_1_0_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_fpn_inner_blocks_1_inner_blocks_1_0_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_fpn_inner_blocks_1_inner_blocks_1_0_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_fpn_inner_blocks_1_inner_blocks_1_0_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__backbone_fpn_inner_blocks_1_inner_blocks_1_0_Conv[] = {
    "_backbone_base_layer3_layer3_1_relu_1_Relu_output_0",
    "onnx__Conv_1430",
    "onnx__Conv_1431"
  };
  uint32_t dimensions__backbone_fpn_inner_blocks_1_inner_blocks_1_0_Conv_output_0[] = {1, 34, 45, 256};
  Qnn_Tensor_t outputs__backbone_fpn_inner_blocks_1_inner_blocks_1_0_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_fpn_inner_blocks_1_inner_blocks_1_0_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_fpn_inner_blocks_1_inner_blocks_1_0_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_fpn_inner_blocks_1_inner_blocks_1_0_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__backbone_fpn_inner_blocks_1_inner_blocks_1_0_Conv, // Node Params
                         4, // Num Node Params
                         inputs__backbone_fpn_inner_blocks_1_inner_blocks_1_0_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__backbone_fpn_inner_blocks_1_inner_blocks_1_0_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__backbone_fpn_Resize(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_fpn_Resize */
  Qnn_Param_t params__backbone_fpn_Resize[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="align_corners",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="half_pixel_centers",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs__backbone_fpn_Resize[] = {
    "_backbone_fpn_inner_blocks_2_inner_blocks_2_0_Conv_output_0"
  };
  uint32_t dimensions__backbone_fpn_Resize_output_0[] = {1, 34, 45, 256};
  Qnn_Tensor_t outputs__backbone_fpn_Resize[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_fpn_Resize_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_fpn_Resize_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_fpn_Resize", // Node Name
                         "qti.aisw", // Package Name
                         "ResizeNearestNeighbor", // Qnn Node Type
                         params__backbone_fpn_Resize, // Node Params
                         2, // Num Node Params
                         inputs__backbone_fpn_Resize, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__backbone_fpn_Resize, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__backbone_fpn_Add(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_fpn_Add */
  Qnn_Param_t params__backbone_fpn_Add[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__backbone_fpn_Add[] = {
    "_backbone_fpn_inner_blocks_1_inner_blocks_1_0_Conv_output_0",
    "_backbone_fpn_Resize_output_0"
  };
  uint32_t dimensions__backbone_fpn_Add_output_0[] = {1, 34, 45, 256};
  Qnn_Tensor_t outputs__backbone_fpn_Add[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_fpn_Add_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_fpn_Add_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_fpn_Add", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__backbone_fpn_Add, // Node Params
                         1, // Num Node Params
                         inputs__backbone_fpn_Add, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__backbone_fpn_Add, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1433(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1433[] = {3, 3, 256, 256};
  VALIDATE(model.addTensor("onnx__Conv_1433", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1433",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_onnx__Conv_1433,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1433),
                                                .dataSize=BINLEN(onnx__Conv_1433)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1434(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1434[] = {256};
  VALIDATE(model.addTensor("onnx__Conv_1434", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1434",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_onnx__Conv_1434,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1434),
                                                .dataSize=BINLEN(onnx__Conv_1434)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__backbone_fpn_layer_blocks_1_layer_blocks_1_0_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_fpn_layer_blocks_1_layer_blocks_1_0_Conv */
  uint32_t dimensions___backbone_fpn_layer_blocks_1_layer_blocks_1_0_Conv_dilation[] = {2};
  uint32_t __backbone_fpn_layer_blocks_1_layer_blocks_1_0_Conv_dilation[] = {1, 1};
  uint32_t dimensions___backbone_fpn_layer_blocks_1_layer_blocks_1_0_Conv_pad_amount[] = {2, 2};
  uint32_t __backbone_fpn_layer_blocks_1_layer_blocks_1_0_Conv_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions___backbone_fpn_layer_blocks_1_layer_blocks_1_0_Conv_stride[] = {2};
  uint32_t __backbone_fpn_layer_blocks_1_layer_blocks_1_0_Conv_stride[] = {1, 1};
  Qnn_Param_t params__backbone_fpn_layer_blocks_1_layer_blocks_1_0_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_fpn_layer_blocks_1_layer_blocks_1_0_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_fpn_layer_blocks_1_layer_blocks_1_0_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_fpn_layer_blocks_1_layer_blocks_1_0_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_fpn_layer_blocks_1_layer_blocks_1_0_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___backbone_fpn_layer_blocks_1_layer_blocks_1_0_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_fpn_layer_blocks_1_layer_blocks_1_0_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_fpn_layer_blocks_1_layer_blocks_1_0_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_fpn_layer_blocks_1_layer_blocks_1_0_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_fpn_layer_blocks_1_layer_blocks_1_0_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__backbone_fpn_layer_blocks_1_layer_blocks_1_0_Conv[] = {
    "_backbone_fpn_Add_output_0",
    "onnx__Conv_1433",
    "onnx__Conv_1434"
  };
  uint32_t dimensions__backbone_fpn_layer_blocks_1_layer_blocks_1_0_Conv_output_0[] = {1, 34, 45, 256};
  Qnn_Tensor_t outputs__backbone_fpn_layer_blocks_1_layer_blocks_1_0_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_fpn_layer_blocks_1_layer_blocks_1_0_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_fpn_layer_blocks_1_layer_blocks_1_0_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_fpn_layer_blocks_1_layer_blocks_1_0_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__backbone_fpn_layer_blocks_1_layer_blocks_1_0_Conv, // Node Params
                         4, // Num Node Params
                         inputs__backbone_fpn_layer_blocks_1_layer_blocks_1_0_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__backbone_fpn_layer_blocks_1_layer_blocks_1_0_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1436(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1436[] = {1, 1, 128, 256};
  VALIDATE(model.addTensor("onnx__Conv_1436", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1436",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_onnx__Conv_1436,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1436),
                                                .dataSize=BINLEN(onnx__Conv_1436)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1437(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1437[] = {256};
  VALIDATE(model.addTensor("onnx__Conv_1437", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1437",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_onnx__Conv_1437,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1437),
                                                .dataSize=BINLEN(onnx__Conv_1437)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__backbone_fpn_inner_blocks_0_inner_blocks_0_0_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_fpn_inner_blocks_0_inner_blocks_0_0_Conv */
  uint32_t dimensions___backbone_fpn_inner_blocks_0_inner_blocks_0_0_Conv_dilation[] = {2};
  uint32_t __backbone_fpn_inner_blocks_0_inner_blocks_0_0_Conv_dilation[] = {1, 1};
  uint32_t dimensions___backbone_fpn_inner_blocks_0_inner_blocks_0_0_Conv_pad_amount[] = {2, 2};
  uint32_t __backbone_fpn_inner_blocks_0_inner_blocks_0_0_Conv_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions___backbone_fpn_inner_blocks_0_inner_blocks_0_0_Conv_stride[] = {2};
  uint32_t __backbone_fpn_inner_blocks_0_inner_blocks_0_0_Conv_stride[] = {1, 1};
  Qnn_Param_t params__backbone_fpn_inner_blocks_0_inner_blocks_0_0_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_fpn_inner_blocks_0_inner_blocks_0_0_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_fpn_inner_blocks_0_inner_blocks_0_0_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_fpn_inner_blocks_0_inner_blocks_0_0_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_fpn_inner_blocks_0_inner_blocks_0_0_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___backbone_fpn_inner_blocks_0_inner_blocks_0_0_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_fpn_inner_blocks_0_inner_blocks_0_0_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_fpn_inner_blocks_0_inner_blocks_0_0_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_fpn_inner_blocks_0_inner_blocks_0_0_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_fpn_inner_blocks_0_inner_blocks_0_0_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__backbone_fpn_inner_blocks_0_inner_blocks_0_0_Conv[] = {
    "_backbone_base_layer2_layer2_1_relu_1_Relu_output_0",
    "onnx__Conv_1436",
    "onnx__Conv_1437"
  };
  uint32_t dimensions__backbone_fpn_inner_blocks_0_inner_blocks_0_0_Conv_output_0[] = {1, 68, 90, 256};
  Qnn_Tensor_t outputs__backbone_fpn_inner_blocks_0_inner_blocks_0_0_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_fpn_inner_blocks_0_inner_blocks_0_0_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_fpn_inner_blocks_0_inner_blocks_0_0_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_fpn_inner_blocks_0_inner_blocks_0_0_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__backbone_fpn_inner_blocks_0_inner_blocks_0_0_Conv, // Node Params
                         4, // Num Node Params
                         inputs__backbone_fpn_inner_blocks_0_inner_blocks_0_0_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__backbone_fpn_inner_blocks_0_inner_blocks_0_0_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__backbone_fpn_Resize_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_fpn_Resize_1 */
  Qnn_Param_t params__backbone_fpn_Resize_1[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="align_corners",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="half_pixel_centers",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs__backbone_fpn_Resize_1[] = {
    "_backbone_fpn_Add_output_0"
  };
  uint32_t dimensions__backbone_fpn_Resize_1_output_0[] = {1, 68, 90, 256};
  Qnn_Tensor_t outputs__backbone_fpn_Resize_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_fpn_Resize_1_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_fpn_Resize_1_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_fpn_Resize_1", // Node Name
                         "qti.aisw", // Package Name
                         "ResizeNearestNeighbor", // Qnn Node Type
                         params__backbone_fpn_Resize_1, // Node Params
                         2, // Num Node Params
                         inputs__backbone_fpn_Resize_1, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__backbone_fpn_Resize_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__backbone_fpn_Add_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_fpn_Add_1 */
  Qnn_Param_t params__backbone_fpn_Add_1[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__backbone_fpn_Add_1[] = {
    "_backbone_fpn_inner_blocks_0_inner_blocks_0_0_Conv_output_0",
    "_backbone_fpn_Resize_1_output_0"
  };
  uint32_t dimensions__backbone_fpn_Add_1_output_0[] = {1, 68, 90, 256};
  Qnn_Tensor_t outputs__backbone_fpn_Add_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_fpn_Add_1_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_fpn_Add_1_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_fpn_Add_1", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__backbone_fpn_Add_1, // Node Params
                         1, // Num Node Params
                         inputs__backbone_fpn_Add_1, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__backbone_fpn_Add_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1439(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1439[] = {3, 3, 256, 256};
  VALIDATE(model.addTensor("onnx__Conv_1439", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1439",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_onnx__Conv_1439,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1439),
                                                .dataSize=BINLEN(onnx__Conv_1439)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1440(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1440[] = {256};
  VALIDATE(model.addTensor("onnx__Conv_1440", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1440",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_onnx__Conv_1440,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1440),
                                                .dataSize=BINLEN(onnx__Conv_1440)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__backbone_fpn_layer_blocks_0_layer_blocks_0_0_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_fpn_layer_blocks_0_layer_blocks_0_0_Conv */
  uint32_t dimensions___backbone_fpn_layer_blocks_0_layer_blocks_0_0_Conv_dilation[] = {2};
  uint32_t __backbone_fpn_layer_blocks_0_layer_blocks_0_0_Conv_dilation[] = {1, 1};
  uint32_t dimensions___backbone_fpn_layer_blocks_0_layer_blocks_0_0_Conv_pad_amount[] = {2, 2};
  uint32_t __backbone_fpn_layer_blocks_0_layer_blocks_0_0_Conv_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions___backbone_fpn_layer_blocks_0_layer_blocks_0_0_Conv_stride[] = {2};
  uint32_t __backbone_fpn_layer_blocks_0_layer_blocks_0_0_Conv_stride[] = {1, 1};
  Qnn_Param_t params__backbone_fpn_layer_blocks_0_layer_blocks_0_0_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_fpn_layer_blocks_0_layer_blocks_0_0_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_fpn_layer_blocks_0_layer_blocks_0_0_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_fpn_layer_blocks_0_layer_blocks_0_0_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_fpn_layer_blocks_0_layer_blocks_0_0_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___backbone_fpn_layer_blocks_0_layer_blocks_0_0_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_fpn_layer_blocks_0_layer_blocks_0_0_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_fpn_layer_blocks_0_layer_blocks_0_0_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_fpn_layer_blocks_0_layer_blocks_0_0_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_fpn_layer_blocks_0_layer_blocks_0_0_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__backbone_fpn_layer_blocks_0_layer_blocks_0_0_Conv[] = {
    "_backbone_fpn_Add_1_output_0",
    "onnx__Conv_1439",
    "onnx__Conv_1440"
  };
  uint32_t dimensions__backbone_fpn_layer_blocks_0_layer_blocks_0_0_Conv_output_0[] = {1, 68, 90, 256};
  Qnn_Tensor_t outputs__backbone_fpn_layer_blocks_0_layer_blocks_0_0_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_fpn_layer_blocks_0_layer_blocks_0_0_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_fpn_layer_blocks_0_layer_blocks_0_0_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_fpn_layer_blocks_0_layer_blocks_0_0_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__backbone_fpn_layer_blocks_0_layer_blocks_0_0_Conv, // Node Params
                         4, // Num Node Params
                         inputs__backbone_fpn_layer_blocks_0_layer_blocks_0_0_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__backbone_fpn_layer_blocks_0_layer_blocks_0_0_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_backbone_fpn_extra_blocks_p6_weight(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_backbone_fpn_extra_blocks_p6_weight[] = {3, 3, 256, 256};
  VALIDATE(model.addTensor("backbone_fpn_extra_blocks_p6_weight", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "backbone_fpn_extra_blocks_p6_weight",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_backbone_fpn_extra_blocks_p6_weight,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(backbone_fpn_extra_blocks_p6_weight),
                                                .dataSize=BINLEN(backbone_fpn_extra_blocks_p6_weight)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_backbone_fpn_extra_blocks_p6_bias(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_backbone_fpn_extra_blocks_p6_bias[] = {256};
  VALIDATE(model.addTensor("backbone_fpn_extra_blocks_p6_bias", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "backbone_fpn_extra_blocks_p6_bias",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_backbone_fpn_extra_blocks_p6_bias,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(backbone_fpn_extra_blocks_p6_bias),
                                                .dataSize=BINLEN(backbone_fpn_extra_blocks_p6_bias)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__backbone_fpn_extra_blocks_p6_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_fpn_extra_blocks_p6_Conv */
  uint32_t dimensions___backbone_fpn_extra_blocks_p6_Conv_dilation[] = {2};
  uint32_t __backbone_fpn_extra_blocks_p6_Conv_dilation[] = {1, 1};
  uint32_t dimensions___backbone_fpn_extra_blocks_p6_Conv_pad_amount[] = {2, 2};
  uint32_t __backbone_fpn_extra_blocks_p6_Conv_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions___backbone_fpn_extra_blocks_p6_Conv_stride[] = {2};
  uint32_t __backbone_fpn_extra_blocks_p6_Conv_stride[] = {2, 2};
  Qnn_Param_t params__backbone_fpn_extra_blocks_p6_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_fpn_extra_blocks_p6_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_fpn_extra_blocks_p6_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_fpn_extra_blocks_p6_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_fpn_extra_blocks_p6_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___backbone_fpn_extra_blocks_p6_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_fpn_extra_blocks_p6_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_fpn_extra_blocks_p6_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_fpn_extra_blocks_p6_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_fpn_extra_blocks_p6_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__backbone_fpn_extra_blocks_p6_Conv[] = {
    "_backbone_fpn_layer_blocks_2_layer_blocks_2_0_Conv_output_0",
    "backbone_fpn_extra_blocks_p6_weight",
    "backbone_fpn_extra_blocks_p6_bias"
  };
  uint32_t dimensions__backbone_fpn_extra_blocks_p6_Conv_output_0[] = {1, 9, 12, 256};
  Qnn_Tensor_t outputs__backbone_fpn_extra_blocks_p6_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_fpn_extra_blocks_p6_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_fpn_extra_blocks_p6_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_fpn_extra_blocks_p6_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__backbone_fpn_extra_blocks_p6_Conv, // Node Params
                         4, // Num Node Params
                         inputs__backbone_fpn_extra_blocks_p6_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__backbone_fpn_extra_blocks_p6_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__backbone_fpn_extra_blocks_Relu(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_fpn_extra_blocks_Relu */
  Qnn_Param_t params__backbone_fpn_extra_blocks_Relu[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 4}}}}
  };
  const char*  inputs__backbone_fpn_extra_blocks_Relu[] = {
    "_backbone_fpn_extra_blocks_p6_Conv_output_0"
  };
  uint32_t dimensions__backbone_fpn_extra_blocks_Relu_output_0[] = {1, 9, 12, 256};
  Qnn_Tensor_t outputs__backbone_fpn_extra_blocks_Relu[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_fpn_extra_blocks_Relu_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_fpn_extra_blocks_Relu_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_fpn_extra_blocks_Relu", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params__backbone_fpn_extra_blocks_Relu, // Node Params
                         1, // Num Node Params
                         inputs__backbone_fpn_extra_blocks_Relu, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__backbone_fpn_extra_blocks_Relu, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_backbone_fpn_extra_blocks_p7_weight(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_backbone_fpn_extra_blocks_p7_weight[] = {3, 3, 256, 256};
  VALIDATE(model.addTensor("backbone_fpn_extra_blocks_p7_weight", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "backbone_fpn_extra_blocks_p7_weight",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_backbone_fpn_extra_blocks_p7_weight,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(backbone_fpn_extra_blocks_p7_weight),
                                                .dataSize=BINLEN(backbone_fpn_extra_blocks_p7_weight)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_backbone_fpn_extra_blocks_p7_bias(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_backbone_fpn_extra_blocks_p7_bias[] = {256};
  VALIDATE(model.addTensor("backbone_fpn_extra_blocks_p7_bias", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "backbone_fpn_extra_blocks_p7_bias",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_backbone_fpn_extra_blocks_p7_bias,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(backbone_fpn_extra_blocks_p7_bias),
                                                .dataSize=BINLEN(backbone_fpn_extra_blocks_p7_bias)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__backbone_fpn_extra_blocks_p7_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _backbone_fpn_extra_blocks_p7_Conv */
  uint32_t dimensions___backbone_fpn_extra_blocks_p7_Conv_dilation[] = {2};
  uint32_t __backbone_fpn_extra_blocks_p7_Conv_dilation[] = {1, 1};
  uint32_t dimensions___backbone_fpn_extra_blocks_p7_Conv_pad_amount[] = {2, 2};
  uint32_t __backbone_fpn_extra_blocks_p7_Conv_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions___backbone_fpn_extra_blocks_p7_Conv_stride[] = {2};
  uint32_t __backbone_fpn_extra_blocks_p7_Conv_stride[] = {2, 2};
  Qnn_Param_t params__backbone_fpn_extra_blocks_p7_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_fpn_extra_blocks_p7_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_fpn_extra_blocks_p7_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_fpn_extra_blocks_p7_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_fpn_extra_blocks_p7_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___backbone_fpn_extra_blocks_p7_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_fpn_extra_blocks_p7_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__backbone_fpn_extra_blocks_p7_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___backbone_fpn_extra_blocks_p7_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__backbone_fpn_extra_blocks_p7_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__backbone_fpn_extra_blocks_p7_Conv[] = {
    "_backbone_fpn_extra_blocks_Relu_output_0",
    "backbone_fpn_extra_blocks_p7_weight",
    "backbone_fpn_extra_blocks_p7_bias"
  };
  uint32_t dimensions__backbone_fpn_extra_blocks_p7_Conv_output_0[] = {1, 5, 6, 256};
  Qnn_Tensor_t outputs__backbone_fpn_extra_blocks_p7_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_backbone_fpn_extra_blocks_p7_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__backbone_fpn_extra_blocks_p7_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_backbone_fpn_extra_blocks_p7_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__backbone_fpn_extra_blocks_p7_Conv, // Node Params
                         4, // Num Node Params
                         inputs__backbone_fpn_extra_blocks_p7_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__backbone_fpn_extra_blocks_p7_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1442(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1442[] = {3, 3, 1, 256};
  VALIDATE(model.addTensor("onnx__Conv_1442", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1442",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_onnx__Conv_1442,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1442),
                                                .dataSize=BINLEN(onnx__Conv_1442)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1443(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1443[] = {256};
  VALIDATE(model.addTensor("onnx__Conv_1443", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1443",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_onnx__Conv_1443,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1443),
                                                .dataSize=BINLEN(onnx__Conv_1443)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv */
  uint32_t dimensions___head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_dilation[] = {2};
  uint32_t __head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_dilation[] = {1, 1};
  uint32_t dimensions___head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_pad_amount[] = {2, 2};
  uint32_t __head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions___head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_stride[] = {2};
  uint32_t __head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_stride[] = {1, 1};
  Qnn_Param_t params__head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv[] = {
    "_backbone_fpn_layer_blocks_0_layer_blocks_0_0_Conv_output_0",
    "onnx__Conv_1442",
    "onnx__Conv_1443"
  };
  uint32_t dimensions__head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_output_0[] = {1, 68, 90, 256};
  Qnn_Tensor_t outputs__head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "DepthWiseConv2d", // Qnn Node Type
                         params__head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv, // Node Params
                         3, // Num Node Params
                         inputs__head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_regression_head_module_list_0_module_list_0_0_module_list_0_0_2_Clip(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_regression_head_module_list_0_module_list_0_0_module_list_0_0_2_Clip */
  Qnn_Param_t params__head_regression_head_module_list_0_module_list_0_0_module_list_0_0_2_Clip[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="max_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 6.000000000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="min_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 0.000000000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 5}}}}
  };
  const char*  inputs__head_regression_head_module_list_0_module_list_0_0_module_list_0_0_2_Clip[] = {
    "_head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_output_0"
  };
  uint32_t dimensions__head_regression_head_module_list_0_module_list_0_0_module_list_0_0_2_Clip_output_0[] = {1, 68, 90, 256};
  Qnn_Tensor_t outputs__head_regression_head_module_list_0_module_list_0_0_module_list_0_0_2_Clip[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_regression_head_module_list_0_module_list_0_0_module_list_0_0_2_Clip_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_regression_head_module_list_0_module_list_0_0_module_list_0_0_2_Clip_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_regression_head_module_list_0_module_list_0_0_module_list_0_0_2_Clip", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params__head_regression_head_module_list_0_module_list_0_0_module_list_0_0_2_Clip, // Node Params
                         3, // Num Node Params
                         inputs__head_regression_head_module_list_0_module_list_0_0_module_list_0_0_2_Clip, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_regression_head_module_list_0_module_list_0_0_module_list_0_0_2_Clip, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_head_regression_head_module_list_0_1_weight(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_head_regression_head_module_list_0_1_weight[] = {1, 1, 256, 24};
  VALIDATE(model.addTensor("head_regression_head_module_list_0_1_weight", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "head_regression_head_module_list_0_1_weight",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_head_regression_head_module_list_0_1_weight,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(head_regression_head_module_list_0_1_weight),
                                                .dataSize=BINLEN(head_regression_head_module_list_0_1_weight)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_head_regression_head_module_list_0_1_bias(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_head_regression_head_module_list_0_1_bias[] = {24};
  VALIDATE(model.addTensor("head_regression_head_module_list_0_1_bias", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "head_regression_head_module_list_0_1_bias",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_head_regression_head_module_list_0_1_bias,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(head_regression_head_module_list_0_1_bias),
                                                .dataSize=BINLEN(head_regression_head_module_list_0_1_bias)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__head_regression_head_module_list_0_module_list_0_1_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_regression_head_module_list_0_module_list_0_1_Conv */
  uint32_t dimensions___head_regression_head_module_list_0_module_list_0_1_Conv_dilation[] = {2};
  uint32_t __head_regression_head_module_list_0_module_list_0_1_Conv_dilation[] = {1, 1};
  uint32_t dimensions___head_regression_head_module_list_0_module_list_0_1_Conv_pad_amount[] = {2, 2};
  uint32_t __head_regression_head_module_list_0_module_list_0_1_Conv_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions___head_regression_head_module_list_0_module_list_0_1_Conv_stride[] = {2};
  uint32_t __head_regression_head_module_list_0_module_list_0_1_Conv_stride[] = {1, 1};
  Qnn_Param_t params__head_regression_head_module_list_0_module_list_0_1_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_module_list_0_module_list_0_1_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_regression_head_module_list_0_module_list_0_1_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_module_list_0_module_list_0_1_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_module_list_0_module_list_0_1_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___head_regression_head_module_list_0_module_list_0_1_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_module_list_0_module_list_0_1_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_module_list_0_module_list_0_1_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_regression_head_module_list_0_module_list_0_1_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_module_list_0_module_list_0_1_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__head_regression_head_module_list_0_module_list_0_1_Conv[] = {
    "_head_regression_head_module_list_0_module_list_0_0_module_list_0_0_2_Clip_output_0",
    "head_regression_head_module_list_0_1_weight",
    "head_regression_head_module_list_0_1_bias"
  };
  uint32_t dimensions__head_regression_head_module_list_0_module_list_0_1_Conv_output_0[] = {1, 68, 90, 24};
  Qnn_Tensor_t outputs__head_regression_head_module_list_0_module_list_0_1_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_regression_head_module_list_0_module_list_0_1_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_regression_head_module_list_0_module_list_0_1_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_regression_head_module_list_0_module_list_0_1_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__head_regression_head_module_list_0_module_list_0_1_Conv, // Node Params
                         4, // Num Node Params
                         inputs__head_regression_head_module_list_0_module_list_0_1_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__head_regression_head_module_list_0_module_list_0_1_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_regression_head_module_list_0_module_list_0_1_Conv_output_0_nchw(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_regression_head_module_list_0_module_list_0_1_Conv_output_0_nchw */
  uint32_t dimensions___head_regression_head_module_list_0_module_list_0_1_Conv_output_0_nchw_perm[] = {4};
  uint32_t __head_regression_head_module_list_0_module_list_0_1_Conv_output_0_nchw_perm[] = {0, 3, 1, 2};
  Qnn_Param_t params__head_regression_head_module_list_0_module_list_0_1_Conv_output_0_nchw[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_module_list_0_module_list_0_1_Conv_output_0_nchw_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_regression_head_module_list_0_module_list_0_1_Conv_output_0_nchw_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_module_list_0_module_list_0_1_Conv_output_0_nchw_perm,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__head_regression_head_module_list_0_module_list_0_1_Conv_output_0_nchw[] = {
    "_head_regression_head_module_list_0_module_list_0_1_Conv_output_0"
  };
  uint32_t dimensions__head_regression_head_module_list_0_module_list_0_1_Conv_output_0_nchw[] = {1, 24, 68, 90};
  Qnn_Tensor_t outputs__head_regression_head_module_list_0_module_list_0_1_Conv_output_0_nchw[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_regression_head_module_list_0_module_list_0_1_Conv_output_0_nchw",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_regression_head_module_list_0_module_list_0_1_Conv_output_0_nchw,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_regression_head_module_list_0_module_list_0_1_Conv_output_0_nchw", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__head_regression_head_module_list_0_module_list_0_1_Conv_output_0_nchw, // Node Params
                         1, // Num Node Params
                         inputs__head_regression_head_module_list_0_module_list_0_1_Conv_output_0_nchw, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_regression_head_module_list_0_module_list_0_1_Conv_output_0_nchw, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_regression_head_Reshape(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_regression_head_Reshape */
  const char*  inputs__head_regression_head_Reshape[] = {
    "_head_regression_head_module_list_0_module_list_0_1_Conv_output_0_nchw"
  };
  uint32_t dimensions__head_regression_head_Reshape_output_0[] = {1, 6, 4, 68, 90};
  Qnn_Tensor_t outputs__head_regression_head_Reshape[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_regression_head_Reshape_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 5,
            .dimensions=dimensions__head_regression_head_Reshape_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_regression_head_Reshape", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__head_regression_head_Reshape, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_regression_head_Reshape, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_regression_head_Transpose(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_regression_head_Transpose */
  uint32_t dimensions___head_regression_head_Transpose_perm[] = {5};
  uint32_t __head_regression_head_Transpose_perm[] = {0, 3, 4, 1, 2};
  Qnn_Param_t params__head_regression_head_Transpose[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_Transpose_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_regression_head_Transpose_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_Transpose_perm,
                           .dataSize=20}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__head_regression_head_Transpose[] = {
    "_head_regression_head_Reshape_output_0"
  };
  uint32_t dimensions__head_regression_head_Transpose_output_0[] = {1, 68, 90, 6, 4};
  Qnn_Tensor_t outputs__head_regression_head_Transpose[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_regression_head_Transpose_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 5,
            .dimensions=dimensions__head_regression_head_Transpose_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_regression_head_Transpose", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__head_regression_head_Transpose, // Node Params
                         1, // Num Node Params
                         inputs__head_regression_head_Transpose, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_regression_head_Transpose, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_regression_head_Reshape_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_regression_head_Reshape_1 */
  const char*  inputs__head_regression_head_Reshape_1[] = {
    "_head_regression_head_Transpose_output_0"
  };
  uint32_t dimensions__head_regression_head_Reshape_1_output_0[] = {1, 36720, 4};
  Qnn_Tensor_t outputs__head_regression_head_Reshape_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_regression_head_Reshape_1_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__head_regression_head_Reshape_1_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_regression_head_Reshape_1", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__head_regression_head_Reshape_1, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_regression_head_Reshape_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1445(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1445[] = {3, 3, 1, 256};
  VALIDATE(model.addTensor("onnx__Conv_1445", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1445",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_onnx__Conv_1445,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1445),
                                                .dataSize=BINLEN(onnx__Conv_1445)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1446(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1446[] = {256};
  VALIDATE(model.addTensor("onnx__Conv_1446", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1446",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_onnx__Conv_1446,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1446),
                                                .dataSize=BINLEN(onnx__Conv_1446)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv */
  uint32_t dimensions___head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_dilation[] = {2};
  uint32_t __head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_dilation[] = {1, 1};
  uint32_t dimensions___head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_pad_amount[] = {2, 2};
  uint32_t __head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions___head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_stride[] = {2};
  uint32_t __head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_stride[] = {1, 1};
  Qnn_Param_t params__head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv[] = {
    "_backbone_fpn_layer_blocks_1_layer_blocks_1_0_Conv_output_0",
    "onnx__Conv_1445",
    "onnx__Conv_1446"
  };
  uint32_t dimensions__head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_output_0[] = {1, 34, 45, 256};
  Qnn_Tensor_t outputs__head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "DepthWiseConv2d", // Qnn Node Type
                         params__head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv, // Node Params
                         3, // Num Node Params
                         inputs__head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_regression_head_module_list_1_module_list_1_0_module_list_1_0_2_Clip(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_regression_head_module_list_1_module_list_1_0_module_list_1_0_2_Clip */
  Qnn_Param_t params__head_regression_head_module_list_1_module_list_1_0_module_list_1_0_2_Clip[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="max_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 6.000000000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="min_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 0.000000000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 5}}}}
  };
  const char*  inputs__head_regression_head_module_list_1_module_list_1_0_module_list_1_0_2_Clip[] = {
    "_head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_output_0"
  };
  uint32_t dimensions__head_regression_head_module_list_1_module_list_1_0_module_list_1_0_2_Clip_output_0[] = {1, 34, 45, 256};
  Qnn_Tensor_t outputs__head_regression_head_module_list_1_module_list_1_0_module_list_1_0_2_Clip[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_regression_head_module_list_1_module_list_1_0_module_list_1_0_2_Clip_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_regression_head_module_list_1_module_list_1_0_module_list_1_0_2_Clip_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_regression_head_module_list_1_module_list_1_0_module_list_1_0_2_Clip", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params__head_regression_head_module_list_1_module_list_1_0_module_list_1_0_2_Clip, // Node Params
                         3, // Num Node Params
                         inputs__head_regression_head_module_list_1_module_list_1_0_module_list_1_0_2_Clip, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_regression_head_module_list_1_module_list_1_0_module_list_1_0_2_Clip, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_head_regression_head_module_list_1_1_weight(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_head_regression_head_module_list_1_1_weight[] = {1, 1, 256, 24};
  VALIDATE(model.addTensor("head_regression_head_module_list_1_1_weight", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "head_regression_head_module_list_1_1_weight",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_head_regression_head_module_list_1_1_weight,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(head_regression_head_module_list_1_1_weight),
                                                .dataSize=BINLEN(head_regression_head_module_list_1_1_weight)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_head_regression_head_module_list_1_1_bias(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_head_regression_head_module_list_1_1_bias[] = {24};
  VALIDATE(model.addTensor("head_regression_head_module_list_1_1_bias", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "head_regression_head_module_list_1_1_bias",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_head_regression_head_module_list_1_1_bias,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(head_regression_head_module_list_1_1_bias),
                                                .dataSize=BINLEN(head_regression_head_module_list_1_1_bias)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__head_regression_head_module_list_1_module_list_1_1_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_regression_head_module_list_1_module_list_1_1_Conv */
  uint32_t dimensions___head_regression_head_module_list_1_module_list_1_1_Conv_dilation[] = {2};
  uint32_t __head_regression_head_module_list_1_module_list_1_1_Conv_dilation[] = {1, 1};
  uint32_t dimensions___head_regression_head_module_list_1_module_list_1_1_Conv_pad_amount[] = {2, 2};
  uint32_t __head_regression_head_module_list_1_module_list_1_1_Conv_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions___head_regression_head_module_list_1_module_list_1_1_Conv_stride[] = {2};
  uint32_t __head_regression_head_module_list_1_module_list_1_1_Conv_stride[] = {1, 1};
  Qnn_Param_t params__head_regression_head_module_list_1_module_list_1_1_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_module_list_1_module_list_1_1_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_regression_head_module_list_1_module_list_1_1_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_module_list_1_module_list_1_1_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_module_list_1_module_list_1_1_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___head_regression_head_module_list_1_module_list_1_1_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_module_list_1_module_list_1_1_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_module_list_1_module_list_1_1_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_regression_head_module_list_1_module_list_1_1_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_module_list_1_module_list_1_1_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__head_regression_head_module_list_1_module_list_1_1_Conv[] = {
    "_head_regression_head_module_list_1_module_list_1_0_module_list_1_0_2_Clip_output_0",
    "head_regression_head_module_list_1_1_weight",
    "head_regression_head_module_list_1_1_bias"
  };
  uint32_t dimensions__head_regression_head_module_list_1_module_list_1_1_Conv_output_0[] = {1, 34, 45, 24};
  Qnn_Tensor_t outputs__head_regression_head_module_list_1_module_list_1_1_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_regression_head_module_list_1_module_list_1_1_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_regression_head_module_list_1_module_list_1_1_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_regression_head_module_list_1_module_list_1_1_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__head_regression_head_module_list_1_module_list_1_1_Conv, // Node Params
                         4, // Num Node Params
                         inputs__head_regression_head_module_list_1_module_list_1_1_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__head_regression_head_module_list_1_module_list_1_1_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_regression_head_module_list_1_module_list_1_1_Conv_output_0_nchw(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_regression_head_module_list_1_module_list_1_1_Conv_output_0_nchw */
  uint32_t dimensions___head_regression_head_module_list_1_module_list_1_1_Conv_output_0_nchw_perm[] = {4};
  uint32_t __head_regression_head_module_list_1_module_list_1_1_Conv_output_0_nchw_perm[] = {0, 3, 1, 2};
  Qnn_Param_t params__head_regression_head_module_list_1_module_list_1_1_Conv_output_0_nchw[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_module_list_1_module_list_1_1_Conv_output_0_nchw_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_regression_head_module_list_1_module_list_1_1_Conv_output_0_nchw_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_module_list_1_module_list_1_1_Conv_output_0_nchw_perm,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__head_regression_head_module_list_1_module_list_1_1_Conv_output_0_nchw[] = {
    "_head_regression_head_module_list_1_module_list_1_1_Conv_output_0"
  };
  uint32_t dimensions__head_regression_head_module_list_1_module_list_1_1_Conv_output_0_nchw[] = {1, 24, 34, 45};
  Qnn_Tensor_t outputs__head_regression_head_module_list_1_module_list_1_1_Conv_output_0_nchw[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_regression_head_module_list_1_module_list_1_1_Conv_output_0_nchw",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_regression_head_module_list_1_module_list_1_1_Conv_output_0_nchw,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_regression_head_module_list_1_module_list_1_1_Conv_output_0_nchw", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__head_regression_head_module_list_1_module_list_1_1_Conv_output_0_nchw, // Node Params
                         1, // Num Node Params
                         inputs__head_regression_head_module_list_1_module_list_1_1_Conv_output_0_nchw, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_regression_head_module_list_1_module_list_1_1_Conv_output_0_nchw, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_regression_head_Reshape_2(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_regression_head_Reshape_2 */
  const char*  inputs__head_regression_head_Reshape_2[] = {
    "_head_regression_head_module_list_1_module_list_1_1_Conv_output_0_nchw"
  };
  uint32_t dimensions__head_regression_head_Reshape_2_output_0[] = {1, 6, 4, 34, 45};
  Qnn_Tensor_t outputs__head_regression_head_Reshape_2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_regression_head_Reshape_2_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 5,
            .dimensions=dimensions__head_regression_head_Reshape_2_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_regression_head_Reshape_2", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__head_regression_head_Reshape_2, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_regression_head_Reshape_2, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_regression_head_Transpose_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_regression_head_Transpose_1 */
  uint32_t dimensions___head_regression_head_Transpose_1_perm[] = {5};
  uint32_t __head_regression_head_Transpose_1_perm[] = {0, 3, 4, 1, 2};
  Qnn_Param_t params__head_regression_head_Transpose_1[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_Transpose_1_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_regression_head_Transpose_1_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_Transpose_1_perm,
                           .dataSize=20}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__head_regression_head_Transpose_1[] = {
    "_head_regression_head_Reshape_2_output_0"
  };
  uint32_t dimensions__head_regression_head_Transpose_1_output_0[] = {1, 34, 45, 6, 4};
  Qnn_Tensor_t outputs__head_regression_head_Transpose_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_regression_head_Transpose_1_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 5,
            .dimensions=dimensions__head_regression_head_Transpose_1_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_regression_head_Transpose_1", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__head_regression_head_Transpose_1, // Node Params
                         1, // Num Node Params
                         inputs__head_regression_head_Transpose_1, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_regression_head_Transpose_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_regression_head_Reshape_3(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_regression_head_Reshape_3 */
  const char*  inputs__head_regression_head_Reshape_3[] = {
    "_head_regression_head_Transpose_1_output_0"
  };
  uint32_t dimensions__head_regression_head_Reshape_3_output_0[] = {1, 9180, 4};
  Qnn_Tensor_t outputs__head_regression_head_Reshape_3[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_regression_head_Reshape_3_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__head_regression_head_Reshape_3_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_regression_head_Reshape_3", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__head_regression_head_Reshape_3, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_regression_head_Reshape_3, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1448(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1448[] = {3, 3, 1, 256};
  VALIDATE(model.addTensor("onnx__Conv_1448", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1448",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_onnx__Conv_1448,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1448),
                                                .dataSize=BINLEN(onnx__Conv_1448)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1449(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1449[] = {256};
  VALIDATE(model.addTensor("onnx__Conv_1449", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1449",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_onnx__Conv_1449,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1449),
                                                .dataSize=BINLEN(onnx__Conv_1449)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv */
  uint32_t dimensions___head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_dilation[] = {2};
  uint32_t __head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_dilation[] = {1, 1};
  uint32_t dimensions___head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_pad_amount[] = {2, 2};
  uint32_t __head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions___head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_stride[] = {2};
  uint32_t __head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_stride[] = {1, 1};
  Qnn_Param_t params__head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv[] = {
    "_backbone_fpn_layer_blocks_2_layer_blocks_2_0_Conv_output_0",
    "onnx__Conv_1448",
    "onnx__Conv_1449"
  };
  uint32_t dimensions__head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_output_0[] = {1, 17, 23, 256};
  Qnn_Tensor_t outputs__head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "DepthWiseConv2d", // Qnn Node Type
                         params__head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv, // Node Params
                         3, // Num Node Params
                         inputs__head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_regression_head_module_list_2_module_list_2_0_module_list_2_0_2_Clip(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_regression_head_module_list_2_module_list_2_0_module_list_2_0_2_Clip */
  Qnn_Param_t params__head_regression_head_module_list_2_module_list_2_0_module_list_2_0_2_Clip[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="max_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 6.000000000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="min_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 0.000000000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 5}}}}
  };
  const char*  inputs__head_regression_head_module_list_2_module_list_2_0_module_list_2_0_2_Clip[] = {
    "_head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_output_0"
  };
  uint32_t dimensions__head_regression_head_module_list_2_module_list_2_0_module_list_2_0_2_Clip_output_0[] = {1, 17, 23, 256};
  Qnn_Tensor_t outputs__head_regression_head_module_list_2_module_list_2_0_module_list_2_0_2_Clip[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_regression_head_module_list_2_module_list_2_0_module_list_2_0_2_Clip_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_regression_head_module_list_2_module_list_2_0_module_list_2_0_2_Clip_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_regression_head_module_list_2_module_list_2_0_module_list_2_0_2_Clip", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params__head_regression_head_module_list_2_module_list_2_0_module_list_2_0_2_Clip, // Node Params
                         3, // Num Node Params
                         inputs__head_regression_head_module_list_2_module_list_2_0_module_list_2_0_2_Clip, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_regression_head_module_list_2_module_list_2_0_module_list_2_0_2_Clip, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_head_regression_head_module_list_2_1_weight(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_head_regression_head_module_list_2_1_weight[] = {1, 1, 256, 24};
  VALIDATE(model.addTensor("head_regression_head_module_list_2_1_weight", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "head_regression_head_module_list_2_1_weight",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_head_regression_head_module_list_2_1_weight,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(head_regression_head_module_list_2_1_weight),
                                                .dataSize=BINLEN(head_regression_head_module_list_2_1_weight)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_head_regression_head_module_list_2_1_bias(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_head_regression_head_module_list_2_1_bias[] = {24};
  VALIDATE(model.addTensor("head_regression_head_module_list_2_1_bias", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "head_regression_head_module_list_2_1_bias",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_head_regression_head_module_list_2_1_bias,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(head_regression_head_module_list_2_1_bias),
                                                .dataSize=BINLEN(head_regression_head_module_list_2_1_bias)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__head_regression_head_module_list_2_module_list_2_1_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_regression_head_module_list_2_module_list_2_1_Conv */
  uint32_t dimensions___head_regression_head_module_list_2_module_list_2_1_Conv_dilation[] = {2};
  uint32_t __head_regression_head_module_list_2_module_list_2_1_Conv_dilation[] = {1, 1};
  uint32_t dimensions___head_regression_head_module_list_2_module_list_2_1_Conv_pad_amount[] = {2, 2};
  uint32_t __head_regression_head_module_list_2_module_list_2_1_Conv_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions___head_regression_head_module_list_2_module_list_2_1_Conv_stride[] = {2};
  uint32_t __head_regression_head_module_list_2_module_list_2_1_Conv_stride[] = {1, 1};
  Qnn_Param_t params__head_regression_head_module_list_2_module_list_2_1_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_module_list_2_module_list_2_1_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_regression_head_module_list_2_module_list_2_1_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_module_list_2_module_list_2_1_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_module_list_2_module_list_2_1_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___head_regression_head_module_list_2_module_list_2_1_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_module_list_2_module_list_2_1_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_module_list_2_module_list_2_1_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_regression_head_module_list_2_module_list_2_1_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_module_list_2_module_list_2_1_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__head_regression_head_module_list_2_module_list_2_1_Conv[] = {
    "_head_regression_head_module_list_2_module_list_2_0_module_list_2_0_2_Clip_output_0",
    "head_regression_head_module_list_2_1_weight",
    "head_regression_head_module_list_2_1_bias"
  };
  uint32_t dimensions__head_regression_head_module_list_2_module_list_2_1_Conv_output_0[] = {1, 17, 23, 24};
  Qnn_Tensor_t outputs__head_regression_head_module_list_2_module_list_2_1_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_regression_head_module_list_2_module_list_2_1_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_regression_head_module_list_2_module_list_2_1_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_regression_head_module_list_2_module_list_2_1_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__head_regression_head_module_list_2_module_list_2_1_Conv, // Node Params
                         4, // Num Node Params
                         inputs__head_regression_head_module_list_2_module_list_2_1_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__head_regression_head_module_list_2_module_list_2_1_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_regression_head_module_list_2_module_list_2_1_Conv_output_0_nchw(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_regression_head_module_list_2_module_list_2_1_Conv_output_0_nchw */
  uint32_t dimensions___head_regression_head_module_list_2_module_list_2_1_Conv_output_0_nchw_perm[] = {4};
  uint32_t __head_regression_head_module_list_2_module_list_2_1_Conv_output_0_nchw_perm[] = {0, 3, 1, 2};
  Qnn_Param_t params__head_regression_head_module_list_2_module_list_2_1_Conv_output_0_nchw[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_module_list_2_module_list_2_1_Conv_output_0_nchw_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_regression_head_module_list_2_module_list_2_1_Conv_output_0_nchw_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_module_list_2_module_list_2_1_Conv_output_0_nchw_perm,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__head_regression_head_module_list_2_module_list_2_1_Conv_output_0_nchw[] = {
    "_head_regression_head_module_list_2_module_list_2_1_Conv_output_0"
  };
  uint32_t dimensions__head_regression_head_module_list_2_module_list_2_1_Conv_output_0_nchw[] = {1, 24, 17, 23};
  Qnn_Tensor_t outputs__head_regression_head_module_list_2_module_list_2_1_Conv_output_0_nchw[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_regression_head_module_list_2_module_list_2_1_Conv_output_0_nchw",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_regression_head_module_list_2_module_list_2_1_Conv_output_0_nchw,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_regression_head_module_list_2_module_list_2_1_Conv_output_0_nchw", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__head_regression_head_module_list_2_module_list_2_1_Conv_output_0_nchw, // Node Params
                         1, // Num Node Params
                         inputs__head_regression_head_module_list_2_module_list_2_1_Conv_output_0_nchw, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_regression_head_module_list_2_module_list_2_1_Conv_output_0_nchw, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_regression_head_Reshape_4(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_regression_head_Reshape_4 */
  const char*  inputs__head_regression_head_Reshape_4[] = {
    "_head_regression_head_module_list_2_module_list_2_1_Conv_output_0_nchw"
  };
  uint32_t dimensions__head_regression_head_Reshape_4_output_0[] = {1, 6, 4, 17, 23};
  Qnn_Tensor_t outputs__head_regression_head_Reshape_4[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_regression_head_Reshape_4_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 5,
            .dimensions=dimensions__head_regression_head_Reshape_4_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_regression_head_Reshape_4", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__head_regression_head_Reshape_4, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_regression_head_Reshape_4, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_regression_head_Transpose_2(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_regression_head_Transpose_2 */
  uint32_t dimensions___head_regression_head_Transpose_2_perm[] = {5};
  uint32_t __head_regression_head_Transpose_2_perm[] = {0, 3, 4, 1, 2};
  Qnn_Param_t params__head_regression_head_Transpose_2[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_Transpose_2_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_regression_head_Transpose_2_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_Transpose_2_perm,
                           .dataSize=20}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__head_regression_head_Transpose_2[] = {
    "_head_regression_head_Reshape_4_output_0"
  };
  uint32_t dimensions__head_regression_head_Transpose_2_output_0[] = {1, 17, 23, 6, 4};
  Qnn_Tensor_t outputs__head_regression_head_Transpose_2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_regression_head_Transpose_2_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 5,
            .dimensions=dimensions__head_regression_head_Transpose_2_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_regression_head_Transpose_2", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__head_regression_head_Transpose_2, // Node Params
                         1, // Num Node Params
                         inputs__head_regression_head_Transpose_2, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_regression_head_Transpose_2, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_regression_head_Reshape_5(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_regression_head_Reshape_5 */
  const char*  inputs__head_regression_head_Reshape_5[] = {
    "_head_regression_head_Transpose_2_output_0"
  };
  uint32_t dimensions__head_regression_head_Reshape_5_output_0[] = {1, 2346, 4};
  Qnn_Tensor_t outputs__head_regression_head_Reshape_5[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_regression_head_Reshape_5_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__head_regression_head_Reshape_5_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_regression_head_Reshape_5", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__head_regression_head_Reshape_5, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_regression_head_Reshape_5, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1452(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1452[] = {256};
  VALIDATE(model.addTensor("onnx__Conv_1452", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1452",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_onnx__Conv_1452,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1452),
                                                .dataSize=BINLEN(onnx__Conv_1452)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv */
  uint32_t dimensions___head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_dilation[] = {2};
  uint32_t __head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_dilation[] = {1, 1};
  uint32_t dimensions___head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_pad_amount[] = {2, 2};
  uint32_t __head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions___head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_stride[] = {2};
  uint32_t __head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_stride[] = {1, 1};
  Qnn_Param_t params__head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv[] = {
    "_backbone_fpn_extra_blocks_p6_Conv_output_0",
    "onnx__Conv_1448",
    "onnx__Conv_1452"
  };
  uint32_t dimensions__head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_output_0[] = {1, 9, 12, 256};
  Qnn_Tensor_t outputs__head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "DepthWiseConv2d", // Qnn Node Type
                         params__head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv, // Node Params
                         3, // Num Node Params
                         inputs__head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_regression_head_module_list_3_module_list_3_0_module_list_3_0_2_Clip(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_regression_head_module_list_3_module_list_3_0_module_list_3_0_2_Clip */
  Qnn_Param_t params__head_regression_head_module_list_3_module_list_3_0_module_list_3_0_2_Clip[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="max_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 6.000000000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="min_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 0.000000000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 5}}}}
  };
  const char*  inputs__head_regression_head_module_list_3_module_list_3_0_module_list_3_0_2_Clip[] = {
    "_head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_output_0"
  };
  uint32_t dimensions__head_regression_head_module_list_3_module_list_3_0_module_list_3_0_2_Clip_output_0[] = {1, 9, 12, 256};
  Qnn_Tensor_t outputs__head_regression_head_module_list_3_module_list_3_0_module_list_3_0_2_Clip[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_regression_head_module_list_3_module_list_3_0_module_list_3_0_2_Clip_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_regression_head_module_list_3_module_list_3_0_module_list_3_0_2_Clip_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_regression_head_module_list_3_module_list_3_0_module_list_3_0_2_Clip", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params__head_regression_head_module_list_3_module_list_3_0_module_list_3_0_2_Clip, // Node Params
                         3, // Num Node Params
                         inputs__head_regression_head_module_list_3_module_list_3_0_module_list_3_0_2_Clip, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_regression_head_module_list_3_module_list_3_0_module_list_3_0_2_Clip, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_head_regression_head_module_list_3_1_weight(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_head_regression_head_module_list_3_1_weight[] = {1, 1, 256, 24};
  VALIDATE(model.addTensor("head_regression_head_module_list_3_1_weight", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "head_regression_head_module_list_3_1_weight",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_head_regression_head_module_list_3_1_weight,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(head_regression_head_module_list_3_1_weight),
                                                .dataSize=BINLEN(head_regression_head_module_list_3_1_weight)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_head_regression_head_module_list_3_1_bias(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_head_regression_head_module_list_3_1_bias[] = {24};
  VALIDATE(model.addTensor("head_regression_head_module_list_3_1_bias", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "head_regression_head_module_list_3_1_bias",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_head_regression_head_module_list_3_1_bias,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(head_regression_head_module_list_3_1_bias),
                                                .dataSize=BINLEN(head_regression_head_module_list_3_1_bias)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__head_regression_head_module_list_3_module_list_3_1_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_regression_head_module_list_3_module_list_3_1_Conv */
  uint32_t dimensions___head_regression_head_module_list_3_module_list_3_1_Conv_dilation[] = {2};
  uint32_t __head_regression_head_module_list_3_module_list_3_1_Conv_dilation[] = {1, 1};
  uint32_t dimensions___head_regression_head_module_list_3_module_list_3_1_Conv_pad_amount[] = {2, 2};
  uint32_t __head_regression_head_module_list_3_module_list_3_1_Conv_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions___head_regression_head_module_list_3_module_list_3_1_Conv_stride[] = {2};
  uint32_t __head_regression_head_module_list_3_module_list_3_1_Conv_stride[] = {1, 1};
  Qnn_Param_t params__head_regression_head_module_list_3_module_list_3_1_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_module_list_3_module_list_3_1_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_regression_head_module_list_3_module_list_3_1_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_module_list_3_module_list_3_1_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_module_list_3_module_list_3_1_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___head_regression_head_module_list_3_module_list_3_1_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_module_list_3_module_list_3_1_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_module_list_3_module_list_3_1_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_regression_head_module_list_3_module_list_3_1_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_module_list_3_module_list_3_1_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__head_regression_head_module_list_3_module_list_3_1_Conv[] = {
    "_head_regression_head_module_list_3_module_list_3_0_module_list_3_0_2_Clip_output_0",
    "head_regression_head_module_list_3_1_weight",
    "head_regression_head_module_list_3_1_bias"
  };
  uint32_t dimensions__head_regression_head_module_list_3_module_list_3_1_Conv_output_0[] = {1, 9, 12, 24};
  Qnn_Tensor_t outputs__head_regression_head_module_list_3_module_list_3_1_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_regression_head_module_list_3_module_list_3_1_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_regression_head_module_list_3_module_list_3_1_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_regression_head_module_list_3_module_list_3_1_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__head_regression_head_module_list_3_module_list_3_1_Conv, // Node Params
                         4, // Num Node Params
                         inputs__head_regression_head_module_list_3_module_list_3_1_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__head_regression_head_module_list_3_module_list_3_1_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_regression_head_module_list_3_module_list_3_1_Conv_output_0_nchw(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_regression_head_module_list_3_module_list_3_1_Conv_output_0_nchw */
  uint32_t dimensions___head_regression_head_module_list_3_module_list_3_1_Conv_output_0_nchw_perm[] = {4};
  uint32_t __head_regression_head_module_list_3_module_list_3_1_Conv_output_0_nchw_perm[] = {0, 3, 1, 2};
  Qnn_Param_t params__head_regression_head_module_list_3_module_list_3_1_Conv_output_0_nchw[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_module_list_3_module_list_3_1_Conv_output_0_nchw_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_regression_head_module_list_3_module_list_3_1_Conv_output_0_nchw_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_module_list_3_module_list_3_1_Conv_output_0_nchw_perm,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__head_regression_head_module_list_3_module_list_3_1_Conv_output_0_nchw[] = {
    "_head_regression_head_module_list_3_module_list_3_1_Conv_output_0"
  };
  uint32_t dimensions__head_regression_head_module_list_3_module_list_3_1_Conv_output_0_nchw[] = {1, 24, 9, 12};
  Qnn_Tensor_t outputs__head_regression_head_module_list_3_module_list_3_1_Conv_output_0_nchw[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_regression_head_module_list_3_module_list_3_1_Conv_output_0_nchw",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_regression_head_module_list_3_module_list_3_1_Conv_output_0_nchw,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_regression_head_module_list_3_module_list_3_1_Conv_output_0_nchw", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__head_regression_head_module_list_3_module_list_3_1_Conv_output_0_nchw, // Node Params
                         1, // Num Node Params
                         inputs__head_regression_head_module_list_3_module_list_3_1_Conv_output_0_nchw, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_regression_head_module_list_3_module_list_3_1_Conv_output_0_nchw, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_regression_head_Reshape_6(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_regression_head_Reshape_6 */
  const char*  inputs__head_regression_head_Reshape_6[] = {
    "_head_regression_head_module_list_3_module_list_3_1_Conv_output_0_nchw"
  };
  uint32_t dimensions__head_regression_head_Reshape_6_output_0[] = {1, 6, 4, 9, 12};
  Qnn_Tensor_t outputs__head_regression_head_Reshape_6[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_regression_head_Reshape_6_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 5,
            .dimensions=dimensions__head_regression_head_Reshape_6_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_regression_head_Reshape_6", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__head_regression_head_Reshape_6, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_regression_head_Reshape_6, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_regression_head_Transpose_3(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_regression_head_Transpose_3 */
  uint32_t dimensions___head_regression_head_Transpose_3_perm[] = {5};
  uint32_t __head_regression_head_Transpose_3_perm[] = {0, 3, 4, 1, 2};
  Qnn_Param_t params__head_regression_head_Transpose_3[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_Transpose_3_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_regression_head_Transpose_3_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_Transpose_3_perm,
                           .dataSize=20}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__head_regression_head_Transpose_3[] = {
    "_head_regression_head_Reshape_6_output_0"
  };
  uint32_t dimensions__head_regression_head_Transpose_3_output_0[] = {1, 9, 12, 6, 4};
  Qnn_Tensor_t outputs__head_regression_head_Transpose_3[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_regression_head_Transpose_3_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 5,
            .dimensions=dimensions__head_regression_head_Transpose_3_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_regression_head_Transpose_3", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__head_regression_head_Transpose_3, // Node Params
                         1, // Num Node Params
                         inputs__head_regression_head_Transpose_3, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_regression_head_Transpose_3, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_regression_head_Reshape_7(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_regression_head_Reshape_7 */
  const char*  inputs__head_regression_head_Reshape_7[] = {
    "_head_regression_head_Transpose_3_output_0"
  };
  uint32_t dimensions__head_regression_head_Reshape_7_output_0[] = {1, 648, 4};
  Qnn_Tensor_t outputs__head_regression_head_Reshape_7[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_regression_head_Reshape_7_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__head_regression_head_Reshape_7_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_regression_head_Reshape_7", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__head_regression_head_Reshape_7, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_regression_head_Reshape_7, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1455(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1455[] = {256};
  VALIDATE(model.addTensor("onnx__Conv_1455", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1455",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_onnx__Conv_1455,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1455),
                                                .dataSize=BINLEN(onnx__Conv_1455)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv */
  uint32_t dimensions___head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_dilation[] = {2};
  uint32_t __head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_dilation[] = {1, 1};
  uint32_t dimensions___head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_pad_amount[] = {2, 2};
  uint32_t __head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions___head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_stride[] = {2};
  uint32_t __head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_stride[] = {1, 1};
  Qnn_Param_t params__head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv[] = {
    "_backbone_fpn_extra_blocks_p7_Conv_output_0",
    "onnx__Conv_1448",
    "onnx__Conv_1455"
  };
  uint32_t dimensions__head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_output_0[] = {1, 5, 6, 256};
  Qnn_Tensor_t outputs__head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "DepthWiseConv2d", // Qnn Node Type
                         params__head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv, // Node Params
                         3, // Num Node Params
                         inputs__head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_regression_head_module_list_4_module_list_4_0_module_list_4_0_2_Clip(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_regression_head_module_list_4_module_list_4_0_module_list_4_0_2_Clip */
  Qnn_Param_t params__head_regression_head_module_list_4_module_list_4_0_module_list_4_0_2_Clip[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="max_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 6.000000000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="min_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 0.000000000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 5}}}}
  };
  const char*  inputs__head_regression_head_module_list_4_module_list_4_0_module_list_4_0_2_Clip[] = {
    "_head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_output_0"
  };
  uint32_t dimensions__head_regression_head_module_list_4_module_list_4_0_module_list_4_0_2_Clip_output_0[] = {1, 5, 6, 256};
  Qnn_Tensor_t outputs__head_regression_head_module_list_4_module_list_4_0_module_list_4_0_2_Clip[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_regression_head_module_list_4_module_list_4_0_module_list_4_0_2_Clip_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_regression_head_module_list_4_module_list_4_0_module_list_4_0_2_Clip_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_regression_head_module_list_4_module_list_4_0_module_list_4_0_2_Clip", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params__head_regression_head_module_list_4_module_list_4_0_module_list_4_0_2_Clip, // Node Params
                         3, // Num Node Params
                         inputs__head_regression_head_module_list_4_module_list_4_0_module_list_4_0_2_Clip, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_regression_head_module_list_4_module_list_4_0_module_list_4_0_2_Clip, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_head_regression_head_module_list_4_1_weight(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_head_regression_head_module_list_4_1_weight[] = {1, 1, 256, 24};
  VALIDATE(model.addTensor("head_regression_head_module_list_4_1_weight", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "head_regression_head_module_list_4_1_weight",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_head_regression_head_module_list_4_1_weight,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(head_regression_head_module_list_4_1_weight),
                                                .dataSize=BINLEN(head_regression_head_module_list_4_1_weight)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_head_regression_head_module_list_4_1_bias(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_head_regression_head_module_list_4_1_bias[] = {24};
  VALIDATE(model.addTensor("head_regression_head_module_list_4_1_bias", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "head_regression_head_module_list_4_1_bias",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_head_regression_head_module_list_4_1_bias,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(head_regression_head_module_list_4_1_bias),
                                                .dataSize=BINLEN(head_regression_head_module_list_4_1_bias)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__head_regression_head_module_list_4_module_list_4_1_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_regression_head_module_list_4_module_list_4_1_Conv */
  uint32_t dimensions___head_regression_head_module_list_4_module_list_4_1_Conv_dilation[] = {2};
  uint32_t __head_regression_head_module_list_4_module_list_4_1_Conv_dilation[] = {1, 1};
  uint32_t dimensions___head_regression_head_module_list_4_module_list_4_1_Conv_pad_amount[] = {2, 2};
  uint32_t __head_regression_head_module_list_4_module_list_4_1_Conv_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions___head_regression_head_module_list_4_module_list_4_1_Conv_stride[] = {2};
  uint32_t __head_regression_head_module_list_4_module_list_4_1_Conv_stride[] = {1, 1};
  Qnn_Param_t params__head_regression_head_module_list_4_module_list_4_1_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_module_list_4_module_list_4_1_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_regression_head_module_list_4_module_list_4_1_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_module_list_4_module_list_4_1_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_module_list_4_module_list_4_1_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___head_regression_head_module_list_4_module_list_4_1_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_module_list_4_module_list_4_1_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_module_list_4_module_list_4_1_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_regression_head_module_list_4_module_list_4_1_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_module_list_4_module_list_4_1_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__head_regression_head_module_list_4_module_list_4_1_Conv[] = {
    "_head_regression_head_module_list_4_module_list_4_0_module_list_4_0_2_Clip_output_0",
    "head_regression_head_module_list_4_1_weight",
    "head_regression_head_module_list_4_1_bias"
  };
  uint32_t dimensions__head_regression_head_module_list_4_module_list_4_1_Conv_output_0[] = {1, 5, 6, 24};
  Qnn_Tensor_t outputs__head_regression_head_module_list_4_module_list_4_1_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_regression_head_module_list_4_module_list_4_1_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_regression_head_module_list_4_module_list_4_1_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_regression_head_module_list_4_module_list_4_1_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__head_regression_head_module_list_4_module_list_4_1_Conv, // Node Params
                         4, // Num Node Params
                         inputs__head_regression_head_module_list_4_module_list_4_1_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__head_regression_head_module_list_4_module_list_4_1_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_regression_head_module_list_4_module_list_4_1_Conv_output_0_nchw(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_regression_head_module_list_4_module_list_4_1_Conv_output_0_nchw */
  uint32_t dimensions___head_regression_head_module_list_4_module_list_4_1_Conv_output_0_nchw_perm[] = {4};
  uint32_t __head_regression_head_module_list_4_module_list_4_1_Conv_output_0_nchw_perm[] = {0, 3, 1, 2};
  Qnn_Param_t params__head_regression_head_module_list_4_module_list_4_1_Conv_output_0_nchw[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_module_list_4_module_list_4_1_Conv_output_0_nchw_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_regression_head_module_list_4_module_list_4_1_Conv_output_0_nchw_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_module_list_4_module_list_4_1_Conv_output_0_nchw_perm,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__head_regression_head_module_list_4_module_list_4_1_Conv_output_0_nchw[] = {
    "_head_regression_head_module_list_4_module_list_4_1_Conv_output_0"
  };
  uint32_t dimensions__head_regression_head_module_list_4_module_list_4_1_Conv_output_0_nchw[] = {1, 24, 5, 6};
  Qnn_Tensor_t outputs__head_regression_head_module_list_4_module_list_4_1_Conv_output_0_nchw[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_regression_head_module_list_4_module_list_4_1_Conv_output_0_nchw",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_regression_head_module_list_4_module_list_4_1_Conv_output_0_nchw,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_regression_head_module_list_4_module_list_4_1_Conv_output_0_nchw", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__head_regression_head_module_list_4_module_list_4_1_Conv_output_0_nchw, // Node Params
                         1, // Num Node Params
                         inputs__head_regression_head_module_list_4_module_list_4_1_Conv_output_0_nchw, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_regression_head_module_list_4_module_list_4_1_Conv_output_0_nchw, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_regression_head_Reshape_8(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_regression_head_Reshape_8 */
  const char*  inputs__head_regression_head_Reshape_8[] = {
    "_head_regression_head_module_list_4_module_list_4_1_Conv_output_0_nchw"
  };
  uint32_t dimensions__head_regression_head_Reshape_8_output_0[] = {1, 6, 4, 5, 6};
  Qnn_Tensor_t outputs__head_regression_head_Reshape_8[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_regression_head_Reshape_8_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 5,
            .dimensions=dimensions__head_regression_head_Reshape_8_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_regression_head_Reshape_8", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__head_regression_head_Reshape_8, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_regression_head_Reshape_8, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_regression_head_Transpose_4(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_regression_head_Transpose_4 */
  uint32_t dimensions___head_regression_head_Transpose_4_perm[] = {5};
  uint32_t __head_regression_head_Transpose_4_perm[] = {0, 3, 4, 1, 2};
  Qnn_Param_t params__head_regression_head_Transpose_4[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_regression_head_Transpose_4_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_regression_head_Transpose_4_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_regression_head_Transpose_4_perm,
                           .dataSize=20}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__head_regression_head_Transpose_4[] = {
    "_head_regression_head_Reshape_8_output_0"
  };
  uint32_t dimensions__head_regression_head_Transpose_4_output_0[] = {1, 5, 6, 6, 4};
  Qnn_Tensor_t outputs__head_regression_head_Transpose_4[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_regression_head_Transpose_4_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 5,
            .dimensions=dimensions__head_regression_head_Transpose_4_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_regression_head_Transpose_4", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__head_regression_head_Transpose_4, // Node Params
                         1, // Num Node Params
                         inputs__head_regression_head_Transpose_4, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_regression_head_Transpose_4, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_regression_head_Reshape_9(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_regression_head_Reshape_9 */
  const char*  inputs__head_regression_head_Reshape_9[] = {
    "_head_regression_head_Transpose_4_output_0"
  };
  uint32_t dimensions__head_regression_head_Reshape_9_output_0[] = {1, 180, 4};
  Qnn_Tensor_t outputs__head_regression_head_Reshape_9[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_regression_head_Reshape_9_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__head_regression_head_Reshape_9_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_regression_head_Reshape_9", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__head_regression_head_Reshape_9, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_regression_head_Reshape_9, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_regression_head_Concat_10(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_regression_head_Concat_10 */
  Qnn_Param_t params__head_regression_head_Concat_10[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="axis",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__head_regression_head_Concat_10[] = {
    "_head_regression_head_Reshape_1_output_0",
    "_head_regression_head_Reshape_3_output_0",
    "_head_regression_head_Reshape_5_output_0",
    "_head_regression_head_Reshape_7_output_0",
    "_head_regression_head_Reshape_9_output_0"
  };
  uint32_t dimensions__head_regression_head_Concat_10_output_0[] = {1, 49074, 4};
  Qnn_Tensor_t outputs__head_regression_head_Concat_10[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_regression_head_Concat_10_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__head_regression_head_Concat_10_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_regression_head_Concat_10", // Node Name
                         "qti.aisw", // Package Name
                         "Concat", // Qnn Node Type
                         params__head_regression_head_Concat_10, // Node Params
                         1, // Num Node Params
                         inputs__head_regression_head_Concat_10, // Input Tensor Names
                         5, // Num Input Tensor Names
                         outputs__head_regression_head_Concat_10, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1457(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1457[] = {3, 3, 1, 256};
  VALIDATE(model.addTensor("onnx__Conv_1457", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1457",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_onnx__Conv_1457,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1457),
                                                .dataSize=BINLEN(onnx__Conv_1457)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1458(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1458[] = {256};
  VALIDATE(model.addTensor("onnx__Conv_1458", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1458",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_onnx__Conv_1458,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1458),
                                                .dataSize=BINLEN(onnx__Conv_1458)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv */
  uint32_t dimensions___head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_dilation[] = {2};
  uint32_t __head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_dilation[] = {1, 1};
  uint32_t dimensions___head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_pad_amount[] = {2, 2};
  uint32_t __head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions___head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_stride[] = {2};
  uint32_t __head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_stride[] = {1, 1};
  Qnn_Param_t params__head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv[] = {
    "_backbone_fpn_layer_blocks_0_layer_blocks_0_0_Conv_output_0",
    "onnx__Conv_1457",
    "onnx__Conv_1458"
  };
  uint32_t dimensions__head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_output_0[] = {1, 68, 90, 256};
  Qnn_Tensor_t outputs__head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "DepthWiseConv2d", // Qnn Node Type
                         params__head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv, // Node Params
                         3, // Num Node Params
                         inputs__head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_classification_head_module_list_0_module_list_0_0_module_list_0_0_2_Clip(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_classification_head_module_list_0_module_list_0_0_module_list_0_0_2_Clip */
  Qnn_Param_t params__head_classification_head_module_list_0_module_list_0_0_module_list_0_0_2_Clip[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="max_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 6.000000000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="min_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 0.000000000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 5}}}}
  };
  const char*  inputs__head_classification_head_module_list_0_module_list_0_0_module_list_0_0_2_Clip[] = {
    "_head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv_output_0"
  };
  uint32_t dimensions__head_classification_head_module_list_0_module_list_0_0_module_list_0_0_2_Clip_output_0[] = {1, 68, 90, 256};
  Qnn_Tensor_t outputs__head_classification_head_module_list_0_module_list_0_0_module_list_0_0_2_Clip[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_module_list_0_module_list_0_0_module_list_0_0_2_Clip_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_classification_head_module_list_0_module_list_0_0_module_list_0_0_2_Clip_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_classification_head_module_list_0_module_list_0_0_module_list_0_0_2_Clip", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params__head_classification_head_module_list_0_module_list_0_0_module_list_0_0_2_Clip, // Node Params
                         3, // Num Node Params
                         inputs__head_classification_head_module_list_0_module_list_0_0_module_list_0_0_2_Clip, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_classification_head_module_list_0_module_list_0_0_module_list_0_0_2_Clip, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_head_classification_head_module_list_0_1_weight(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_head_classification_head_module_list_0_1_weight[] = {1, 1, 256, 24};
  VALIDATE(model.addTensor("head_classification_head_module_list_0_1_weight", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "head_classification_head_module_list_0_1_weight",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_head_classification_head_module_list_0_1_weight,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(head_classification_head_module_list_0_1_weight),
                                                .dataSize=BINLEN(head_classification_head_module_list_0_1_weight)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_head_classification_head_module_list_0_1_bias(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_head_classification_head_module_list_0_1_bias[] = {24};
  VALIDATE(model.addTensor("head_classification_head_module_list_0_1_bias", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "head_classification_head_module_list_0_1_bias",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_head_classification_head_module_list_0_1_bias,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(head_classification_head_module_list_0_1_bias),
                                                .dataSize=BINLEN(head_classification_head_module_list_0_1_bias)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__head_classification_head_module_list_0_module_list_0_1_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_classification_head_module_list_0_module_list_0_1_Conv */
  uint32_t dimensions___head_classification_head_module_list_0_module_list_0_1_Conv_dilation[] = {2};
  uint32_t __head_classification_head_module_list_0_module_list_0_1_Conv_dilation[] = {1, 1};
  uint32_t dimensions___head_classification_head_module_list_0_module_list_0_1_Conv_pad_amount[] = {2, 2};
  uint32_t __head_classification_head_module_list_0_module_list_0_1_Conv_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions___head_classification_head_module_list_0_module_list_0_1_Conv_stride[] = {2};
  uint32_t __head_classification_head_module_list_0_module_list_0_1_Conv_stride[] = {1, 1};
  Qnn_Param_t params__head_classification_head_module_list_0_module_list_0_1_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_module_list_0_module_list_0_1_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_classification_head_module_list_0_module_list_0_1_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_module_list_0_module_list_0_1_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_module_list_0_module_list_0_1_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___head_classification_head_module_list_0_module_list_0_1_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_module_list_0_module_list_0_1_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_module_list_0_module_list_0_1_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_classification_head_module_list_0_module_list_0_1_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_module_list_0_module_list_0_1_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__head_classification_head_module_list_0_module_list_0_1_Conv[] = {
    "_head_classification_head_module_list_0_module_list_0_0_module_list_0_0_2_Clip_output_0",
    "head_classification_head_module_list_0_1_weight",
    "head_classification_head_module_list_0_1_bias"
  };
  uint32_t dimensions__head_classification_head_module_list_0_module_list_0_1_Conv_output_0[] = {1, 68, 90, 24};
  Qnn_Tensor_t outputs__head_classification_head_module_list_0_module_list_0_1_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_module_list_0_module_list_0_1_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_classification_head_module_list_0_module_list_0_1_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_classification_head_module_list_0_module_list_0_1_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__head_classification_head_module_list_0_module_list_0_1_Conv, // Node Params
                         4, // Num Node Params
                         inputs__head_classification_head_module_list_0_module_list_0_1_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__head_classification_head_module_list_0_module_list_0_1_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_classification_head_module_list_0_module_list_0_1_Conv_output_0_nchw(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_classification_head_module_list_0_module_list_0_1_Conv_output_0_nchw */
  uint32_t dimensions___head_classification_head_module_list_0_module_list_0_1_Conv_output_0_nchw_perm[] = {4};
  uint32_t __head_classification_head_module_list_0_module_list_0_1_Conv_output_0_nchw_perm[] = {0, 3, 1, 2};
  Qnn_Param_t params__head_classification_head_module_list_0_module_list_0_1_Conv_output_0_nchw[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_module_list_0_module_list_0_1_Conv_output_0_nchw_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_classification_head_module_list_0_module_list_0_1_Conv_output_0_nchw_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_module_list_0_module_list_0_1_Conv_output_0_nchw_perm,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__head_classification_head_module_list_0_module_list_0_1_Conv_output_0_nchw[] = {
    "_head_classification_head_module_list_0_module_list_0_1_Conv_output_0"
  };
  uint32_t dimensions__head_classification_head_module_list_0_module_list_0_1_Conv_output_0_nchw[] = {1, 24, 68, 90};
  Qnn_Tensor_t outputs__head_classification_head_module_list_0_module_list_0_1_Conv_output_0_nchw[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_module_list_0_module_list_0_1_Conv_output_0_nchw",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_classification_head_module_list_0_module_list_0_1_Conv_output_0_nchw,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_classification_head_module_list_0_module_list_0_1_Conv_output_0_nchw", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__head_classification_head_module_list_0_module_list_0_1_Conv_output_0_nchw, // Node Params
                         1, // Num Node Params
                         inputs__head_classification_head_module_list_0_module_list_0_1_Conv_output_0_nchw, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_classification_head_module_list_0_module_list_0_1_Conv_output_0_nchw, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_classification_head_Reshape(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_classification_head_Reshape */
  const char*  inputs__head_classification_head_Reshape[] = {
    "_head_classification_head_module_list_0_module_list_0_1_Conv_output_0_nchw"
  };
  uint32_t dimensions__head_classification_head_Reshape_output_0[] = {1, 6, 4, 68, 90};
  Qnn_Tensor_t outputs__head_classification_head_Reshape[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_Reshape_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 5,
            .dimensions=dimensions__head_classification_head_Reshape_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_classification_head_Reshape", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__head_classification_head_Reshape, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_classification_head_Reshape, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_classification_head_Transpose(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_classification_head_Transpose */
  uint32_t dimensions___head_classification_head_Transpose_perm[] = {5};
  uint32_t __head_classification_head_Transpose_perm[] = {0, 3, 4, 1, 2};
  Qnn_Param_t params__head_classification_head_Transpose[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_Transpose_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_classification_head_Transpose_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_Transpose_perm,
                           .dataSize=20}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__head_classification_head_Transpose[] = {
    "_head_classification_head_Reshape_output_0"
  };
  uint32_t dimensions__head_classification_head_Transpose_output_0[] = {1, 68, 90, 6, 4};
  Qnn_Tensor_t outputs__head_classification_head_Transpose[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_Transpose_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 5,
            .dimensions=dimensions__head_classification_head_Transpose_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_classification_head_Transpose", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__head_classification_head_Transpose, // Node Params
                         1, // Num Node Params
                         inputs__head_classification_head_Transpose, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_classification_head_Transpose, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_classification_head_Reshape_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_classification_head_Reshape_1 */
  const char*  inputs__head_classification_head_Reshape_1[] = {
    "_head_classification_head_Transpose_output_0"
  };
  uint32_t dimensions__head_classification_head_Reshape_1_output_0[] = {1, 36720, 4};
  Qnn_Tensor_t outputs__head_classification_head_Reshape_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_Reshape_1_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__head_classification_head_Reshape_1_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_classification_head_Reshape_1", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__head_classification_head_Reshape_1, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_classification_head_Reshape_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1460(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1460[] = {3, 3, 1, 256};
  VALIDATE(model.addTensor("onnx__Conv_1460", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1460",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_onnx__Conv_1460,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1460),
                                                .dataSize=BINLEN(onnx__Conv_1460)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1461(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1461[] = {256};
  VALIDATE(model.addTensor("onnx__Conv_1461", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1461",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_onnx__Conv_1461,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1461),
                                                .dataSize=BINLEN(onnx__Conv_1461)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv */
  uint32_t dimensions___head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_dilation[] = {2};
  uint32_t __head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_dilation[] = {1, 1};
  uint32_t dimensions___head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_pad_amount[] = {2, 2};
  uint32_t __head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions___head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_stride[] = {2};
  uint32_t __head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_stride[] = {1, 1};
  Qnn_Param_t params__head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv[] = {
    "_backbone_fpn_layer_blocks_1_layer_blocks_1_0_Conv_output_0",
    "onnx__Conv_1460",
    "onnx__Conv_1461"
  };
  uint32_t dimensions__head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_output_0[] = {1, 34, 45, 256};
  Qnn_Tensor_t outputs__head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "DepthWiseConv2d", // Qnn Node Type
                         params__head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv, // Node Params
                         3, // Num Node Params
                         inputs__head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_classification_head_module_list_1_module_list_1_0_module_list_1_0_2_Clip(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_classification_head_module_list_1_module_list_1_0_module_list_1_0_2_Clip */
  Qnn_Param_t params__head_classification_head_module_list_1_module_list_1_0_module_list_1_0_2_Clip[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="max_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 6.000000000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="min_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 0.000000000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 5}}}}
  };
  const char*  inputs__head_classification_head_module_list_1_module_list_1_0_module_list_1_0_2_Clip[] = {
    "_head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv_output_0"
  };
  uint32_t dimensions__head_classification_head_module_list_1_module_list_1_0_module_list_1_0_2_Clip_output_0[] = {1, 34, 45, 256};
  Qnn_Tensor_t outputs__head_classification_head_module_list_1_module_list_1_0_module_list_1_0_2_Clip[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_module_list_1_module_list_1_0_module_list_1_0_2_Clip_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_classification_head_module_list_1_module_list_1_0_module_list_1_0_2_Clip_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_classification_head_module_list_1_module_list_1_0_module_list_1_0_2_Clip", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params__head_classification_head_module_list_1_module_list_1_0_module_list_1_0_2_Clip, // Node Params
                         3, // Num Node Params
                         inputs__head_classification_head_module_list_1_module_list_1_0_module_list_1_0_2_Clip, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_classification_head_module_list_1_module_list_1_0_module_list_1_0_2_Clip, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_head_classification_head_module_list_1_1_weight(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_head_classification_head_module_list_1_1_weight[] = {1, 1, 256, 24};
  VALIDATE(model.addTensor("head_classification_head_module_list_1_1_weight", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "head_classification_head_module_list_1_1_weight",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_head_classification_head_module_list_1_1_weight,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(head_classification_head_module_list_1_1_weight),
                                                .dataSize=BINLEN(head_classification_head_module_list_1_1_weight)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_head_classification_head_module_list_1_1_bias(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_head_classification_head_module_list_1_1_bias[] = {24};
  VALIDATE(model.addTensor("head_classification_head_module_list_1_1_bias", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "head_classification_head_module_list_1_1_bias",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_head_classification_head_module_list_1_1_bias,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(head_classification_head_module_list_1_1_bias),
                                                .dataSize=BINLEN(head_classification_head_module_list_1_1_bias)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__head_classification_head_module_list_1_module_list_1_1_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_classification_head_module_list_1_module_list_1_1_Conv */
  uint32_t dimensions___head_classification_head_module_list_1_module_list_1_1_Conv_dilation[] = {2};
  uint32_t __head_classification_head_module_list_1_module_list_1_1_Conv_dilation[] = {1, 1};
  uint32_t dimensions___head_classification_head_module_list_1_module_list_1_1_Conv_pad_amount[] = {2, 2};
  uint32_t __head_classification_head_module_list_1_module_list_1_1_Conv_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions___head_classification_head_module_list_1_module_list_1_1_Conv_stride[] = {2};
  uint32_t __head_classification_head_module_list_1_module_list_1_1_Conv_stride[] = {1, 1};
  Qnn_Param_t params__head_classification_head_module_list_1_module_list_1_1_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_module_list_1_module_list_1_1_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_classification_head_module_list_1_module_list_1_1_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_module_list_1_module_list_1_1_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_module_list_1_module_list_1_1_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___head_classification_head_module_list_1_module_list_1_1_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_module_list_1_module_list_1_1_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_module_list_1_module_list_1_1_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_classification_head_module_list_1_module_list_1_1_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_module_list_1_module_list_1_1_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__head_classification_head_module_list_1_module_list_1_1_Conv[] = {
    "_head_classification_head_module_list_1_module_list_1_0_module_list_1_0_2_Clip_output_0",
    "head_classification_head_module_list_1_1_weight",
    "head_classification_head_module_list_1_1_bias"
  };
  uint32_t dimensions__head_classification_head_module_list_1_module_list_1_1_Conv_output_0[] = {1, 34, 45, 24};
  Qnn_Tensor_t outputs__head_classification_head_module_list_1_module_list_1_1_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_module_list_1_module_list_1_1_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_classification_head_module_list_1_module_list_1_1_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_classification_head_module_list_1_module_list_1_1_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__head_classification_head_module_list_1_module_list_1_1_Conv, // Node Params
                         4, // Num Node Params
                         inputs__head_classification_head_module_list_1_module_list_1_1_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__head_classification_head_module_list_1_module_list_1_1_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_classification_head_module_list_1_module_list_1_1_Conv_output_0_nchw(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_classification_head_module_list_1_module_list_1_1_Conv_output_0_nchw */
  uint32_t dimensions___head_classification_head_module_list_1_module_list_1_1_Conv_output_0_nchw_perm[] = {4};
  uint32_t __head_classification_head_module_list_1_module_list_1_1_Conv_output_0_nchw_perm[] = {0, 3, 1, 2};
  Qnn_Param_t params__head_classification_head_module_list_1_module_list_1_1_Conv_output_0_nchw[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_module_list_1_module_list_1_1_Conv_output_0_nchw_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_classification_head_module_list_1_module_list_1_1_Conv_output_0_nchw_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_module_list_1_module_list_1_1_Conv_output_0_nchw_perm,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__head_classification_head_module_list_1_module_list_1_1_Conv_output_0_nchw[] = {
    "_head_classification_head_module_list_1_module_list_1_1_Conv_output_0"
  };
  uint32_t dimensions__head_classification_head_module_list_1_module_list_1_1_Conv_output_0_nchw[] = {1, 24, 34, 45};
  Qnn_Tensor_t outputs__head_classification_head_module_list_1_module_list_1_1_Conv_output_0_nchw[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_module_list_1_module_list_1_1_Conv_output_0_nchw",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_classification_head_module_list_1_module_list_1_1_Conv_output_0_nchw,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_classification_head_module_list_1_module_list_1_1_Conv_output_0_nchw", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__head_classification_head_module_list_1_module_list_1_1_Conv_output_0_nchw, // Node Params
                         1, // Num Node Params
                         inputs__head_classification_head_module_list_1_module_list_1_1_Conv_output_0_nchw, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_classification_head_module_list_1_module_list_1_1_Conv_output_0_nchw, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_classification_head_Reshape_2(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_classification_head_Reshape_2 */
  const char*  inputs__head_classification_head_Reshape_2[] = {
    "_head_classification_head_module_list_1_module_list_1_1_Conv_output_0_nchw"
  };
  uint32_t dimensions__head_classification_head_Reshape_2_output_0[] = {1, 6, 4, 34, 45};
  Qnn_Tensor_t outputs__head_classification_head_Reshape_2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_Reshape_2_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 5,
            .dimensions=dimensions__head_classification_head_Reshape_2_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_classification_head_Reshape_2", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__head_classification_head_Reshape_2, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_classification_head_Reshape_2, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_classification_head_Transpose_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_classification_head_Transpose_1 */
  uint32_t dimensions___head_classification_head_Transpose_1_perm[] = {5};
  uint32_t __head_classification_head_Transpose_1_perm[] = {0, 3, 4, 1, 2};
  Qnn_Param_t params__head_classification_head_Transpose_1[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_Transpose_1_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_classification_head_Transpose_1_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_Transpose_1_perm,
                           .dataSize=20}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__head_classification_head_Transpose_1[] = {
    "_head_classification_head_Reshape_2_output_0"
  };
  uint32_t dimensions__head_classification_head_Transpose_1_output_0[] = {1, 34, 45, 6, 4};
  Qnn_Tensor_t outputs__head_classification_head_Transpose_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_Transpose_1_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 5,
            .dimensions=dimensions__head_classification_head_Transpose_1_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_classification_head_Transpose_1", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__head_classification_head_Transpose_1, // Node Params
                         1, // Num Node Params
                         inputs__head_classification_head_Transpose_1, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_classification_head_Transpose_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_classification_head_Reshape_3(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_classification_head_Reshape_3 */
  const char*  inputs__head_classification_head_Reshape_3[] = {
    "_head_classification_head_Transpose_1_output_0"
  };
  uint32_t dimensions__head_classification_head_Reshape_3_output_0[] = {1, 9180, 4};
  Qnn_Tensor_t outputs__head_classification_head_Reshape_3[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_Reshape_3_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__head_classification_head_Reshape_3_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_classification_head_Reshape_3", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__head_classification_head_Reshape_3, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_classification_head_Reshape_3, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1463(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1463[] = {3, 3, 1, 256};
  VALIDATE(model.addTensor("onnx__Conv_1463", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1463",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_onnx__Conv_1463,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1463),
                                                .dataSize=BINLEN(onnx__Conv_1463)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1464(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1464[] = {256};
  VALIDATE(model.addTensor("onnx__Conv_1464", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1464",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_onnx__Conv_1464,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1464),
                                                .dataSize=BINLEN(onnx__Conv_1464)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv */
  uint32_t dimensions___head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_dilation[] = {2};
  uint32_t __head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_dilation[] = {1, 1};
  uint32_t dimensions___head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_pad_amount[] = {2, 2};
  uint32_t __head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions___head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_stride[] = {2};
  uint32_t __head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_stride[] = {1, 1};
  Qnn_Param_t params__head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv[] = {
    "_backbone_fpn_layer_blocks_2_layer_blocks_2_0_Conv_output_0",
    "onnx__Conv_1463",
    "onnx__Conv_1464"
  };
  uint32_t dimensions__head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_output_0[] = {1, 17, 23, 256};
  Qnn_Tensor_t outputs__head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "DepthWiseConv2d", // Qnn Node Type
                         params__head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv, // Node Params
                         3, // Num Node Params
                         inputs__head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_classification_head_module_list_2_module_list_2_0_module_list_2_0_2_Clip(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_classification_head_module_list_2_module_list_2_0_module_list_2_0_2_Clip */
  Qnn_Param_t params__head_classification_head_module_list_2_module_list_2_0_module_list_2_0_2_Clip[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="max_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 6.000000000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="min_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 0.000000000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 5}}}}
  };
  const char*  inputs__head_classification_head_module_list_2_module_list_2_0_module_list_2_0_2_Clip[] = {
    "_head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv_output_0"
  };
  uint32_t dimensions__head_classification_head_module_list_2_module_list_2_0_module_list_2_0_2_Clip_output_0[] = {1, 17, 23, 256};
  Qnn_Tensor_t outputs__head_classification_head_module_list_2_module_list_2_0_module_list_2_0_2_Clip[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_module_list_2_module_list_2_0_module_list_2_0_2_Clip_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_classification_head_module_list_2_module_list_2_0_module_list_2_0_2_Clip_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_classification_head_module_list_2_module_list_2_0_module_list_2_0_2_Clip", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params__head_classification_head_module_list_2_module_list_2_0_module_list_2_0_2_Clip, // Node Params
                         3, // Num Node Params
                         inputs__head_classification_head_module_list_2_module_list_2_0_module_list_2_0_2_Clip, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_classification_head_module_list_2_module_list_2_0_module_list_2_0_2_Clip, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_head_classification_head_module_list_2_1_weight(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_head_classification_head_module_list_2_1_weight[] = {1, 1, 256, 24};
  VALIDATE(model.addTensor("head_classification_head_module_list_2_1_weight", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "head_classification_head_module_list_2_1_weight",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_head_classification_head_module_list_2_1_weight,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(head_classification_head_module_list_2_1_weight),
                                                .dataSize=BINLEN(head_classification_head_module_list_2_1_weight)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_head_classification_head_module_list_2_1_bias(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_head_classification_head_module_list_2_1_bias[] = {24};
  VALIDATE(model.addTensor("head_classification_head_module_list_2_1_bias", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "head_classification_head_module_list_2_1_bias",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_head_classification_head_module_list_2_1_bias,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(head_classification_head_module_list_2_1_bias),
                                                .dataSize=BINLEN(head_classification_head_module_list_2_1_bias)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__head_classification_head_module_list_2_module_list_2_1_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_classification_head_module_list_2_module_list_2_1_Conv */
  uint32_t dimensions___head_classification_head_module_list_2_module_list_2_1_Conv_dilation[] = {2};
  uint32_t __head_classification_head_module_list_2_module_list_2_1_Conv_dilation[] = {1, 1};
  uint32_t dimensions___head_classification_head_module_list_2_module_list_2_1_Conv_pad_amount[] = {2, 2};
  uint32_t __head_classification_head_module_list_2_module_list_2_1_Conv_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions___head_classification_head_module_list_2_module_list_2_1_Conv_stride[] = {2};
  uint32_t __head_classification_head_module_list_2_module_list_2_1_Conv_stride[] = {1, 1};
  Qnn_Param_t params__head_classification_head_module_list_2_module_list_2_1_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_module_list_2_module_list_2_1_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_classification_head_module_list_2_module_list_2_1_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_module_list_2_module_list_2_1_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_module_list_2_module_list_2_1_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___head_classification_head_module_list_2_module_list_2_1_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_module_list_2_module_list_2_1_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_module_list_2_module_list_2_1_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_classification_head_module_list_2_module_list_2_1_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_module_list_2_module_list_2_1_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__head_classification_head_module_list_2_module_list_2_1_Conv[] = {
    "_head_classification_head_module_list_2_module_list_2_0_module_list_2_0_2_Clip_output_0",
    "head_classification_head_module_list_2_1_weight",
    "head_classification_head_module_list_2_1_bias"
  };
  uint32_t dimensions__head_classification_head_module_list_2_module_list_2_1_Conv_output_0[] = {1, 17, 23, 24};
  Qnn_Tensor_t outputs__head_classification_head_module_list_2_module_list_2_1_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_module_list_2_module_list_2_1_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_classification_head_module_list_2_module_list_2_1_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_classification_head_module_list_2_module_list_2_1_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__head_classification_head_module_list_2_module_list_2_1_Conv, // Node Params
                         4, // Num Node Params
                         inputs__head_classification_head_module_list_2_module_list_2_1_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__head_classification_head_module_list_2_module_list_2_1_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_classification_head_module_list_2_module_list_2_1_Conv_output_0_nchw(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_classification_head_module_list_2_module_list_2_1_Conv_output_0_nchw */
  uint32_t dimensions___head_classification_head_module_list_2_module_list_2_1_Conv_output_0_nchw_perm[] = {4};
  uint32_t __head_classification_head_module_list_2_module_list_2_1_Conv_output_0_nchw_perm[] = {0, 3, 1, 2};
  Qnn_Param_t params__head_classification_head_module_list_2_module_list_2_1_Conv_output_0_nchw[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_module_list_2_module_list_2_1_Conv_output_0_nchw_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_classification_head_module_list_2_module_list_2_1_Conv_output_0_nchw_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_module_list_2_module_list_2_1_Conv_output_0_nchw_perm,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__head_classification_head_module_list_2_module_list_2_1_Conv_output_0_nchw[] = {
    "_head_classification_head_module_list_2_module_list_2_1_Conv_output_0"
  };
  uint32_t dimensions__head_classification_head_module_list_2_module_list_2_1_Conv_output_0_nchw[] = {1, 24, 17, 23};
  Qnn_Tensor_t outputs__head_classification_head_module_list_2_module_list_2_1_Conv_output_0_nchw[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_module_list_2_module_list_2_1_Conv_output_0_nchw",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_classification_head_module_list_2_module_list_2_1_Conv_output_0_nchw,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_classification_head_module_list_2_module_list_2_1_Conv_output_0_nchw", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__head_classification_head_module_list_2_module_list_2_1_Conv_output_0_nchw, // Node Params
                         1, // Num Node Params
                         inputs__head_classification_head_module_list_2_module_list_2_1_Conv_output_0_nchw, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_classification_head_module_list_2_module_list_2_1_Conv_output_0_nchw, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_classification_head_Reshape_4(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_classification_head_Reshape_4 */
  const char*  inputs__head_classification_head_Reshape_4[] = {
    "_head_classification_head_module_list_2_module_list_2_1_Conv_output_0_nchw"
  };
  uint32_t dimensions__head_classification_head_Reshape_4_output_0[] = {1, 6, 4, 17, 23};
  Qnn_Tensor_t outputs__head_classification_head_Reshape_4[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_Reshape_4_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 5,
            .dimensions=dimensions__head_classification_head_Reshape_4_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_classification_head_Reshape_4", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__head_classification_head_Reshape_4, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_classification_head_Reshape_4, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_classification_head_Transpose_2(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_classification_head_Transpose_2 */
  uint32_t dimensions___head_classification_head_Transpose_2_perm[] = {5};
  uint32_t __head_classification_head_Transpose_2_perm[] = {0, 3, 4, 1, 2};
  Qnn_Param_t params__head_classification_head_Transpose_2[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_Transpose_2_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_classification_head_Transpose_2_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_Transpose_2_perm,
                           .dataSize=20}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__head_classification_head_Transpose_2[] = {
    "_head_classification_head_Reshape_4_output_0"
  };
  uint32_t dimensions__head_classification_head_Transpose_2_output_0[] = {1, 17, 23, 6, 4};
  Qnn_Tensor_t outputs__head_classification_head_Transpose_2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_Transpose_2_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 5,
            .dimensions=dimensions__head_classification_head_Transpose_2_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_classification_head_Transpose_2", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__head_classification_head_Transpose_2, // Node Params
                         1, // Num Node Params
                         inputs__head_classification_head_Transpose_2, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_classification_head_Transpose_2, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_classification_head_Reshape_5(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_classification_head_Reshape_5 */
  const char*  inputs__head_classification_head_Reshape_5[] = {
    "_head_classification_head_Transpose_2_output_0"
  };
  uint32_t dimensions__head_classification_head_Reshape_5_output_0[] = {1, 2346, 4};
  Qnn_Tensor_t outputs__head_classification_head_Reshape_5[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_Reshape_5_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__head_classification_head_Reshape_5_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_classification_head_Reshape_5", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__head_classification_head_Reshape_5, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_classification_head_Reshape_5, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1467(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1467[] = {256};
  VALIDATE(model.addTensor("onnx__Conv_1467", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1467",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_onnx__Conv_1467,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1467),
                                                .dataSize=BINLEN(onnx__Conv_1467)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv */
  uint32_t dimensions___head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_dilation[] = {2};
  uint32_t __head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_dilation[] = {1, 1};
  uint32_t dimensions___head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_pad_amount[] = {2, 2};
  uint32_t __head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions___head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_stride[] = {2};
  uint32_t __head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_stride[] = {1, 1};
  Qnn_Param_t params__head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv[] = {
    "_backbone_fpn_extra_blocks_p6_Conv_output_0",
    "onnx__Conv_1448",
    "onnx__Conv_1467"
  };
  uint32_t dimensions__head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_output_0[] = {1, 9, 12, 256};
  Qnn_Tensor_t outputs__head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "DepthWiseConv2d", // Qnn Node Type
                         params__head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv, // Node Params
                         3, // Num Node Params
                         inputs__head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_classification_head_module_list_3_module_list_3_0_module_list_3_0_2_Clip(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_classification_head_module_list_3_module_list_3_0_module_list_3_0_2_Clip */
  Qnn_Param_t params__head_classification_head_module_list_3_module_list_3_0_module_list_3_0_2_Clip[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="max_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 6.000000000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="min_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 0.000000000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 5}}}}
  };
  const char*  inputs__head_classification_head_module_list_3_module_list_3_0_module_list_3_0_2_Clip[] = {
    "_head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv_output_0"
  };
  uint32_t dimensions__head_classification_head_module_list_3_module_list_3_0_module_list_3_0_2_Clip_output_0[] = {1, 9, 12, 256};
  Qnn_Tensor_t outputs__head_classification_head_module_list_3_module_list_3_0_module_list_3_0_2_Clip[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_module_list_3_module_list_3_0_module_list_3_0_2_Clip_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_classification_head_module_list_3_module_list_3_0_module_list_3_0_2_Clip_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_classification_head_module_list_3_module_list_3_0_module_list_3_0_2_Clip", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params__head_classification_head_module_list_3_module_list_3_0_module_list_3_0_2_Clip, // Node Params
                         3, // Num Node Params
                         inputs__head_classification_head_module_list_3_module_list_3_0_module_list_3_0_2_Clip, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_classification_head_module_list_3_module_list_3_0_module_list_3_0_2_Clip, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_head_classification_head_module_list_3_1_weight(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_head_classification_head_module_list_3_1_weight[] = {1, 1, 256, 24};
  VALIDATE(model.addTensor("head_classification_head_module_list_3_1_weight", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "head_classification_head_module_list_3_1_weight",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_head_classification_head_module_list_3_1_weight,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(head_classification_head_module_list_3_1_weight),
                                                .dataSize=BINLEN(head_classification_head_module_list_3_1_weight)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_head_classification_head_module_list_3_1_bias(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_head_classification_head_module_list_3_1_bias[] = {24};
  VALIDATE(model.addTensor("head_classification_head_module_list_3_1_bias", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "head_classification_head_module_list_3_1_bias",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_head_classification_head_module_list_3_1_bias,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(head_classification_head_module_list_3_1_bias),
                                                .dataSize=BINLEN(head_classification_head_module_list_3_1_bias)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__head_classification_head_module_list_3_module_list_3_1_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_classification_head_module_list_3_module_list_3_1_Conv */
  uint32_t dimensions___head_classification_head_module_list_3_module_list_3_1_Conv_dilation[] = {2};
  uint32_t __head_classification_head_module_list_3_module_list_3_1_Conv_dilation[] = {1, 1};
  uint32_t dimensions___head_classification_head_module_list_3_module_list_3_1_Conv_pad_amount[] = {2, 2};
  uint32_t __head_classification_head_module_list_3_module_list_3_1_Conv_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions___head_classification_head_module_list_3_module_list_3_1_Conv_stride[] = {2};
  uint32_t __head_classification_head_module_list_3_module_list_3_1_Conv_stride[] = {1, 1};
  Qnn_Param_t params__head_classification_head_module_list_3_module_list_3_1_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_module_list_3_module_list_3_1_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_classification_head_module_list_3_module_list_3_1_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_module_list_3_module_list_3_1_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_module_list_3_module_list_3_1_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___head_classification_head_module_list_3_module_list_3_1_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_module_list_3_module_list_3_1_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_module_list_3_module_list_3_1_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_classification_head_module_list_3_module_list_3_1_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_module_list_3_module_list_3_1_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__head_classification_head_module_list_3_module_list_3_1_Conv[] = {
    "_head_classification_head_module_list_3_module_list_3_0_module_list_3_0_2_Clip_output_0",
    "head_classification_head_module_list_3_1_weight",
    "head_classification_head_module_list_3_1_bias"
  };
  uint32_t dimensions__head_classification_head_module_list_3_module_list_3_1_Conv_output_0[] = {1, 9, 12, 24};
  Qnn_Tensor_t outputs__head_classification_head_module_list_3_module_list_3_1_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_module_list_3_module_list_3_1_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_classification_head_module_list_3_module_list_3_1_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_classification_head_module_list_3_module_list_3_1_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__head_classification_head_module_list_3_module_list_3_1_Conv, // Node Params
                         4, // Num Node Params
                         inputs__head_classification_head_module_list_3_module_list_3_1_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__head_classification_head_module_list_3_module_list_3_1_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_classification_head_module_list_3_module_list_3_1_Conv_output_0_nchw(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_classification_head_module_list_3_module_list_3_1_Conv_output_0_nchw */
  uint32_t dimensions___head_classification_head_module_list_3_module_list_3_1_Conv_output_0_nchw_perm[] = {4};
  uint32_t __head_classification_head_module_list_3_module_list_3_1_Conv_output_0_nchw_perm[] = {0, 3, 1, 2};
  Qnn_Param_t params__head_classification_head_module_list_3_module_list_3_1_Conv_output_0_nchw[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_module_list_3_module_list_3_1_Conv_output_0_nchw_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_classification_head_module_list_3_module_list_3_1_Conv_output_0_nchw_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_module_list_3_module_list_3_1_Conv_output_0_nchw_perm,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__head_classification_head_module_list_3_module_list_3_1_Conv_output_0_nchw[] = {
    "_head_classification_head_module_list_3_module_list_3_1_Conv_output_0"
  };
  uint32_t dimensions__head_classification_head_module_list_3_module_list_3_1_Conv_output_0_nchw[] = {1, 24, 9, 12};
  Qnn_Tensor_t outputs__head_classification_head_module_list_3_module_list_3_1_Conv_output_0_nchw[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_module_list_3_module_list_3_1_Conv_output_0_nchw",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_classification_head_module_list_3_module_list_3_1_Conv_output_0_nchw,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_classification_head_module_list_3_module_list_3_1_Conv_output_0_nchw", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__head_classification_head_module_list_3_module_list_3_1_Conv_output_0_nchw, // Node Params
                         1, // Num Node Params
                         inputs__head_classification_head_module_list_3_module_list_3_1_Conv_output_0_nchw, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_classification_head_module_list_3_module_list_3_1_Conv_output_0_nchw, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_classification_head_Reshape_6(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_classification_head_Reshape_6 */
  const char*  inputs__head_classification_head_Reshape_6[] = {
    "_head_classification_head_module_list_3_module_list_3_1_Conv_output_0_nchw"
  };
  uint32_t dimensions__head_classification_head_Reshape_6_output_0[] = {1, 6, 4, 9, 12};
  Qnn_Tensor_t outputs__head_classification_head_Reshape_6[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_Reshape_6_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 5,
            .dimensions=dimensions__head_classification_head_Reshape_6_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_classification_head_Reshape_6", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__head_classification_head_Reshape_6, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_classification_head_Reshape_6, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_classification_head_Transpose_3(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_classification_head_Transpose_3 */
  uint32_t dimensions___head_classification_head_Transpose_3_perm[] = {5};
  uint32_t __head_classification_head_Transpose_3_perm[] = {0, 3, 4, 1, 2};
  Qnn_Param_t params__head_classification_head_Transpose_3[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_Transpose_3_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_classification_head_Transpose_3_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_Transpose_3_perm,
                           .dataSize=20}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__head_classification_head_Transpose_3[] = {
    "_head_classification_head_Reshape_6_output_0"
  };
  uint32_t dimensions__head_classification_head_Transpose_3_output_0[] = {1, 9, 12, 6, 4};
  Qnn_Tensor_t outputs__head_classification_head_Transpose_3[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_Transpose_3_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 5,
            .dimensions=dimensions__head_classification_head_Transpose_3_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_classification_head_Transpose_3", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__head_classification_head_Transpose_3, // Node Params
                         1, // Num Node Params
                         inputs__head_classification_head_Transpose_3, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_classification_head_Transpose_3, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_classification_head_Reshape_7(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_classification_head_Reshape_7 */
  const char*  inputs__head_classification_head_Reshape_7[] = {
    "_head_classification_head_Transpose_3_output_0"
  };
  uint32_t dimensions__head_classification_head_Reshape_7_output_0[] = {1, 648, 4};
  Qnn_Tensor_t outputs__head_classification_head_Reshape_7[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_Reshape_7_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__head_classification_head_Reshape_7_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_classification_head_Reshape_7", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__head_classification_head_Reshape_7, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_classification_head_Reshape_7, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__Conv_1470(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__Conv_1470[] = {256};
  VALIDATE(model.addTensor("onnx__Conv_1470", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__Conv_1470",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_onnx__Conv_1470,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__Conv_1470),
                                                .dataSize=BINLEN(onnx__Conv_1470)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv */
  uint32_t dimensions___head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_dilation[] = {2};
  uint32_t __head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_dilation[] = {1, 1};
  uint32_t dimensions___head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_pad_amount[] = {2, 2};
  uint32_t __head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions___head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_stride[] = {2};
  uint32_t __head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_stride[] = {1, 1};
  Qnn_Param_t params__head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv[] = {
    "_backbone_fpn_extra_blocks_p7_Conv_output_0",
    "onnx__Conv_1448",
    "onnx__Conv_1470"
  };
  uint32_t dimensions__head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_output_0[] = {1, 5, 6, 256};
  Qnn_Tensor_t outputs__head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "DepthWiseConv2d", // Qnn Node Type
                         params__head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv, // Node Params
                         3, // Num Node Params
                         inputs__head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_classification_head_module_list_4_module_list_4_0_module_list_4_0_2_Clip(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_classification_head_module_list_4_module_list_4_0_module_list_4_0_2_Clip */
  Qnn_Param_t params__head_classification_head_module_list_4_module_list_4_0_module_list_4_0_2_Clip[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="max_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 6.000000000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="min_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 0.000000000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 5}}}}
  };
  const char*  inputs__head_classification_head_module_list_4_module_list_4_0_module_list_4_0_2_Clip[] = {
    "_head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv_output_0"
  };
  uint32_t dimensions__head_classification_head_module_list_4_module_list_4_0_module_list_4_0_2_Clip_output_0[] = {1, 5, 6, 256};
  Qnn_Tensor_t outputs__head_classification_head_module_list_4_module_list_4_0_module_list_4_0_2_Clip[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_module_list_4_module_list_4_0_module_list_4_0_2_Clip_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_classification_head_module_list_4_module_list_4_0_module_list_4_0_2_Clip_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_classification_head_module_list_4_module_list_4_0_module_list_4_0_2_Clip", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params__head_classification_head_module_list_4_module_list_4_0_module_list_4_0_2_Clip, // Node Params
                         3, // Num Node Params
                         inputs__head_classification_head_module_list_4_module_list_4_0_module_list_4_0_2_Clip, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_classification_head_module_list_4_module_list_4_0_module_list_4_0_2_Clip, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_head_classification_head_module_list_4_1_weight(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_head_classification_head_module_list_4_1_weight[] = {1, 1, 256, 24};
  VALIDATE(model.addTensor("head_classification_head_module_list_4_1_weight", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "head_classification_head_module_list_4_1_weight",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_head_classification_head_module_list_4_1_weight,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(head_classification_head_module_list_4_1_weight),
                                                .dataSize=BINLEN(head_classification_head_module_list_4_1_weight)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_head_classification_head_module_list_4_1_bias(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_head_classification_head_module_list_4_1_bias[] = {24};
  VALIDATE(model.addTensor("head_classification_head_module_list_4_1_bias", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "head_classification_head_module_list_4_1_bias",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_head_classification_head_module_list_4_1_bias,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(head_classification_head_module_list_4_1_bias),
                                                .dataSize=BINLEN(head_classification_head_module_list_4_1_bias)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__head_classification_head_module_list_4_module_list_4_1_Conv(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_classification_head_module_list_4_module_list_4_1_Conv */
  uint32_t dimensions___head_classification_head_module_list_4_module_list_4_1_Conv_dilation[] = {2};
  uint32_t __head_classification_head_module_list_4_module_list_4_1_Conv_dilation[] = {1, 1};
  uint32_t dimensions___head_classification_head_module_list_4_module_list_4_1_Conv_pad_amount[] = {2, 2};
  uint32_t __head_classification_head_module_list_4_module_list_4_1_Conv_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions___head_classification_head_module_list_4_module_list_4_1_Conv_stride[] = {2};
  uint32_t __head_classification_head_module_list_4_module_list_4_1_Conv_stride[] = {1, 1};
  Qnn_Param_t params__head_classification_head_module_list_4_module_list_4_1_Conv[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_module_list_4_module_list_4_1_Conv_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_classification_head_module_list_4_module_list_4_1_Conv_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_module_list_4_module_list_4_1_Conv_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_module_list_4_module_list_4_1_Conv_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___head_classification_head_module_list_4_module_list_4_1_Conv_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_module_list_4_module_list_4_1_Conv_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_module_list_4_module_list_4_1_Conv_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_classification_head_module_list_4_module_list_4_1_Conv_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_module_list_4_module_list_4_1_Conv_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__head_classification_head_module_list_4_module_list_4_1_Conv[] = {
    "_head_classification_head_module_list_4_module_list_4_0_module_list_4_0_2_Clip_output_0",
    "head_classification_head_module_list_4_1_weight",
    "head_classification_head_module_list_4_1_bias"
  };
  uint32_t dimensions__head_classification_head_module_list_4_module_list_4_1_Conv_output_0[] = {1, 5, 6, 24};
  Qnn_Tensor_t outputs__head_classification_head_module_list_4_module_list_4_1_Conv[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_module_list_4_module_list_4_1_Conv_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_classification_head_module_list_4_module_list_4_1_Conv_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_classification_head_module_list_4_module_list_4_1_Conv", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params__head_classification_head_module_list_4_module_list_4_1_Conv, // Node Params
                         4, // Num Node Params
                         inputs__head_classification_head_module_list_4_module_list_4_1_Conv, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__head_classification_head_module_list_4_module_list_4_1_Conv, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_classification_head_module_list_4_module_list_4_1_Conv_output_0_nchw(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_classification_head_module_list_4_module_list_4_1_Conv_output_0_nchw */
  uint32_t dimensions___head_classification_head_module_list_4_module_list_4_1_Conv_output_0_nchw_perm[] = {4};
  uint32_t __head_classification_head_module_list_4_module_list_4_1_Conv_output_0_nchw_perm[] = {0, 3, 1, 2};
  Qnn_Param_t params__head_classification_head_module_list_4_module_list_4_1_Conv_output_0_nchw[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_module_list_4_module_list_4_1_Conv_output_0_nchw_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_classification_head_module_list_4_module_list_4_1_Conv_output_0_nchw_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_module_list_4_module_list_4_1_Conv_output_0_nchw_perm,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__head_classification_head_module_list_4_module_list_4_1_Conv_output_0_nchw[] = {
    "_head_classification_head_module_list_4_module_list_4_1_Conv_output_0"
  };
  uint32_t dimensions__head_classification_head_module_list_4_module_list_4_1_Conv_output_0_nchw[] = {1, 24, 5, 6};
  Qnn_Tensor_t outputs__head_classification_head_module_list_4_module_list_4_1_Conv_output_0_nchw[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_module_list_4_module_list_4_1_Conv_output_0_nchw",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__head_classification_head_module_list_4_module_list_4_1_Conv_output_0_nchw,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_classification_head_module_list_4_module_list_4_1_Conv_output_0_nchw", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__head_classification_head_module_list_4_module_list_4_1_Conv_output_0_nchw, // Node Params
                         1, // Num Node Params
                         inputs__head_classification_head_module_list_4_module_list_4_1_Conv_output_0_nchw, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_classification_head_module_list_4_module_list_4_1_Conv_output_0_nchw, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_classification_head_Reshape_8(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_classification_head_Reshape_8 */
  const char*  inputs__head_classification_head_Reshape_8[] = {
    "_head_classification_head_module_list_4_module_list_4_1_Conv_output_0_nchw"
  };
  uint32_t dimensions__head_classification_head_Reshape_8_output_0[] = {1, 6, 4, 5, 6};
  Qnn_Tensor_t outputs__head_classification_head_Reshape_8[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_Reshape_8_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 5,
            .dimensions=dimensions__head_classification_head_Reshape_8_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_classification_head_Reshape_8", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__head_classification_head_Reshape_8, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_classification_head_Reshape_8, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_classification_head_Transpose_4(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_classification_head_Transpose_4 */
  uint32_t dimensions___head_classification_head_Transpose_4_perm[] = {5};
  uint32_t __head_classification_head_Transpose_4_perm[] = {0, 3, 4, 1, 2};
  Qnn_Param_t params__head_classification_head_Transpose_4[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__head_classification_head_Transpose_4_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___head_classification_head_Transpose_4_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__head_classification_head_Transpose_4_perm,
                           .dataSize=20}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__head_classification_head_Transpose_4[] = {
    "_head_classification_head_Reshape_8_output_0"
  };
  uint32_t dimensions__head_classification_head_Transpose_4_output_0[] = {1, 5, 6, 6, 4};
  Qnn_Tensor_t outputs__head_classification_head_Transpose_4[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_Transpose_4_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 5,
            .dimensions=dimensions__head_classification_head_Transpose_4_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_classification_head_Transpose_4", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__head_classification_head_Transpose_4, // Node Params
                         1, // Num Node Params
                         inputs__head_classification_head_Transpose_4, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_classification_head_Transpose_4, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_classification_head_Reshape_9(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_classification_head_Reshape_9 */
  const char*  inputs__head_classification_head_Reshape_9[] = {
    "_head_classification_head_Transpose_4_output_0"
  };
  uint32_t dimensions__head_classification_head_Reshape_9_output_0[] = {1, 180, 4};
  Qnn_Tensor_t outputs__head_classification_head_Reshape_9[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_Reshape_9_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__head_classification_head_Reshape_9_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_classification_head_Reshape_9", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__head_classification_head_Reshape_9, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__head_classification_head_Reshape_9, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__head_classification_head_Concat_10(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _head_classification_head_Concat_10 */
  Qnn_Param_t params__head_classification_head_Concat_10[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="axis",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__head_classification_head_Concat_10[] = {
    "_head_classification_head_Reshape_1_output_0",
    "_head_classification_head_Reshape_3_output_0",
    "_head_classification_head_Reshape_5_output_0",
    "_head_classification_head_Reshape_7_output_0",
    "_head_classification_head_Reshape_9_output_0"
  };
  uint32_t dimensions__head_classification_head_Concat_10_output_0_reshape[] = {1, 49074, 4};
  Qnn_Tensor_t outputs__head_classification_head_Concat_10[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_Concat_10_output_0_reshape",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__head_classification_head_Concat_10_output_0_reshape,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_head_classification_head_Concat_10", // Node Name
                         "qti.aisw", // Package Name
                         "Concat", // Qnn Node Type
                         params__head_classification_head_Concat_10, // Node Params
                         1, // Num Node Params
                         inputs__head_classification_head_Concat_10, // Input Tensor Names
                         5, // Num Input Tensor Names
                         outputs__head_classification_head_Concat_10, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Softmax(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Softmax */
  Qnn_Param_t params__Softmax[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="axis",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 2}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="beta",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 1.000000000000f}}}}
  };
  const char*  inputs__Softmax[] = {
    "_head_classification_head_Concat_10_output_0_reshape"
  };
  uint32_t dimensions__head_classification_head_Concat_10_output_0_softmax[] = {1, 49074, 4};
  Qnn_Tensor_t outputs__Softmax[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_head_classification_head_Concat_10_output_0_softmax",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__head_classification_head_Concat_10_output_0_softmax,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Softmax", // Node Name
                         "qti.aisw", // Package Name
                         "Softmax", // Qnn Node Type
                         params__Softmax, // Node Params
                         2, // Num Node Params
                         inputs__Softmax, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Softmax, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Softmax_output_0_reshape(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Softmax_output_0_reshape */
  const char*  inputs__Softmax_output_0_reshape[] = {
    "_head_classification_head_Concat_10_output_0_softmax"
  };
  uint32_t dimensions__Squeeze_2_output_0[] = {49074, 4};
  Qnn_Tensor_t outputs__Softmax_output_0_reshape[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Squeeze_2_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Squeeze_2_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Softmax_output_0_reshape", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__Softmax_output_0_reshape, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Softmax_output_0_reshape, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Squeeze_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Squeeze_1 */
  const char*  inputs__Squeeze_1[] = {
    "_head_regression_head_Concat_10_output_0"
  };
  uint32_t dimensions__Squeeze_1_output_0[] = {49074, 4};
  Qnn_Tensor_t outputs__Squeeze_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Squeeze_1_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Squeeze_1_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Squeeze_1", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__Squeeze_1, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Squeeze_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Slice(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Slice */
  uint32_t dimensions___Slice_ranges[] = {2, 3};
  int32_t __Slice_ranges[] = {0, 49074, 1, 0, 4, 4};
  Qnn_Param_t params__Slice[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="ranges",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__Slice_ranges",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_INT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___Slice_ranges,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__Slice_ranges,
                           .dataSize=24}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="begin_mask",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="end_mask",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="new_axes_mask",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="shrink_axes",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__Slice[] = {
    "_Squeeze_1_output_0"
  };
  uint32_t dimensions__Slice_output_0[] = {49074, 1};
  Qnn_Tensor_t outputs__Slice[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Slice_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Slice_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Slice", // Node Name
                         "qti.aisw", // Package Name
                         "StridedSlice", // Qnn Node Type
                         params__Slice, // Node Params
                         5, // Num Node Params
                         inputs__Slice, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Slice, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor__Constant_9_output_0(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions__Constant_9_output_0[] = {1};
  VALIDATE(model.addTensor("_Constant_9_output_0", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "_Constant_9_output_0",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions__Constant_9_output_0,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(_Constant_9_output_0),
                                                .dataSize=BINLEN(_Constant_9_output_0)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__Div(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Div */
  Qnn_Param_t params__Div[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 2}}}}
  };
  const char*  inputs__Div[] = {
    "_Slice_output_0",
    "_Constant_9_output_0"
  };
  uint32_t dimensions__Div_output_0[] = {49074, 1};
  Qnn_Tensor_t outputs__Div[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Div_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Div_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Div", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__Div, // Node Params
                         1, // Num Node Params
                         inputs__Div, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Div, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Slice_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Slice_1 */
  uint32_t dimensions___Slice_1_ranges[] = {2, 3};
  int32_t __Slice_1_ranges[] = {0, 49074, 1, 1, 4, 4};
  Qnn_Param_t params__Slice_1[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="ranges",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__Slice_1_ranges",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_INT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___Slice_1_ranges,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__Slice_1_ranges,
                           .dataSize=24}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="begin_mask",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="end_mask",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="new_axes_mask",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="shrink_axes",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__Slice_1[] = {
    "_Squeeze_1_output_0"
  };
  uint32_t dimensions__Slice_1_output_0[] = {49074, 1};
  Qnn_Tensor_t outputs__Slice_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Slice_1_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Slice_1_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Slice_1", // Node Name
                         "qti.aisw", // Package Name
                         "StridedSlice", // Qnn Node Type
                         params__Slice_1, // Node Params
                         5, // Num Node Params
                         inputs__Slice_1, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Slice_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Div_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Div_1 */
  Qnn_Param_t params__Div_1[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 2}}}}
  };
  const char*  inputs__Div_1[] = {
    "_Slice_1_output_0",
    "_Constant_9_output_0"
  };
  uint32_t dimensions__Div_1_output_0[] = {49074, 1};
  Qnn_Tensor_t outputs__Div_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Div_1_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Div_1_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Div_1", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__Div_1, // Node Params
                         1, // Num Node Params
                         inputs__Div_1, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Div_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Slice_2(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Slice_2 */
  uint32_t dimensions___Slice_2_ranges[] = {2, 3};
  int32_t __Slice_2_ranges[] = {0, 49074, 1, 2, 4, 4};
  Qnn_Param_t params__Slice_2[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="ranges",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__Slice_2_ranges",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_INT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___Slice_2_ranges,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__Slice_2_ranges,
                           .dataSize=24}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="begin_mask",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="end_mask",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="new_axes_mask",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="shrink_axes",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__Slice_2[] = {
    "_Squeeze_1_output_0"
  };
  uint32_t dimensions__Slice_2_output_0[] = {49074, 1};
  Qnn_Tensor_t outputs__Slice_2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Slice_2_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Slice_2_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Slice_2", // Node Name
                         "qti.aisw", // Package Name
                         "StridedSlice", // Qnn Node Type
                         params__Slice_2, // Node Params
                         5, // Num Node Params
                         inputs__Slice_2, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Slice_2, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor__Constant_19_output_0(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions__Constant_19_output_0[] = {1};
  VALIDATE(model.addTensor("_Constant_19_output_0", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "_Constant_19_output_0",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions__Constant_19_output_0,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(_Constant_19_output_0),
                                                .dataSize=BINLEN(_Constant_19_output_0)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__Div_2(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Div_2 */
  Qnn_Param_t params__Div_2[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 2}}}}
  };
  const char*  inputs__Div_2[] = {
    "_Slice_2_output_0",
    "_Constant_19_output_0"
  };
  uint32_t dimensions__Div_2_output_0[] = {49074, 1};
  Qnn_Tensor_t outputs__Div_2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Div_2_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Div_2_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Div_2", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__Div_2, // Node Params
                         1, // Num Node Params
                         inputs__Div_2, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Div_2, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Slice_3(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Slice_3 */
  uint32_t dimensions___Slice_3_ranges[] = {2, 3};
  int32_t __Slice_3_ranges[] = {0, 49074, 1, 3, 4, 4};
  Qnn_Param_t params__Slice_3[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="ranges",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__Slice_3_ranges",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_INT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___Slice_3_ranges,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__Slice_3_ranges,
                           .dataSize=24}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="begin_mask",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="end_mask",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="new_axes_mask",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="shrink_axes",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__Slice_3[] = {
    "_Squeeze_1_output_0"
  };
  uint32_t dimensions__Slice_3_output_0[] = {49074, 1};
  Qnn_Tensor_t outputs__Slice_3[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Slice_3_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Slice_3_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Slice_3", // Node Name
                         "qti.aisw", // Package Name
                         "StridedSlice", // Qnn Node Type
                         params__Slice_3, // Node Params
                         5, // Num Node Params
                         inputs__Slice_3, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Slice_3, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Div_3(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Div_3 */
  Qnn_Param_t params__Div_3[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 2}}}}
  };
  const char*  inputs__Div_3[] = {
    "_Slice_3_output_0",
    "_Constant_19_output_0"
  };
  uint32_t dimensions__Div_3_output_0[] = {49074, 1};
  Qnn_Tensor_t outputs__Div_3[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Div_3_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Div_3_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Div_3", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__Div_3, // Node Params
                         1, // Num Node Params
                         inputs__Div_3, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Div_3, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Clip(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Clip */
  Qnn_Param_t params__Clip[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="max_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 4.135166645050f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="min_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = -340282346638528859811704183484516925440.000000000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 5}}}}
  };
  const char*  inputs__Clip[] = {
    "_Div_2_output_0"
  };
  uint32_t dimensions__Clip_output_0[] = {49074, 1};
  Qnn_Tensor_t outputs__Clip[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Clip_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Clip_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Clip", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params__Clip, // Node Params
                         3, // Num Node Params
                         inputs__Clip, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Clip, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Clip_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Clip_1 */
  Qnn_Param_t params__Clip_1[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="max_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 4.135166645050f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="min_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = -340282346638528859811704183484516925440.000000000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 5}}}}
  };
  const char*  inputs__Clip_1[] = {
    "_Div_3_output_0"
  };
  uint32_t dimensions__Clip_1_output_0[] = {49074, 1};
  Qnn_Tensor_t outputs__Clip_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Clip_1_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Clip_1_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Clip_1", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params__Clip_1, // Node Params
                         3, // Num Node Params
                         inputs__Clip_1, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Clip_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor__Unsqueeze_output_0(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions__Unsqueeze_output_0[] = {49074, 1};
  VALIDATE(model.addTensor("_Unsqueeze_output_0", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "_Unsqueeze_output_0",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 2,
                                 .dimensions=dimensions__Unsqueeze_output_0,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(_Unsqueeze_output_0),
                                                .dataSize=BINLEN(_Unsqueeze_output_0)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__Mul_2(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Mul_2 */
  Qnn_Param_t params__Mul_2[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 13}}}}
  };
  const char*  inputs__Mul_2[] = {
    "_Div_output_0",
    "_Unsqueeze_output_0"
  };
  uint32_t dimensions__Mul_2_output_0[] = {49074, 1};
  Qnn_Tensor_t outputs__Mul_2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Mul_2_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Mul_2_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Mul_2", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__Mul_2, // Node Params
                         1, // Num Node Params
                         inputs__Mul_2, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Mul_2, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor__Unsqueeze_1_output_0(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions__Unsqueeze_1_output_0[] = {49074, 1};
  VALIDATE(model.addTensor("_Unsqueeze_1_output_0", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "_Unsqueeze_1_output_0",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 2,
                                 .dimensions=dimensions__Unsqueeze_1_output_0,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(_Unsqueeze_1_output_0),
                                                .dataSize=BINLEN(_Unsqueeze_1_output_0)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__Add_2(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Add_2 */
  Qnn_Param_t params__Add_2[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__Add_2[] = {
    "_Mul_2_output_0",
    "_Unsqueeze_1_output_0"
  };
  uint32_t dimensions__Add_2_output_0[] = {49074, 1};
  Qnn_Tensor_t outputs__Add_2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Add_2_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Add_2_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Add_2", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__Add_2, // Node Params
                         1, // Num Node Params
                         inputs__Add_2, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Add_2, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor__Unsqueeze_2_output_0(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions__Unsqueeze_2_output_0[] = {49074, 1};
  VALIDATE(model.addTensor("_Unsqueeze_2_output_0", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "_Unsqueeze_2_output_0",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 2,
                                 .dimensions=dimensions__Unsqueeze_2_output_0,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(_Unsqueeze_2_output_0),
                                                .dataSize=BINLEN(_Unsqueeze_2_output_0)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__Mul_3(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Mul_3 */
  Qnn_Param_t params__Mul_3[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 13}}}}
  };
  const char*  inputs__Mul_3[] = {
    "_Div_1_output_0",
    "_Unsqueeze_2_output_0"
  };
  uint32_t dimensions__Mul_3_output_0[] = {49074, 1};
  Qnn_Tensor_t outputs__Mul_3[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Mul_3_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Mul_3_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Mul_3", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__Mul_3, // Node Params
                         1, // Num Node Params
                         inputs__Mul_3, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Mul_3, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor__Unsqueeze_3_output_0(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions__Unsqueeze_3_output_0[] = {49074, 1};
  VALIDATE(model.addTensor("_Unsqueeze_3_output_0", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "_Unsqueeze_3_output_0",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 2,
                                 .dimensions=dimensions__Unsqueeze_3_output_0,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(_Unsqueeze_3_output_0),
                                                .dataSize=BINLEN(_Unsqueeze_3_output_0)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__Add_3(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Add_3 */
  Qnn_Param_t params__Add_3[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__Add_3[] = {
    "_Mul_3_output_0",
    "_Unsqueeze_3_output_0"
  };
  uint32_t dimensions__Add_3_output_0[] = {49074, 1};
  Qnn_Tensor_t outputs__Add_3[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Add_3_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Add_3_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Add_3", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__Add_3, // Node Params
                         1, // Num Node Params
                         inputs__Add_3, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Add_3, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Exp(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Exp */
  Qnn_Param_t params__Exp[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 5}}}}
  };
  const char*  inputs__Exp[] = {
    "_Clip_output_0"
  };
  uint32_t dimensions__Exp_output_0[] = {49074, 1};
  Qnn_Tensor_t outputs__Exp[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Exp_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Exp_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Exp", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseUnary", // Qnn Node Type
                         params__Exp, // Node Params
                         1, // Num Node Params
                         inputs__Exp, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Exp, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Mul_4(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Mul_4 */
  Qnn_Param_t params__Mul_4[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 13}}}}
  };
  const char*  inputs__Mul_4[] = {
    "_Exp_output_0",
    "_Unsqueeze_output_0"
  };
  uint32_t dimensions__Mul_4_output_0[] = {49074, 1};
  Qnn_Tensor_t outputs__Mul_4[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Mul_4_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Mul_4_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Mul_4", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__Mul_4, // Node Params
                         1, // Num Node Params
                         inputs__Mul_4, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Mul_4, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Exp_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Exp_1 */
  Qnn_Param_t params__Exp_1[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 5}}}}
  };
  const char*  inputs__Exp_1[] = {
    "_Clip_1_output_0"
  };
  uint32_t dimensions__Exp_1_output_0[] = {49074, 1};
  Qnn_Tensor_t outputs__Exp_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Exp_1_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Exp_1_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Exp_1", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseUnary", // Qnn Node Type
                         params__Exp_1, // Node Params
                         1, // Num Node Params
                         inputs__Exp_1, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Exp_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Mul_5(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Mul_5 */
  Qnn_Param_t params__Mul_5[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 13}}}}
  };
  const char*  inputs__Mul_5[] = {
    "_Exp_1_output_0",
    "_Unsqueeze_2_output_0"
  };
  uint32_t dimensions__Mul_5_output_0[] = {49074, 1};
  Qnn_Tensor_t outputs__Mul_5[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Mul_5_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Mul_5_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Mul_5", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__Mul_5, // Node Params
                         1, // Num Node Params
                         inputs__Mul_5, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Mul_5, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor__anchor_generator_Constant_12_output_0(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions__anchor_generator_Constant_12_output_0[] = {1};
  VALIDATE(model.addTensor("_anchor_generator_Constant_12_output_0", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "_anchor_generator_Constant_12_output_0",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions__anchor_generator_Constant_12_output_0,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(_anchor_generator_Constant_12_output_0),
                                                .dataSize=BINLEN(_anchor_generator_Constant_12_output_0)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__Mul_6(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Mul_6 */
  Qnn_Param_t params__Mul_6[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 13}}}}
  };
  const char*  inputs__Mul_6[] = {
    "_anchor_generator_Constant_12_output_0",
    "_Mul_5_output_0"
  };
  uint32_t dimensions__Mul_6_output_0[] = {49074, 1};
  Qnn_Tensor_t outputs__Mul_6[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Mul_6_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Mul_6_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Mul_6", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__Mul_6, // Node Params
                         1, // Num Node Params
                         inputs__Mul_6, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Mul_6, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Mul_7(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Mul_7 */
  Qnn_Param_t params__Mul_7[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 13}}}}
  };
  const char*  inputs__Mul_7[] = {
    "_anchor_generator_Constant_12_output_0",
    "_Mul_4_output_0"
  };
  uint32_t dimensions__Mul_7_output_0[] = {49074, 1};
  Qnn_Tensor_t outputs__Mul_7[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Mul_7_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Mul_7_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Mul_7", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__Mul_7, // Node Params
                         1, // Num Node Params
                         inputs__Mul_7, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Mul_7, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Sub_2(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Sub_2 */
  Qnn_Param_t params__Sub_2[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 18}}}}
  };
  const char*  inputs__Sub_2[] = {
    "_Add_2_output_0",
    "_Mul_7_output_0"
  };
  uint32_t dimensions__Sub_2_output_0[] = {49074, 1};
  Qnn_Tensor_t outputs__Sub_2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Sub_2_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Sub_2_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Sub_2", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__Sub_2, // Node Params
                         1, // Num Node Params
                         inputs__Sub_2, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Sub_2, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Sub_3(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Sub_3 */
  Qnn_Param_t params__Sub_3[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 18}}}}
  };
  const char*  inputs__Sub_3[] = {
    "_Add_3_output_0",
    "_Mul_6_output_0"
  };
  uint32_t dimensions__Sub_3_output_0[] = {49074, 1};
  Qnn_Tensor_t outputs__Sub_3[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Sub_3_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Sub_3_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Sub_3", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__Sub_3, // Node Params
                         1, // Num Node Params
                         inputs__Sub_3, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Sub_3, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Add_4(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Add_4 */
  Qnn_Param_t params__Add_4[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__Add_4[] = {
    "_Add_2_output_0",
    "_Mul_7_output_0"
  };
  uint32_t dimensions__Add_4_output_0[] = {49074, 1};
  Qnn_Tensor_t outputs__Add_4[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Add_4_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Add_4_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Add_4", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__Add_4, // Node Params
                         1, // Num Node Params
                         inputs__Add_4, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Add_4, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Add_5(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Add_5 */
  Qnn_Param_t params__Add_5[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__Add_5[] = {
    "_Add_3_output_0",
    "_Mul_6_output_0"
  };
  uint32_t dimensions__Add_5_output_0[] = {49074, 1};
  Qnn_Tensor_t outputs__Add_5[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Add_5_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Add_5_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Add_5", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__Add_5, // Node Params
                         1, // Num Node Params
                         inputs__Add_5, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Add_5, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Unsqueeze_4(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Unsqueeze_4 */
  const char*  inputs__Unsqueeze_4[] = {
    "_Sub_2_output_0"
  };
  uint32_t dimensions__Unsqueeze_4_output_0[] = {49074, 1, 1};
  Qnn_Tensor_t outputs__Unsqueeze_4[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Unsqueeze_4_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__Unsqueeze_4_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Unsqueeze_4", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__Unsqueeze_4, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Unsqueeze_4, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Unsqueeze_5(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Unsqueeze_5 */
  const char*  inputs__Unsqueeze_5[] = {
    "_Sub_3_output_0"
  };
  uint32_t dimensions__Unsqueeze_5_output_0[] = {49074, 1, 1};
  Qnn_Tensor_t outputs__Unsqueeze_5[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Unsqueeze_5_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__Unsqueeze_5_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Unsqueeze_5", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__Unsqueeze_5, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Unsqueeze_5, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Unsqueeze_6(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Unsqueeze_6 */
  const char*  inputs__Unsqueeze_6[] = {
    "_Add_4_output_0"
  };
  uint32_t dimensions__Unsqueeze_6_output_0[] = {49074, 1, 1};
  Qnn_Tensor_t outputs__Unsqueeze_6[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Unsqueeze_6_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__Unsqueeze_6_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Unsqueeze_6", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__Unsqueeze_6, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Unsqueeze_6, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Unsqueeze_7(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Unsqueeze_7 */
  const char*  inputs__Unsqueeze_7[] = {
    "_Add_5_output_0"
  };
  uint32_t dimensions__Unsqueeze_7_output_0[] = {49074, 1, 1};
  Qnn_Tensor_t outputs__Unsqueeze_7[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Unsqueeze_7_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__Unsqueeze_7_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Unsqueeze_7", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__Unsqueeze_7, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Unsqueeze_7, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Concat(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Concat */
  Qnn_Param_t params__Concat[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="axis",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 2}}}}
  };
  const char*  inputs__Concat[] = {
    "_Unsqueeze_4_output_0",
    "_Unsqueeze_5_output_0",
    "_Unsqueeze_6_output_0",
    "_Unsqueeze_7_output_0"
  };
  uint32_t dimensions__Concat_output_0[] = {49074, 1, 4};
  Qnn_Tensor_t outputs__Concat[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Concat_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__Concat_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Concat", // Node Name
                         "qti.aisw", // Package Name
                         "Concat", // Qnn Node Type
                         params__Concat, // Node Params
                         1, // Num Node Params
                         inputs__Concat, // Input Tensor Names
                         4, // Num Input Tensor Names
                         outputs__Concat, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Flatten(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Flatten */
  const char*  inputs__Flatten[] = {
    "_Concat_output_0"
  };
  uint32_t dimensions__Flatten_output_0[] = {49074, 4};
  Qnn_Tensor_t outputs__Flatten[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Flatten_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Flatten_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Flatten", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__Flatten, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Flatten, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Slice_4(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Slice_4 */
  uint32_t dimensions___Slice_4_ranges[] = {2, 3};
  int32_t __Slice_4_ranges[] = {0, 49074, 1, 0, 4, 2};
  Qnn_Param_t params__Slice_4[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="ranges",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__Slice_4_ranges",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_INT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___Slice_4_ranges,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__Slice_4_ranges,
                           .dataSize=24}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="begin_mask",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="end_mask",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="new_axes_mask",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="shrink_axes",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__Slice_4[] = {
    "_Flatten_output_0"
  };
  uint32_t dimensions__Slice_4_output_0[] = {49074, 2};
  Qnn_Tensor_t outputs__Slice_4[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Slice_4_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Slice_4_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Slice_4", // Node Name
                         "qti.aisw", // Package Name
                         "StridedSlice", // Qnn Node Type
                         params__Slice_4, // Node Params
                         5, // Num Node Params
                         inputs__Slice_4, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Slice_4, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Slice_5(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Slice_5 */
  uint32_t dimensions___Slice_5_ranges[] = {2, 3};
  int32_t __Slice_5_ranges[] = {0, 49074, 1, 1, 4, 2};
  Qnn_Param_t params__Slice_5[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="ranges",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__Slice_5_ranges",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_INT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___Slice_5_ranges,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__Slice_5_ranges,
                           .dataSize=24}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="begin_mask",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="end_mask",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="new_axes_mask",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="shrink_axes",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__Slice_5[] = {
    "_Flatten_output_0"
  };
  uint32_t dimensions__Slice_5_output_0[] = {49074, 2};
  Qnn_Tensor_t outputs__Slice_5[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Slice_5_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Slice_5_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Slice_5", // Node Name
                         "qti.aisw", // Package Name
                         "StridedSlice", // Qnn Node Type
                         params__Slice_5, // Node Params
                         5, // Num Node Params
                         inputs__Slice_5, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Slice_5, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_elementwiseneuron_30(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR elementwiseneuron_30 */
  Qnn_Param_t params_elementwiseneuron_30[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="max_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 720.000000000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="min_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 0.000000000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 5}}}}
  };
  const char*  inputs_elementwiseneuron_30[] = {
    "_Slice_4_output_0"
  };
  uint32_t dimensions__Max_output_0[] = {49074, 2};
  Qnn_Tensor_t outputs_elementwiseneuron_30[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Max_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Max_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "elementwiseneuron_30", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params_elementwiseneuron_30, // Node Params
                         3, // Num Node Params
                         inputs_elementwiseneuron_30, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_elementwiseneuron_30, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_elementwiseneuron_32(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR elementwiseneuron_32 */
  Qnn_Param_t params_elementwiseneuron_32[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="max_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 540.000000000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="min_value",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 0.000000000000f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 5}}}}
  };
  const char*  inputs_elementwiseneuron_32[] = {
    "_Slice_5_output_0"
  };
  uint32_t dimensions__Max_1_output_0[] = {49074, 2};
  Qnn_Tensor_t outputs_elementwiseneuron_32[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Max_1_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Max_1_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "elementwiseneuron_32", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params_elementwiseneuron_32, // Node Params
                         3, // Num Node Params
                         inputs_elementwiseneuron_32, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_elementwiseneuron_32, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Unsqueeze_8(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Unsqueeze_8 */
  const char*  inputs__Unsqueeze_8[] = {
    "_Max_output_0"
  };
  uint32_t dimensions__Unsqueeze_8_output_0[] = {49074, 2, 1};
  Qnn_Tensor_t outputs__Unsqueeze_8[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Unsqueeze_8_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__Unsqueeze_8_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Unsqueeze_8", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__Unsqueeze_8, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Unsqueeze_8, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Unsqueeze_9(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Unsqueeze_9 */
  const char*  inputs__Unsqueeze_9[] = {
    "_Max_1_output_0"
  };
  uint32_t dimensions__Unsqueeze_9_output_0[] = {49074, 2, 1};
  Qnn_Tensor_t outputs__Unsqueeze_9[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Unsqueeze_9_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__Unsqueeze_9_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Unsqueeze_9", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__Unsqueeze_9, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Unsqueeze_9, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Concat_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Concat_1 */
  Qnn_Param_t params__Concat_1[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="axis",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 2}}}}
  };
  const char*  inputs__Concat_1[] = {
    "_Unsqueeze_8_output_0",
    "_Unsqueeze_9_output_0"
  };
  uint32_t dimensions__Concat_1_output_0[] = {49074, 2, 2};
  Qnn_Tensor_t outputs__Concat_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Concat_1_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__Concat_1_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Concat_1", // Node Name
                         "qti.aisw", // Package Name
                         "Concat", // Qnn Node Type
                         params__Concat_1, // Node Params
                         1, // Num Node Params
                         inputs__Concat_1, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Concat_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Reshape(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Reshape */
  const char*  inputs__Reshape[] = {
    "_Concat_1_output_0"
  };
  uint32_t dimensions__Reshape_output_0[] = {49074, 4};
  Qnn_Tensor_t outputs__Reshape[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Reshape_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Reshape_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Reshape", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__Reshape, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Reshape, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor__Constant_1_output_0(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions__Constant_1_output_0[] = {1};
  VALIDATE(model.addTensor("_Constant_1_output_0", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "_Constant_1_output_0",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_INT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions__Constant_1_output_0,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(_Constant_1_output_0),
                                                .dataSize=BINLEN(_Constant_1_output_0)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__Gather_6(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Gather_6 */
  Qnn_Param_t params__Gather_6[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="axis",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_INT_32, {.int32Value = 1}}}}
  };
  const char*  inputs__Gather_6[] = {
    "_Squeeze_2_output_0",
    "_Constant_1_output_0"
  };
  uint32_t dimensions__Gather_6_output_0_pre_reshape[] = {49074, 1};
  Qnn_Tensor_t outputs__Gather_6[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Gather_6_output_0_pre_reshape",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Gather_6_output_0_pre_reshape,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Gather_6", // Node Name
                         "qti.aisw", // Package Name
                         "Gather", // Qnn Node Type
                         params__Gather_6, // Node Params
                         1, // Num Node Params
                         inputs__Gather_6, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Gather_6, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_Reshape_post__Gather_6(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR Reshape_post__Gather_6 */
  const char*  inputs_Reshape_post__Gather_6[] = {
    "_Gather_6_output_0_pre_reshape"
  };
  uint32_t dimensions__Gather_6_output_0[] = {49074};
  Qnn_Tensor_t outputs_Reshape_post__Gather_6[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Gather_6_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__Gather_6_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "Reshape_post__Gather_6", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs_Reshape_post__Gather_6, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_Reshape_post__Gather_6, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor__Constant_43_output_0(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions__Constant_43_output_0[] = {1};
  VALIDATE(model.addTensor("_Constant_43_output_0", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "_Constant_43_output_0",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions__Constant_43_output_0,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(_Constant_43_output_0),
                                                .dataSize=BINLEN(_Constant_43_output_0)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__Greater(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Greater */
  Qnn_Param_t params__Greater[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 6}}}}
  };
  const char*  inputs__Greater[] = {
    "_Gather_6_output_0",
    "_Constant_43_output_0"
  };
  uint32_t dimensions__Greater_output_0[] = {49074};
  Qnn_Tensor_t outputs__Greater[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Greater_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_BOOL_8,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__Greater_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Greater", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__Greater, // Node Params
                         1, // Num Node Params
                         inputs__Greater, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Greater, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__NonZero(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _NonZero */
  const char*  inputs__NonZero[] = {
    "_Greater_output_0"
  };
  uint32_t dimensions__Transpose_output_0[] = {49074, 1};
  Qnn_Tensor_t outputs__NonZero[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Transpose_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_INT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Transpose_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_NonZero", // Node Name
                         "qti.aisw", // Package Name
                         "NonZero", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__NonZero, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__NonZero, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__GatherND(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _GatherND */
  Qnn_Param_t params__GatherND[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="batch_dims",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__GatherND[] = {
    "_Gather_6_output_0",
    "_Transpose_output_0"
  };
  uint32_t dimensions__GatherND_output_0[] = {49074};
  Qnn_Tensor_t outputs__GatherND[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_GatherND_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__GatherND_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_GatherND", // Node Name
                         "qti.aisw", // Package Name
                         "GatherNd", // Qnn Node Type
                         params__GatherND, // Node Params
                         1, // Num Node Params
                         inputs__GatherND, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__GatherND, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__GatherND_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _GatherND_1 */
  Qnn_Param_t params__GatherND_1[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="batch_dims",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__GatherND_1[] = {
    "_Reshape_output_0",
    "_Transpose_output_0"
  };
  uint32_t dimensions__GatherND_1_output_0[] = {49074, 4};
  Qnn_Tensor_t outputs__GatherND_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_GatherND_1_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__GatherND_1_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_GatherND_1", // Node Name
                         "qti.aisw", // Package Name
                         "GatherNd", // Qnn Node Type
                         params__GatherND_1, // Node Params
                         1, // Num Node Params
                         inputs__GatherND_1, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__GatherND_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__TopK(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _TopK */
  Qnn_Param_t params__TopK[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="k",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 400}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="largest",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 1}}}}
  };
  const char*  inputs__TopK[] = {
    "_GatherND_output_0"
  };
  uint32_t dimensions__TopK_output_0[] = {400};
  uint32_t dimensions__TopK_output_1[] = {400};
  Qnn_Tensor_t outputs__TopK[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_TopK_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__TopK_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}},
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_TopK_output_1",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_INT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__TopK_output_1,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_TopK", // Node Name
                         "qti.aisw", // Package Name
                         "TopK", // Qnn Node Type
                         params__TopK, // Node Params
                         2, // Num Node Params
                         inputs__TopK, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__TopK, // Output Tensors 
                         2// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Gather_8(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Gather_8 */
  Qnn_Param_t params__Gather_8[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="axis",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_INT_32, {.int32Value = 0}}}}
  };
  const char*  inputs__Gather_8[] = {
    "_GatherND_1_output_0",
    "_TopK_output_1"
  };
  uint32_t dimensions__Gather_8_output_0[] = {400, 4};
  Qnn_Tensor_t outputs__Gather_8[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Gather_8_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Gather_8_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Gather_8", // Node Name
                         "qti.aisw", // Package Name
                         "Gather", // Qnn Node Type
                         params__Gather_8, // Node Params
                         1, // Num Node Params
                         inputs__Gather_8, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Gather_8, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor__Constant_2_output_0(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions__Constant_2_output_0[] = {1};
  VALIDATE(model.addTensor("_Constant_2_output_0", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "_Constant_2_output_0",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_INT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions__Constant_2_output_0,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(_Constant_2_output_0),
                                                .dataSize=BINLEN(_Constant_2_output_0)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__Gather_9(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Gather_9 */
  Qnn_Param_t params__Gather_9[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="axis",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_INT_32, {.int32Value = 1}}}}
  };
  const char*  inputs__Gather_9[] = {
    "_Squeeze_2_output_0",
    "_Constant_2_output_0"
  };
  uint32_t dimensions__Gather_9_output_0_pre_reshape[] = {49074, 1};
  Qnn_Tensor_t outputs__Gather_9[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Gather_9_output_0_pre_reshape",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Gather_9_output_0_pre_reshape,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Gather_9", // Node Name
                         "qti.aisw", // Package Name
                         "Gather", // Qnn Node Type
                         params__Gather_9, // Node Params
                         1, // Num Node Params
                         inputs__Gather_9, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Gather_9, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_Reshape_post__Gather_9(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR Reshape_post__Gather_9 */
  const char*  inputs_Reshape_post__Gather_9[] = {
    "_Gather_9_output_0_pre_reshape"
  };
  uint32_t dimensions__Gather_9_output_0[] = {49074};
  Qnn_Tensor_t outputs_Reshape_post__Gather_9[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Gather_9_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__Gather_9_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "Reshape_post__Gather_9", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs_Reshape_post__Gather_9, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_Reshape_post__Gather_9, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Greater_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Greater_1 */
  Qnn_Param_t params__Greater_1[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 6}}}}
  };
  const char*  inputs__Greater_1[] = {
    "_Gather_9_output_0",
    "_Constant_43_output_0"
  };
  uint32_t dimensions__Greater_1_output_0[] = {49074};
  Qnn_Tensor_t outputs__Greater_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Greater_1_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_BOOL_8,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__Greater_1_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Greater_1", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__Greater_1, // Node Params
                         1, // Num Node Params
                         inputs__Greater_1, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Greater_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__NonZero_2(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _NonZero_2 */
  const char*  inputs__NonZero_2[] = {
    "_Greater_1_output_0"
  };
  uint32_t dimensions__Transpose_2_output_0[] = {49074, 1};
  Qnn_Tensor_t outputs__NonZero_2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Transpose_2_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_INT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Transpose_2_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_NonZero_2", // Node Name
                         "qti.aisw", // Package Name
                         "NonZero", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__NonZero_2, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__NonZero_2, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__GatherND_2(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _GatherND_2 */
  Qnn_Param_t params__GatherND_2[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="batch_dims",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__GatherND_2[] = {
    "_Gather_9_output_0",
    "_Transpose_2_output_0"
  };
  uint32_t dimensions__GatherND_2_output_0[] = {49074};
  Qnn_Tensor_t outputs__GatherND_2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_GatherND_2_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__GatherND_2_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_GatherND_2", // Node Name
                         "qti.aisw", // Package Name
                         "GatherNd", // Qnn Node Type
                         params__GatherND_2, // Node Params
                         1, // Num Node Params
                         inputs__GatherND_2, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__GatherND_2, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__GatherND_3(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _GatherND_3 */
  Qnn_Param_t params__GatherND_3[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="batch_dims",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__GatherND_3[] = {
    "_Reshape_output_0",
    "_Transpose_2_output_0"
  };
  uint32_t dimensions__GatherND_3_output_0[] = {49074, 4};
  Qnn_Tensor_t outputs__GatherND_3[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_GatherND_3_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__GatherND_3_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_GatherND_3", // Node Name
                         "qti.aisw", // Package Name
                         "GatherNd", // Qnn Node Type
                         params__GatherND_3, // Node Params
                         1, // Num Node Params
                         inputs__GatherND_3, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__GatherND_3, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__TopK_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _TopK_1 */
  Qnn_Param_t params__TopK_1[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="k",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 400}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="largest",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 1}}}}
  };
  const char*  inputs__TopK_1[] = {
    "_GatherND_2_output_0"
  };
  uint32_t dimensions__TopK_1_output_0[] = {400};
  uint32_t dimensions__TopK_1_output_1[] = {400};
  Qnn_Tensor_t outputs__TopK_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_TopK_1_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__TopK_1_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}},
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_TopK_1_output_1",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_INT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__TopK_1_output_1,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_TopK_1", // Node Name
                         "qti.aisw", // Package Name
                         "TopK", // Qnn Node Type
                         params__TopK_1, // Node Params
                         2, // Num Node Params
                         inputs__TopK_1, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__TopK_1, // Output Tensors 
                         2// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Gather_11(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Gather_11 */
  Qnn_Param_t params__Gather_11[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="axis",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_INT_32, {.int32Value = 0}}}}
  };
  const char*  inputs__Gather_11[] = {
    "_GatherND_3_output_0",
    "_TopK_1_output_1"
  };
  uint32_t dimensions__Gather_11_output_0[] = {400, 4};
  Qnn_Tensor_t outputs__Gather_11[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Gather_11_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Gather_11_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Gather_11", // Node Name
                         "qti.aisw", // Package Name
                         "Gather", // Qnn Node Type
                         params__Gather_11, // Node Params
                         1, // Num Node Params
                         inputs__Gather_11, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Gather_11, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor__transform_Constant_13_output_0(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions__transform_Constant_13_output_0[] = {1};
  VALIDATE(model.addTensor("_transform_Constant_13_output_0", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "_transform_Constant_13_output_0",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_INT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions__transform_Constant_13_output_0,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(_transform_Constant_13_output_0),
                                                .dataSize=BINLEN(_transform_Constant_13_output_0)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__Gather_12(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Gather_12 */
  Qnn_Param_t params__Gather_12[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="axis",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_INT_32, {.int32Value = 1}}}}
  };
  const char*  inputs__Gather_12[] = {
    "_Squeeze_2_output_0",
    "_transform_Constant_13_output_0"
  };
  uint32_t dimensions__Gather_12_output_0_pre_reshape[] = {49074, 1};
  Qnn_Tensor_t outputs__Gather_12[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Gather_12_output_0_pre_reshape",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Gather_12_output_0_pre_reshape,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Gather_12", // Node Name
                         "qti.aisw", // Package Name
                         "Gather", // Qnn Node Type
                         params__Gather_12, // Node Params
                         1, // Num Node Params
                         inputs__Gather_12, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Gather_12, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_Reshape_post__Gather_12(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR Reshape_post__Gather_12 */
  const char*  inputs_Reshape_post__Gather_12[] = {
    "_Gather_12_output_0_pre_reshape"
  };
  uint32_t dimensions__Gather_12_output_0[] = {49074};
  Qnn_Tensor_t outputs_Reshape_post__Gather_12[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Gather_12_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__Gather_12_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "Reshape_post__Gather_12", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs_Reshape_post__Gather_12, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_Reshape_post__Gather_12, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Greater_2(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Greater_2 */
  Qnn_Param_t params__Greater_2[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 6}}}}
  };
  const char*  inputs__Greater_2[] = {
    "_Gather_12_output_0",
    "_Constant_43_output_0"
  };
  uint32_t dimensions__Greater_2_output_0[] = {49074};
  Qnn_Tensor_t outputs__Greater_2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Greater_2_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_BOOL_8,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__Greater_2_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Greater_2", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__Greater_2, // Node Params
                         1, // Num Node Params
                         inputs__Greater_2, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Greater_2, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__NonZero_4(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _NonZero_4 */
  const char*  inputs__NonZero_4[] = {
    "_Greater_2_output_0"
  };
  uint32_t dimensions__Transpose_4_output_0[] = {49074, 1};
  Qnn_Tensor_t outputs__NonZero_4[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Transpose_4_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_INT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Transpose_4_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_NonZero_4", // Node Name
                         "qti.aisw", // Package Name
                         "NonZero", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__NonZero_4, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__NonZero_4, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__GatherND_4(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _GatherND_4 */
  Qnn_Param_t params__GatherND_4[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="batch_dims",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__GatherND_4[] = {
    "_Gather_12_output_0",
    "_Transpose_4_output_0"
  };
  uint32_t dimensions__GatherND_4_output_0[] = {49074};
  Qnn_Tensor_t outputs__GatherND_4[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_GatherND_4_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__GatherND_4_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_GatherND_4", // Node Name
                         "qti.aisw", // Package Name
                         "GatherNd", // Qnn Node Type
                         params__GatherND_4, // Node Params
                         1, // Num Node Params
                         inputs__GatherND_4, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__GatherND_4, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__GatherND_5(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _GatherND_5 */
  Qnn_Param_t params__GatherND_5[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="batch_dims",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__GatherND_5[] = {
    "_Reshape_output_0",
    "_Transpose_4_output_0"
  };
  uint32_t dimensions__GatherND_5_output_0[] = {49074, 4};
  Qnn_Tensor_t outputs__GatherND_5[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_GatherND_5_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__GatherND_5_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_GatherND_5", // Node Name
                         "qti.aisw", // Package Name
                         "GatherNd", // Qnn Node Type
                         params__GatherND_5, // Node Params
                         1, // Num Node Params
                         inputs__GatherND_5, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__GatherND_5, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__TopK_2(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _TopK_2 */
  Qnn_Param_t params__TopK_2[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="k",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 400}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="largest",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 1}}}}
  };
  const char*  inputs__TopK_2[] = {
    "_GatherND_4_output_0"
  };
  uint32_t dimensions__TopK_2_output_0[] = {400};
  uint32_t dimensions__TopK_2_output_1[] = {400};
  Qnn_Tensor_t outputs__TopK_2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_TopK_2_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__TopK_2_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}},
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_TopK_2_output_1",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_INT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__TopK_2_output_1,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_TopK_2", // Node Name
                         "qti.aisw", // Package Name
                         "TopK", // Qnn Node Type
                         params__TopK_2, // Node Params
                         2, // Num Node Params
                         inputs__TopK_2, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__TopK_2, // Output Tensors 
                         2// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Gather_14(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Gather_14 */
  Qnn_Param_t params__Gather_14[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="axis",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_INT_32, {.int32Value = 0}}}}
  };
  const char*  inputs__Gather_14[] = {
    "_GatherND_5_output_0",
    "_TopK_2_output_1"
  };
  uint32_t dimensions__Gather_14_output_0[] = {400, 4};
  Qnn_Tensor_t outputs__Gather_14[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Gather_14_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Gather_14_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Gather_14", // Node Name
                         "qti.aisw", // Package Name
                         "Gather", // Qnn Node Type
                         params__Gather_14, // Node Params
                         1, // Num Node Params
                         inputs__Gather_14, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Gather_14, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Concat_6(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Concat_6 */
  Qnn_Param_t params__Concat_6[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="axis",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__Concat_6[] = {
    "_Gather_8_output_0",
    "_Gather_11_output_0",
    "_Gather_14_output_0"
  };
  uint32_t dimensions__Concat_6_output_0[] = {1200, 4};
  Qnn_Tensor_t outputs__Concat_6[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Concat_6_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Concat_6_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Concat_6", // Node Name
                         "qti.aisw", // Package Name
                         "Concat", // Qnn Node Type
                         params__Concat_6, // Node Params
                         1, // Num Node Params
                         inputs__Concat_6, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__Concat_6, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Concat_7(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Concat_7 */
  Qnn_Param_t params__Concat_7[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="axis",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__Concat_7[] = {
    "_TopK_output_0",
    "_TopK_1_output_0",
    "_TopK_2_output_0"
  };
  uint32_t dimensions__Concat_7_output_0[] = {1200};
  Qnn_Tensor_t outputs__Concat_7[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Concat_7_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__Concat_7_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Concat_7", // Node Name
                         "qti.aisw", // Package Name
                         "Concat", // Qnn Node Type
                         params__Concat_7, // Node Params
                         1, // Num Node Params
                         inputs__Concat_7, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__Concat_7, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_ReduceMax_1317(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR ReduceMax_1317 */
  uint32_t dimensions_ReduceMax_1317_axes[] = {2};
  uint32_t ReduceMax_1317_axes[] = {0, 1};
  Qnn_Param_t params_ReduceMax_1317[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="axes",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "ReduceMax_1317_axes",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_ReduceMax_1317_axes,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)ReduceMax_1317_axes,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="keep_dims",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs_ReduceMax_1317[] = {
    "_Concat_6_output_0"
  };
  uint32_t dimensions_max_coordinate[] = {1};
  Qnn_Tensor_t outputs_ReduceMax_1317[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "max_coordinate",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_max_coordinate,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "ReduceMax_1317", // Node Name
                         "qti.aisw", // Package Name
                         "ReduceMax", // Qnn Node Type
                         params_ReduceMax_1317, // Node Params
                         2, // Num Node Params
                         inputs_ReduceMax_1317, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_ReduceMax_1317, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor__1312(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions__1312[] = {1};
  VALIDATE(model.addTensor("_1312", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "_1312",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions__1312,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(_1312),
                                                .dataSize=BINLEN(_1312)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode_Add_1320(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR Add_1320 */
  Qnn_Param_t params_Add_1320[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs_Add_1320[] = {
    "max_coordinate",
    "_1312"
  };
  uint32_t dimensions__1313[] = {1};
  Qnn_Tensor_t outputs_Add_1320[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_1313",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__1313,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "Add_1320", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params_Add_1320, // Node Params
                         1, // Num Node Params
                         inputs_Add_1320, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_Add_1320, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor__1311(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions__1311[] = {1200};
  VALIDATE(model.addTensor("_1311", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "_1311",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions__1311,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(_1311),
                                                .dataSize=BINLEN(_1311)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode_Mul_1321(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR Mul_1321 */
  Qnn_Param_t params_Mul_1321[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 13}}}}
  };
  const char*  inputs_Mul_1321[] = {
    "_1311",
    "_1313"
  };
  uint32_t dimensions_offsets[] = {1200};
  Qnn_Tensor_t outputs_Mul_1321[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "offsets",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_offsets,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "Mul_1321", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params_Mul_1321, // Node Params
                         1, // Num Node Params
                         inputs_Mul_1321, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_Mul_1321, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_Slice_1326(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR Slice_1326 */
  uint32_t dimensions_Slice_1326_ranges[] = {1, 3};
  int32_t Slice_1326_ranges[] = {0, 1200, 1};
  Qnn_Param_t params_Slice_1326[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="ranges",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "Slice_1326_ranges",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_INT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Slice_1326_ranges,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Slice_1326_ranges,
                           .dataSize=12}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="begin_mask",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="end_mask",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="new_axes_mask",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="shrink_axes",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs_Slice_1326[] = {
    "offsets"
  };
  uint32_t dimensions__1319[] = {1200};
  Qnn_Tensor_t outputs_Slice_1326[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_1319",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__1319,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "Slice_1326", // Node Name
                         "qti.aisw", // Package Name
                         "StridedSlice", // Qnn Node Type
                         params_Slice_1326, // Node Params
                         5, // Num Node Params
                         inputs_Slice_1326, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_Slice_1326, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_Unsqueeze_1327(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR Unsqueeze_1327 */
  const char*  inputs_Unsqueeze_1327[] = {
    "_1319"
  };
  uint32_t dimensions__1320[] = {1200, 1};
  Qnn_Tensor_t outputs_Unsqueeze_1327[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_1320",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__1320,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "Unsqueeze_1327", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs_Unsqueeze_1327, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_Unsqueeze_1327, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_Add_1328(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR Add_1328 */
  Qnn_Param_t params_Add_1328[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs_Add_1328[] = {
    "_Concat_6_output_0",
    "_1320"
  };
  uint32_t dimensions_boxes_for_nms[] = {1200, 4};
  Qnn_Tensor_t outputs_Add_1328[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "boxes_for_nms",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_boxes_for_nms,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "Add_1328", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params_Add_1328, // Node Params
                         1, // Num Node Params
                         inputs_Add_1328, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_Add_1328, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_Unsqueeze_1329(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR Unsqueeze_1329 */
  const char*  inputs_Unsqueeze_1329[] = {
    "boxes_for_nms"
  };
  uint32_t dimensions__1322[] = {1, 1200, 4};
  Qnn_Tensor_t outputs_Unsqueeze_1329[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_1322",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__1322,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "Unsqueeze_1329", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs_Unsqueeze_1329, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_Unsqueeze_1329, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_Unsqueeze_1330(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR Unsqueeze_1330 */
  const char*  inputs_Unsqueeze_1330[] = {
    "_Concat_7_output_0"
  };
  uint32_t dimensions__1324[] = {1, 1, 1200};
  Qnn_Tensor_t outputs_Unsqueeze_1330[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_1324",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__1324,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "Unsqueeze_1330", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs_Unsqueeze_1330, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_Unsqueeze_1330, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_NonMaxSuppression_1336(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR NonMaxSuppression_1336 */
  Qnn_Param_t params_NonMaxSuppression_1336[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="iou_threshold",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 0.449999988079f}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="max_boxes_selected",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1200}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="score_threshold",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 0.000000000000f}}}}
  };
  const char*  inputs_NonMaxSuppression_1336[] = {
    "_1322",
    "_1324"
  };
  uint32_t dimensions__1329[] = {1200, 3};
  uint32_t dimensions__1329_valid_num_selected_indices[] = {1};
  Qnn_Tensor_t outputs_NonMaxSuppression_1336[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_1329",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_INT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__1329,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}},
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_1329_valid_num_selected_indices",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__1329_valid_num_selected_indices,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "NonMaxSuppression_1336", // Node Name
                         "qti.aisw", // Package Name
                         "NonMaxSuppression", // Qnn Node Type
                         params_NonMaxSuppression_1336, // Node Params
                         3, // Num Node Params
                         inputs_NonMaxSuppression_1336, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_NonMaxSuppression_1336, // Output Tensors 
                         2// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor__1330(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions__1330[] = {1};
  VALIDATE(model.addTensor("_1330", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "_1330",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_INT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions__1330,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(_1330),
                                                .dataSize=BINLEN(_1330)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode_Gather_1338(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR Gather_1338 */
  Qnn_Param_t params_Gather_1338[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="axis",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_INT_32, {.int32Value = 1}}}}
  };
  const char*  inputs_Gather_1338[] = {
    "_1329",
    "_1330"
  };
  uint32_t dimensions__1331[] = {1200, 1};
  Qnn_Tensor_t outputs_Gather_1338[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_1331",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_INT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__1331,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "Gather_1338", // Node Name
                         "qti.aisw", // Package Name
                         "Gather", // Qnn Node Type
                         params_Gather_1338, // Node Params
                         1, // Num Node Params
                         inputs_Gather_1338, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_Gather_1338, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_Squeeze_1339(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR Squeeze_1339 */
  const char*  inputs_Squeeze_1339[] = {
    "_1331"
  };
  uint32_t dimensions_onnx__Slice_1308[] = {1200};
  Qnn_Tensor_t outputs_Squeeze_1339[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "onnx__Slice_1308",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_INT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_onnx__Slice_1308,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "Squeeze_1339", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs_Squeeze_1339, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_Squeeze_1339, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Slice_6(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Slice_6 */
  uint32_t dimensions___Slice_6_ranges[] = {1, 3};
  int32_t __Slice_6_ranges[] = {0, 200, 1};
  Qnn_Param_t params__Slice_6[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="ranges",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__Slice_6_ranges",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_INT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions___Slice_6_ranges,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__Slice_6_ranges,
                           .dataSize=12}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="begin_mask",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="end_mask",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="new_axes_mask",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="shrink_axes",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__Slice_6[] = {
    "onnx__Slice_1308"
  };
  uint32_t dimensions__Slice_6_output_0[] = {200};
  Qnn_Tensor_t outputs__Slice_6[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Slice_6_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_INT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__Slice_6_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Slice_6", // Node Name
                         "qti.aisw", // Package Name
                         "StridedSlice", // Qnn Node Type
                         params__Slice_6, // Node Params
                         5, // Num Node Params
                         inputs__Slice_6, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Slice_6, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Gather_15(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Gather_15 */
  Qnn_Param_t params__Gather_15[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="axis",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_INT_32, {.int32Value = 0}}}}
  };
  const char*  inputs__Gather_15[] = {
    "_Concat_6_output_0",
    "_Slice_6_output_0"
  };
  uint32_t dimensions__Gather_15_output_0[] = {200, 4};
  Qnn_Tensor_t outputs__Gather_15[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Gather_15_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Gather_15_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Gather_15", // Node Name
                         "qti.aisw", // Package Name
                         "Gather", // Qnn Node Type
                         params__Gather_15, // Node Params
                         1, // Num Node Params
                         inputs__Gather_15, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Gather_15, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Gather_16(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Gather_16 */
  Qnn_Param_t params__Gather_16[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="axis",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_INT_32, {.int32Value = 0}}}}
  };
  const char*  inputs__Gather_16[] = {
    "_Concat_7_output_0",
    "_Slice_6_output_0"
  };
  uint32_t dimensions__1340[] = {200};
  Qnn_Tensor_t outputs__Gather_16[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_1340",
            .type= QNN_TENSOR_TYPE_APP_READ,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__1340,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Gather_16", // Node Name
                         "qti.aisw", // Package Name
                         "Gather", // Qnn Node Type
                         params__Gather_16, // Node Params
                         1, // Num Node Params
                         inputs__Gather_16, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Gather_16, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor__Concat_8_output_0(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions__Concat_8_output_0[] = {1200};
  VALIDATE(model.addTensor("_Concat_8_output_0", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "_Concat_8_output_0",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_INT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions__Concat_8_output_0,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(_Concat_8_output_0),
                                                .dataSize=BINLEN(_Concat_8_output_0)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__Gather_17(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Gather_17 */
  Qnn_Param_t params__Gather_17[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="axis",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_INT_32, {.int32Value = 0}}}}
  };
  const char*  inputs__Gather_17[] = {
    "_Concat_8_output_0",
    "_Slice_6_output_0"
  };
  uint32_t dimensions__1341[] = {200};
  Qnn_Tensor_t outputs__Gather_17[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_1341",
            .type= QNN_TENSOR_TYPE_APP_READ,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_INT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__1341,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Gather_17", // Node Name
                         "qti.aisw", // Package Name
                         "Gather", // Qnn Node Type
                         params__Gather_17, // Node Params
                         1, // Num Node Params
                         inputs__Gather_17, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__Gather_17, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Split_3(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Split_3 */
  uint32_t dimensions___Split_3_split_index[] = {3};
  uint32_t __Split_3_split_index[] = {1, 2, 3};
  Qnn_Param_t params__Split_3[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="split_index",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "__Split_3_split_index",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___Split_3_split_index,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__Split_3_split_index,
                           .dataSize=12}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="axis",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__Split_3[] = {
    "_Gather_15_output_0"
  };
  uint32_t dimensions__Unsqueeze_15_output_0[] = {200, 1};
  uint32_t dimensions__Unsqueeze_16_output_0[] = {200, 1};
  uint32_t dimensions__Unsqueeze_17_output_0[] = {200, 1};
  uint32_t dimensions__Unsqueeze_18_output_0[] = {200, 1};
  Qnn_Tensor_t outputs__Split_3[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Unsqueeze_15_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Unsqueeze_15_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}},
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Unsqueeze_16_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Unsqueeze_16_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}},
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Unsqueeze_17_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Unsqueeze_17_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}},
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Unsqueeze_18_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Unsqueeze_18_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Split_3", // Node Name
                         "qti.aisw", // Package Name
                         "Split", // Qnn Node Type
                         params__Split_3, // Node Params
                         2, // Num Node Params
                         inputs__Split_3, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__Split_3, // Output Tensors 
                         4// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__Concat_9(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _Concat_9 */
  Qnn_Param_t params__Concat_9[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="axis",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__Concat_9[] = {
    "_Unsqueeze_15_output_0",
    "_Unsqueeze_16_output_0",
    "_Unsqueeze_17_output_0",
    "_Unsqueeze_18_output_0"
  };
  uint32_t dimensions__1362[] = {200, 4};
  Qnn_Tensor_t outputs__Concat_9[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_1362",
            .type= QNN_TENSOR_TYPE_APP_READ,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__1362,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_Concat_9", // Node Name
                         "qti.aisw", // Package Name
                         "Concat", // Qnn Node Type
                         params__Concat_9, // Node Params
                         1, // Num Node Params
                         inputs__Concat_9, // Input Tensor Names
                         4, // Num Input Tensor Names
                         outputs__Concat_9, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

QNN_API
ModelError_t QnnModel_composeGraphs(Qnn_BackendHandle_t backendHandle,
                                    QNN_INTERFACE_VER_TYPE interface,
                                    Qnn_ContextHandle_t contextHandle,
                                    const GraphConfigInfo_t** graphsConfigInfo,
                                    const uint32_t numGraphsConfigInfo,
                                    GraphInfoPtr_t** graphsInfo,
                                    uint32_t* numGraphsInfo,
                                    bool debug,
                                    QnnLog_Callback_t logCallback,
                                    QnnLog_Level_t maxLogLevel) {

  ModelError_t err = MODEL_NO_ERROR;

  /* model/graph for carbspray_2_v3_model*/
  QnnModel carbspray_2_v3_model;
  const QnnGraph_Config_t** graphConfigs = nullptr;
  VALIDATE(getQnnGraphConfigFromInfo("carbspray_2_v3_model", graphsConfigInfo, numGraphsConfigInfo, graphConfigs), err);
  VALIDATE(carbspray_2_v3_model.initialize(backendHandle, interface, contextHandle, "carbspray_2_v3_model", debug, DO_GRAPH_NODE_VALIDATIONS, graphConfigs), err);
  VALIDATE(addTensor_images(carbspray_2_v3_model), err);
  VALIDATE(addNode_images_nchw(carbspray_2_v3_model), err);
  VALIDATE(addNode__Squeeze(carbspray_2_v3_model), err);
  VALIDATE(addTensor__transform_Constant_output_0(carbspray_2_v3_model), err);
  VALIDATE(addNode__transform_Sub(carbspray_2_v3_model), err);
  VALIDATE(addTensor__transform_Constant_1_output_0(carbspray_2_v3_model), err);
  VALIDATE(addNode__transform_Div(carbspray_2_v3_model), err);
  VALIDATE(addNode__transform_Unsqueeze(carbspray_2_v3_model), err);
  VALIDATE(addNode__transform_Unsqueeze_output_0_nhwc(carbspray_2_v3_model), err);
  VALIDATE(addNode__transform_Resize(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1364(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1365(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_conv1_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_relu_Relu(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_maxpool_MaxPool(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1367(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1368(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer1_layer1_0_conv1_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer1_layer1_0_relu_Relu(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1370(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1371(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer1_layer1_0_conv2_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer1_layer1_0_Add(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer1_layer1_0_relu_1_Relu(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1373(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1374(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer1_layer1_1_conv1_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer1_layer1_1_relu_Relu(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1376(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1377(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer1_layer1_1_conv2_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer1_layer1_1_Add(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer1_layer1_1_relu_1_Relu(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1379(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1380(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer2_layer2_0_conv1_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer2_layer2_0_relu_Relu(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1382(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1383(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer2_layer2_0_conv2_Conv(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1385(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1386(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer2_layer2_0_downsample_downsample_0_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer2_layer2_0_Add(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer2_layer2_0_relu_1_Relu(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1388(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1389(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer2_layer2_1_conv1_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer2_layer2_1_relu_Relu(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1391(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1392(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer2_layer2_1_conv2_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer2_layer2_1_Add(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer2_layer2_1_relu_1_Relu(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1394(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1395(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer3_layer3_0_conv1_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer3_layer3_0_relu_Relu(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1397(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1398(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer3_layer3_0_conv2_Conv(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1400(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1401(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer3_layer3_0_downsample_downsample_0_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer3_layer3_0_Add(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer3_layer3_0_relu_1_Relu(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1403(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1404(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer3_layer3_1_conv1_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer3_layer3_1_relu_Relu(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1406(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1407(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer3_layer3_1_conv2_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer3_layer3_1_Add(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer3_layer3_1_relu_1_Relu(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1409(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1410(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer4_layer4_0_conv1_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer4_layer4_0_relu_Relu(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1412(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1413(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer4_layer4_0_conv2_Conv(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1415(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1416(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer4_layer4_0_downsample_downsample_0_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer4_layer4_0_Add(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer4_layer4_0_relu_1_Relu(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1418(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1419(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer4_layer4_1_conv1_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer4_layer4_1_relu_Relu(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1421(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1422(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer4_layer4_1_conv2_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer4_layer4_1_Add(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_base_layer4_layer4_1_relu_1_Relu(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1424(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1425(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_fpn_inner_blocks_2_inner_blocks_2_0_Conv(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1427(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1428(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_fpn_layer_blocks_2_layer_blocks_2_0_Conv(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1430(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1431(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_fpn_inner_blocks_1_inner_blocks_1_0_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_fpn_Resize(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_fpn_Add(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1433(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1434(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_fpn_layer_blocks_1_layer_blocks_1_0_Conv(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1436(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1437(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_fpn_inner_blocks_0_inner_blocks_0_0_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_fpn_Resize_1(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_fpn_Add_1(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1439(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1440(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_fpn_layer_blocks_0_layer_blocks_0_0_Conv(carbspray_2_v3_model), err);
  VALIDATE(addTensor_backbone_fpn_extra_blocks_p6_weight(carbspray_2_v3_model), err);
  VALIDATE(addTensor_backbone_fpn_extra_blocks_p6_bias(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_fpn_extra_blocks_p6_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_fpn_extra_blocks_Relu(carbspray_2_v3_model), err);
  VALIDATE(addTensor_backbone_fpn_extra_blocks_p7_weight(carbspray_2_v3_model), err);
  VALIDATE(addTensor_backbone_fpn_extra_blocks_p7_bias(carbspray_2_v3_model), err);
  VALIDATE(addNode__backbone_fpn_extra_blocks_p7_Conv(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1442(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1443(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_regression_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_regression_head_module_list_0_module_list_0_0_module_list_0_0_2_Clip(carbspray_2_v3_model), err);
  VALIDATE(addTensor_head_regression_head_module_list_0_1_weight(carbspray_2_v3_model), err);
  VALIDATE(addTensor_head_regression_head_module_list_0_1_bias(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_regression_head_module_list_0_module_list_0_1_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_regression_head_module_list_0_module_list_0_1_Conv_output_0_nchw(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_regression_head_Reshape(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_regression_head_Transpose(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_regression_head_Reshape_1(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1445(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1446(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_regression_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_regression_head_module_list_1_module_list_1_0_module_list_1_0_2_Clip(carbspray_2_v3_model), err);
  VALIDATE(addTensor_head_regression_head_module_list_1_1_weight(carbspray_2_v3_model), err);
  VALIDATE(addTensor_head_regression_head_module_list_1_1_bias(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_regression_head_module_list_1_module_list_1_1_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_regression_head_module_list_1_module_list_1_1_Conv_output_0_nchw(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_regression_head_Reshape_2(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_regression_head_Transpose_1(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_regression_head_Reshape_3(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1448(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1449(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_regression_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_regression_head_module_list_2_module_list_2_0_module_list_2_0_2_Clip(carbspray_2_v3_model), err);
  VALIDATE(addTensor_head_regression_head_module_list_2_1_weight(carbspray_2_v3_model), err);
  VALIDATE(addTensor_head_regression_head_module_list_2_1_bias(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_regression_head_module_list_2_module_list_2_1_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_regression_head_module_list_2_module_list_2_1_Conv_output_0_nchw(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_regression_head_Reshape_4(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_regression_head_Transpose_2(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_regression_head_Reshape_5(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1452(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_regression_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_regression_head_module_list_3_module_list_3_0_module_list_3_0_2_Clip(carbspray_2_v3_model), err);
  VALIDATE(addTensor_head_regression_head_module_list_3_1_weight(carbspray_2_v3_model), err);
  VALIDATE(addTensor_head_regression_head_module_list_3_1_bias(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_regression_head_module_list_3_module_list_3_1_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_regression_head_module_list_3_module_list_3_1_Conv_output_0_nchw(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_regression_head_Reshape_6(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_regression_head_Transpose_3(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_regression_head_Reshape_7(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1455(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_regression_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_regression_head_module_list_4_module_list_4_0_module_list_4_0_2_Clip(carbspray_2_v3_model), err);
  VALIDATE(addTensor_head_regression_head_module_list_4_1_weight(carbspray_2_v3_model), err);
  VALIDATE(addTensor_head_regression_head_module_list_4_1_bias(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_regression_head_module_list_4_module_list_4_1_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_regression_head_module_list_4_module_list_4_1_Conv_output_0_nchw(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_regression_head_Reshape_8(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_regression_head_Transpose_4(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_regression_head_Reshape_9(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_regression_head_Concat_10(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1457(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1458(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_classification_head_module_list_0_module_list_0_0_module_list_0_0_0_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_classification_head_module_list_0_module_list_0_0_module_list_0_0_2_Clip(carbspray_2_v3_model), err);
  VALIDATE(addTensor_head_classification_head_module_list_0_1_weight(carbspray_2_v3_model), err);
  VALIDATE(addTensor_head_classification_head_module_list_0_1_bias(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_classification_head_module_list_0_module_list_0_1_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_classification_head_module_list_0_module_list_0_1_Conv_output_0_nchw(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_classification_head_Reshape(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_classification_head_Transpose(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_classification_head_Reshape_1(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1460(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1461(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_classification_head_module_list_1_module_list_1_0_module_list_1_0_0_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_classification_head_module_list_1_module_list_1_0_module_list_1_0_2_Clip(carbspray_2_v3_model), err);
  VALIDATE(addTensor_head_classification_head_module_list_1_1_weight(carbspray_2_v3_model), err);
  VALIDATE(addTensor_head_classification_head_module_list_1_1_bias(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_classification_head_module_list_1_module_list_1_1_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_classification_head_module_list_1_module_list_1_1_Conv_output_0_nchw(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_classification_head_Reshape_2(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_classification_head_Transpose_1(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_classification_head_Reshape_3(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1463(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1464(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_classification_head_module_list_2_module_list_2_0_module_list_2_0_0_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_classification_head_module_list_2_module_list_2_0_module_list_2_0_2_Clip(carbspray_2_v3_model), err);
  VALIDATE(addTensor_head_classification_head_module_list_2_1_weight(carbspray_2_v3_model), err);
  VALIDATE(addTensor_head_classification_head_module_list_2_1_bias(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_classification_head_module_list_2_module_list_2_1_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_classification_head_module_list_2_module_list_2_1_Conv_output_0_nchw(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_classification_head_Reshape_4(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_classification_head_Transpose_2(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_classification_head_Reshape_5(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1467(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_classification_head_module_list_3_module_list_3_0_module_list_3_0_0_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_classification_head_module_list_3_module_list_3_0_module_list_3_0_2_Clip(carbspray_2_v3_model), err);
  VALIDATE(addTensor_head_classification_head_module_list_3_1_weight(carbspray_2_v3_model), err);
  VALIDATE(addTensor_head_classification_head_module_list_3_1_bias(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_classification_head_module_list_3_module_list_3_1_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_classification_head_module_list_3_module_list_3_1_Conv_output_0_nchw(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_classification_head_Reshape_6(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_classification_head_Transpose_3(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_classification_head_Reshape_7(carbspray_2_v3_model), err);
  VALIDATE(addTensor_onnx__Conv_1470(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_classification_head_module_list_4_module_list_4_0_module_list_4_0_0_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_classification_head_module_list_4_module_list_4_0_module_list_4_0_2_Clip(carbspray_2_v3_model), err);
  VALIDATE(addTensor_head_classification_head_module_list_4_1_weight(carbspray_2_v3_model), err);
  VALIDATE(addTensor_head_classification_head_module_list_4_1_bias(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_classification_head_module_list_4_module_list_4_1_Conv(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_classification_head_module_list_4_module_list_4_1_Conv_output_0_nchw(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_classification_head_Reshape_8(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_classification_head_Transpose_4(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_classification_head_Reshape_9(carbspray_2_v3_model), err);
  VALIDATE(addNode__head_classification_head_Concat_10(carbspray_2_v3_model), err);
  VALIDATE(addNode__Softmax(carbspray_2_v3_model), err);
  VALIDATE(addNode__Softmax_output_0_reshape(carbspray_2_v3_model), err);
  VALIDATE(addNode__Squeeze_1(carbspray_2_v3_model), err);
  VALIDATE(addNode__Slice(carbspray_2_v3_model), err);
  VALIDATE(addTensor__Constant_9_output_0(carbspray_2_v3_model), err);
  VALIDATE(addNode__Div(carbspray_2_v3_model), err);
  VALIDATE(addNode__Slice_1(carbspray_2_v3_model), err);
  VALIDATE(addNode__Div_1(carbspray_2_v3_model), err);
  VALIDATE(addNode__Slice_2(carbspray_2_v3_model), err);
  VALIDATE(addTensor__Constant_19_output_0(carbspray_2_v3_model), err);
  VALIDATE(addNode__Div_2(carbspray_2_v3_model), err);
  VALIDATE(addNode__Slice_3(carbspray_2_v3_model), err);
  VALIDATE(addNode__Div_3(carbspray_2_v3_model), err);
  VALIDATE(addNode__Clip(carbspray_2_v3_model), err);
  VALIDATE(addNode__Clip_1(carbspray_2_v3_model), err);
  VALIDATE(addTensor__Unsqueeze_output_0(carbspray_2_v3_model), err);
  VALIDATE(addNode__Mul_2(carbspray_2_v3_model), err);
  VALIDATE(addTensor__Unsqueeze_1_output_0(carbspray_2_v3_model), err);
  VALIDATE(addNode__Add_2(carbspray_2_v3_model), err);
  VALIDATE(addTensor__Unsqueeze_2_output_0(carbspray_2_v3_model), err);
  VALIDATE(addNode__Mul_3(carbspray_2_v3_model), err);
  VALIDATE(addTensor__Unsqueeze_3_output_0(carbspray_2_v3_model), err);
  VALIDATE(addNode__Add_3(carbspray_2_v3_model), err);
  VALIDATE(addNode__Exp(carbspray_2_v3_model), err);
  VALIDATE(addNode__Mul_4(carbspray_2_v3_model), err);
  VALIDATE(addNode__Exp_1(carbspray_2_v3_model), err);
  VALIDATE(addNode__Mul_5(carbspray_2_v3_model), err);
  VALIDATE(addTensor__anchor_generator_Constant_12_output_0(carbspray_2_v3_model), err);
  VALIDATE(addNode__Mul_6(carbspray_2_v3_model), err);
  VALIDATE(addNode__Mul_7(carbspray_2_v3_model), err);
  VALIDATE(addNode__Sub_2(carbspray_2_v3_model), err);
  VALIDATE(addNode__Sub_3(carbspray_2_v3_model), err);
  VALIDATE(addNode__Add_4(carbspray_2_v3_model), err);
  VALIDATE(addNode__Add_5(carbspray_2_v3_model), err);
  VALIDATE(addNode__Unsqueeze_4(carbspray_2_v3_model), err);
  VALIDATE(addNode__Unsqueeze_5(carbspray_2_v3_model), err);
  VALIDATE(addNode__Unsqueeze_6(carbspray_2_v3_model), err);
  VALIDATE(addNode__Unsqueeze_7(carbspray_2_v3_model), err);
  VALIDATE(addNode__Concat(carbspray_2_v3_model), err);
  VALIDATE(addNode__Flatten(carbspray_2_v3_model), err);
  VALIDATE(addNode__Slice_4(carbspray_2_v3_model), err);
  VALIDATE(addNode__Slice_5(carbspray_2_v3_model), err);
  VALIDATE(addNode_elementwiseneuron_30(carbspray_2_v3_model), err);
  VALIDATE(addNode_elementwiseneuron_32(carbspray_2_v3_model), err);
  VALIDATE(addNode__Unsqueeze_8(carbspray_2_v3_model), err);
  VALIDATE(addNode__Unsqueeze_9(carbspray_2_v3_model), err);
  VALIDATE(addNode__Concat_1(carbspray_2_v3_model), err);
  VALIDATE(addNode__Reshape(carbspray_2_v3_model), err);
  VALIDATE(addTensor__Constant_1_output_0(carbspray_2_v3_model), err);
  VALIDATE(addNode__Gather_6(carbspray_2_v3_model), err);
  VALIDATE(addNode_Reshape_post__Gather_6(carbspray_2_v3_model), err);
  VALIDATE(addTensor__Constant_43_output_0(carbspray_2_v3_model), err);
  VALIDATE(addNode__Greater(carbspray_2_v3_model), err);
  VALIDATE(addNode__NonZero(carbspray_2_v3_model), err);
  VALIDATE(addNode__GatherND(carbspray_2_v3_model), err);
  VALIDATE(addNode__GatherND_1(carbspray_2_v3_model), err);
  VALIDATE(addNode__TopK(carbspray_2_v3_model), err);
  VALIDATE(addNode__Gather_8(carbspray_2_v3_model), err);
  VALIDATE(addTensor__Constant_2_output_0(carbspray_2_v3_model), err);
  VALIDATE(addNode__Gather_9(carbspray_2_v3_model), err);
  VALIDATE(addNode_Reshape_post__Gather_9(carbspray_2_v3_model), err);
  VALIDATE(addNode__Greater_1(carbspray_2_v3_model), err);
  VALIDATE(addNode__NonZero_2(carbspray_2_v3_model), err);
  VALIDATE(addNode__GatherND_2(carbspray_2_v3_model), err);
  VALIDATE(addNode__GatherND_3(carbspray_2_v3_model), err);
  VALIDATE(addNode__TopK_1(carbspray_2_v3_model), err);
  VALIDATE(addNode__Gather_11(carbspray_2_v3_model), err);
  VALIDATE(addTensor__transform_Constant_13_output_0(carbspray_2_v3_model), err);
  VALIDATE(addNode__Gather_12(carbspray_2_v3_model), err);
  VALIDATE(addNode_Reshape_post__Gather_12(carbspray_2_v3_model), err);
  VALIDATE(addNode__Greater_2(carbspray_2_v3_model), err);
  VALIDATE(addNode__NonZero_4(carbspray_2_v3_model), err);
  VALIDATE(addNode__GatherND_4(carbspray_2_v3_model), err);
  VALIDATE(addNode__GatherND_5(carbspray_2_v3_model), err);
  VALIDATE(addNode__TopK_2(carbspray_2_v3_model), err);
  VALIDATE(addNode__Gather_14(carbspray_2_v3_model), err);
  VALIDATE(addNode__Concat_6(carbspray_2_v3_model), err);
  VALIDATE(addNode__Concat_7(carbspray_2_v3_model), err);
  VALIDATE(addNode_ReduceMax_1317(carbspray_2_v3_model), err);
  VALIDATE(addTensor__1312(carbspray_2_v3_model), err);
  VALIDATE(addNode_Add_1320(carbspray_2_v3_model), err);
  VALIDATE(addTensor__1311(carbspray_2_v3_model), err);
  VALIDATE(addNode_Mul_1321(carbspray_2_v3_model), err);
  VALIDATE(addNode_Slice_1326(carbspray_2_v3_model), err);
  VALIDATE(addNode_Unsqueeze_1327(carbspray_2_v3_model), err);
  VALIDATE(addNode_Add_1328(carbspray_2_v3_model), err);
  VALIDATE(addNode_Unsqueeze_1329(carbspray_2_v3_model), err);
  VALIDATE(addNode_Unsqueeze_1330(carbspray_2_v3_model), err);
  VALIDATE(addNode_NonMaxSuppression_1336(carbspray_2_v3_model), err);
  VALIDATE(addTensor__1330(carbspray_2_v3_model), err);
  VALIDATE(addNode_Gather_1338(carbspray_2_v3_model), err);
  VALIDATE(addNode_Squeeze_1339(carbspray_2_v3_model), err);
  VALIDATE(addNode__Slice_6(carbspray_2_v3_model), err);
  VALIDATE(addNode__Gather_15(carbspray_2_v3_model), err);
  VALIDATE(addNode__Gather_16(carbspray_2_v3_model), err);
  VALIDATE(addTensor__Concat_8_output_0(carbspray_2_v3_model), err);
  VALIDATE(addNode__Gather_17(carbspray_2_v3_model), err);
  VALIDATE(addNode__Split_3(carbspray_2_v3_model), err);
  VALIDATE(addNode__Concat_9(carbspray_2_v3_model), err);

  // Add all models to array to get graphsInfo
  QnnModel* models [] = {&carbspray_2_v3_model};
  uint32_t numModels = 1;

  // Populate the constructed graphs in provided output variables
  VALIDATE(getGraphInfoFromModels(*models, numModels, graphsInfo), err);
  *numGraphsInfo = numModels;

  return err;

} // PREPARE_GRAPHS

QNN_API
ModelError_t QnnModel_freeGraphsInfo(GraphInfoPtr_t** graphsInfo, uint32_t numGraphsInfo){
  return qnn_wrapper_api::freeGraphsInfo(graphsInfo, numGraphsInfo);
} // FREEGRAPHINFO

}