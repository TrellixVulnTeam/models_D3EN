# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: object_detection/protos/prob_two_stage.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from object_detection.protos import anchor_generator_pb2 as object__detection_dot_protos_dot_anchor__generator__pb2
from object_detection.protos import box_predictor_pb2 as object__detection_dot_protos_dot_box__predictor__pb2
from object_detection.protos import hyperparams_pb2 as object__detection_dot_protos_dot_hyperparams__pb2
from object_detection.protos import image_resizer_pb2 as object__detection_dot_protos_dot_image__resizer__pb2
from object_detection.protos import losses_pb2 as object__detection_dot_protos_dot_losses__pb2
from object_detection.protos import post_processing_pb2 as object__detection_dot_protos_dot_post__processing__pb2
from object_detection.protos import fpn_pb2 as object__detection_dot_protos_dot_fpn__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='object_detection/protos/prob_two_stage.proto',
  package='object_detection.protos',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n,object_detection/protos/prob_two_stage.proto\x12\x17object_detection.protos\x1a.object_detection/protos/anchor_generator.proto\x1a+object_detection/protos/box_predictor.proto\x1a)object_detection/protos/hyperparams.proto\x1a+object_detection/protos/image_resizer.proto\x1a$object_detection/protos/losses.proto\x1a-object_detection/protos/post_processing.proto\x1a!object_detection/protos/fpn.proto\"\x86\x0f\n\x15ProbabilisticTwoStage\x12\x1b\n\x10number_of_stages\x18\x01 \x01(\x05:\x01\x32\x12%\n\x16\x61\x64\x64_weight_information\x18\x02 \x01(\x08:\x05\x66\x61lse\x12\x13\n\x0bnum_classes\x18\x03 \x01(\x05\x12<\n\rimage_resizer\x18\x04 \x01(\x0b\x32%.object_detection.protos.ImageResizer\x12Y\n\x11\x66\x65\x61ture_extractor\x18\x05 \x01(\x0b\x32>.object_detection.protos.ProbabilisticTwoStageFeatureExtractor\x12N\n\x1c\x66irst_stage_anchor_generator\x18\x06 \x01(\x0b\x32(.object_detection.protos.AnchorGenerator\x12H\n\x19\x66irst_stage_box_predictor\x18\n \x01(\x0b\x32%.object_detection.protos.BoxPredictor\x12\'\n\x1a\x66irst_stage_minibatch_size\x18\x0b \x01(\x05:\x03\x32\x35\x36\x12\x32\n%first_stage_positive_balance_fraction\x18\x0c \x01(\x02:\x03\x30.5\x12*\n\x1f\x66irst_stage_nms_score_threshold\x18\r \x01(\x02:\x01\x30\x12*\n\x1d\x66irst_stage_nms_iou_threshold\x18\x0e \x01(\x02:\x03\x30.7\x12&\n\x19\x66irst_stage_max_proposals\x18\x0f \x01(\x05:\x03\x33\x30\x30\x12/\n$first_stage_localization_loss_weight\x18\x10 \x01(\x02:\x01\x31\x12-\n\"first_stage_objectness_loss_weight\x18\x11 \x01(\x02:\x01\x31\x12\x19\n\x11initial_crop_size\x18\x12 \x01(\x05\x12\x1b\n\x13maxpool_kernel_size\x18\x13 \x01(\x05\x12\x16\n\x0emaxpool_stride\x18\x14 \x01(\x05\x12I\n\x1asecond_stage_box_predictor\x18\x15 \x01(\x0b\x32%.object_detection.protos.BoxPredictor\x12#\n\x17second_stage_batch_size\x18\x16 \x01(\x05:\x02\x36\x34\x12+\n\x1dsecond_stage_balance_fraction\x18\x17 \x01(\x02:\x04\x30.25\x12M\n\x1csecond_stage_post_processing\x18\x18 \x01(\x0b\x32\'.object_detection.protos.PostProcessing\x12\x30\n%second_stage_localization_loss_weight\x18\x19 \x01(\x02:\x01\x31\x12\x32\n\'second_stage_classification_loss_weight\x18\x1a \x01(\x02:\x01\x31\x12\x33\n(second_stage_mask_prediction_loss_weight\x18\x1b \x01(\x02:\x01\x31\x12\x45\n\x12hard_example_miner\x18\x1c \x01(\x0b\x32).object_detection.protos.HardExampleMiner\x12U\n second_stage_classification_loss\x18\x1d \x01(\x0b\x32+.object_detection.protos.ClassificationLoss\x12\'\n\x18inplace_batchnorm_update\x18\x1e \x01(\x08:\x05\x66\x61lse\x12)\n\x1ause_matmul_crop_and_resize\x18\x1f \x01(\x08:\x05\x66\x61lse\x12$\n\x15\x63lip_anchors_to_image\x18  \x01(\x08:\x05\x66\x61lse\x12+\n\x1cuse_matmul_gather_in_matcher\x18! \x01(\x08:\x05\x66\x61lse\x12\x30\n!use_static_balanced_label_sampler\x18\" \x01(\x08:\x05\x66\x61lse\x12 \n\x11use_static_shapes\x18# \x01(\x08:\x05\x66\x61lse\x12\x1a\n\x0cresize_masks\x18$ \x01(\x08:\x04true\x12)\n\x1ause_static_shapes_for_eval\x18% \x01(\x08:\x05\x66\x61lse\x12\x30\n\"use_partitioned_nms_in_first_stage\x18& \x01(\x08:\x04true\x12\x33\n$return_raw_detections_during_predict\x18\' \x01(\x08:\x05\x66\x61lse\x12.\n\x1fuse_combined_nms_in_first_stage\x18( \x01(\x08:\x05\x66\x61lse\x12(\n\x19output_final_box_features\x18* \x01(\x08:\x05\x66\x61lse\x12,\n\x1doutput_final_box_rpn_features\x18+ \x01(\x08:\x05\x66\x61lse\"\xfc\x02\n%ProbabilisticTwoStageFeatureExtractor\x12\x0c\n\x04type\x18\x01 \x01(\t\x12\'\n\x1b\x66irst_stage_features_stride\x18\x02 \x01(\x05:\x02\x31\x36\x12\x1f\n\x10\x66reeze_batchnorm\x18\x03 \x01(\x08:\x05\x66\x61lse\x12>\n\x10\x63onv_hyperparams\x18\x04 \x01(\x0b\x32$.object_detection.protos.Hyperparams\x12:\n+override_base_feature_extractor_hyperparams\x18\x05 \x01(\x08:\x05\x66\x61lse\x12\x1b\n\x0fpad_to_multiple\x18\x06 \x01(\x05:\x02\x33\x32\x12K\n\x05\x62ifpn\x18\x07 \x01(\x0b\x32<.object_detection.protos.BidirectionalFeaturePyramidNetworks\x12\x15\n\tmin_depth\x18\x08 \x01(\x05:\x02\x31\x36')
  ,
  dependencies=[object__detection_dot_protos_dot_anchor__generator__pb2.DESCRIPTOR,object__detection_dot_protos_dot_box__predictor__pb2.DESCRIPTOR,object__detection_dot_protos_dot_hyperparams__pb2.DESCRIPTOR,object__detection_dot_protos_dot_image__resizer__pb2.DESCRIPTOR,object__detection_dot_protos_dot_losses__pb2.DESCRIPTOR,object__detection_dot_protos_dot_post__processing__pb2.DESCRIPTOR,object__detection_dot_protos_dot_fpn__pb2.DESCRIPTOR,])




_PROBABILISTICTWOSTAGE = _descriptor.Descriptor(
  name='ProbabilisticTwoStage',
  full_name='object_detection.protos.ProbabilisticTwoStage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='number_of_stages', full_name='object_detection.protos.ProbabilisticTwoStage.number_of_stages', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=2,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='add_weight_information', full_name='object_detection.protos.ProbabilisticTwoStage.add_weight_information', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_classes', full_name='object_detection.protos.ProbabilisticTwoStage.num_classes', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='image_resizer', full_name='object_detection.protos.ProbabilisticTwoStage.image_resizer', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='feature_extractor', full_name='object_detection.protos.ProbabilisticTwoStage.feature_extractor', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='first_stage_anchor_generator', full_name='object_detection.protos.ProbabilisticTwoStage.first_stage_anchor_generator', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='first_stage_box_predictor', full_name='object_detection.protos.ProbabilisticTwoStage.first_stage_box_predictor', index=6,
      number=10, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='first_stage_minibatch_size', full_name='object_detection.protos.ProbabilisticTwoStage.first_stage_minibatch_size', index=7,
      number=11, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=256,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='first_stage_positive_balance_fraction', full_name='object_detection.protos.ProbabilisticTwoStage.first_stage_positive_balance_fraction', index=8,
      number=12, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.5),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='first_stage_nms_score_threshold', full_name='object_detection.protos.ProbabilisticTwoStage.first_stage_nms_score_threshold', index=9,
      number=13, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='first_stage_nms_iou_threshold', full_name='object_detection.protos.ProbabilisticTwoStage.first_stage_nms_iou_threshold', index=10,
      number=14, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.7),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='first_stage_max_proposals', full_name='object_detection.protos.ProbabilisticTwoStage.first_stage_max_proposals', index=11,
      number=15, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=300,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='first_stage_localization_loss_weight', full_name='object_detection.protos.ProbabilisticTwoStage.first_stage_localization_loss_weight', index=12,
      number=16, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='first_stage_objectness_loss_weight', full_name='object_detection.protos.ProbabilisticTwoStage.first_stage_objectness_loss_weight', index=13,
      number=17, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='initial_crop_size', full_name='object_detection.protos.ProbabilisticTwoStage.initial_crop_size', index=14,
      number=18, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='maxpool_kernel_size', full_name='object_detection.protos.ProbabilisticTwoStage.maxpool_kernel_size', index=15,
      number=19, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='maxpool_stride', full_name='object_detection.protos.ProbabilisticTwoStage.maxpool_stride', index=16,
      number=20, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='second_stage_box_predictor', full_name='object_detection.protos.ProbabilisticTwoStage.second_stage_box_predictor', index=17,
      number=21, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='second_stage_batch_size', full_name='object_detection.protos.ProbabilisticTwoStage.second_stage_batch_size', index=18,
      number=22, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=64,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='second_stage_balance_fraction', full_name='object_detection.protos.ProbabilisticTwoStage.second_stage_balance_fraction', index=19,
      number=23, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.25),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='second_stage_post_processing', full_name='object_detection.protos.ProbabilisticTwoStage.second_stage_post_processing', index=20,
      number=24, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='second_stage_localization_loss_weight', full_name='object_detection.protos.ProbabilisticTwoStage.second_stage_localization_loss_weight', index=21,
      number=25, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='second_stage_classification_loss_weight', full_name='object_detection.protos.ProbabilisticTwoStage.second_stage_classification_loss_weight', index=22,
      number=26, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='second_stage_mask_prediction_loss_weight', full_name='object_detection.protos.ProbabilisticTwoStage.second_stage_mask_prediction_loss_weight', index=23,
      number=27, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='hard_example_miner', full_name='object_detection.protos.ProbabilisticTwoStage.hard_example_miner', index=24,
      number=28, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='second_stage_classification_loss', full_name='object_detection.protos.ProbabilisticTwoStage.second_stage_classification_loss', index=25,
      number=29, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='inplace_batchnorm_update', full_name='object_detection.protos.ProbabilisticTwoStage.inplace_batchnorm_update', index=26,
      number=30, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='use_matmul_crop_and_resize', full_name='object_detection.protos.ProbabilisticTwoStage.use_matmul_crop_and_resize', index=27,
      number=31, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='clip_anchors_to_image', full_name='object_detection.protos.ProbabilisticTwoStage.clip_anchors_to_image', index=28,
      number=32, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='use_matmul_gather_in_matcher', full_name='object_detection.protos.ProbabilisticTwoStage.use_matmul_gather_in_matcher', index=29,
      number=33, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='use_static_balanced_label_sampler', full_name='object_detection.protos.ProbabilisticTwoStage.use_static_balanced_label_sampler', index=30,
      number=34, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='use_static_shapes', full_name='object_detection.protos.ProbabilisticTwoStage.use_static_shapes', index=31,
      number=35, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='resize_masks', full_name='object_detection.protos.ProbabilisticTwoStage.resize_masks', index=32,
      number=36, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='use_static_shapes_for_eval', full_name='object_detection.protos.ProbabilisticTwoStage.use_static_shapes_for_eval', index=33,
      number=37, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='use_partitioned_nms_in_first_stage', full_name='object_detection.protos.ProbabilisticTwoStage.use_partitioned_nms_in_first_stage', index=34,
      number=38, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='return_raw_detections_during_predict', full_name='object_detection.protos.ProbabilisticTwoStage.return_raw_detections_during_predict', index=35,
      number=39, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='use_combined_nms_in_first_stage', full_name='object_detection.protos.ProbabilisticTwoStage.use_combined_nms_in_first_stage', index=36,
      number=40, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='output_final_box_features', full_name='object_detection.protos.ProbabilisticTwoStage.output_final_box_features', index=37,
      number=42, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='output_final_box_rpn_features', full_name='object_detection.protos.ProbabilisticTwoStage.output_final_box_rpn_features', index=38,
      number=43, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=375,
  serialized_end=2301,
)


_PROBABILISTICTWOSTAGEFEATUREEXTRACTOR = _descriptor.Descriptor(
  name='ProbabilisticTwoStageFeatureExtractor',
  full_name='object_detection.protos.ProbabilisticTwoStageFeatureExtractor',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='object_detection.protos.ProbabilisticTwoStageFeatureExtractor.type', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='first_stage_features_stride', full_name='object_detection.protos.ProbabilisticTwoStageFeatureExtractor.first_stage_features_stride', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=16,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='freeze_batchnorm', full_name='object_detection.protos.ProbabilisticTwoStageFeatureExtractor.freeze_batchnorm', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='conv_hyperparams', full_name='object_detection.protos.ProbabilisticTwoStageFeatureExtractor.conv_hyperparams', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='override_base_feature_extractor_hyperparams', full_name='object_detection.protos.ProbabilisticTwoStageFeatureExtractor.override_base_feature_extractor_hyperparams', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pad_to_multiple', full_name='object_detection.protos.ProbabilisticTwoStageFeatureExtractor.pad_to_multiple', index=5,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=32,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bifpn', full_name='object_detection.protos.ProbabilisticTwoStageFeatureExtractor.bifpn', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='min_depth', full_name='object_detection.protos.ProbabilisticTwoStageFeatureExtractor.min_depth', index=7,
      number=8, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=16,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=2304,
  serialized_end=2684,
)

_PROBABILISTICTWOSTAGE.fields_by_name['image_resizer'].message_type = object__detection_dot_protos_dot_image__resizer__pb2._IMAGERESIZER
_PROBABILISTICTWOSTAGE.fields_by_name['feature_extractor'].message_type = _PROBABILISTICTWOSTAGEFEATUREEXTRACTOR
_PROBABILISTICTWOSTAGE.fields_by_name['first_stage_anchor_generator'].message_type = object__detection_dot_protos_dot_anchor__generator__pb2._ANCHORGENERATOR
_PROBABILISTICTWOSTAGE.fields_by_name['first_stage_box_predictor'].message_type = object__detection_dot_protos_dot_box__predictor__pb2._BOXPREDICTOR
_PROBABILISTICTWOSTAGE.fields_by_name['second_stage_box_predictor'].message_type = object__detection_dot_protos_dot_box__predictor__pb2._BOXPREDICTOR
_PROBABILISTICTWOSTAGE.fields_by_name['second_stage_post_processing'].message_type = object__detection_dot_protos_dot_post__processing__pb2._POSTPROCESSING
_PROBABILISTICTWOSTAGE.fields_by_name['hard_example_miner'].message_type = object__detection_dot_protos_dot_losses__pb2._HARDEXAMPLEMINER
_PROBABILISTICTWOSTAGE.fields_by_name['second_stage_classification_loss'].message_type = object__detection_dot_protos_dot_losses__pb2._CLASSIFICATIONLOSS
_PROBABILISTICTWOSTAGEFEATUREEXTRACTOR.fields_by_name['conv_hyperparams'].message_type = object__detection_dot_protos_dot_hyperparams__pb2._HYPERPARAMS
_PROBABILISTICTWOSTAGEFEATUREEXTRACTOR.fields_by_name['bifpn'].message_type = object__detection_dot_protos_dot_fpn__pb2._BIDIRECTIONALFEATUREPYRAMIDNETWORKS
DESCRIPTOR.message_types_by_name['ProbabilisticTwoStage'] = _PROBABILISTICTWOSTAGE
DESCRIPTOR.message_types_by_name['ProbabilisticTwoStageFeatureExtractor'] = _PROBABILISTICTWOSTAGEFEATUREEXTRACTOR
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ProbabilisticTwoStage = _reflection.GeneratedProtocolMessageType('ProbabilisticTwoStage', (_message.Message,), dict(
  DESCRIPTOR = _PROBABILISTICTWOSTAGE,
  __module__ = 'object_detection.protos.prob_two_stage_pb2'
  # @@protoc_insertion_point(class_scope:object_detection.protos.ProbabilisticTwoStage)
  ))
_sym_db.RegisterMessage(ProbabilisticTwoStage)

ProbabilisticTwoStageFeatureExtractor = _reflection.GeneratedProtocolMessageType('ProbabilisticTwoStageFeatureExtractor', (_message.Message,), dict(
  DESCRIPTOR = _PROBABILISTICTWOSTAGEFEATUREEXTRACTOR,
  __module__ = 'object_detection.protos.prob_two_stage_pb2'
  # @@protoc_insertion_point(class_scope:object_detection.protos.ProbabilisticTwoStageFeatureExtractor)
  ))
_sym_db.RegisterMessage(ProbabilisticTwoStageFeatureExtractor)


# @@protoc_insertion_point(module_scope)
