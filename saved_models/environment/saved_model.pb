??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8??
?
latent_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_namelatent_layer/kernel
{
'latent_layer/kernel/Read/ReadVariableOpReadVariableOplatent_layer/kernel*
_output_shapes

:*
dtype0
z
latent_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namelatent_layer/bias
s
%latent_layer/bias/Read/ReadVariableOpReadVariableOplatent_layer/bias*
_output_shapes
:*
dtype0
?
dense_layer_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_namedense_layer_2/kernel
}
(dense_layer_2/kernel/Read/ReadVariableOpReadVariableOpdense_layer_2/kernel*
_output_shapes

:*
dtype0
|
dense_layer_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namedense_layer_2/bias
u
&dense_layer_2/bias/Read/ReadVariableOpReadVariableOpdense_layer_2/bias*
_output_shapes
:*
dtype0
?
dense_layer_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *%
shared_namedense_layer_1/kernel
}
(dense_layer_1/kernel/Read/ReadVariableOpReadVariableOpdense_layer_1/kernel*
_output_shapes

: *
dtype0
|
dense_layer_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_namedense_layer_1/bias
u
&dense_layer_1/bias/Read/ReadVariableOpReadVariableOpdense_layer_1/bias*
_output_shapes
: *
dtype0
?
decoder_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_namedecoder_output/kernel

)decoder_output/kernel/Read/ReadVariableOpReadVariableOpdecoder_output/kernel*
_output_shapes

: *
dtype0
~
decoder_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namedecoder_output/bias
w
'decoder_output/bias/Read/ReadVariableOpReadVariableOpdecoder_output/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
loss
trainable_variables
	regularization_losses

	variables
	keras_api

signatures
 
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
 bias
!trainable_variables
"regularization_losses
#	variables
$	keras_api
 
 
8
0
1
2
3
4
5
6
 7
 
8
0
1
2
3
4
5
6
 7
?
trainable_variables
%layer_regularization_losses
	regularization_losses
&metrics

'layers
(layer_metrics

	variables
)non_trainable_variables
 
_]
VARIABLE_VALUElatent_layer/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUElatent_layer/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
trainable_variables
*layer_regularization_losses
+metrics
regularization_losses

,layers
-layer_metrics
	variables
.non_trainable_variables
`^
VARIABLE_VALUEdense_layer_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEdense_layer_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
trainable_variables
/layer_regularization_losses
0metrics
regularization_losses

1layers
2layer_metrics
	variables
3non_trainable_variables
`^
VARIABLE_VALUEdense_layer_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEdense_layer_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
trainable_variables
4layer_regularization_losses
5metrics
regularization_losses

6layers
7layer_metrics
	variables
8non_trainable_variables
a_
VARIABLE_VALUEdecoder_output/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEdecoder_output/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
 1
 

0
 1
?
!trainable_variables
9layer_regularization_losses
:metrics
"regularization_losses

;layers
<layer_metrics
#	variables
=non_trainable_variables
 
 
#
0
1
2
3
4
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_2Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2latent_layer/kernellatent_layer/biasdense_layer_2/kerneldense_layer_2/biasdense_layer_1/kerneldense_layer_1/biasdecoder_output/kerneldecoder_output/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_31995
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'latent_layer/kernel/Read/ReadVariableOp%latent_layer/bias/Read/ReadVariableOp(dense_layer_2/kernel/Read/ReadVariableOp&dense_layer_2/bias/Read/ReadVariableOp(dense_layer_1/kernel/Read/ReadVariableOp&dense_layer_1/bias/Read/ReadVariableOp)decoder_output/kernel/Read/ReadVariableOp'decoder_output/bias/Read/ReadVariableOpConst*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_32228
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelatent_layer/kernellatent_layer/biasdense_layer_2/kerneldense_layer_2/biasdense_layer_1/kerneldense_layer_1/biasdecoder_output/kerneldecoder_output/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_32262??
?&
?
!__inference__traced_restore_32262
file_prefix(
$assignvariableop_latent_layer_kernel(
$assignvariableop_1_latent_layer_bias+
'assignvariableop_2_dense_layer_2_kernel)
%assignvariableop_3_dense_layer_2_bias+
'assignvariableop_4_dense_layer_1_kernel)
%assignvariableop_5_dense_layer_1_bias,
(assignvariableop_6_decoder_output_kernel*
&assignvariableop_7_decoder_output_bias

identity_9??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*?
value?B?	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp$assignvariableop_latent_layer_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp$assignvariableop_1_latent_layer_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp'assignvariableop_2_dense_layer_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp%assignvariableop_3_dense_layer_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp'assignvariableop_4_dense_layer_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp%assignvariableop_5_dense_layer_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp(assignvariableop_6_decoder_output_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp&assignvariableop_7_decoder_output_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8?

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*
T0*
_output_shapes
: 2

Identity_9"!

identity_9Identity_9:output:0*5
_input_shapes$
": ::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
'__inference_decoder_layer_call_fn_31927
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_319082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?
?
B__inference_decoder_layer_call_and_return_conditional_losses_31881
input_2
latent_layer_31860
latent_layer_31862
dense_layer_2_31865
dense_layer_2_31867
dense_layer_1_31870
dense_layer_1_31872
decoder_output_31875
decoder_output_31877
identity??&decoder_output/StatefulPartitionedCall?%dense_layer_1/StatefulPartitionedCall?%dense_layer_2/StatefulPartitionedCall?$latent_layer/StatefulPartitionedCall?
$latent_layer/StatefulPartitionedCallStatefulPartitionedCallinput_2latent_layer_31860latent_layer_31862*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_latent_layer_layer_call_and_return_conditional_losses_317592&
$latent_layer/StatefulPartitionedCall?
%dense_layer_2/StatefulPartitionedCallStatefulPartitionedCall-latent_layer/StatefulPartitionedCall:output:0dense_layer_2_31865dense_layer_2_31867*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dense_layer_2_layer_call_and_return_conditional_losses_317862'
%dense_layer_2/StatefulPartitionedCall?
%dense_layer_1/StatefulPartitionedCallStatefulPartitionedCall.dense_layer_2/StatefulPartitionedCall:output:0dense_layer_1_31870dense_layer_1_31872*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_318132'
%dense_layer_1/StatefulPartitionedCall?
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall.dense_layer_1/StatefulPartitionedCall:output:0decoder_output_31875decoder_output_31877*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_318402(
&decoder_output/StatefulPartitionedCall?
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0'^decoder_output/StatefulPartitionedCall&^dense_layer_1/StatefulPartitionedCall&^dense_layer_2/StatefulPartitionedCall%^latent_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall2N
%dense_layer_1/StatefulPartitionedCall%dense_layer_1/StatefulPartitionedCall2N
%dense_layer_2/StatefulPartitionedCall%dense_layer_2/StatefulPartitionedCall2L
$latent_layer/StatefulPartitionedCall$latent_layer/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?
?
B__inference_decoder_layer_call_and_return_conditional_losses_31908

inputs
latent_layer_31887
latent_layer_31889
dense_layer_2_31892
dense_layer_2_31894
dense_layer_1_31897
dense_layer_1_31899
decoder_output_31902
decoder_output_31904
identity??&decoder_output/StatefulPartitionedCall?%dense_layer_1/StatefulPartitionedCall?%dense_layer_2/StatefulPartitionedCall?$latent_layer/StatefulPartitionedCall?
$latent_layer/StatefulPartitionedCallStatefulPartitionedCallinputslatent_layer_31887latent_layer_31889*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_latent_layer_layer_call_and_return_conditional_losses_317592&
$latent_layer/StatefulPartitionedCall?
%dense_layer_2/StatefulPartitionedCallStatefulPartitionedCall-latent_layer/StatefulPartitionedCall:output:0dense_layer_2_31892dense_layer_2_31894*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dense_layer_2_layer_call_and_return_conditional_losses_317862'
%dense_layer_2/StatefulPartitionedCall?
%dense_layer_1/StatefulPartitionedCallStatefulPartitionedCall.dense_layer_2/StatefulPartitionedCall:output:0dense_layer_1_31897dense_layer_1_31899*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_318132'
%dense_layer_1/StatefulPartitionedCall?
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall.dense_layer_1/StatefulPartitionedCall:output:0decoder_output_31902decoder_output_31904*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_318402(
&decoder_output/StatefulPartitionedCall?
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0'^decoder_output/StatefulPartitionedCall&^dense_layer_1/StatefulPartitionedCall&^dense_layer_2/StatefulPartitionedCall%^latent_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall2N
%dense_layer_1/StatefulPartitionedCall%dense_layer_1/StatefulPartitionedCall2N
%dense_layer_2/StatefulPartitionedCall%dense_layer_2/StatefulPartitionedCall2L
$latent_layer/StatefulPartitionedCall$latent_layer/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_31813

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
G__inference_latent_layer_layer_call_and_return_conditional_losses_32112

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
I__inference_decoder_output_layer_call_and_return_conditional_losses_32172

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
G__inference_latent_layer_layer_call_and_return_conditional_losses_31759

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_decoder_layer_call_fn_32080

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_319082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
I__inference_decoder_output_layer_call_and_return_conditional_losses_31840

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
'__inference_decoder_layer_call_fn_32101

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_319532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?+
?
B__inference_decoder_layer_call_and_return_conditional_losses_32027

inputs/
+latent_layer_matmul_readvariableop_resource0
,latent_layer_biasadd_readvariableop_resource0
,dense_layer_2_matmul_readvariableop_resource1
-dense_layer_2_biasadd_readvariableop_resource0
,dense_layer_1_matmul_readvariableop_resource1
-dense_layer_1_biasadd_readvariableop_resource1
-decoder_output_matmul_readvariableop_resource2
.decoder_output_biasadd_readvariableop_resource
identity??%decoder_output/BiasAdd/ReadVariableOp?$decoder_output/MatMul/ReadVariableOp?$dense_layer_1/BiasAdd/ReadVariableOp?#dense_layer_1/MatMul/ReadVariableOp?$dense_layer_2/BiasAdd/ReadVariableOp?#dense_layer_2/MatMul/ReadVariableOp?#latent_layer/BiasAdd/ReadVariableOp?"latent_layer/MatMul/ReadVariableOp?
"latent_layer/MatMul/ReadVariableOpReadVariableOp+latent_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"latent_layer/MatMul/ReadVariableOp?
latent_layer/MatMulMatMulinputs*latent_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
latent_layer/MatMul?
#latent_layer/BiasAdd/ReadVariableOpReadVariableOp,latent_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#latent_layer/BiasAdd/ReadVariableOp?
latent_layer/BiasAddBiasAddlatent_layer/MatMul:product:0+latent_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
latent_layer/BiasAdd
latent_layer/ReluRelulatent_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
latent_layer/Relu?
#dense_layer_2/MatMul/ReadVariableOpReadVariableOp,dense_layer_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#dense_layer_2/MatMul/ReadVariableOp?
dense_layer_2/MatMulMatMullatent_layer/Relu:activations:0+dense_layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_layer_2/MatMul?
$dense_layer_2/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$dense_layer_2/BiasAdd/ReadVariableOp?
dense_layer_2/BiasAddBiasAdddense_layer_2/MatMul:product:0,dense_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_layer_2/BiasAdd?
dense_layer_2/ReluReludense_layer_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_layer_2/Relu?
#dense_layer_1/MatMul/ReadVariableOpReadVariableOp,dense_layer_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02%
#dense_layer_1/MatMul/ReadVariableOp?
dense_layer_1/MatMulMatMul dense_layer_2/Relu:activations:0+dense_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_layer_1/MatMul?
$dense_layer_1/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$dense_layer_1/BiasAdd/ReadVariableOp?
dense_layer_1/BiasAddBiasAdddense_layer_1/MatMul:product:0,dense_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_layer_1/BiasAdd?
dense_layer_1/ReluReludense_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_layer_1/Relu?
$decoder_output/MatMul/ReadVariableOpReadVariableOp-decoder_output_matmul_readvariableop_resource*
_output_shapes

: *
dtype02&
$decoder_output/MatMul/ReadVariableOp?
decoder_output/MatMulMatMul dense_layer_1/Relu:activations:0,decoder_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
decoder_output/MatMul?
%decoder_output/BiasAdd/ReadVariableOpReadVariableOp.decoder_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%decoder_output/BiasAdd/ReadVariableOp?
decoder_output/BiasAddBiasAdddecoder_output/MatMul:product:0-decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
decoder_output/BiasAdd?
decoder_output/SigmoidSigmoiddecoder_output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
decoder_output/Sigmoid?
IdentityIdentitydecoder_output/Sigmoid:y:0&^decoder_output/BiasAdd/ReadVariableOp%^decoder_output/MatMul/ReadVariableOp%^dense_layer_1/BiasAdd/ReadVariableOp$^dense_layer_1/MatMul/ReadVariableOp%^dense_layer_2/BiasAdd/ReadVariableOp$^dense_layer_2/MatMul/ReadVariableOp$^latent_layer/BiasAdd/ReadVariableOp#^latent_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2N
%decoder_output/BiasAdd/ReadVariableOp%decoder_output/BiasAdd/ReadVariableOp2L
$decoder_output/MatMul/ReadVariableOp$decoder_output/MatMul/ReadVariableOp2L
$dense_layer_1/BiasAdd/ReadVariableOp$dense_layer_1/BiasAdd/ReadVariableOp2J
#dense_layer_1/MatMul/ReadVariableOp#dense_layer_1/MatMul/ReadVariableOp2L
$dense_layer_2/BiasAdd/ReadVariableOp$dense_layer_2/BiasAdd/ReadVariableOp2J
#dense_layer_2/MatMul/ReadVariableOp#dense_layer_2/MatMul/ReadVariableOp2J
#latent_layer/BiasAdd/ReadVariableOp#latent_layer/BiasAdd/ReadVariableOp2H
"latent_layer/MatMul/ReadVariableOp"latent_layer/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
-__inference_dense_layer_2_layer_call_fn_32141

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dense_layer_2_layer_call_and_return_conditional_losses_317862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
.__inference_decoder_output_layer_call_fn_32181

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_318402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
,__inference_latent_layer_layer_call_fn_32121

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_latent_layer_layer_call_and_return_conditional_losses_317592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?1
?
 __inference__wrapped_model_31744
input_27
3decoder_latent_layer_matmul_readvariableop_resource8
4decoder_latent_layer_biasadd_readvariableop_resource8
4decoder_dense_layer_2_matmul_readvariableop_resource9
5decoder_dense_layer_2_biasadd_readvariableop_resource8
4decoder_dense_layer_1_matmul_readvariableop_resource9
5decoder_dense_layer_1_biasadd_readvariableop_resource9
5decoder_decoder_output_matmul_readvariableop_resource:
6decoder_decoder_output_biasadd_readvariableop_resource
identity??-decoder/decoder_output/BiasAdd/ReadVariableOp?,decoder/decoder_output/MatMul/ReadVariableOp?,decoder/dense_layer_1/BiasAdd/ReadVariableOp?+decoder/dense_layer_1/MatMul/ReadVariableOp?,decoder/dense_layer_2/BiasAdd/ReadVariableOp?+decoder/dense_layer_2/MatMul/ReadVariableOp?+decoder/latent_layer/BiasAdd/ReadVariableOp?*decoder/latent_layer/MatMul/ReadVariableOp?
*decoder/latent_layer/MatMul/ReadVariableOpReadVariableOp3decoder_latent_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*decoder/latent_layer/MatMul/ReadVariableOp?
decoder/latent_layer/MatMulMatMulinput_22decoder/latent_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
decoder/latent_layer/MatMul?
+decoder/latent_layer/BiasAdd/ReadVariableOpReadVariableOp4decoder_latent_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+decoder/latent_layer/BiasAdd/ReadVariableOp?
decoder/latent_layer/BiasAddBiasAdd%decoder/latent_layer/MatMul:product:03decoder/latent_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
decoder/latent_layer/BiasAdd?
decoder/latent_layer/ReluRelu%decoder/latent_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
decoder/latent_layer/Relu?
+decoder/dense_layer_2/MatMul/ReadVariableOpReadVariableOp4decoder_dense_layer_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02-
+decoder/dense_layer_2/MatMul/ReadVariableOp?
decoder/dense_layer_2/MatMulMatMul'decoder/latent_layer/Relu:activations:03decoder/dense_layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
decoder/dense_layer_2/MatMul?
,decoder/dense_layer_2/BiasAdd/ReadVariableOpReadVariableOp5decoder_dense_layer_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,decoder/dense_layer_2/BiasAdd/ReadVariableOp?
decoder/dense_layer_2/BiasAddBiasAdd&decoder/dense_layer_2/MatMul:product:04decoder/dense_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
decoder/dense_layer_2/BiasAdd?
decoder/dense_layer_2/ReluRelu&decoder/dense_layer_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
decoder/dense_layer_2/Relu?
+decoder/dense_layer_1/MatMul/ReadVariableOpReadVariableOp4decoder_dense_layer_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02-
+decoder/dense_layer_1/MatMul/ReadVariableOp?
decoder/dense_layer_1/MatMulMatMul(decoder/dense_layer_2/Relu:activations:03decoder/dense_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
decoder/dense_layer_1/MatMul?
,decoder/dense_layer_1/BiasAdd/ReadVariableOpReadVariableOp5decoder_dense_layer_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,decoder/dense_layer_1/BiasAdd/ReadVariableOp?
decoder/dense_layer_1/BiasAddBiasAdd&decoder/dense_layer_1/MatMul:product:04decoder/dense_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
decoder/dense_layer_1/BiasAdd?
decoder/dense_layer_1/ReluRelu&decoder/dense_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
decoder/dense_layer_1/Relu?
,decoder/decoder_output/MatMul/ReadVariableOpReadVariableOp5decoder_decoder_output_matmul_readvariableop_resource*
_output_shapes

: *
dtype02.
,decoder/decoder_output/MatMul/ReadVariableOp?
decoder/decoder_output/MatMulMatMul(decoder/dense_layer_1/Relu:activations:04decoder/decoder_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
decoder/decoder_output/MatMul?
-decoder/decoder_output/BiasAdd/ReadVariableOpReadVariableOp6decoder_decoder_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-decoder/decoder_output/BiasAdd/ReadVariableOp?
decoder/decoder_output/BiasAddBiasAdd'decoder/decoder_output/MatMul:product:05decoder/decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
decoder/decoder_output/BiasAdd?
decoder/decoder_output/SigmoidSigmoid'decoder/decoder_output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2 
decoder/decoder_output/Sigmoid?
IdentityIdentity"decoder/decoder_output/Sigmoid:y:0.^decoder/decoder_output/BiasAdd/ReadVariableOp-^decoder/decoder_output/MatMul/ReadVariableOp-^decoder/dense_layer_1/BiasAdd/ReadVariableOp,^decoder/dense_layer_1/MatMul/ReadVariableOp-^decoder/dense_layer_2/BiasAdd/ReadVariableOp,^decoder/dense_layer_2/MatMul/ReadVariableOp,^decoder/latent_layer/BiasAdd/ReadVariableOp+^decoder/latent_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2^
-decoder/decoder_output/BiasAdd/ReadVariableOp-decoder/decoder_output/BiasAdd/ReadVariableOp2\
,decoder/decoder_output/MatMul/ReadVariableOp,decoder/decoder_output/MatMul/ReadVariableOp2\
,decoder/dense_layer_1/BiasAdd/ReadVariableOp,decoder/dense_layer_1/BiasAdd/ReadVariableOp2Z
+decoder/dense_layer_1/MatMul/ReadVariableOp+decoder/dense_layer_1/MatMul/ReadVariableOp2\
,decoder/dense_layer_2/BiasAdd/ReadVariableOp,decoder/dense_layer_2/BiasAdd/ReadVariableOp2Z
+decoder/dense_layer_2/MatMul/ReadVariableOp+decoder/dense_layer_2/MatMul/ReadVariableOp2Z
+decoder/latent_layer/BiasAdd/ReadVariableOp+decoder/latent_layer/BiasAdd/ReadVariableOp2X
*decoder/latent_layer/MatMul/ReadVariableOp*decoder/latent_layer/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?+
?
B__inference_decoder_layer_call_and_return_conditional_losses_32059

inputs/
+latent_layer_matmul_readvariableop_resource0
,latent_layer_biasadd_readvariableop_resource0
,dense_layer_2_matmul_readvariableop_resource1
-dense_layer_2_biasadd_readvariableop_resource0
,dense_layer_1_matmul_readvariableop_resource1
-dense_layer_1_biasadd_readvariableop_resource1
-decoder_output_matmul_readvariableop_resource2
.decoder_output_biasadd_readvariableop_resource
identity??%decoder_output/BiasAdd/ReadVariableOp?$decoder_output/MatMul/ReadVariableOp?$dense_layer_1/BiasAdd/ReadVariableOp?#dense_layer_1/MatMul/ReadVariableOp?$dense_layer_2/BiasAdd/ReadVariableOp?#dense_layer_2/MatMul/ReadVariableOp?#latent_layer/BiasAdd/ReadVariableOp?"latent_layer/MatMul/ReadVariableOp?
"latent_layer/MatMul/ReadVariableOpReadVariableOp+latent_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"latent_layer/MatMul/ReadVariableOp?
latent_layer/MatMulMatMulinputs*latent_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
latent_layer/MatMul?
#latent_layer/BiasAdd/ReadVariableOpReadVariableOp,latent_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#latent_layer/BiasAdd/ReadVariableOp?
latent_layer/BiasAddBiasAddlatent_layer/MatMul:product:0+latent_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
latent_layer/BiasAdd
latent_layer/ReluRelulatent_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
latent_layer/Relu?
#dense_layer_2/MatMul/ReadVariableOpReadVariableOp,dense_layer_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#dense_layer_2/MatMul/ReadVariableOp?
dense_layer_2/MatMulMatMullatent_layer/Relu:activations:0+dense_layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_layer_2/MatMul?
$dense_layer_2/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$dense_layer_2/BiasAdd/ReadVariableOp?
dense_layer_2/BiasAddBiasAdddense_layer_2/MatMul:product:0,dense_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_layer_2/BiasAdd?
dense_layer_2/ReluReludense_layer_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_layer_2/Relu?
#dense_layer_1/MatMul/ReadVariableOpReadVariableOp,dense_layer_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02%
#dense_layer_1/MatMul/ReadVariableOp?
dense_layer_1/MatMulMatMul dense_layer_2/Relu:activations:0+dense_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_layer_1/MatMul?
$dense_layer_1/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$dense_layer_1/BiasAdd/ReadVariableOp?
dense_layer_1/BiasAddBiasAdddense_layer_1/MatMul:product:0,dense_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_layer_1/BiasAdd?
dense_layer_1/ReluReludense_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_layer_1/Relu?
$decoder_output/MatMul/ReadVariableOpReadVariableOp-decoder_output_matmul_readvariableop_resource*
_output_shapes

: *
dtype02&
$decoder_output/MatMul/ReadVariableOp?
decoder_output/MatMulMatMul dense_layer_1/Relu:activations:0,decoder_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
decoder_output/MatMul?
%decoder_output/BiasAdd/ReadVariableOpReadVariableOp.decoder_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%decoder_output/BiasAdd/ReadVariableOp?
decoder_output/BiasAddBiasAdddecoder_output/MatMul:product:0-decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
decoder_output/BiasAdd?
decoder_output/SigmoidSigmoiddecoder_output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
decoder_output/Sigmoid?
IdentityIdentitydecoder_output/Sigmoid:y:0&^decoder_output/BiasAdd/ReadVariableOp%^decoder_output/MatMul/ReadVariableOp%^dense_layer_1/BiasAdd/ReadVariableOp$^dense_layer_1/MatMul/ReadVariableOp%^dense_layer_2/BiasAdd/ReadVariableOp$^dense_layer_2/MatMul/ReadVariableOp$^latent_layer/BiasAdd/ReadVariableOp#^latent_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2N
%decoder_output/BiasAdd/ReadVariableOp%decoder_output/BiasAdd/ReadVariableOp2L
$decoder_output/MatMul/ReadVariableOp$decoder_output/MatMul/ReadVariableOp2L
$dense_layer_1/BiasAdd/ReadVariableOp$dense_layer_1/BiasAdd/ReadVariableOp2J
#dense_layer_1/MatMul/ReadVariableOp#dense_layer_1/MatMul/ReadVariableOp2L
$dense_layer_2/BiasAdd/ReadVariableOp$dense_layer_2/BiasAdd/ReadVariableOp2J
#dense_layer_2/MatMul/ReadVariableOp#dense_layer_2/MatMul/ReadVariableOp2J
#latent_layer/BiasAdd/ReadVariableOp#latent_layer/BiasAdd/ReadVariableOp2H
"latent_layer/MatMul/ReadVariableOp"latent_layer/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
H__inference_dense_layer_2_layer_call_and_return_conditional_losses_32132

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference__traced_save_32228
file_prefix2
.savev2_latent_layer_kernel_read_readvariableop0
,savev2_latent_layer_bias_read_readvariableop3
/savev2_dense_layer_2_kernel_read_readvariableop1
-savev2_dense_layer_2_bias_read_readvariableop3
/savev2_dense_layer_1_kernel_read_readvariableop1
-savev2_dense_layer_1_bias_read_readvariableop4
0savev2_decoder_output_kernel_read_readvariableop2
.savev2_decoder_output_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*?
value?B?	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_latent_layer_kernel_read_readvariableop,savev2_latent_layer_bias_read_readvariableop/savev2_dense_layer_2_kernel_read_readvariableop-savev2_dense_layer_2_bias_read_readvariableop/savev2_dense_layer_1_kernel_read_readvariableop-savev2_dense_layer_1_bias_read_readvariableop0savev2_decoder_output_kernel_read_readvariableop.savev2_decoder_output_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*W
_input_shapesF
D: ::::: : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::	

_output_shapes
: 
?
?
B__inference_decoder_layer_call_and_return_conditional_losses_31857
input_2
latent_layer_31770
latent_layer_31772
dense_layer_2_31797
dense_layer_2_31799
dense_layer_1_31824
dense_layer_1_31826
decoder_output_31851
decoder_output_31853
identity??&decoder_output/StatefulPartitionedCall?%dense_layer_1/StatefulPartitionedCall?%dense_layer_2/StatefulPartitionedCall?$latent_layer/StatefulPartitionedCall?
$latent_layer/StatefulPartitionedCallStatefulPartitionedCallinput_2latent_layer_31770latent_layer_31772*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_latent_layer_layer_call_and_return_conditional_losses_317592&
$latent_layer/StatefulPartitionedCall?
%dense_layer_2/StatefulPartitionedCallStatefulPartitionedCall-latent_layer/StatefulPartitionedCall:output:0dense_layer_2_31797dense_layer_2_31799*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dense_layer_2_layer_call_and_return_conditional_losses_317862'
%dense_layer_2/StatefulPartitionedCall?
%dense_layer_1/StatefulPartitionedCallStatefulPartitionedCall.dense_layer_2/StatefulPartitionedCall:output:0dense_layer_1_31824dense_layer_1_31826*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_318132'
%dense_layer_1/StatefulPartitionedCall?
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall.dense_layer_1/StatefulPartitionedCall:output:0decoder_output_31851decoder_output_31853*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_318402(
&decoder_output/StatefulPartitionedCall?
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0'^decoder_output/StatefulPartitionedCall&^dense_layer_1/StatefulPartitionedCall&^dense_layer_2/StatefulPartitionedCall%^latent_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall2N
%dense_layer_1/StatefulPartitionedCall%dense_layer_1/StatefulPartitionedCall2N
%dense_layer_2/StatefulPartitionedCall%dense_layer_2/StatefulPartitionedCall2L
$latent_layer/StatefulPartitionedCall$latent_layer/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?
?
'__inference_decoder_layer_call_fn_31972
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_319532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?
?
#__inference_signature_wrapper_31995
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_317442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?	
?
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_32152

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
H__inference_dense_layer_2_layer_call_and_return_conditional_losses_31786

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
-__inference_dense_layer_1_layer_call_fn_32161

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_318132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
B__inference_decoder_layer_call_and_return_conditional_losses_31953

inputs
latent_layer_31932
latent_layer_31934
dense_layer_2_31937
dense_layer_2_31939
dense_layer_1_31942
dense_layer_1_31944
decoder_output_31947
decoder_output_31949
identity??&decoder_output/StatefulPartitionedCall?%dense_layer_1/StatefulPartitionedCall?%dense_layer_2/StatefulPartitionedCall?$latent_layer/StatefulPartitionedCall?
$latent_layer/StatefulPartitionedCallStatefulPartitionedCallinputslatent_layer_31932latent_layer_31934*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_latent_layer_layer_call_and_return_conditional_losses_317592&
$latent_layer/StatefulPartitionedCall?
%dense_layer_2/StatefulPartitionedCallStatefulPartitionedCall-latent_layer/StatefulPartitionedCall:output:0dense_layer_2_31937dense_layer_2_31939*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dense_layer_2_layer_call_and_return_conditional_losses_317862'
%dense_layer_2/StatefulPartitionedCall?
%dense_layer_1/StatefulPartitionedCallStatefulPartitionedCall.dense_layer_2/StatefulPartitionedCall:output:0dense_layer_1_31942dense_layer_1_31944*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_318132'
%dense_layer_1/StatefulPartitionedCall?
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall.dense_layer_1/StatefulPartitionedCall:output:0decoder_output_31947decoder_output_31949*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_318402(
&decoder_output/StatefulPartitionedCall?
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0'^decoder_output/StatefulPartitionedCall&^dense_layer_1/StatefulPartitionedCall&^dense_layer_2/StatefulPartitionedCall%^latent_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall2N
%dense_layer_1/StatefulPartitionedCall%dense_layer_1/StatefulPartitionedCall2N
%dense_layer_2/StatefulPartitionedCall%dense_layer_2/StatefulPartitionedCall2L
$latent_layer/StatefulPartitionedCall$latent_layer/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_20
serving_default_input_2:0?????????B
decoder_output0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?.
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
loss
trainable_variables
	regularization_losses

	variables
	keras_api

signatures
>__call__
?_default_save_signature
*@&call_and_return_all_conditional_losses"?+
_tf_keras_network?+{"class_name": "Functional", "name": "decoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "latent_layer", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "latent_layer", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_layer_2", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_layer_2", "inbound_nodes": [[["latent_layer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_layer_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_layer_1", "inbound_nodes": [[["dense_layer_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "decoder_output", "trainable": true, "dtype": "float32", "units": 25, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_output", "inbound_nodes": [[["dense_layer_1", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["decoder_output", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "latent_layer", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "latent_layer", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_layer_2", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_layer_2", "inbound_nodes": [[["latent_layer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_layer_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_layer_1", "inbound_nodes": [[["dense_layer_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "decoder_output", "trainable": true, "dtype": "float32", "units": 25, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_output", "inbound_nodes": [[["dense_layer_1", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["decoder_output", 0, 0]]}}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.001, "decay": 0.0, "rho": 0.9, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
?

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
A__call__
*B&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "latent_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "latent_layer", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
?

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
C__call__
*D&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_layer_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_layer_2", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}}
?

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
E__call__
*F&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_layer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_layer_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
?

kernel
 bias
!trainable_variables
"regularization_losses
#	variables
$	keras_api
G__call__
*H&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "decoder_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "decoder_output", "trainable": true, "dtype": "float32", "units": 25, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
"
	optimizer
 "
trackable_dict_wrapper
X
0
1
2
3
4
5
6
 7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
 7"
trackable_list_wrapper
?
trainable_variables
%layer_regularization_losses
	regularization_losses
&metrics

'layers
(layer_metrics

	variables
)non_trainable_variables
>__call__
?_default_save_signature
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
,
Iserving_default"
signature_map
%:#2latent_layer/kernel
:2latent_layer/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
trainable_variables
*layer_regularization_losses
+metrics
regularization_losses

,layers
-layer_metrics
	variables
.non_trainable_variables
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
&:$2dense_layer_2/kernel
 :2dense_layer_2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
trainable_variables
/layer_regularization_losses
0metrics
regularization_losses

1layers
2layer_metrics
	variables
3non_trainable_variables
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
&:$ 2dense_layer_1/kernel
 : 2dense_layer_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
trainable_variables
4layer_regularization_losses
5metrics
regularization_losses

6layers
7layer_metrics
	variables
8non_trainable_variables
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
':% 2decoder_output/kernel
!:2decoder_output/bias
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
?
!trainable_variables
9layer_regularization_losses
:metrics
"regularization_losses

;layers
<layer_metrics
#	variables
=non_trainable_variables
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?2?
'__inference_decoder_layer_call_fn_32080
'__inference_decoder_layer_call_fn_32101
'__inference_decoder_layer_call_fn_31927
'__inference_decoder_layer_call_fn_31972?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_31744?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_2?????????
?2?
B__inference_decoder_layer_call_and_return_conditional_losses_32059
B__inference_decoder_layer_call_and_return_conditional_losses_32027
B__inference_decoder_layer_call_and_return_conditional_losses_31881
B__inference_decoder_layer_call_and_return_conditional_losses_31857?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_latent_layer_layer_call_fn_32121?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_latent_layer_layer_call_and_return_conditional_losses_32112?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_dense_layer_2_layer_call_fn_32141?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_dense_layer_2_layer_call_and_return_conditional_losses_32132?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_dense_layer_1_layer_call_fn_32161?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_32152?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_decoder_output_layer_call_fn_32181?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_decoder_output_layer_call_and_return_conditional_losses_32172?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_31995input_2"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_31744} 0?-
&?#
!?
input_2?????????
? "??<
:
decoder_output(?%
decoder_output??????????
B__inference_decoder_layer_call_and_return_conditional_losses_31857k 8?5
.?+
!?
input_2?????????
p

 
? "%?"
?
0?????????
? ?
B__inference_decoder_layer_call_and_return_conditional_losses_31881k 8?5
.?+
!?
input_2?????????
p 

 
? "%?"
?
0?????????
? ?
B__inference_decoder_layer_call_and_return_conditional_losses_32027j 7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
B__inference_decoder_layer_call_and_return_conditional_losses_32059j 7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
'__inference_decoder_layer_call_fn_31927^ 8?5
.?+
!?
input_2?????????
p

 
? "???????????
'__inference_decoder_layer_call_fn_31972^ 8?5
.?+
!?
input_2?????????
p 

 
? "???????????
'__inference_decoder_layer_call_fn_32080] 7?4
-?*
 ?
inputs?????????
p

 
? "???????????
'__inference_decoder_layer_call_fn_32101] 7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
I__inference_decoder_output_layer_call_and_return_conditional_losses_32172\ /?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? ?
.__inference_decoder_output_layer_call_fn_32181O /?,
%?"
 ?
inputs????????? 
? "???????????
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_32152\/?,
%?"
 ?
inputs?????????
? "%?"
?
0????????? 
? ?
-__inference_dense_layer_1_layer_call_fn_32161O/?,
%?"
 ?
inputs?????????
? "?????????? ?
H__inference_dense_layer_2_layer_call_and_return_conditional_losses_32132\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
-__inference_dense_layer_2_layer_call_fn_32141O/?,
%?"
 ?
inputs?????????
? "???????????
G__inference_latent_layer_layer_call_and_return_conditional_losses_32112\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? 
,__inference_latent_layer_layer_call_fn_32121O/?,
%?"
 ?
inputs?????????
? "???????????
#__inference_signature_wrapper_31995? ;?8
? 
1?.
,
input_2!?
input_2?????????"??<
:
decoder_output(?%
decoder_output?????????