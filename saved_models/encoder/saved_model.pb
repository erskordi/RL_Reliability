äË
ã
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
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
,
Exp
x"T
y"T"
Ttype:

2
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
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68Öñ
v
x_mean/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namex_mean/kernel
o
!x_mean/kernel/Read/ReadVariableOpReadVariableOpx_mean/kernel*
_output_shapes

:*
dtype0
n
x_mean/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namex_mean/bias
g
x_mean/bias/Read/ReadVariableOpReadVariableOpx_mean/bias*
_output_shapes
:*
dtype0
z
x_logvar/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namex_logvar/kernel
s
#x_logvar/kernel/Read/ReadVariableOpReadVariableOpx_logvar/kernel*
_output_shapes

:*
dtype0
r
x_logvar/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namex_logvar/bias
k
!x_logvar/bias/Read/ReadVariableOpReadVariableOpx_logvar/bias*
_output_shapes
:*
dtype0
J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?

NoOpNoOp
Ö
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB Bû

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
¦

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses*

%	keras_api* 

&	keras_api* 

'	keras_api* 

(	keras_api* 

)	keras_api* 

*	keras_api* 

+	keras_api* 

,	keras_api* 

-	keras_api* 
 
0
1
2
3*
 
0
1
2
3*
* 
°
.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

3serving_default* 
]W
VARIABLE_VALUEx_mean/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEx_mean/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEx_logvar/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEx_logvar/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
Z
0
1
2
3
4
5
6
7
	8

9
10
11*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
z
serving_default_input_1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
¨
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1x_mean/kernelx_mean/biasx_logvar/kernelx_logvar/biasConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_52961
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¯
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!x_mean/kernel/Read/ReadVariableOpx_mean/bias/Read/ReadVariableOp#x_logvar/kernel/Read/ReadVariableOp!x_logvar/bias/Read/ReadVariableOpConst_1*
Tin

2*
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
GPU2*0J 8 *'
f"R 
__inference__traced_save_53037
Ø
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamex_mean/kernelx_mean/biasx_logvar/kernelx_logvar/bias*
Tin	
2*
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
GPU2*0J 8 **
f%R#
!__inference__traced_restore_53059ÿÉ

¥
__inference__traced_save_53037
file_prefix,
(savev2_x_mean_kernel_read_readvariableop*
&savev2_x_mean_bias_read_readvariableop.
*savev2_x_logvar_kernel_read_readvariableop,
(savev2_x_logvar_bias_read_readvariableop
savev2_const_1

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: æ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHw
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B Þ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_x_mean_kernel_read_readvariableop&savev2_x_mean_bias_read_readvariableop*savev2_x_logvar_kernel_read_readvariableop(savev2_x_logvar_bias_read_readvariableopsavev2_const_1"/device:CPU:0*
_output_shapes
 *
dtypes	
2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*7
_input_shapes&
$: ::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
¿(
Ï
B__inference_encoder_layer_call_and_return_conditional_losses_52784
input_1
x_mean_52749:
x_mean_52751: 
x_logvar_52756:
x_logvar_52758:
unknown
identity

identity_1

identity_2¢ x_logvar/StatefulPartitionedCall¢x_mean/StatefulPartitionedCallé
x_mean/StatefulPartitionedCallStatefulPartitionedCallinput_1x_mean_52749x_mean_52751*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_x_mean_layer_call_and_return_conditional_losses_52567q
tf.compat.v1.shape_1/ShapeShape'x_mean/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:o
tf.compat.v1.shape/ShapeShape'x_mean/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:ñ
 x_logvar/StatefulPartitionedCallStatefulPartitionedCallinput_1x_logvar_52756x_logvar_52758*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_x_logvar_layer_call_and_return_conditional_losses_52585v
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:È
&tf.__operators__.getitem/strided_sliceStridedSlice!tf.compat.v1.shape/Shape:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
(tf.__operators__.getitem_1/strided_sliceStridedSlice#tf.compat.v1.shape_1/Shape:output:07tf.__operators__.getitem_1/strided_slice/stack:output:09tf.__operators__.getitem_1/strided_slice/stack_1:output:09tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
tf.math.multiply/MulMulunknown)x_logvar/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
tf.math.exp/ExpExptf.math.multiply/Mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
$tf.random.normal/random_normal/shapePack/tf.__operators__.getitem/strided_slice:output:01tf.__operators__.getitem_1/strided_slice:output:0*
N*
T0*
_output_shapes
:h
#tf.random.normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    j
%tf.random.normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ø
3tf.random.normal/random_normal/RandomStandardNormalRandomStandardNormal-tf.random.normal/random_normal/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2§ÌäÉ
"tf.random.normal/random_normal/mulMul<tf.random.normal/random_normal/RandomStandardNormal:output:0.tf.random.normal/random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
tf.random.normal/random_normalAddV2&tf.random.normal/random_normal/mul:z:0,tf.random.normal/random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
tf.math.multiply_1/MulMultf.math.exp/Exp:y:0"tf.random.normal/random_normal:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
tf.__operators__.add/AddV2AddV2'x_mean/StatefulPartitionedCall:output:0tf.math.multiply_1/Mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
IdentityIdentity'x_mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz

Identity_1Identity)x_logvar/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo

Identity_2Identitytf.__operators__.add/AddV2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp!^x_logvar/StatefulPartitionedCall^x_mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : : 2D
 x_logvar/StatefulPartitionedCall x_logvar/StatefulPartitionedCall2@
x_mean/StatefulPartitionedCallx_mean/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:

_output_shapes
: 
 2

 __inference__wrapped_model_52550
input_1?
-encoder_x_mean_matmul_readvariableop_resource:<
.encoder_x_mean_biasadd_readvariableop_resource:A
/encoder_x_logvar_matmul_readvariableop_resource:>
0encoder_x_logvar_biasadd_readvariableop_resource:
encoder_52535
identity

identity_1

identity_2¢'encoder/x_logvar/BiasAdd/ReadVariableOp¢&encoder/x_logvar/MatMul/ReadVariableOp¢%encoder/x_mean/BiasAdd/ReadVariableOp¢$encoder/x_mean/MatMul/ReadVariableOp
$encoder/x_mean/MatMul/ReadVariableOpReadVariableOp-encoder_x_mean_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
encoder/x_mean/MatMulMatMulinput_1,encoder/x_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%encoder/x_mean/BiasAdd/ReadVariableOpReadVariableOp.encoder_x_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0£
encoder/x_mean/BiasAddBiasAddencoder/x_mean/MatMul:product:0-encoder/x_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
"encoder/tf.compat.v1.shape_1/ShapeShapeencoder/x_mean/BiasAdd:output:0*
T0*
_output_shapes
:o
 encoder/tf.compat.v1.shape/ShapeShapeencoder/x_mean/BiasAdd:output:0*
T0*
_output_shapes
:
&encoder/x_logvar/MatMul/ReadVariableOpReadVariableOp/encoder_x_logvar_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
encoder/x_logvar/MatMulMatMulinput_1.encoder/x_logvar/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'encoder/x_logvar/BiasAdd/ReadVariableOpReadVariableOp0encoder_x_logvar_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0©
encoder/x_logvar/BiasAddBiasAdd!encoder/x_logvar/MatMul:product:0/encoder/x_logvar/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
4encoder/tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6encoder/tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6encoder/tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ð
.encoder/tf.__operators__.getitem/strided_sliceStridedSlice)encoder/tf.compat.v1.shape/Shape:output:0=encoder/tf.__operators__.getitem/strided_slice/stack:output:0?encoder/tf.__operators__.getitem/strided_slice/stack_1:output:0?encoder/tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
6encoder/tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
8encoder/tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
8encoder/tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ú
0encoder/tf.__operators__.getitem_1/strided_sliceStridedSlice+encoder/tf.compat.v1.shape_1/Shape:output:0?encoder/tf.__operators__.getitem_1/strided_slice/stack:output:0Aencoder/tf.__operators__.getitem_1/strided_slice/stack_1:output:0Aencoder/tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
encoder/tf.math.multiply/MulMulencoder_52535!encoder/x_logvar/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
encoder/tf.math.exp/ExpExp encoder/tf.math.multiply/Mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÖ
,encoder/tf.random.normal/random_normal/shapePack7encoder/tf.__operators__.getitem/strided_slice:output:09encoder/tf.__operators__.getitem_1/strided_slice:output:0*
N*
T0*
_output_shapes
:p
+encoder/tf.random.normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    r
-encoder/tf.random.normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?è
;encoder/tf.random.normal/random_normal/RandomStandardNormalRandomStandardNormal5encoder/tf.random.normal/random_normal/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2§Ìäá
*encoder/tf.random.normal/random_normal/mulMulDencoder/tf.random.normal/random_normal/RandomStandardNormal:output:06encoder/tf.random.normal/random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
&encoder/tf.random.normal/random_normalAddV2.encoder/tf.random.normal/random_normal/mul:z:04encoder/tf.random.normal/random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
encoder/tf.math.multiply_1/MulMulencoder/tf.math.exp/Exp:y:0*encoder/tf.random.normal/random_normal:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
"encoder/tf.__operators__.add/AddV2AddV2encoder/x_mean/BiasAdd:output:0"encoder/tf.math.multiply_1/Mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
IdentityIdentity&encoder/tf.__operators__.add/AddV2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_1Identity!encoder/x_logvar/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp

Identity_2Identityencoder/x_mean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
NoOpNoOp(^encoder/x_logvar/BiasAdd/ReadVariableOp'^encoder/x_logvar/MatMul/ReadVariableOp&^encoder/x_mean/BiasAdd/ReadVariableOp%^encoder/x_mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : : 2R
'encoder/x_logvar/BiasAdd/ReadVariableOp'encoder/x_logvar/BiasAdd/ReadVariableOp2P
&encoder/x_logvar/MatMul/ReadVariableOp&encoder/x_logvar/MatMul/ReadVariableOp2N
%encoder/x_mean/BiasAdd/ReadVariableOp%encoder/x_mean/BiasAdd/ReadVariableOp2L
$encoder/x_mean/MatMul/ReadVariableOp$encoder/x_mean/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:

_output_shapes
: 
Á
ß
!__inference__traced_restore_53059
file_prefix0
assignvariableop_x_mean_kernel:,
assignvariableop_1_x_mean_bias:4
"assignvariableop_2_x_logvar_kernel:.
 assignvariableop_3_x_logvar_bias:

identity_5¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3é
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHz
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B ·
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*(
_output_shapes
:::::*
dtypes	
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_x_mean_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_x_mean_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_x_logvar_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_x_logvar_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ¬

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_5IdentityIdentity_4:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3*"
_acd_function_control_output(*
_output_shapes
 "!

identity_5Identity_5:output:0*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_3:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Æ	
ô
C__inference_x_logvar_layer_call_and_return_conditional_losses_52585

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡
ö
#__inference_signature_wrapper_52961
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3
identity

identity_1

identity_2¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_52550o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:

_output_shapes
: 
Ä	
ò
A__inference_x_mean_layer_call_and_return_conditional_losses_52567

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã

(__inference_x_logvar_layer_call_fn_52989

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_x_logvar_layer_call_and_return_conditional_losses_52585o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä
ù
'__inference_encoder_layer_call_fn_52860

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3
identity

identity_1

identity_2¢StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_52710o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
Æ	
ô
C__inference_x_logvar_layer_call_and_return_conditional_losses_52999

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä
ù
'__inference_encoder_layer_call_fn_52841

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3
identity

identity_1

identity_2¢StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_52614o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
»(
Î
B__inference_encoder_layer_call_and_return_conditional_losses_52710

inputs
x_mean_52675:
x_mean_52677: 
x_logvar_52682:
x_logvar_52684:
unknown
identity

identity_1

identity_2¢ x_logvar/StatefulPartitionedCall¢x_mean/StatefulPartitionedCallè
x_mean/StatefulPartitionedCallStatefulPartitionedCallinputsx_mean_52675x_mean_52677*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_x_mean_layer_call_and_return_conditional_losses_52567q
tf.compat.v1.shape_1/ShapeShape'x_mean/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:o
tf.compat.v1.shape/ShapeShape'x_mean/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:ð
 x_logvar/StatefulPartitionedCallStatefulPartitionedCallinputsx_logvar_52682x_logvar_52684*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_x_logvar_layer_call_and_return_conditional_losses_52585v
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:È
&tf.__operators__.getitem/strided_sliceStridedSlice!tf.compat.v1.shape/Shape:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
(tf.__operators__.getitem_1/strided_sliceStridedSlice#tf.compat.v1.shape_1/Shape:output:07tf.__operators__.getitem_1/strided_slice/stack:output:09tf.__operators__.getitem_1/strided_slice/stack_1:output:09tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
tf.math.multiply/MulMulunknown)x_logvar/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
tf.math.exp/ExpExptf.math.multiply/Mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
$tf.random.normal/random_normal/shapePack/tf.__operators__.getitem/strided_slice:output:01tf.__operators__.getitem_1/strided_slice:output:0*
N*
T0*
_output_shapes
:h
#tf.random.normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    j
%tf.random.normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ø
3tf.random.normal/random_normal/RandomStandardNormalRandomStandardNormal-tf.random.normal/random_normal/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2§ÌäÉ
"tf.random.normal/random_normal/mulMul<tf.random.normal/random_normal/RandomStandardNormal:output:0.tf.random.normal/random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
tf.random.normal/random_normalAddV2&tf.random.normal/random_normal/mul:z:0,tf.random.normal/random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
tf.math.multiply_1/MulMultf.math.exp/Exp:y:0"tf.random.normal/random_normal:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
tf.__operators__.add/AddV2AddV2'x_mean/StatefulPartitionedCall:output:0tf.math.multiply_1/Mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
IdentityIdentity'x_mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz

Identity_1Identity)x_logvar/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo

Identity_2Identitytf.__operators__.add/AddV2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp!^x_logvar/StatefulPartitionedCall^x_mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : : 2D
 x_logvar/StatefulPartitionedCall x_logvar/StatefulPartitionedCall2@
x_mean/StatefulPartitionedCallx_mean/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
»(
Î
B__inference_encoder_layer_call_and_return_conditional_losses_52614

inputs
x_mean_52568:
x_mean_52570: 
x_logvar_52586:
x_logvar_52588:
unknown
identity

identity_1

identity_2¢ x_logvar/StatefulPartitionedCall¢x_mean/StatefulPartitionedCallè
x_mean/StatefulPartitionedCallStatefulPartitionedCallinputsx_mean_52568x_mean_52570*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_x_mean_layer_call_and_return_conditional_losses_52567q
tf.compat.v1.shape_1/ShapeShape'x_mean/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:o
tf.compat.v1.shape/ShapeShape'x_mean/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:ð
 x_logvar/StatefulPartitionedCallStatefulPartitionedCallinputsx_logvar_52586x_logvar_52588*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_x_logvar_layer_call_and_return_conditional_losses_52585v
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:È
&tf.__operators__.getitem/strided_sliceStridedSlice!tf.compat.v1.shape/Shape:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
(tf.__operators__.getitem_1/strided_sliceStridedSlice#tf.compat.v1.shape_1/Shape:output:07tf.__operators__.getitem_1/strided_slice/stack:output:09tf.__operators__.getitem_1/strided_slice/stack_1:output:09tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
tf.math.multiply/MulMulunknown)x_logvar/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
tf.math.exp/ExpExptf.math.multiply/Mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
$tf.random.normal/random_normal/shapePack/tf.__operators__.getitem/strided_slice:output:01tf.__operators__.getitem_1/strided_slice:output:0*
N*
T0*
_output_shapes
:h
#tf.random.normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    j
%tf.random.normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ø
3tf.random.normal/random_normal/RandomStandardNormalRandomStandardNormal-tf.random.normal/random_normal/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2§ÌäÉ
"tf.random.normal/random_normal/mulMul<tf.random.normal/random_normal/RandomStandardNormal:output:0.tf.random.normal/random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
tf.random.normal/random_normalAddV2&tf.random.normal/random_normal/mul:z:0,tf.random.normal/random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
tf.math.multiply_1/MulMultf.math.exp/Exp:y:0"tf.random.normal/random_normal:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
tf.__operators__.add/AddV2AddV2'x_mean/StatefulPartitionedCall:output:0tf.math.multiply_1/Mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
IdentityIdentity'x_mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz

Identity_1Identity)x_logvar/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo

Identity_2Identitytf.__operators__.add/AddV2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp!^x_logvar/StatefulPartitionedCall^x_mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : : 2D
 x_logvar/StatefulPartitionedCall x_logvar/StatefulPartitionedCall2@
x_mean/StatefulPartitionedCallx_mean/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
Ç
ú
'__inference_encoder_layer_call_fn_52746
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3
identity

identity_1

identity_2¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_52710o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:

_output_shapes
: 
Ä	
ò
A__inference_x_mean_layer_call_and_return_conditional_losses_52980

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
-
ò
B__inference_encoder_layer_call_and_return_conditional_losses_52900

inputs7
%x_mean_matmul_readvariableop_resource:4
&x_mean_biasadd_readvariableop_resource:9
'x_logvar_matmul_readvariableop_resource:6
(x_logvar_biasadd_readvariableop_resource:
unknown
identity

identity_1

identity_2¢x_logvar/BiasAdd/ReadVariableOp¢x_logvar/MatMul/ReadVariableOp¢x_mean/BiasAdd/ReadVariableOp¢x_mean/MatMul/ReadVariableOp
x_mean/MatMul/ReadVariableOpReadVariableOp%x_mean_matmul_readvariableop_resource*
_output_shapes

:*
dtype0w
x_mean/MatMulMatMulinputs$x_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
x_mean/BiasAdd/ReadVariableOpReadVariableOp&x_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
x_mean/BiasAddBiasAddx_mean/MatMul:product:0%x_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
tf.compat.v1.shape_1/ShapeShapex_mean/BiasAdd:output:0*
T0*
_output_shapes
:_
tf.compat.v1.shape/ShapeShapex_mean/BiasAdd:output:0*
T0*
_output_shapes
:
x_logvar/MatMul/ReadVariableOpReadVariableOp'x_logvar_matmul_readvariableop_resource*
_output_shapes

:*
dtype0{
x_logvar/MatMulMatMulinputs&x_logvar/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
x_logvar/BiasAdd/ReadVariableOpReadVariableOp(x_logvar_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
x_logvar/BiasAddBiasAddx_logvar/MatMul:product:0'x_logvar/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:È
&tf.__operators__.getitem/strided_sliceStridedSlice!tf.compat.v1.shape/Shape:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
(tf.__operators__.getitem_1/strided_sliceStridedSlice#tf.compat.v1.shape_1/Shape:output:07tf.__operators__.getitem_1/strided_slice/stack:output:09tf.__operators__.getitem_1/strided_slice/stack_1:output:09tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
tf.math.multiply/MulMulunknownx_logvar/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
tf.math.exp/ExpExptf.math.multiply/Mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
$tf.random.normal/random_normal/shapePack/tf.__operators__.getitem/strided_slice:output:01tf.__operators__.getitem_1/strided_slice:output:0*
N*
T0*
_output_shapes
:h
#tf.random.normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    j
%tf.random.normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ø
3tf.random.normal/random_normal/RandomStandardNormalRandomStandardNormal-tf.random.normal/random_normal/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2§ÌäÉ
"tf.random.normal/random_normal/mulMul<tf.random.normal/random_normal/RandomStandardNormal:output:0.tf.random.normal/random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
tf.random.normal/random_normalAddV2&tf.random.normal/random_normal/mul:z:0,tf.random.normal/random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
tf.math.multiply_1/MulMultf.math.exp/Exp:y:0"tf.random.normal/random_normal:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
tf.__operators__.add/AddV2AddV2x_mean/BiasAdd:output:0tf.math.multiply_1/Mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
IdentityIdentityx_mean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj

Identity_1Identityx_logvar/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo

Identity_2Identitytf.__operators__.add/AddV2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
NoOpNoOp ^x_logvar/BiasAdd/ReadVariableOp^x_logvar/MatMul/ReadVariableOp^x_mean/BiasAdd/ReadVariableOp^x_mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : : 2B
x_logvar/BiasAdd/ReadVariableOpx_logvar/BiasAdd/ReadVariableOp2@
x_logvar/MatMul/ReadVariableOpx_logvar/MatMul/ReadVariableOp2>
x_mean/BiasAdd/ReadVariableOpx_mean/BiasAdd/ReadVariableOp2<
x_mean/MatMul/ReadVariableOpx_mean/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
¿(
Ï
B__inference_encoder_layer_call_and_return_conditional_losses_52822
input_1
x_mean_52787:
x_mean_52789: 
x_logvar_52794:
x_logvar_52796:
unknown
identity

identity_1

identity_2¢ x_logvar/StatefulPartitionedCall¢x_mean/StatefulPartitionedCallé
x_mean/StatefulPartitionedCallStatefulPartitionedCallinput_1x_mean_52787x_mean_52789*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_x_mean_layer_call_and_return_conditional_losses_52567q
tf.compat.v1.shape_1/ShapeShape'x_mean/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:o
tf.compat.v1.shape/ShapeShape'x_mean/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:ñ
 x_logvar/StatefulPartitionedCallStatefulPartitionedCallinput_1x_logvar_52794x_logvar_52796*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_x_logvar_layer_call_and_return_conditional_losses_52585v
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:È
&tf.__operators__.getitem/strided_sliceStridedSlice!tf.compat.v1.shape/Shape:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
(tf.__operators__.getitem_1/strided_sliceStridedSlice#tf.compat.v1.shape_1/Shape:output:07tf.__operators__.getitem_1/strided_slice/stack:output:09tf.__operators__.getitem_1/strided_slice/stack_1:output:09tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
tf.math.multiply/MulMulunknown)x_logvar/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
tf.math.exp/ExpExptf.math.multiply/Mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
$tf.random.normal/random_normal/shapePack/tf.__operators__.getitem/strided_slice:output:01tf.__operators__.getitem_1/strided_slice:output:0*
N*
T0*
_output_shapes
:h
#tf.random.normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    j
%tf.random.normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ø
3tf.random.normal/random_normal/RandomStandardNormalRandomStandardNormal-tf.random.normal/random_normal/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2§ÌäÉ
"tf.random.normal/random_normal/mulMul<tf.random.normal/random_normal/RandomStandardNormal:output:0.tf.random.normal/random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
tf.random.normal/random_normalAddV2&tf.random.normal/random_normal/mul:z:0,tf.random.normal/random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
tf.math.multiply_1/MulMultf.math.exp/Exp:y:0"tf.random.normal/random_normal:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
tf.__operators__.add/AddV2AddV2'x_mean/StatefulPartitionedCall:output:0tf.math.multiply_1/Mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
IdentityIdentity'x_mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz

Identity_1Identity)x_logvar/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo

Identity_2Identitytf.__operators__.add/AddV2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp!^x_logvar/StatefulPartitionedCall^x_mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : : 2D
 x_logvar/StatefulPartitionedCall x_logvar/StatefulPartitionedCall2@
x_mean/StatefulPartitionedCallx_mean/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:

_output_shapes
: 
Ç
ú
'__inference_encoder_layer_call_fn_52631
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3
identity

identity_1

identity_2¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_52614o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:

_output_shapes
: 
¿

&__inference_x_mean_layer_call_fn_52970

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_x_mean_layer_call_and_return_conditional_losses_52567o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
-
ò
B__inference_encoder_layer_call_and_return_conditional_losses_52940

inputs7
%x_mean_matmul_readvariableop_resource:4
&x_mean_biasadd_readvariableop_resource:9
'x_logvar_matmul_readvariableop_resource:6
(x_logvar_biasadd_readvariableop_resource:
unknown
identity

identity_1

identity_2¢x_logvar/BiasAdd/ReadVariableOp¢x_logvar/MatMul/ReadVariableOp¢x_mean/BiasAdd/ReadVariableOp¢x_mean/MatMul/ReadVariableOp
x_mean/MatMul/ReadVariableOpReadVariableOp%x_mean_matmul_readvariableop_resource*
_output_shapes

:*
dtype0w
x_mean/MatMulMatMulinputs$x_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
x_mean/BiasAdd/ReadVariableOpReadVariableOp&x_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
x_mean/BiasAddBiasAddx_mean/MatMul:product:0%x_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
tf.compat.v1.shape_1/ShapeShapex_mean/BiasAdd:output:0*
T0*
_output_shapes
:_
tf.compat.v1.shape/ShapeShapex_mean/BiasAdd:output:0*
T0*
_output_shapes
:
x_logvar/MatMul/ReadVariableOpReadVariableOp'x_logvar_matmul_readvariableop_resource*
_output_shapes

:*
dtype0{
x_logvar/MatMulMatMulinputs&x_logvar/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
x_logvar/BiasAdd/ReadVariableOpReadVariableOp(x_logvar_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
x_logvar/BiasAddBiasAddx_logvar/MatMul:product:0'x_logvar/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:È
&tf.__operators__.getitem/strided_sliceStridedSlice!tf.compat.v1.shape/Shape:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
(tf.__operators__.getitem_1/strided_sliceStridedSlice#tf.compat.v1.shape_1/Shape:output:07tf.__operators__.getitem_1/strided_slice/stack:output:09tf.__operators__.getitem_1/strided_slice/stack_1:output:09tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
tf.math.multiply/MulMulunknownx_logvar/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
tf.math.exp/ExpExptf.math.multiply/Mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
$tf.random.normal/random_normal/shapePack/tf.__operators__.getitem/strided_slice:output:01tf.__operators__.getitem_1/strided_slice:output:0*
N*
T0*
_output_shapes
:h
#tf.random.normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    j
%tf.random.normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ø
3tf.random.normal/random_normal/RandomStandardNormalRandomStandardNormal-tf.random.normal/random_normal/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2§ÌäÉ
"tf.random.normal/random_normal/mulMul<tf.random.normal/random_normal/RandomStandardNormal:output:0.tf.random.normal/random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
tf.random.normal/random_normalAddV2&tf.random.normal/random_normal/mul:z:0,tf.random.normal/random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
tf.math.multiply_1/MulMultf.math.exp/Exp:y:0"tf.random.normal/random_normal:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
tf.__operators__.add/AddV2AddV2x_mean/BiasAdd:output:0tf.math.multiply_1/Mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
IdentityIdentityx_mean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj

Identity_1Identityx_logvar/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo

Identity_2Identitytf.__operators__.add/AddV2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
NoOpNoOp ^x_logvar/BiasAdd/ReadVariableOp^x_logvar/MatMul/ReadVariableOp^x_mean/BiasAdd/ReadVariableOp^x_mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : : 2B
x_logvar/BiasAdd/ReadVariableOpx_logvar/BiasAdd/ReadVariableOp2@
x_logvar/MatMul/ReadVariableOpx_logvar/MatMul/ReadVariableOp2>
x_mean/BiasAdd/ReadVariableOpx_mean/BiasAdd/ReadVariableOp2<
x_mean/MatMul/ReadVariableOpx_mean/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: "ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*±
serving_default
;
input_10
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿH
tf.__operators__.add0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ<
x_logvar0
StatefulPartitionedCall:1ÿÿÿÿÿÿÿÿÿ:
x_mean0
StatefulPartitionedCall:2ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:D
¦
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses"
_tf_keras_layer
(
%	keras_api"
_tf_keras_layer
(
&	keras_api"
_tf_keras_layer
(
'	keras_api"
_tf_keras_layer
(
(	keras_api"
_tf_keras_layer
(
)	keras_api"
_tf_keras_layer
(
*	keras_api"
_tf_keras_layer
(
+	keras_api"
_tf_keras_layer
(
,	keras_api"
_tf_keras_layer
(
-	keras_api"
_tf_keras_layer
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ê2ç
'__inference_encoder_layer_call_fn_52631
'__inference_encoder_layer_call_fn_52841
'__inference_encoder_layer_call_fn_52860
'__inference_encoder_layer_call_fn_52746À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ö2Ó
B__inference_encoder_layer_call_and_return_conditional_losses_52900
B__inference_encoder_layer_call_and_return_conditional_losses_52940
B__inference_encoder_layer_call_and_return_conditional_losses_52784
B__inference_encoder_layer_call_and_return_conditional_losses_52822À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ËBÈ
 __inference__wrapped_model_52550input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
3serving_default"
signature_map
:2x_mean/kernel
:2x_mean/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ð2Í
&__inference_x_mean_layer_call_fn_52970¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_x_mean_layer_call_and_return_conditional_losses_52980¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
!:2x_logvar/kernel
:2x_logvar/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_x_logvar_layer_call_fn_52989¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_x_logvar_layer_call_and_return_conditional_losses_52999¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
	8

9
10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÊBÇ
#__inference_signature_wrapper_52961input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
	J
Const
 __inference__wrapped_model_52550ä>0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "¨ª¤
F
tf.__operators__.add.+
tf.__operators__.addÿÿÿÿÿÿÿÿÿ
.
x_logvar"
x_logvarÿÿÿÿÿÿÿÿÿ
*
x_mean 
x_meanÿÿÿÿÿÿÿÿÿô
B__inference_encoder_layer_call_and_return_conditional_losses_52784­>8¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "j¢g
`]

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ

0/2ÿÿÿÿÿÿÿÿÿ
 ô
B__inference_encoder_layer_call_and_return_conditional_losses_52822­>8¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "j¢g
`]

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ

0/2ÿÿÿÿÿÿÿÿÿ
 ó
B__inference_encoder_layer_call_and_return_conditional_losses_52900¬>7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "j¢g
`]

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ

0/2ÿÿÿÿÿÿÿÿÿ
 ó
B__inference_encoder_layer_call_and_return_conditional_losses_52940¬>7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "j¢g
`]

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ

0/2ÿÿÿÿÿÿÿÿÿ
 É
'__inference_encoder_layer_call_fn_52631>8¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ZW

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ

2ÿÿÿÿÿÿÿÿÿÉ
'__inference_encoder_layer_call_fn_52746>8¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ZW

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ

2ÿÿÿÿÿÿÿÿÿÈ
'__inference_encoder_layer_call_fn_52841>7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ZW

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ

2ÿÿÿÿÿÿÿÿÿÈ
'__inference_encoder_layer_call_fn_52860>7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ZW

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ

2ÿÿÿÿÿÿÿÿÿ
#__inference_signature_wrapper_52961ï>;¢8
¢ 
1ª.
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ"¨ª¤
F
tf.__operators__.add.+
tf.__operators__.addÿÿÿÿÿÿÿÿÿ
.
x_logvar"
x_logvarÿÿÿÿÿÿÿÿÿ
*
x_mean 
x_meanÿÿÿÿÿÿÿÿÿ£
C__inference_x_logvar_layer_call_and_return_conditional_losses_52999\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
(__inference_x_logvar_layer_call_fn_52989O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¡
A__inference_x_mean_layer_call_and_return_conditional_losses_52980\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 y
&__inference_x_mean_layer_call_fn_52970O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ