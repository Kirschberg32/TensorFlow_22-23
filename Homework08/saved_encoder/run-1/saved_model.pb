??
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
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
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

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
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
?
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ?
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
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
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
executor_typestring ??
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
 ?"serve*2.10.02unknown8??

l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:
*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	?
*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:0*
dtype0
?
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:00* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:00*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:0*
dtype0
?
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:0*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:*
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasdense/kernel
dense/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_52795

NoOpNoOp
?_
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?^
value?^B?^ B?^
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

block1

	block2

flatten
out
call

signatures*
J
0
1
2
3
4
5
6
7
8
9*
J
0
1
2
3
4
5
6
7
8
9*
* 
?
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
trace_0
trace_1
trace_2
 trace_3* 
6
!trace_0
"trace_1
#trace_2
$trace_3* 
* 
?
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses
+conv_layers
,pool
-call*
?
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses
4conv_layers
5pool
6call*
?
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses* 
?
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses

kernel
bias*

Ctrace_0
Dtrace_1* 

Eserving_default* 
MG
VARIABLE_VALUEconv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEconv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_3/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_3/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
dense/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
	1

2
3*
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
 
0
1
2
3*
 
0
1
2
3*
* 
?
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*

Ktrace_0
Ltrace_1* 

Mtrace_0
Ntrace_1* 

O0
P1*
?
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses* 

Wtrace_0
Xtrace_1* 
 
0
1
2
3*
 
0
1
2
3*
* 
?
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*

^trace_0
_trace_1* 

`trace_0
atrace_1* 

b0
c1*
?
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses* 

jtrace_0
ktrace_1* 
* 
* 
* 
?
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses* 

qtrace_0* 

rtrace_0* 

0
1*

0
1*
* 
?
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses*

xtrace_0* 

ytrace_0* 
* 
* 
* 
* 

O0
P1
,2*
* 
* 
* 
* 
* 
* 
* 
?
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses
?
conv_layer
?
activation
	?call*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?
conv_layer
?
activation
	?call*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 

b0
c1
52*
* 
* 
* 
* 
* 
* 
* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?
conv_layer
?
activation
	?call*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?
conv_layer
?
activation
	?call*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
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
* 
* 

0
1*

0
1*


?0* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 

?trace_0
?trace_1* 

0
1*

0
1*


?0* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 

?trace_0
?trace_1* 
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*


?0* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 

?trace_0
?trace_1* 

0
1*

0
1*


?0* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 

?trace_0
?trace_1* 
* 
* 
* 
* 
* 
* 
* 

?trace_0* 
* 

?0
?1*
* 
* 
* 

0
1*

0
1*


?0* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
* 

?trace_0* 
* 

?0
?1*
* 
* 
* 

0
1*

0
1*


?0* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
* 

?trace_0* 
* 

?0
?1*
* 
* 
* 

0
1*

0
1*


?0* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
* 

?trace_0* 
* 

?0
?1*
* 
* 
* 

0
1*

0
1*


?0* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
* 


?0* 
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


?0* 
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


?0* 
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


?0* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_53271
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasdense/kernel
dense/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_53311??	
?
?
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_52534	
input:
 my_cnn_normalization_layer_52514:.
 my_cnn_normalization_layer_52516:<
"my_cnn_normalization_layer_1_52519:0
"my_cnn_normalization_layer_1_52521:
identity??/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp?2my_cnn_normalization_layer/StatefulPartitionedCall?4my_cnn_normalization_layer_1/StatefulPartitionedCall?
2my_cnn_normalization_layer/StatefulPartitionedCallStatefulPartitionedCallinput my_cnn_normalization_layer_52514 my_cnn_normalization_layer_52516*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_48095?
4my_cnn_normalization_layer_1/StatefulPartitionedCallStatefulPartitionedCall;my_cnn_normalization_layer/StatefulPartitionedCall:output:0"my_cnn_normalization_layer_1_52519"my_cnn_normalization_layer_1_52521*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_48111?
max_pooling2d/MaxPoolMaxPool=my_cnn_normalization_layer_1/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
?
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp my_cnn_normalization_layer_52514*&
_output_shapes
:*
dtype0?
 conv2d/kernel/Regularizer/L2LossL2Loss7conv2d/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0)conv2d/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ?
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp"my_cnn_normalization_layer_1_52519*&
_output_shapes
:*
dtype0?
"conv2d_1/kernel/Regularizer/L2LossL2Loss9conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0+conv2d_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: u
IdentityIdentitymax_pooling2d/MaxPool:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp0^conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp3^my_cnn_normalization_layer/StatefulPartitionedCall5^my_cnn_normalization_layer_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2b
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2h
2my_cnn_normalization_layer/StatefulPartitionedCall2my_cnn_normalization_layer/StatefulPartitionedCall2l
4my_cnn_normalization_layer_1/StatefulPartitionedCall4my_cnn_normalization_layer_1/StatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_nameinput
?
?
__inference__traced_save_53271
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
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
_temp/part?
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
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
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

identity_1Identity_1:output:0*?
_input_shapesw
u: :::::0:0:00:0:	?
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:0: 

_output_shapes
:0:,(
&
_output_shapes
:00: 

_output_shapes
:0:%	!

_output_shapes
:	?
: 


_output_shapes
:
:

_output_shapes
: 
?
?
__inference_call_51347
x?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????j
activation/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????t
IdentityIdentityactivation/Relu:activations:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:R N
/
_output_shapes
:?????????

_user_specified_namex
?
?
__inference_call_47643
xA
'conv2d_2_conv2d_readvariableop_resource:06
(conv2d_2_biasadd_readvariableop_resource:0
identity??conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0?
conv2d_2/Conv2DConv2Dx&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0n
activation_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????0v
IdentityIdentityactivation_2/Relu:activations:0^NoOp*
T0*/
_output_shapes
:?????????0?
NoOpNoOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:R N
/
_output_shapes
:?????????

_user_specified_namex
?
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_52288

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
__forward_call_49825
x_0?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:
identity
activation_relu
x 
conv2d_conv2d_readvariableop??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d/Conv2DConv2Dx_0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????j
activation/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????t
IdentityIdentityactivation/Relu:activations:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "0
activation_reluactivation/Relu:activations:0"D
conv2d_conv2d_readvariableop$conv2d/Conv2D/ReadVariableOp:value:0"
identityIdentity:output:0"
xx_0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : *C
backward_function_name)'__inference___backward_call_49809_498262>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:R N
/
_output_shapes
:?????????

_user_specified_namex
?
?
.__inference_my_cnn_block_1_layer_call_fn_53053	
input!
unknown:0
	unknown_0:0#
	unknown_1:00
	unknown_2:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_my_cnn_block_1_layer_call_and_return_conditional_losses_52484w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_nameinput
?/
?
A__inference_my_cnn_layer_call_and_return_conditional_losses_52708
input_1,
my_cnn_block_52667: 
my_cnn_block_52669:,
my_cnn_block_52671: 
my_cnn_block_52673:.
my_cnn_block_1_52676:0"
my_cnn_block_1_52678:0.
my_cnn_block_1_52680:00"
my_cnn_block_1_52682:0
dense_52686:	?

dense_52688:

identity??/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp?dense/StatefulPartitionedCall?$my_cnn_block/StatefulPartitionedCall?&my_cnn_block_1/StatefulPartitionedCall?
$my_cnn_block/StatefulPartitionedCallStatefulPartitionedCallinput_1my_cnn_block_52667my_cnn_block_52669my_cnn_block_52671my_cnn_block_52673*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_52321?
&my_cnn_block_1/StatefulPartitionedCallStatefulPartitionedCall-my_cnn_block/StatefulPartitionedCall:output:0my_cnn_block_1_52676my_cnn_block_1_52678my_cnn_block_1_52680my_cnn_block_1_52682*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_my_cnn_block_1_layer_call_and_return_conditional_losses_52354?
flatten/PartitionedCallPartitionedCall/my_cnn_block_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_52370?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_52686dense_52688*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_52383?
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_52667*&
_output_shapes
:*
dtype0?
 conv2d/kernel/Regularizer/L2LossL2Loss7conv2d/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0)conv2d/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ?
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_52671*&
_output_shapes
:*
dtype0?
"conv2d_1/kernel/Regularizer/L2LossL2Loss9conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0+conv2d_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ?
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_1_52676*&
_output_shapes
:0*
dtype0?
"conv2d_2/kernel/Regularizer/L2LossL2Loss9conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0+conv2d_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ?
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_1_52680*&
_output_shapes
:00*
dtype0?
"conv2d_3/kernel/Regularizer/L2LossL2Loss9conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0+conv2d_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp0^conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp^dense/StatefulPartitionedCall%^my_cnn_block/StatefulPartitionedCall'^my_cnn_block_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2b
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2L
$my_cnn_block/StatefulPartitionedCall$my_cnn_block/StatefulPartitionedCall2P
&my_cnn_block_1/StatefulPartitionedCall&my_cnn_block_1/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
__inference_call_47687
x,
my_cnn_block_47622: 
my_cnn_block_47624:,
my_cnn_block_47626: 
my_cnn_block_47628:.
my_cnn_block_1_47668:0"
my_cnn_block_1_47670:0.
my_cnn_block_1_47672:00"
my_cnn_block_1_47674:07
$dense_matmul_readvariableop_resource:	?
3
%dense_biasadd_readvariableop_resource:

identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?$my_cnn_block/StatefulPartitionedCall?&my_cnn_block_1/StatefulPartitionedCall?
$my_cnn_block/StatefulPartitionedCallStatefulPartitionedCallxmy_cnn_block_47622my_cnn_block_47624my_cnn_block_47626my_cnn_block_47628*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_47621?
&my_cnn_block_1/StatefulPartitionedCallStatefulPartitionedCall-my_cnn_block/StatefulPartitionedCall:output:0my_cnn_block_1_47668my_cnn_block_1_47670my_cnn_block_1_47672my_cnn_block_1_47674*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_47667^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????0	  ?
flatten/ReshapeReshape/my_cnn_block_1/StatefulPartitionedCall:output:0flatten/Const:output:0*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
g
IdentityIdentitydense/Relu:activations:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp%^my_cnn_block/StatefulPartitionedCall'^my_cnn_block_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2L
$my_cnn_block/StatefulPartitionedCall$my_cnn_block/StatefulPartitionedCall2P
&my_cnn_block_1/StatefulPartitionedCall&my_cnn_block_1/StatefulPartitionedCall:R N
/
_output_shapes
:?????????

_user_specified_namex
?
?
__inference_call_51362
xA
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:
identity??conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_1/Conv2DConv2Dx&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????n
activation_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????v
IdentityIdentityactivation_1/Relu:activations:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:R N
/
_output_shapes
:?????????

_user_specified_namex
?	
?
__inference_loss_fn_2_53209T
:conv2d_2_kernel_regularizer_l2loss_readvariableop_resource:0
identity??1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp?
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp:conv2d_2_kernel_regularizer_l2loss_readvariableop_resource*&
_output_shapes
:0*
dtype0?
"conv2d_2/kernel/Regularizer/L2LossL2Loss9conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0+conv2d_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentity#conv2d_2/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp
?
?
__inference_call_50507
x,
my_cnn_block_50479: 
my_cnn_block_50481:,
my_cnn_block_50483: 
my_cnn_block_50485:.
my_cnn_block_1_50488:0"
my_cnn_block_1_50490:0.
my_cnn_block_1_50492:00"
my_cnn_block_1_50494:07
$dense_matmul_readvariableop_resource:	?
3
%dense_biasadd_readvariableop_resource:

identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?$my_cnn_block/StatefulPartitionedCall?&my_cnn_block_1/StatefulPartitionedCall?
$my_cnn_block/StatefulPartitionedCallStatefulPartitionedCallxmy_cnn_block_50479my_cnn_block_50481my_cnn_block_50483my_cnn_block_50485*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_49032?
&my_cnn_block_1/StatefulPartitionedCallStatefulPartitionedCall-my_cnn_block/StatefulPartitionedCall:output:0my_cnn_block_1_50488my_cnn_block_1_50490my_cnn_block_1_50492my_cnn_block_1_50494*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_49056^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????0	  ?
flatten/ReshapeReshape/my_cnn_block_1/StatefulPartitionedCall:output:0flatten/Const:output:0*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
g
IdentityIdentitydense/Relu:activations:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp%^my_cnn_block/StatefulPartitionedCall'^my_cnn_block_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2L
$my_cnn_block/StatefulPartitionedCall$my_cnn_block/StatefulPartitionedCall2P
&my_cnn_block_1/StatefulPartitionedCall&my_cnn_block_1/StatefulPartitionedCall:R N
/
_output_shapes
:?????????

_user_specified_namex
?0
?
A__inference_my_cnn_layer_call_and_return_conditional_losses_52892
x,
my_cnn_block_52848: 
my_cnn_block_52850:,
my_cnn_block_52852: 
my_cnn_block_52854:.
my_cnn_block_1_52857:0"
my_cnn_block_1_52859:0.
my_cnn_block_1_52861:00"
my_cnn_block_1_52863:07
$dense_matmul_readvariableop_resource:	?
3
%dense_biasadd_readvariableop_resource:

identity??/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?$my_cnn_block/StatefulPartitionedCall?&my_cnn_block_1/StatefulPartitionedCall?
$my_cnn_block/StatefulPartitionedCallStatefulPartitionedCallxmy_cnn_block_52848my_cnn_block_52850my_cnn_block_52852my_cnn_block_52854*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_47621?
&my_cnn_block_1/StatefulPartitionedCallStatefulPartitionedCall-my_cnn_block/StatefulPartitionedCall:output:0my_cnn_block_1_52857my_cnn_block_1_52859my_cnn_block_1_52861my_cnn_block_1_52863*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_47667^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????0	  ?
flatten/ReshapeReshape/my_cnn_block_1/StatefulPartitionedCall:output:0flatten/Const:output:0*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_52848*&
_output_shapes
:*
dtype0?
 conv2d/kernel/Regularizer/L2LossL2Loss7conv2d/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0)conv2d/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ?
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_52852*&
_output_shapes
:*
dtype0?
"conv2d_1/kernel/Regularizer/L2LossL2Loss9conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0+conv2d_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ?
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_1_52857*&
_output_shapes
:0*
dtype0?
"conv2d_2/kernel/Regularizer/L2LossL2Loss9conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0+conv2d_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ?
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_1_52861*&
_output_shapes
:00*
dtype0?
"conv2d_3/kernel/Regularizer/L2LossL2Loss9conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0+conv2d_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: g
IdentityIdentitydense/Relu:activations:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp0^conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp%^my_cnn_block/StatefulPartitionedCall'^my_cnn_block_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2b
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2L
$my_cnn_block/StatefulPartitionedCall$my_cnn_block/StatefulPartitionedCall2P
&my_cnn_block_1/StatefulPartitionedCall&my_cnn_block_1/StatefulPartitionedCall:R N
/
_output_shapes
:?????????

_user_specified_namex
?
?
,__inference_my_cnn_block_layer_call_fn_52968	
input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_52321w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_nameinput
?

?
@__inference_dense_layer_call_and_return_conditional_losses_53130

inputs1
matmul_readvariableop_resource:	?
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
%__inference_dense_layer_call_fn_53119

inputs
unknown:	?

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_52383o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
I
-__inference_max_pooling2d_layer_call_fn_53143

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_52276?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_53148

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_52370

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????0	  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????0:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?

?
#__inference_signature_wrapper_52795
input_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:0
	unknown_4:0#
	unknown_5:00
	unknown_6:0
	unknown_7:	?

	unknown_8:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_52267o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
__inference_call_51373
xA
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:
identity??conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_1/Conv2DConv2Dx&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????n
activation_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????v
IdentityIdentityactivation_1/Relu:activations:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:R N
/
_output_shapes
:?????????

_user_specified_namex
?
?
__inference_call_47659
xA
'conv2d_3_conv2d_readvariableop_resource:006
(conv2d_3_biasadd_readvariableop_resource:0
identity??conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype0?
conv2d_3/Conv2DConv2Dx&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0n
activation_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????0v
IdentityIdentityactivation_3/Relu:activations:0^NoOp*
T0*/
_output_shapes
:?????????0?
NoOpNoOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????0: : 2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp:R N
/
_output_shapes
:?????????0

_user_specified_namex
?
?
I__inference_my_cnn_block_1_layer_call_and_return_conditional_losses_53099	
input<
"my_cnn_normalization_layer_2_53079:00
"my_cnn_normalization_layer_2_53081:0<
"my_cnn_normalization_layer_3_53084:000
"my_cnn_normalization_layer_3_53086:0
identity??1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp?4my_cnn_normalization_layer_2/StatefulPartitionedCall?4my_cnn_normalization_layer_3/StatefulPartitionedCall?
4my_cnn_normalization_layer_2/StatefulPartitionedCallStatefulPartitionedCallinput"my_cnn_normalization_layer_2_53079"my_cnn_normalization_layer_2_53081*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_48023?
4my_cnn_normalization_layer_3/StatefulPartitionedCallStatefulPartitionedCall=my_cnn_normalization_layer_2/StatefulPartitionedCall:output:0"my_cnn_normalization_layer_3_53084"my_cnn_normalization_layer_3_53086*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_48039?
max_pooling2d_1/MaxPoolMaxPool=my_cnn_normalization_layer_3/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????0*
ksize
*
paddingVALID*
strides
?
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp"my_cnn_normalization_layer_2_53079*&
_output_shapes
:0*
dtype0?
"conv2d_2/kernel/Regularizer/L2LossL2Loss9conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0+conv2d_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ?
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp"my_cnn_normalization_layer_3_53084*&
_output_shapes
:00*
dtype0?
"conv2d_3/kernel/Regularizer/L2LossL2Loss9conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0+conv2d_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: w
IdentityIdentity max_pooling2d_1/MaxPool:output:0^NoOp*
T0*/
_output_shapes
:?????????0?
NoOpNoOp2^conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp5^my_cnn_normalization_layer_2/StatefulPartitionedCall5^my_cnn_normalization_layer_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2f
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp2l
4my_cnn_normalization_layer_2/StatefulPartitionedCall4my_cnn_normalization_layer_2/StatefulPartitionedCall2l
4my_cnn_normalization_layer_3/StatefulPartitionedCall4my_cnn_normalization_layer_3/StatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_nameinput
?
?
__forward_call_49798
x_0A
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:
identity
activation_1_relu
x"
conv2d_1_conv2d_readvariableop??conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_1/Conv2DConv2Dx_0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????n
activation_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????v
IdentityIdentityactivation_1/Relu:activations:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "4
activation_1_reluactivation_1/Relu:activations:0"H
conv2d_1_conv2d_readvariableop&conv2d_1/Conv2D/ReadVariableOp:value:0"
identityIdentity:output:0"
xx_0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : *C
backward_function_name)'__inference___backward_call_49782_497992B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:R N
/
_output_shapes
:?????????

_user_specified_namex
?
?
__inference_call_48095
x?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????j
activation/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????t
IdentityIdentityactivation/Relu:activations:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:R N
/
_output_shapes
:?????????

_user_specified_namex
?
?
__inference_call_48111
xA
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:
identity??conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_1/Conv2DConv2Dx&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????n
activation_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????v
IdentityIdentityactivation_1/Relu:activations:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:R N
/
_output_shapes
:?????????

_user_specified_namex
?
?
__inference_call_50874	
input<
"my_cnn_normalization_layer_2_50862:00
"my_cnn_normalization_layer_2_50864:0<
"my_cnn_normalization_layer_3_50867:000
"my_cnn_normalization_layer_3_50869:0
identity??4my_cnn_normalization_layer_2/StatefulPartitionedCall?4my_cnn_normalization_layer_3/StatefulPartitionedCall?
4my_cnn_normalization_layer_2/StatefulPartitionedCallStatefulPartitionedCallinput"my_cnn_normalization_layer_2_50862"my_cnn_normalization_layer_2_50864*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_47643?
4my_cnn_normalization_layer_3/StatefulPartitionedCallStatefulPartitionedCall=my_cnn_normalization_layer_2/StatefulPartitionedCall:output:0"my_cnn_normalization_layer_3_50867"my_cnn_normalization_layer_3_50869*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_47659?
max_pooling2d_1/MaxPoolMaxPool=my_cnn_normalization_layer_3/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????0*
ksize
*
paddingVALID*
strides
w
IdentityIdentity max_pooling2d_1/MaxPool:output:0^NoOp*
T0*/
_output_shapes
:?????????0?
NoOpNoOp5^my_cnn_normalization_layer_2/StatefulPartitionedCall5^my_cnn_normalization_layer_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2l
4my_cnn_normalization_layer_2/StatefulPartitionedCall4my_cnn_normalization_layer_2/StatefulPartitionedCall2l
4my_cnn_normalization_layer_3/StatefulPartitionedCall4my_cnn_normalization_layer_3/StatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_nameinput
?
?
__inference_call_47667	
input<
"my_cnn_normalization_layer_2_47644:00
"my_cnn_normalization_layer_2_47646:0<
"my_cnn_normalization_layer_3_47660:000
"my_cnn_normalization_layer_3_47662:0
identity??4my_cnn_normalization_layer_2/StatefulPartitionedCall?4my_cnn_normalization_layer_3/StatefulPartitionedCall?
4my_cnn_normalization_layer_2/StatefulPartitionedCallStatefulPartitionedCallinput"my_cnn_normalization_layer_2_47644"my_cnn_normalization_layer_2_47646*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_47643?
4my_cnn_normalization_layer_3/StatefulPartitionedCallStatefulPartitionedCall=my_cnn_normalization_layer_2/StatefulPartitionedCall:output:0"my_cnn_normalization_layer_3_47660"my_cnn_normalization_layer_3_47662*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_47659?
max_pooling2d_1/MaxPoolMaxPool=my_cnn_normalization_layer_3/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????0*
ksize
*
paddingVALID*
strides
w
IdentityIdentity max_pooling2d_1/MaxPool:output:0^NoOp*
T0*/
_output_shapes
:?????????0?
NoOpNoOp5^my_cnn_normalization_layer_2/StatefulPartitionedCall5^my_cnn_normalization_layer_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2l
4my_cnn_normalization_layer_2/StatefulPartitionedCall4my_cnn_normalization_layer_2/StatefulPartitionedCall2l
4my_cnn_normalization_layer_3/StatefulPartitionedCall4my_cnn_normalization_layer_3/StatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_nameinput
?
?
__inference_call_51424
xA
'conv2d_3_conv2d_readvariableop_resource:006
(conv2d_3_biasadd_readvariableop_resource:0
identity??conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype0?
conv2d_3/Conv2DConv2Dx&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0n
activation_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????0v
IdentityIdentityactivation_3/Relu:activations:0^NoOp*
T0*/
_output_shapes
:?????????0?
NoOpNoOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????0: : 2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp:R N
/
_output_shapes
:?????????0

_user_specified_namex
?

?
&__inference_my_cnn_layer_call_fn_52820
x!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:0
	unknown_4:0#
	unknown_5:00
	unknown_6:0
	unknown_7:	?

	unknown_8:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_my_cnn_layer_call_and_return_conditional_losses_52406o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:?????????

_user_specified_namex
?
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_53166

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
__inference_call_51409
xA
'conv2d_2_conv2d_readvariableop_resource:06
(conv2d_2_biasadd_readvariableop_resource:0
identity??conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0?
conv2d_2/Conv2DConv2Dx&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0n
activation_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????0v
IdentityIdentityactivation_2/Relu:activations:0^NoOp*
T0*/
_output_shapes
:?????????0?
NoOpNoOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:R N
/
_output_shapes
:?????????

_user_specified_namex
?.
?
A__inference_my_cnn_layer_call_and_return_conditional_losses_52406
x,
my_cnn_block_52322: 
my_cnn_block_52324:,
my_cnn_block_52326: 
my_cnn_block_52328:.
my_cnn_block_1_52355:0"
my_cnn_block_1_52357:0.
my_cnn_block_1_52359:00"
my_cnn_block_1_52361:0
dense_52384:	?

dense_52386:

identity??/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp?dense/StatefulPartitionedCall?$my_cnn_block/StatefulPartitionedCall?&my_cnn_block_1/StatefulPartitionedCall?
$my_cnn_block/StatefulPartitionedCallStatefulPartitionedCallxmy_cnn_block_52322my_cnn_block_52324my_cnn_block_52326my_cnn_block_52328*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_52321?
&my_cnn_block_1/StatefulPartitionedCallStatefulPartitionedCall-my_cnn_block/StatefulPartitionedCall:output:0my_cnn_block_1_52355my_cnn_block_1_52357my_cnn_block_1_52359my_cnn_block_1_52361*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_my_cnn_block_1_layer_call_and_return_conditional_losses_52354?
flatten/PartitionedCallPartitionedCall/my_cnn_block_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_52370?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_52384dense_52386*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_52383?
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_52322*&
_output_shapes
:*
dtype0?
 conv2d/kernel/Regularizer/L2LossL2Loss7conv2d/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0)conv2d/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ?
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_52326*&
_output_shapes
:*
dtype0?
"conv2d_1/kernel/Regularizer/L2LossL2Loss9conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0+conv2d_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ?
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_1_52355*&
_output_shapes
:0*
dtype0?
"conv2d_2/kernel/Regularizer/L2LossL2Loss9conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0+conv2d_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ?
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_1_52359*&
_output_shapes
:00*
dtype0?
"conv2d_3/kernel/Regularizer/L2LossL2Loss9conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0+conv2d_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp0^conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp^dense/StatefulPartitionedCall%^my_cnn_block/StatefulPartitionedCall'^my_cnn_block_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2b
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2L
$my_cnn_block/StatefulPartitionedCall$my_cnn_block/StatefulPartitionedCall2P
&my_cnn_block_1/StatefulPartitionedCall&my_cnn_block_1/StatefulPartitionedCall:R N
/
_output_shapes
:?????????

_user_specified_namex
?*
?
!__inference__traced_restore_53311
file_prefix8
assignvariableop_conv2d_kernel:,
assignvariableop_1_conv2d_bias:<
"assignvariableop_2_conv2d_1_kernel:.
 assignvariableop_3_conv2d_1_bias:<
"assignvariableop_4_conv2d_2_kernel:0.
 assignvariableop_5_conv2d_2_bias:0<
"assignvariableop_6_conv2d_3_kernel:00.
 assignvariableop_7_conv2d_3_bias:02
assignvariableop_8_dense_kernel:	?
+
assignvariableop_9_dense_bias:

identity_11??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_dense_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_11IdentityIdentity_10:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_11Identity_11:output:0*)
_input_shapes
: : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_52276

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_53004	
input:
 my_cnn_normalization_layer_52984:.
 my_cnn_normalization_layer_52986:<
"my_cnn_normalization_layer_1_52989:0
"my_cnn_normalization_layer_1_52991:
identity??/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp?2my_cnn_normalization_layer/StatefulPartitionedCall?4my_cnn_normalization_layer_1/StatefulPartitionedCall?
2my_cnn_normalization_layer/StatefulPartitionedCallStatefulPartitionedCallinput my_cnn_normalization_layer_52984 my_cnn_normalization_layer_52986*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_47597?
4my_cnn_normalization_layer_1/StatefulPartitionedCallStatefulPartitionedCall;my_cnn_normalization_layer/StatefulPartitionedCall:output:0"my_cnn_normalization_layer_1_52989"my_cnn_normalization_layer_1_52991*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_47613?
max_pooling2d/MaxPoolMaxPool=my_cnn_normalization_layer_1/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
?
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp my_cnn_normalization_layer_52984*&
_output_shapes
:*
dtype0?
 conv2d/kernel/Regularizer/L2LossL2Loss7conv2d/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0)conv2d/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ?
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp"my_cnn_normalization_layer_1_52989*&
_output_shapes
:*
dtype0?
"conv2d_1/kernel/Regularizer/L2LossL2Loss9conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0+conv2d_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: u
IdentityIdentitymax_pooling2d/MaxPool:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp0^conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp3^my_cnn_normalization_layer/StatefulPartitionedCall5^my_cnn_normalization_layer_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2b
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2h
2my_cnn_normalization_layer/StatefulPartitionedCall2my_cnn_normalization_layer/StatefulPartitionedCall2l
4my_cnn_normalization_layer_1/StatefulPartitionedCall4my_cnn_normalization_layer_1/StatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_nameinput
?
?
__inference_call_50538
x,
my_cnn_block_50510: 
my_cnn_block_50512:,
my_cnn_block_50514: 
my_cnn_block_50516:.
my_cnn_block_1_50519:0"
my_cnn_block_1_50521:0.
my_cnn_block_1_50523:00"
my_cnn_block_1_50525:07
$dense_matmul_readvariableop_resource:	?
3
%dense_biasadd_readvariableop_resource:

identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?$my_cnn_block/StatefulPartitionedCall?&my_cnn_block_1/StatefulPartitionedCall?
$my_cnn_block/StatefulPartitionedCallStatefulPartitionedCallxmy_cnn_block_50510my_cnn_block_50512my_cnn_block_50514my_cnn_block_50516*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_47621?
&my_cnn_block_1/StatefulPartitionedCallStatefulPartitionedCall-my_cnn_block/StatefulPartitionedCall:output:0my_cnn_block_1_50519my_cnn_block_1_50521my_cnn_block_1_50523my_cnn_block_1_50525*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_47667^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????0	  ?
flatten/ReshapeReshape/my_cnn_block_1/StatefulPartitionedCall:output:0flatten/Const:output:0*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
g
IdentityIdentitydense/Relu:activations:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp%^my_cnn_block/StatefulPartitionedCall'^my_cnn_block_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2L
$my_cnn_block/StatefulPartitionedCall$my_cnn_block/StatefulPartitionedCall2P
&my_cnn_block_1/StatefulPartitionedCall&my_cnn_block_1/StatefulPartitionedCall:R N
/
_output_shapes
:?????????

_user_specified_namex
?
?
__inference_call_51336
x?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????j
activation/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????t
IdentityIdentityactivation/Relu:activations:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:R N
/
_output_shapes
:?????????

_user_specified_namex
?
?
__forward_call_49709
x_0A
'conv2d_3_conv2d_readvariableop_resource:006
(conv2d_3_biasadd_readvariableop_resource:0
identity
activation_3_relu
x"
conv2d_3_conv2d_readvariableop??conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype0?
conv2d_3/Conv2DConv2Dx_0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0n
activation_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????0v
IdentityIdentityactivation_3/Relu:activations:0^NoOp*
T0*/
_output_shapes
:?????????0?
NoOpNoOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "4
activation_3_reluactivation_3/Relu:activations:0"H
conv2d_3_conv2d_readvariableop&conv2d_3/Conv2D/ReadVariableOp:value:0"
identityIdentity:output:0"
xx_0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????0: : *C
backward_function_name)'__inference___backward_call_49693_497102B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp:R N
/
_output_shapes
:?????????0

_user_specified_namex
?
?
__inference_call_47621	
input:
 my_cnn_normalization_layer_47598:.
 my_cnn_normalization_layer_47600:<
"my_cnn_normalization_layer_1_47614:0
"my_cnn_normalization_layer_1_47616:
identity??2my_cnn_normalization_layer/StatefulPartitionedCall?4my_cnn_normalization_layer_1/StatefulPartitionedCall?
2my_cnn_normalization_layer/StatefulPartitionedCallStatefulPartitionedCallinput my_cnn_normalization_layer_47598 my_cnn_normalization_layer_47600*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_47597?
4my_cnn_normalization_layer_1/StatefulPartitionedCallStatefulPartitionedCall;my_cnn_normalization_layer/StatefulPartitionedCall:output:0"my_cnn_normalization_layer_1_47614"my_cnn_normalization_layer_1_47616*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_47613?
max_pooling2d/MaxPoolMaxPool=my_cnn_normalization_layer_1/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
u
IdentityIdentitymax_pooling2d/MaxPool:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp3^my_cnn_normalization_layer/StatefulPartitionedCall5^my_cnn_normalization_layer_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2h
2my_cnn_normalization_layer/StatefulPartitionedCall2my_cnn_normalization_layer/StatefulPartitionedCall2l
4my_cnn_normalization_layer_1/StatefulPartitionedCall4my_cnn_normalization_layer_1/StatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_nameinput
?
?
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_53027	
input:
 my_cnn_normalization_layer_53007:.
 my_cnn_normalization_layer_53009:<
"my_cnn_normalization_layer_1_53012:0
"my_cnn_normalization_layer_1_53014:
identity??/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp?2my_cnn_normalization_layer/StatefulPartitionedCall?4my_cnn_normalization_layer_1/StatefulPartitionedCall?
2my_cnn_normalization_layer/StatefulPartitionedCallStatefulPartitionedCallinput my_cnn_normalization_layer_53007 my_cnn_normalization_layer_53009*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_48095?
4my_cnn_normalization_layer_1/StatefulPartitionedCallStatefulPartitionedCall;my_cnn_normalization_layer/StatefulPartitionedCall:output:0"my_cnn_normalization_layer_1_53012"my_cnn_normalization_layer_1_53014*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_48111?
max_pooling2d/MaxPoolMaxPool=my_cnn_normalization_layer_1/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
?
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp my_cnn_normalization_layer_53007*&
_output_shapes
:*
dtype0?
 conv2d/kernel/Regularizer/L2LossL2Loss7conv2d/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0)conv2d/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ?
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp"my_cnn_normalization_layer_1_53012*&
_output_shapes
:*
dtype0?
"conv2d_1/kernel/Regularizer/L2LossL2Loss9conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0+conv2d_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: u
IdentityIdentitymax_pooling2d/MaxPool:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp0^conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp3^my_cnn_normalization_layer/StatefulPartitionedCall5^my_cnn_normalization_layer_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2b
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2h
2my_cnn_normalization_layer/StatefulPartitionedCall2my_cnn_normalization_layer/StatefulPartitionedCall2l
4my_cnn_normalization_layer_1/StatefulPartitionedCall4my_cnn_normalization_layer_1/StatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_nameinput
?0
?
A__inference_my_cnn_layer_call_and_return_conditional_losses_52939
x,
my_cnn_block_52895: 
my_cnn_block_52897:,
my_cnn_block_52899: 
my_cnn_block_52901:.
my_cnn_block_1_52904:0"
my_cnn_block_1_52906:0.
my_cnn_block_1_52908:00"
my_cnn_block_1_52910:07
$dense_matmul_readvariableop_resource:	?
3
%dense_biasadd_readvariableop_resource:

identity??/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?$my_cnn_block/StatefulPartitionedCall?&my_cnn_block_1/StatefulPartitionedCall?
$my_cnn_block/StatefulPartitionedCallStatefulPartitionedCallxmy_cnn_block_52895my_cnn_block_52897my_cnn_block_52899my_cnn_block_52901*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_49032?
&my_cnn_block_1/StatefulPartitionedCallStatefulPartitionedCall-my_cnn_block/StatefulPartitionedCall:output:0my_cnn_block_1_52904my_cnn_block_1_52906my_cnn_block_1_52908my_cnn_block_1_52910*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_49056^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????0	  ?
flatten/ReshapeReshape/my_cnn_block_1/StatefulPartitionedCall:output:0flatten/Const:output:0*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_52895*&
_output_shapes
:*
dtype0?
 conv2d/kernel/Regularizer/L2LossL2Loss7conv2d/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0)conv2d/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ?
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_52899*&
_output_shapes
:*
dtype0?
"conv2d_1/kernel/Regularizer/L2LossL2Loss9conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0+conv2d_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ?
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_1_52904*&
_output_shapes
:0*
dtype0?
"conv2d_2/kernel/Regularizer/L2LossL2Loss9conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0+conv2d_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ?
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_1_52908*&
_output_shapes
:00*
dtype0?
"conv2d_3/kernel/Regularizer/L2LossL2Loss9conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0+conv2d_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: g
IdentityIdentitydense/Relu:activations:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp0^conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp%^my_cnn_block/StatefulPartitionedCall'^my_cnn_block_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2b
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2L
$my_cnn_block/StatefulPartitionedCall$my_cnn_block/StatefulPartitionedCall2P
&my_cnn_block_1/StatefulPartitionedCall&my_cnn_block_1/StatefulPartitionedCall:R N
/
_output_shapes
:?????????

_user_specified_namex
?
?
__inference_call_50821	
input:
 my_cnn_normalization_layer_50809:.
 my_cnn_normalization_layer_50811:<
"my_cnn_normalization_layer_1_50814:0
"my_cnn_normalization_layer_1_50816:
identity??2my_cnn_normalization_layer/StatefulPartitionedCall?4my_cnn_normalization_layer_1/StatefulPartitionedCall?
2my_cnn_normalization_layer/StatefulPartitionedCallStatefulPartitionedCallinput my_cnn_normalization_layer_50809 my_cnn_normalization_layer_50811*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_48095?
4my_cnn_normalization_layer_1/StatefulPartitionedCallStatefulPartitionedCall;my_cnn_normalization_layer/StatefulPartitionedCall:output:0"my_cnn_normalization_layer_1_50814"my_cnn_normalization_layer_1_50816*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_48111?
max_pooling2d/MaxPoolMaxPool=my_cnn_normalization_layer_1/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
u
IdentityIdentitymax_pooling2d/MaxPool:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp3^my_cnn_normalization_layer/StatefulPartitionedCall5^my_cnn_normalization_layer_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2h
2my_cnn_normalization_layer/StatefulPartitionedCall2my_cnn_normalization_layer/StatefulPartitionedCall2l
4my_cnn_normalization_layer_1/StatefulPartitionedCall4my_cnn_normalization_layer_1/StatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_nameinput
?
?
__forward_call_49736
x_0A
'conv2d_2_conv2d_readvariableop_resource:06
(conv2d_2_biasadd_readvariableop_resource:0
identity
activation_2_relu
x"
conv2d_2_conv2d_readvariableop??conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0?
conv2d_2/Conv2DConv2Dx_0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0n
activation_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????0v
IdentityIdentityactivation_2/Relu:activations:0^NoOp*
T0*/
_output_shapes
:?????????0?
NoOpNoOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "4
activation_2_reluactivation_2/Relu:activations:0"H
conv2d_2_conv2d_readvariableop&conv2d_2/Conv2D/ReadVariableOp:value:0"
identityIdentity:output:0"
xx_0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : *C
backward_function_name)'__inference___backward_call_49720_497372B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:R N
/
_output_shapes
:?????????

_user_specified_namex
?
?
__inference_call_49056	
input<
"my_cnn_normalization_layer_2_49044:00
"my_cnn_normalization_layer_2_49046:0<
"my_cnn_normalization_layer_3_49049:000
"my_cnn_normalization_layer_3_49051:0
identity??4my_cnn_normalization_layer_2/StatefulPartitionedCall?4my_cnn_normalization_layer_3/StatefulPartitionedCall?
4my_cnn_normalization_layer_2/StatefulPartitionedCallStatefulPartitionedCallinput"my_cnn_normalization_layer_2_49044"my_cnn_normalization_layer_2_49046*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_48023?
4my_cnn_normalization_layer_3/StatefulPartitionedCallStatefulPartitionedCall=my_cnn_normalization_layer_2/StatefulPartitionedCall:output:0"my_cnn_normalization_layer_3_49049"my_cnn_normalization_layer_3_49051*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_48039?
max_pooling2d_1/MaxPoolMaxPool=my_cnn_normalization_layer_3/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????0*
ksize
*
paddingVALID*
strides
w
IdentityIdentity max_pooling2d_1/MaxPool:output:0^NoOp*
T0*/
_output_shapes
:?????????0?
NoOpNoOp5^my_cnn_normalization_layer_2/StatefulPartitionedCall5^my_cnn_normalization_layer_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2l
4my_cnn_normalization_layer_2/StatefulPartitionedCall4my_cnn_normalization_layer_2/StatefulPartitionedCall2l
4my_cnn_normalization_layer_3/StatefulPartitionedCall4my_cnn_normalization_layer_3/StatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_nameinput
?
C
'__inference_flatten_layer_call_fn_53104

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_52370a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????0:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?

?
&__inference_my_cnn_layer_call_fn_52429
input_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:0
	unknown_4:0#
	unknown_5:00
	unknown_6:0
	unknown_7:	?

	unknown_8:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_my_cnn_layer_call_and_return_conditional_losses_52406o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?

?
 __inference__wrapped_model_52267
input_1&
my_cnn_52245:
my_cnn_52247:&
my_cnn_52249:
my_cnn_52251:&
my_cnn_52253:0
my_cnn_52255:0&
my_cnn_52257:00
my_cnn_52259:0
my_cnn_52261:	?

my_cnn_52263:

identity??my_cnn/StatefulPartitionedCall?
my_cnn/StatefulPartitionedCallStatefulPartitionedCallinput_1my_cnn_52245my_cnn_52247my_cnn_52249my_cnn_52251my_cnn_52253my_cnn_52255my_cnn_52257my_cnn_52259my_cnn_52261my_cnn_52263*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_47687v
IdentityIdentity'my_cnn/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
g
NoOpNoOp^my_cnn/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2@
my_cnn/StatefulPartitionedCallmy_cnn/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
I__inference_my_cnn_block_1_layer_call_and_return_conditional_losses_53076	
input<
"my_cnn_normalization_layer_2_53056:00
"my_cnn_normalization_layer_2_53058:0<
"my_cnn_normalization_layer_3_53061:000
"my_cnn_normalization_layer_3_53063:0
identity??1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp?4my_cnn_normalization_layer_2/StatefulPartitionedCall?4my_cnn_normalization_layer_3/StatefulPartitionedCall?
4my_cnn_normalization_layer_2/StatefulPartitionedCallStatefulPartitionedCallinput"my_cnn_normalization_layer_2_53056"my_cnn_normalization_layer_2_53058*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_47643?
4my_cnn_normalization_layer_3/StatefulPartitionedCallStatefulPartitionedCall=my_cnn_normalization_layer_2/StatefulPartitionedCall:output:0"my_cnn_normalization_layer_3_53061"my_cnn_normalization_layer_3_53063*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_47659?
max_pooling2d_1/MaxPoolMaxPool=my_cnn_normalization_layer_3/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????0*
ksize
*
paddingVALID*
strides
?
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp"my_cnn_normalization_layer_2_53056*&
_output_shapes
:0*
dtype0?
"conv2d_2/kernel/Regularizer/L2LossL2Loss9conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0+conv2d_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ?
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp"my_cnn_normalization_layer_3_53061*&
_output_shapes
:00*
dtype0?
"conv2d_3/kernel/Regularizer/L2LossL2Loss9conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0+conv2d_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: w
IdentityIdentity max_pooling2d_1/MaxPool:output:0^NoOp*
T0*/
_output_shapes
:?????????0?
NoOpNoOp2^conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp5^my_cnn_normalization_layer_2/StatefulPartitionedCall5^my_cnn_normalization_layer_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2f
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp2l
4my_cnn_normalization_layer_2/StatefulPartitionedCall4my_cnn_normalization_layer_2/StatefulPartitionedCall2l
4my_cnn_normalization_layer_3/StatefulPartitionedCall4my_cnn_normalization_layer_3/StatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_nameinput
?

?
&__inference_my_cnn_layer_call_fn_52845
x!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:0
	unknown_4:0#
	unknown_5:00
	unknown_6:0
	unknown_7:	?

	unknown_8:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_my_cnn_layer_call_and_return_conditional_losses_52616o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:?????????

_user_specified_namex
?
?
__inference_call_51398
xA
'conv2d_2_conv2d_readvariableop_resource:06
(conv2d_2_biasadd_readvariableop_resource:0
identity??conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0?
conv2d_2/Conv2DConv2Dx&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0n
activation_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????0v
IdentityIdentityactivation_2/Relu:activations:0^NoOp*
T0*/
_output_shapes
:?????????0?
NoOpNoOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:R N
/
_output_shapes
:?????????

_user_specified_namex
?
?
__inference_call_50859	
input<
"my_cnn_normalization_layer_2_50847:00
"my_cnn_normalization_layer_2_50849:0<
"my_cnn_normalization_layer_3_50852:000
"my_cnn_normalization_layer_3_50854:0
identity??4my_cnn_normalization_layer_2/StatefulPartitionedCall?4my_cnn_normalization_layer_3/StatefulPartitionedCall?
4my_cnn_normalization_layer_2/StatefulPartitionedCallStatefulPartitionedCallinput"my_cnn_normalization_layer_2_50847"my_cnn_normalization_layer_2_50849*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_48023?
4my_cnn_normalization_layer_3/StatefulPartitionedCallStatefulPartitionedCall=my_cnn_normalization_layer_2/StatefulPartitionedCall:output:0"my_cnn_normalization_layer_3_50852"my_cnn_normalization_layer_3_50854*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_48039?
max_pooling2d_1/MaxPoolMaxPool=my_cnn_normalization_layer_3/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????0*
ksize
*
paddingVALID*
strides
w
IdentityIdentity max_pooling2d_1/MaxPool:output:0^NoOp*
T0*/
_output_shapes
:?????????0?
NoOpNoOp5^my_cnn_normalization_layer_2/StatefulPartitionedCall5^my_cnn_normalization_layer_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2l
4my_cnn_normalization_layer_2/StatefulPartitionedCall4my_cnn_normalization_layer_2/StatefulPartitionedCall2l
4my_cnn_normalization_layer_3/StatefulPartitionedCall4my_cnn_normalization_layer_3/StatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_nameinput
?
?
I__inference_my_cnn_block_1_layer_call_and_return_conditional_losses_52354	
input<
"my_cnn_normalization_layer_2_52334:00
"my_cnn_normalization_layer_2_52336:0<
"my_cnn_normalization_layer_3_52339:000
"my_cnn_normalization_layer_3_52341:0
identity??1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp?4my_cnn_normalization_layer_2/StatefulPartitionedCall?4my_cnn_normalization_layer_3/StatefulPartitionedCall?
4my_cnn_normalization_layer_2/StatefulPartitionedCallStatefulPartitionedCallinput"my_cnn_normalization_layer_2_52334"my_cnn_normalization_layer_2_52336*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_47643?
4my_cnn_normalization_layer_3/StatefulPartitionedCallStatefulPartitionedCall=my_cnn_normalization_layer_2/StatefulPartitionedCall:output:0"my_cnn_normalization_layer_3_52339"my_cnn_normalization_layer_3_52341*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_47659?
max_pooling2d_1/MaxPoolMaxPool=my_cnn_normalization_layer_3/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????0*
ksize
*
paddingVALID*
strides
?
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp"my_cnn_normalization_layer_2_52334*&
_output_shapes
:0*
dtype0?
"conv2d_2/kernel/Regularizer/L2LossL2Loss9conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0+conv2d_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ?
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp"my_cnn_normalization_layer_3_52339*&
_output_shapes
:00*
dtype0?
"conv2d_3/kernel/Regularizer/L2LossL2Loss9conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0+conv2d_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: w
IdentityIdentity max_pooling2d_1/MaxPool:output:0^NoOp*
T0*/
_output_shapes
:?????????0?
NoOpNoOp2^conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp5^my_cnn_normalization_layer_2/StatefulPartitionedCall5^my_cnn_normalization_layer_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2f
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp2l
4my_cnn_normalization_layer_2/StatefulPartitionedCall4my_cnn_normalization_layer_2/StatefulPartitionedCall2l
4my_cnn_normalization_layer_3/StatefulPartitionedCall4my_cnn_normalization_layer_3/StatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_nameinput
?
?
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_52321	
input:
 my_cnn_normalization_layer_52301:.
 my_cnn_normalization_layer_52303:<
"my_cnn_normalization_layer_1_52306:0
"my_cnn_normalization_layer_1_52308:
identity??/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp?2my_cnn_normalization_layer/StatefulPartitionedCall?4my_cnn_normalization_layer_1/StatefulPartitionedCall?
2my_cnn_normalization_layer/StatefulPartitionedCallStatefulPartitionedCallinput my_cnn_normalization_layer_52301 my_cnn_normalization_layer_52303*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_47597?
4my_cnn_normalization_layer_1/StatefulPartitionedCallStatefulPartitionedCall;my_cnn_normalization_layer/StatefulPartitionedCall:output:0"my_cnn_normalization_layer_1_52306"my_cnn_normalization_layer_1_52308*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_47613?
max_pooling2d/MaxPoolMaxPool=my_cnn_normalization_layer_1/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
?
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp my_cnn_normalization_layer_52301*&
_output_shapes
:*
dtype0?
 conv2d/kernel/Regularizer/L2LossL2Loss7conv2d/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0)conv2d/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ?
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp"my_cnn_normalization_layer_1_52306*&
_output_shapes
:*
dtype0?
"conv2d_1/kernel/Regularizer/L2LossL2Loss9conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0+conv2d_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: u
IdentityIdentitymax_pooling2d/MaxPool:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp0^conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp3^my_cnn_normalization_layer/StatefulPartitionedCall5^my_cnn_normalization_layer_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2b
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2h
2my_cnn_normalization_layer/StatefulPartitionedCall2my_cnn_normalization_layer/StatefulPartitionedCall2l
4my_cnn_normalization_layer_1/StatefulPartitionedCall4my_cnn_normalization_layer_1/StatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_nameinput
?
?
__inference_call_48023
xA
'conv2d_2_conv2d_readvariableop_resource:06
(conv2d_2_biasadd_readvariableop_resource:0
identity??conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0?
conv2d_2/Conv2DConv2Dx&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0n
activation_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????0v
IdentityIdentityactivation_2/Relu:activations:0^NoOp*
T0*/
_output_shapes
:?????????0?
NoOpNoOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:R N
/
_output_shapes
:?????????

_user_specified_namex
?.
?
A__inference_my_cnn_layer_call_and_return_conditional_losses_52616
x,
my_cnn_block_52575: 
my_cnn_block_52577:,
my_cnn_block_52579: 
my_cnn_block_52581:.
my_cnn_block_1_52584:0"
my_cnn_block_1_52586:0.
my_cnn_block_1_52588:00"
my_cnn_block_1_52590:0
dense_52594:	?

dense_52596:

identity??/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp?dense/StatefulPartitionedCall?$my_cnn_block/StatefulPartitionedCall?&my_cnn_block_1/StatefulPartitionedCall?
$my_cnn_block/StatefulPartitionedCallStatefulPartitionedCallxmy_cnn_block_52575my_cnn_block_52577my_cnn_block_52579my_cnn_block_52581*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_52534?
&my_cnn_block_1/StatefulPartitionedCallStatefulPartitionedCall-my_cnn_block/StatefulPartitionedCall:output:0my_cnn_block_1_52584my_cnn_block_1_52586my_cnn_block_1_52588my_cnn_block_1_52590*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_my_cnn_block_1_layer_call_and_return_conditional_losses_52484?
flatten/PartitionedCallPartitionedCall/my_cnn_block_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_52370?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_52594dense_52596*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_52383?
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_52575*&
_output_shapes
:*
dtype0?
 conv2d/kernel/Regularizer/L2LossL2Loss7conv2d/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0)conv2d/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ?
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_52579*&
_output_shapes
:*
dtype0?
"conv2d_1/kernel/Regularizer/L2LossL2Loss9conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0+conv2d_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ?
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_1_52584*&
_output_shapes
:0*
dtype0?
"conv2d_2/kernel/Regularizer/L2LossL2Loss9conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0+conv2d_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ?
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_1_52588*&
_output_shapes
:00*
dtype0?
"conv2d_3/kernel/Regularizer/L2LossL2Loss9conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0+conv2d_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp0^conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp^dense/StatefulPartitionedCall%^my_cnn_block/StatefulPartitionedCall'^my_cnn_block_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2b
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2L
$my_cnn_block/StatefulPartitionedCall$my_cnn_block/StatefulPartitionedCall2P
&my_cnn_block_1/StatefulPartitionedCall&my_cnn_block_1/StatefulPartitionedCall:R N
/
_output_shapes
:?????????

_user_specified_namex
?

?
@__inference_dense_layer_call_and_return_conditional_losses_52383

inputs1
matmul_readvariableop_resource:	?
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference_call_50836	
input:
 my_cnn_normalization_layer_50824:.
 my_cnn_normalization_layer_50826:<
"my_cnn_normalization_layer_1_50829:0
"my_cnn_normalization_layer_1_50831:
identity??2my_cnn_normalization_layer/StatefulPartitionedCall?4my_cnn_normalization_layer_1/StatefulPartitionedCall?
2my_cnn_normalization_layer/StatefulPartitionedCallStatefulPartitionedCallinput my_cnn_normalization_layer_50824 my_cnn_normalization_layer_50826*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_47597?
4my_cnn_normalization_layer_1/StatefulPartitionedCallStatefulPartitionedCall;my_cnn_normalization_layer/StatefulPartitionedCall:output:0"my_cnn_normalization_layer_1_50829"my_cnn_normalization_layer_1_50831*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_47613?
max_pooling2d/MaxPoolMaxPool=my_cnn_normalization_layer_1/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
u
IdentityIdentitymax_pooling2d/MaxPool:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp3^my_cnn_normalization_layer/StatefulPartitionedCall5^my_cnn_normalization_layer_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2h
2my_cnn_normalization_layer/StatefulPartitionedCall2my_cnn_normalization_layer/StatefulPartitionedCall2l
4my_cnn_normalization_layer_1/StatefulPartitionedCall4my_cnn_normalization_layer_1/StatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_nameinput
?
?
__inference_call_51435
xA
'conv2d_3_conv2d_readvariableop_resource:006
(conv2d_3_biasadd_readvariableop_resource:0
identity??conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype0?
conv2d_3/Conv2DConv2Dx&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0n
activation_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????0v
IdentityIdentityactivation_3/Relu:activations:0^NoOp*
T0*/
_output_shapes
:?????????0?
NoOpNoOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????0: : 2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp:R N
/
_output_shapes
:?????????0

_user_specified_namex
?	
?
__inference_loss_fn_3_53218T
:conv2d_3_kernel_regularizer_l2loss_readvariableop_resource:00
identity??1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp?
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp:conv2d_3_kernel_regularizer_l2loss_readvariableop_resource*&
_output_shapes
:00*
dtype0?
"conv2d_3/kernel/Regularizer/L2LossL2Loss9conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0+conv2d_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentity#conv2d_3/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_53110

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????0	  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????0:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
?
.__inference_my_cnn_block_1_layer_call_fn_53040	
input!
unknown:0
	unknown_0:0#
	unknown_1:00
	unknown_2:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_my_cnn_block_1_layer_call_and_return_conditional_losses_52354w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_nameinput
?
?
__inference_call_48039
xA
'conv2d_3_conv2d_readvariableop_resource:006
(conv2d_3_biasadd_readvariableop_resource:0
identity??conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype0?
conv2d_3/Conv2DConv2Dx&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0n
activation_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????0v
IdentityIdentityactivation_3/Relu:activations:0^NoOp*
T0*/
_output_shapes
:?????????0?
NoOpNoOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????0: : 2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp:R N
/
_output_shapes
:?????????0

_user_specified_namex
?
?
__inference_call_47613
xA
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:
identity??conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_1/Conv2DConv2Dx&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????n
activation_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????v
IdentityIdentityactivation_1/Relu:activations:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:R N
/
_output_shapes
:?????????

_user_specified_namex
?
?
__inference_call_49032	
input:
 my_cnn_normalization_layer_49020:.
 my_cnn_normalization_layer_49022:<
"my_cnn_normalization_layer_1_49025:0
"my_cnn_normalization_layer_1_49027:
identity??2my_cnn_normalization_layer/StatefulPartitionedCall?4my_cnn_normalization_layer_1/StatefulPartitionedCall?
2my_cnn_normalization_layer/StatefulPartitionedCallStatefulPartitionedCallinput my_cnn_normalization_layer_49020 my_cnn_normalization_layer_49022*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_48095?
4my_cnn_normalization_layer_1/StatefulPartitionedCallStatefulPartitionedCall;my_cnn_normalization_layer/StatefulPartitionedCall:output:0"my_cnn_normalization_layer_1_49025"my_cnn_normalization_layer_1_49027*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_48111?
max_pooling2d/MaxPoolMaxPool=my_cnn_normalization_layer_1/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
u
IdentityIdentitymax_pooling2d/MaxPool:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp3^my_cnn_normalization_layer/StatefulPartitionedCall5^my_cnn_normalization_layer_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2h
2my_cnn_normalization_layer/StatefulPartitionedCall2my_cnn_normalization_layer/StatefulPartitionedCall2l
4my_cnn_normalization_layer_1/StatefulPartitionedCall4my_cnn_normalization_layer_1/StatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_nameinput
?
?
,__inference_my_cnn_block_layer_call_fn_52981	
input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_52534w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_nameinput
?	
?
__inference_loss_fn_0_53191R
8conv2d_kernel_regularizer_l2loss_readvariableop_resource:
identity??/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp?
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp8conv2d_kernel_regularizer_l2loss_readvariableop_resource*&
_output_shapes
:*
dtype0?
 conv2d/kernel/Regularizer/L2LossL2Loss7conv2d/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0)conv2d/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentity!conv2d/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: x
NoOpNoOp0^conv2d/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp
?	
?
__inference_loss_fn_1_53200T
:conv2d_1_kernel_regularizer_l2loss_readvariableop_resource:
identity??1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp?
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp:conv2d_1_kernel_regularizer_l2loss_readvariableop_resource*&
_output_shapes
:*
dtype0?
"conv2d_1/kernel/Regularizer/L2LossL2Loss9conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0+conv2d_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentity#conv2d_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp
?
?
__inference_call_47597
x?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????j
activation/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????t
IdentityIdentityactivation/Relu:activations:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:R N
/
_output_shapes
:?????????

_user_specified_namex
?
K
/__inference_max_pooling2d_1_layer_call_fn_53161

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_52288?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?/
?
A__inference_my_cnn_layer_call_and_return_conditional_losses_52752
input_1,
my_cnn_block_52711: 
my_cnn_block_52713:,
my_cnn_block_52715: 
my_cnn_block_52717:.
my_cnn_block_1_52720:0"
my_cnn_block_1_52722:0.
my_cnn_block_1_52724:00"
my_cnn_block_1_52726:0
dense_52730:	?

dense_52732:

identity??/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp?dense/StatefulPartitionedCall?$my_cnn_block/StatefulPartitionedCall?&my_cnn_block_1/StatefulPartitionedCall?
$my_cnn_block/StatefulPartitionedCallStatefulPartitionedCallinput_1my_cnn_block_52711my_cnn_block_52713my_cnn_block_52715my_cnn_block_52717*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_52534?
&my_cnn_block_1/StatefulPartitionedCallStatefulPartitionedCall-my_cnn_block/StatefulPartitionedCall:output:0my_cnn_block_1_52720my_cnn_block_1_52722my_cnn_block_1_52724my_cnn_block_1_52726*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_my_cnn_block_1_layer_call_and_return_conditional_losses_52484?
flatten/PartitionedCallPartitionedCall/my_cnn_block_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_52370?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_52730dense_52732*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_52383?
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_52711*&
_output_shapes
:*
dtype0?
 conv2d/kernel/Regularizer/L2LossL2Loss7conv2d/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0)conv2d/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ?
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_52715*&
_output_shapes
:*
dtype0?
"conv2d_1/kernel/Regularizer/L2LossL2Loss9conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0+conv2d_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ?
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_1_52720*&
_output_shapes
:0*
dtype0?
"conv2d_2/kernel/Regularizer/L2LossL2Loss9conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0+conv2d_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ?
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_1_52724*&
_output_shapes
:00*
dtype0?
"conv2d_3/kernel/Regularizer/L2LossL2Loss9conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0+conv2d_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp0^conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp^dense/StatefulPartitionedCall%^my_cnn_block/StatefulPartitionedCall'^my_cnn_block_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2b
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2L
$my_cnn_block/StatefulPartitionedCall$my_cnn_block/StatefulPartitionedCall2P
&my_cnn_block_1/StatefulPartitionedCall&my_cnn_block_1/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?

?
&__inference_my_cnn_layer_call_fn_52664
input_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:0
	unknown_4:0#
	unknown_5:00
	unknown_6:0
	unknown_7:	?

	unknown_8:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_my_cnn_layer_call_and_return_conditional_losses_52616o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
I__inference_my_cnn_block_1_layer_call_and_return_conditional_losses_52484	
input<
"my_cnn_normalization_layer_2_52464:00
"my_cnn_normalization_layer_2_52466:0<
"my_cnn_normalization_layer_3_52469:000
"my_cnn_normalization_layer_3_52471:0
identity??1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp?4my_cnn_normalization_layer_2/StatefulPartitionedCall?4my_cnn_normalization_layer_3/StatefulPartitionedCall?
4my_cnn_normalization_layer_2/StatefulPartitionedCallStatefulPartitionedCallinput"my_cnn_normalization_layer_2_52464"my_cnn_normalization_layer_2_52466*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_48023?
4my_cnn_normalization_layer_3/StatefulPartitionedCallStatefulPartitionedCall=my_cnn_normalization_layer_2/StatefulPartitionedCall:output:0"my_cnn_normalization_layer_3_52469"my_cnn_normalization_layer_3_52471*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_48039?
max_pooling2d_1/MaxPoolMaxPool=my_cnn_normalization_layer_3/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????0*
ksize
*
paddingVALID*
strides
?
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp"my_cnn_normalization_layer_2_52464*&
_output_shapes
:0*
dtype0?
"conv2d_2/kernel/Regularizer/L2LossL2Loss9conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0+conv2d_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ?
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp"my_cnn_normalization_layer_3_52469*&
_output_shapes
:00*
dtype0?
"conv2d_3/kernel/Regularizer/L2LossL2Loss9conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;?
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0+conv2d_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: w
IdentityIdentity max_pooling2d_1/MaxPool:output:0^NoOp*
T0*/
_output_shapes
:?????????0?
NoOpNoOp2^conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp5^my_cnn_normalization_layer_2/StatefulPartitionedCall5^my_cnn_normalization_layer_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2f
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp2l
4my_cnn_normalization_layer_2/StatefulPartitionedCall4my_cnn_normalization_layer_2/StatefulPartitionedCall2l
4my_cnn_normalization_layer_3/StatefulPartitionedCall4my_cnn_normalization_layer_3/StatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_nameinput"?	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_18
serving_default_input_1:0?????????<
output_10
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:??
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

block1

	block2

flatten
out
call

signatures"
_tf_keras_model
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
?
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
trace_0
trace_1
trace_2
 trace_32?
&__inference_my_cnn_layer_call_fn_52429
&__inference_my_cnn_layer_call_fn_52820
&__inference_my_cnn_layer_call_fn_52845
&__inference_my_cnn_layer_call_fn_52664?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ztrace_0ztrace_1ztrace_2z trace_3
?
!trace_0
"trace_1
#trace_2
$trace_32?
A__inference_my_cnn_layer_call_and_return_conditional_losses_52892
A__inference_my_cnn_layer_call_and_return_conditional_losses_52939
A__inference_my_cnn_layer_call_and_return_conditional_losses_52708
A__inference_my_cnn_layer_call_and_return_conditional_losses_52752?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z!trace_0z"trace_1z#trace_2z$trace_3
?B?
 __inference__wrapped_model_52267input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses
+conv_layers
,pool
-call"
_tf_keras_layer
?
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses
4conv_layers
5pool
6call"
_tf_keras_layer
?
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses"
_tf_keras_layer
?
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
?
Ctrace_0
Dtrace_12?
__inference_call_50507
__inference_call_50538?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zCtrace_0zDtrace_1
,
Eserving_default"
signature_map
':%2conv2d/kernel
:2conv2d/bias
):'2conv2d_1/kernel
:2conv2d_1/bias
):'02conv2d_2/kernel
:02conv2d_2/bias
):'002conv2d_3/kernel
:02conv2d_3/bias
:	?
2dense/kernel
:
2
dense/bias
 "
trackable_list_wrapper
<
0
	1

2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
&__inference_my_cnn_layer_call_fn_52429input_1"?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_my_cnn_layer_call_fn_52820x"?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_my_cnn_layer_call_fn_52845x"?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_my_cnn_layer_call_fn_52664input_1"?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
A__inference_my_cnn_layer_call_and_return_conditional_losses_52892x"?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
A__inference_my_cnn_layer_call_and_return_conditional_losses_52939x"?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
A__inference_my_cnn_layer_call_and_return_conditional_losses_52708input_1"?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
A__inference_my_cnn_layer_call_and_return_conditional_losses_52752input_1"?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
?
Ktrace_0
Ltrace_12?
,__inference_my_cnn_block_layer_call_fn_52968
,__inference_my_cnn_block_layer_call_fn_52981?
???
FullArgSpec(
args ?
jself
jinput

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zKtrace_0zLtrace_1
?
Mtrace_0
Ntrace_12?
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_53004
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_53027?
???
FullArgSpec(
args ?
jself
jinput

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zMtrace_0zNtrace_1
.
O0
P1"
trackable_list_wrapper
?
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Wtrace_0
Xtrace_12?
__inference_call_50821
__inference_call_50836?
???
FullArgSpec(
args ?
jself
jinput

jtraining
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zWtrace_0zXtrace_1
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
?
^trace_0
_trace_12?
.__inference_my_cnn_block_1_layer_call_fn_53040
.__inference_my_cnn_block_1_layer_call_fn_53053?
???
FullArgSpec(
args ?
jself
jinput

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z^trace_0z_trace_1
?
`trace_0
atrace_12?
I__inference_my_cnn_block_1_layer_call_and_return_conditional_losses_53076
I__inference_my_cnn_block_1_layer_call_and_return_conditional_losses_53099?
???
FullArgSpec(
args ?
jself
jinput

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z`trace_0zatrace_1
.
b0
c1"
trackable_list_wrapper
?
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses"
_tf_keras_layer
?
jtrace_0
ktrace_12?
__inference_call_50859
__inference_call_50874?
???
FullArgSpec(
args ?
jself
jinput

jtraining
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zjtrace_0zktrace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
?
qtrace_02?
'__inference_flatten_layer_call_fn_53104?
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
 zqtrace_0
?
rtrace_02?
B__inference_flatten_layer_call_and_return_conditional_losses_53110?
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
 zrtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
?
xtrace_02?
%__inference_dense_layer_call_fn_53119?
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
 zxtrace_0
?
ytrace_02?
@__inference_dense_layer_call_and_return_conditional_losses_53130?
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
 zytrace_0
?B?
__inference_call_50507x"?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
__inference_call_50538x"?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_52795input_1"?
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
 
 "
trackable_list_wrapper
5
O0
P1
,2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
,__inference_my_cnn_block_layer_call_fn_52968input"?
???
FullArgSpec(
args ?
jself
jinput

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
,__inference_my_cnn_block_layer_call_fn_52981input"?
???
FullArgSpec(
args ?
jself
jinput

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_53004input"?
???
FullArgSpec(
args ?
jself
jinput

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_53027input"?
???
FullArgSpec(
args ?
jself
jinput

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses
?
conv_layer
?
activation
	?call"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?
conv_layer
?
activation
	?call"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
-__inference_max_pooling2d_layer_call_fn_53143?
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
 z?trace_0
?
?trace_02?
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_53148?
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
 z?trace_0
?B?
__inference_call_50821input"?
???
FullArgSpec(
args ?
jself
jinput

jtraining
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
__inference_call_50836input"?
???
FullArgSpec(
args ?
jself
jinput

jtraining
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
5
b0
c1
52"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
.__inference_my_cnn_block_1_layer_call_fn_53040input"?
???
FullArgSpec(
args ?
jself
jinput

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
.__inference_my_cnn_block_1_layer_call_fn_53053input"?
???
FullArgSpec(
args ?
jself
jinput

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
I__inference_my_cnn_block_1_layer_call_and_return_conditional_losses_53076input"?
???
FullArgSpec(
args ?
jself
jinput

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
I__inference_my_cnn_block_1_layer_call_and_return_conditional_losses_53099input"?
???
FullArgSpec(
args ?
jself
jinput

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?
conv_layer
?
activation
	?call"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?
conv_layer
?
activation
	?call"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
/__inference_max_pooling2d_1_layer_call_fn_53161?
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
 z?trace_0
?
?trace_02?
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_53166?
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
 z?trace_0
?B?
__inference_call_50859input"?
???
FullArgSpec(
args ?
jself
jinput

jtraining
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
__inference_call_50874input"?
???
FullArgSpec(
args ?
jself
jinput

jtraining
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
'__inference_flatten_layer_call_fn_53104inputs"?
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
B__inference_flatten_layer_call_and_return_conditional_losses_53110inputs"?
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
?B?
%__inference_dense_layer_call_fn_53119inputs"?
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
@__inference_dense_layer_call_and_return_conditional_losses_53130inputs"?
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
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?trace_0
?trace_12?
__inference_call_51336
__inference_call_51347?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?trace_0
?trace_12?
__inference_call_51362
__inference_call_51373?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
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
?B?
-__inference_max_pooling2d_layer_call_fn_53143inputs"?
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
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_53148inputs"?
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
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?trace_0
?trace_12?
__inference_call_51398
__inference_call_51409?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?trace_0
?trace_12?
__inference_call_51424
__inference_call_51435?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
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
?B?
/__inference_max_pooling2d_1_layer_call_fn_53161inputs"?
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
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_53166inputs"?
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
?
?trace_02?
__inference_loss_fn_0_53191?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
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
?2??
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
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
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
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
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
?2??
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
__inference_call_51336x"?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
__inference_call_51347x"?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
?trace_02?
__inference_loss_fn_1_53200?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
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
?2??
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
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
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
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
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
?2??
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
__inference_call_51362x"?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
__inference_call_51373x"?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
?trace_02?
__inference_loss_fn_2_53209?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
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
?2??
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
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
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
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
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
?2??
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
__inference_call_51398x"?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
__inference_call_51409x"?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
?trace_02?
__inference_loss_fn_3_53218?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
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
?2??
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
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
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
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
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
?2??
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
__inference_call_51424x"?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
__inference_call_51435x"?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
__inference_loss_fn_0_53191"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
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
?B?
__inference_loss_fn_1_53200"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
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
?B?
__inference_loss_fn_2_53209"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
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
?B?
__inference_loss_fn_3_53218"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
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
trackable_dict_wrapper?
 __inference__wrapped_model_52267{
8?5
.?+
)?&
input_1?????????
? "3?0
.
output_1"?
output_1?????????
x
__inference_call_50507^
6?3
,?)
#? 
x?????????
p
? "??????????
x
__inference_call_50538^
6?3
,?)
#? 
x?????????
p 
? "??????????
~
__inference_call_50821d:?7
0?-
'?$
input?????????
p
? " ??????????~
__inference_call_50836d:?7
0?-
'?$
input?????????
p 
? " ??????????~
__inference_call_50859d:?7
0?-
'?$
input?????????
p
? " ??????????0~
__inference_call_50874d:?7
0?-
'?$
input?????????
p 
? " ??????????0x
__inference_call_51336^6?3
,?)
#? 
x?????????
p
? " ??????????x
__inference_call_51347^6?3
,?)
#? 
x?????????
p 
? " ??????????x
__inference_call_51362^6?3
,?)
#? 
x?????????
p
? " ??????????x
__inference_call_51373^6?3
,?)
#? 
x?????????
p 
? " ??????????x
__inference_call_51398^6?3
,?)
#? 
x?????????
p
? " ??????????0x
__inference_call_51409^6?3
,?)
#? 
x?????????
p 
? " ??????????0x
__inference_call_51424^6?3
,?)
#? 
x?????????0
p
? " ??????????0x
__inference_call_51435^6?3
,?)
#? 
x?????????0
p 
? " ??????????0?
@__inference_dense_layer_call_and_return_conditional_losses_53130]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????

? y
%__inference_dense_layer_call_fn_53119P0?-
&?#
!?
inputs??????????
? "??????????
?
B__inference_flatten_layer_call_and_return_conditional_losses_53110a7?4
-?*
(?%
inputs?????????0
? "&?#
?
0??????????
? 
'__inference_flatten_layer_call_fn_53104T7?4
-?*
(?%
inputs?????????0
? "???????????:
__inference_loss_fn_0_53191?

? 
? "? :
__inference_loss_fn_1_53200?

? 
? "? :
__inference_loss_fn_2_53209?

? 
? "? :
__inference_loss_fn_3_53218?

? 
? "? ?
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_53166?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_1_layer_call_fn_53161?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_53148?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
-__inference_max_pooling2d_layer_call_fn_53143?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
I__inference_my_cnn_block_1_layer_call_and_return_conditional_losses_53076q:?7
0?-
'?$
input?????????
p 
? "-?*
#? 
0?????????0
? ?
I__inference_my_cnn_block_1_layer_call_and_return_conditional_losses_53099q:?7
0?-
'?$
input?????????
p
? "-?*
#? 
0?????????0
? ?
.__inference_my_cnn_block_1_layer_call_fn_53040d:?7
0?-
'?$
input?????????
p 
? " ??????????0?
.__inference_my_cnn_block_1_layer_call_fn_53053d:?7
0?-
'?$
input?????????
p
? " ??????????0?
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_53004q:?7
0?-
'?$
input?????????
p 
? "-?*
#? 
0?????????
? ?
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_53027q:?7
0?-
'?$
input?????????
p
? "-?*
#? 
0?????????
? ?
,__inference_my_cnn_block_layer_call_fn_52968d:?7
0?-
'?$
input?????????
p 
? " ???????????
,__inference_my_cnn_block_layer_call_fn_52981d:?7
0?-
'?$
input?????????
p
? " ???????????
A__inference_my_cnn_layer_call_and_return_conditional_losses_52708q
<?9
2?/
)?&
input_1?????????
p 
? "%?"
?
0?????????

? ?
A__inference_my_cnn_layer_call_and_return_conditional_losses_52752q
<?9
2?/
)?&
input_1?????????
p
? "%?"
?
0?????????

? ?
A__inference_my_cnn_layer_call_and_return_conditional_losses_52892k
6?3
,?)
#? 
x?????????
p 
? "%?"
?
0?????????

? ?
A__inference_my_cnn_layer_call_and_return_conditional_losses_52939k
6?3
,?)
#? 
x?????????
p
? "%?"
?
0?????????

? ?
&__inference_my_cnn_layer_call_fn_52429d
<?9
2?/
)?&
input_1?????????
p 
? "??????????
?
&__inference_my_cnn_layer_call_fn_52664d
<?9
2?/
)?&
input_1?????????
p
? "??????????
?
&__inference_my_cnn_layer_call_fn_52820^
6?3
,?)
#? 
x?????????
p 
? "??????????
?
&__inference_my_cnn_layer_call_fn_52845^
6?3
,?)
#? 
x?????????
p
? "??????????
?
#__inference_signature_wrapper_52795?
C?@
? 
9?6
4
input_1)?&
input_1?????????"3?0
.
output_1"?
output_1?????????
