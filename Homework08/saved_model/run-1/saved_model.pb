??(
??
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
?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
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
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.10.02unknown8??#
?
Adam/conv2d_transpose_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_1/bias/v
?
2Adam/conv2d_transpose_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_1/bias/v*
_output_shapes
:*
dtype0
?
 Adam/conv2d_transpose_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2d_transpose_1/kernel/v
?
4Adam/conv2d_transpose_1/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_1/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_transpose/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/conv2d_transpose/bias/v
?
0Adam/conv2d_transpose/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_transpose/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*/
shared_name Adam/conv2d_transpose/kernel/v
?
2Adam/conv2d_transpose/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/kernel/v*&
_output_shapes
:0*
dtype0

Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_1/bias/v
x
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
?*&
shared_nameAdam/dense_1/kernel/v
?
)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes
:	
?*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:
*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	?
*
dtype0
?
Adam/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*%
shared_nameAdam/conv2d_3/bias/v
y
(Adam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/v*
_output_shapes
:0*
dtype0
?
Adam/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:00*'
shared_nameAdam/conv2d_3/kernel/v
?
*Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/v*&
_output_shapes
:00*
dtype0
?
Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*%
shared_nameAdam/conv2d_2/bias/v
y
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes
:0*
dtype0
?
Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*'
shared_nameAdam/conv2d_2/kernel/v
?
*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*&
_output_shapes
:0*
dtype0
?
Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_1/kernel/v
?
*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/v
?
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_transpose_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_1/bias/m
?
2Adam/conv2d_transpose_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_1/bias/m*
_output_shapes
:*
dtype0
?
 Adam/conv2d_transpose_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2d_transpose_1/kernel/m
?
4Adam/conv2d_transpose_1/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_1/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_transpose/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/conv2d_transpose/bias/m
?
0Adam/conv2d_transpose/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_transpose/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*/
shared_name Adam/conv2d_transpose/kernel/m
?
2Adam/conv2d_transpose/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/kernel/m*&
_output_shapes
:0*
dtype0

Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_1/bias/m
x
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
?*&
shared_nameAdam/dense_1/kernel/m
?
)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes
:	
?*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:
*
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	?
*
dtype0
?
Adam/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*%
shared_nameAdam/conv2d_3/bias/m
y
(Adam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/m*
_output_shapes
:0*
dtype0
?
Adam/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:00*'
shared_nameAdam/conv2d_3/kernel/m
?
*Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/m*&
_output_shapes
:00*
dtype0
?
Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*%
shared_nameAdam/conv2d_2/bias/m
y
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
_output_shapes
:0*
dtype0
?
Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*'
shared_nameAdam/conv2d_2/kernel/m
?
*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*&
_output_shapes
:0*
dtype0
?
Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_1/kernel/m
?
*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/m
?
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
?
conv2d_transpose_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_1/bias

+conv2d_transpose_1/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/bias*
_output_shapes
:*
dtype0
?
conv2d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv2d_transpose_1/kernel
?
-conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/kernel*&
_output_shapes
:*
dtype0
?
conv2d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameconv2d_transpose/bias
{
)conv2d_transpose/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose/bias*
_output_shapes
:*
dtype0
?
conv2d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*(
shared_nameconv2d_transpose/kernel
?
+conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose/kernel*&
_output_shapes
:0*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:?*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
?*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	
?*
dtype0
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
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasconv2d_transpose/kernelconv2d_transpose/biasconv2d_transpose_1/kernelconv2d_transpose_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_50278

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
encoder
	decoder

metrices
	optimizer
call
	test_step

train_step

signatures*
?
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
 16
!17
"18
#19*
z
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15*
* 
?
$non_trainable_variables

%layers
&metrics
'layer_regularization_losses
(layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
)trace_0
*trace_1
+trace_2
,trace_3* 
6
-trace_0
.trace_1
/trace_2
0trace_3* 
* 
?
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

7block1

8block2
9flatten
:out
;call*
?
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses

Bdense1
Creshape

Dtrans1

Etrans2
Foutput_layer
Gcall*

H0
I1*
?
Jiter

Kbeta_1

Lbeta_2
	Mdecay
Nlearning_ratem?m?m?m?m?m?m?m?m?m?m?m?m?m?m?m?v?v?v?v?v?v?v?v?v?v?v?v?v?v?v?v?*

Otrace_0
Ptrace_1* 

Qtrace_0* 

Rtrace_0* 

Sserving_default* 
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
OI
VARIABLE_VALUEdense_1/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_1/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv2d_transpose/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEconv2d_transpose/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2d_transpose_1/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv2d_transpose_1/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
HB
VARIABLE_VALUEtotal_2'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
HB
VARIABLE_VALUEcount_2'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
HB
VARIABLE_VALUEtotal_1'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
HB
VARIABLE_VALUEcount_1'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
 
 0
!1
"2
#3*

0
	1*

T0
H1
I2*
* 

Hloss
Iaccuracy*
* 
* 
* 
* 
* 
* 
* 
* 
J
0
1
2
3
4
5
6
7
8
9*
J
0
1
2
3
4
5
6
7
8
9*
* 
?
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*
6
Ztrace_0
[trace_1
\trace_2
]trace_3* 
6
^trace_0
_trace_1
`trace_2
atrace_3* 
?
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses
hconv_layers
ipool
jcall*
?
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses
qconv_layers
rpool
scall*
?
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses* 
?
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses

kernel
bias*

?trace_0
?trace_1* 
.
0
1
2
3
4
5*
.
0
1
2
3
4
5*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses*
:
?trace_0
?trace_1
?trace_2
?trace_3* 
:
?trace_0
?trace_1
?trace_2
?trace_3* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias
!?_jit_compiled_convolution_op*
5
?	keras_api
!?_jit_compiled_convolution_op* 

?trace_0
?trace_1* 
:
?	variables
?	keras_api
	 total
	!count*
K
?	variables
?	keras_api
	"total
	#count
?
_fn_kwargs*
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
<
?	variables
?	keras_api

?total

?count*
* 
 
70
81
92
:3*
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
 
0
1
2
3*
 
0
1
2
3*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 

?0
?1*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 

?trace_0
?trace_1* 
 
0
1
2
3*
 
0
1
2
3*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 

?0
?1*
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

0
1*

0
1*
* 
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

?trace_0* 

?trace_0* 
* 
* 
* 
'
B0
C1
D2
E3
F4*
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
0
1*

0
1*
* 
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

?trace_0* 

?trace_0* 
* 
* 
* 
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
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

0
1*

0
1*
* 
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

?trace_0* 

?trace_0* 
* 

0
1*

0
1*
* 
?
?non_trainable_variables
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

?trace_0* 

?trace_0* 
* 
* 
* 
* 
* 

 0
!1*

?	variables*

"0
#1*

?	variables*
* 

?0
?1*

?	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0
?1
i2*
* 
* 
* 
* 
* 
* 
* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?
conv_layer
?
activation
	?call*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?
conv_layer
?
activation
	?call*
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

?trace_0* 

?trace_0* 
* 
* 
* 

?0
?1
r2*
* 
* 
* 
* 
* 
* 
* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?
conv_layer
?
activation
	?call*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?
conv_layer
?
activation
	?call*
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

?trace_0* 

?trace_0* 
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
0
1*

0
1*


?0* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 

?trace_0
?trace_1* 

0
1*

0
1*


?0* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 

?trace_0
?trace_1* 
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*


?0* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 

?trace_0
?trace_1* 

0
1*

0
1*


?0* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 

?trace_0
?trace_1* 
* 
* 
* 
* 
* 
* 
* 

?trace_0* 
* 

?0
?1*
* 
* 
* 

0
1*

0
1*


?0* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
* 

?trace_0* 
* 

?0
?1*
* 
* 
* 

0
1*

0
1*


?0* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
* 

?trace_0* 
* 

?0
?1*
* 
* 
* 

0
1*

0
1*


?0* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
* 

?trace_0* 
* 

?0
?1*
* 
* 
* 

0
1*

0
1*


?0* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
* 


?0* 
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


?0* 
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


?0* 
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


?0* 
* 
* 
* 
* 
* 
* 
pj
VARIABLE_VALUEAdam/conv2d/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/conv2d/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_2/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_2/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_3/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_3/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/dense/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEAdam/dense/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_1/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_1/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/conv2d_transpose/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/conv2d_transpose_1/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose_1/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/conv2d/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_2/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_2/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_3/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_3/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/dense/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEAdam/dense/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_1/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_1/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/conv2d_transpose/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/conv2d_transpose_1/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose_1/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp+conv2d_transpose/kernel/Read/ReadVariableOp)conv2d_transpose/bias/Read/ReadVariableOp-conv2d_transpose_1/kernel/Read/ReadVariableOp+conv2d_transpose_1/bias/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp*Adam/conv2d_3/kernel/m/Read/ReadVariableOp(Adam/conv2d_3/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp2Adam/conv2d_transpose/kernel/m/Read/ReadVariableOp0Adam/conv2d_transpose/bias/m/Read/ReadVariableOp4Adam/conv2d_transpose_1/kernel/m/Read/ReadVariableOp2Adam/conv2d_transpose_1/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp*Adam/conv2d_3/kernel/v/Read/ReadVariableOp(Adam/conv2d_3/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp2Adam/conv2d_transpose/kernel/v/Read/ReadVariableOp0Adam/conv2d_transpose/bias/v/Read/ReadVariableOp4Adam/conv2d_transpose_1/kernel/v/Read/ReadVariableOp2Adam/conv2d_transpose_1/bias/v/Read/ReadVariableOpConst*H
TinA
?2=	*
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
__inference__traced_save_51797
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasconv2d_transpose/kernelconv2d_transpose/biasconv2d_transpose_1/kernelconv2d_transpose_1/biastotal_2count_2total_1count_1	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/mAdam/conv2d_3/kernel/mAdam/conv2d_3/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/conv2d_transpose/kernel/mAdam/conv2d_transpose/bias/m Adam/conv2d_transpose_1/kernel/mAdam/conv2d_transpose_1/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/vAdam/conv2d_3/kernel/vAdam/conv2d_3/bias/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/conv2d_transpose/kernel/vAdam/conv2d_transpose/bias/v Adam/conv2d_transpose_1/kernel/vAdam/conv2d_transpose_1/bias/v*G
Tin@
>2<*
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
!__inference__traced_restore_51984?? 
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
?
.__inference_my_autoencoder_layer_call_fn_50315
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

	unknown_9:	
?

unknown_10:	?$

unknown_11:0

unknown_12:$

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_my_autoencoder_layer_call_and_return_conditional_losses_48683w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:?????????

_user_specified_namex
?o
?
__inference__traced_save_51797
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
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop6
2savev2_conv2d_transpose_kernel_read_readvariableop4
0savev2_conv2d_transpose_bias_read_readvariableop8
4savev2_conv2d_transpose_1_kernel_read_readvariableop6
2savev2_conv2d_transpose_1_bias_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableop5
1savev2_adam_conv2d_3_kernel_m_read_readvariableop3
/savev2_adam_conv2d_3_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop=
9savev2_adam_conv2d_transpose_kernel_m_read_readvariableop;
7savev2_adam_conv2d_transpose_bias_m_read_readvariableop?
;savev2_adam_conv2d_transpose_1_kernel_m_read_readvariableop=
9savev2_adam_conv2d_transpose_1_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableop5
1savev2_adam_conv2d_3_kernel_v_read_readvariableop3
/savev2_adam_conv2d_3_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop=
9savev2_adam_conv2d_transpose_kernel_v_read_readvariableop;
7savev2_adam_conv2d_transpose_bias_v_read_readvariableop?
;savev2_adam_conv2d_transpose_1_kernel_v_read_readvariableop=
9savev2_adam_conv2d_transpose_1_bias_v_read_readvariableop
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
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*?
value?B?<B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*?
value?B?<B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop2savev2_conv2d_transpose_kernel_read_readvariableop0savev2_conv2d_transpose_bias_read_readvariableop4savev2_conv2d_transpose_1_kernel_read_readvariableop2savev2_conv2d_transpose_1_bias_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop1savev2_adam_conv2d_3_kernel_m_read_readvariableop/savev2_adam_conv2d_3_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop9savev2_adam_conv2d_transpose_kernel_m_read_readvariableop7savev2_adam_conv2d_transpose_bias_m_read_readvariableop;savev2_adam_conv2d_transpose_1_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_1_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop1savev2_adam_conv2d_3_kernel_v_read_readvariableop/savev2_adam_conv2d_3_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop9savev2_adam_conv2d_transpose_kernel_v_read_readvariableop7savev2_adam_conv2d_transpose_bias_v_read_readvariableop;savev2_adam_conv2d_transpose_1_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *J
dtypes@
>2<	?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :::::0:0:00:0:	?
:
:	
?:?:0:::: : : : : : : : : : : :::::0:0:00:0:	?
:
:	
?:?:0::::::::0:0:00:0:	?
:
:	
?:?:0:::: 2(
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
:%!

_output_shapes
:	
?:!

_output_shapes	
:?:,(
&
_output_shapes
:0: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::, (
&
_output_shapes
:0: !

_output_shapes
:0:,"(
&
_output_shapes
:00: #

_output_shapes
:0:%$!

_output_shapes
:	?
: %

_output_shapes
:
:%&!

_output_shapes
:	
?:!'

_output_shapes	
:?:,((
&
_output_shapes
:0: )

_output_shapes
::,*(
&
_output_shapes
:: +

_output_shapes
::,,(
&
_output_shapes
:: -

_output_shapes
::,.(
&
_output_shapes
:: /

_output_shapes
::,0(
&
_output_shapes
:0: 1

_output_shapes
:0:,2(
&
_output_shapes
:00: 3

_output_shapes
:0:%4!

_output_shapes
:	?
: 5

_output_shapes
:
:%6!

_output_shapes
:	
?:!7

_output_shapes	
:?:,8(
&
_output_shapes
:0: 9

_output_shapes
::,:(
&
_output_shapes
:: ;

_output_shapes
::<

_output_shapes
: 
?	
?
__inference_loss_fn_0_51470R
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
?0
?
__forward_call_49875
x,
my_cnn_block_49033: 
my_cnn_block_49035:,
my_cnn_block_49037: 
my_cnn_block_49039:.
my_cnn_block_1_49057:0"
my_cnn_block_1_49059:0.
my_cnn_block_1_49061:00"
my_cnn_block_1_49063:07
$dense_matmul_readvariableop_resource:	?
3
%dense_biasadd_readvariableop_resource:

identity

dense_relu
dense_matmul_readvariableop
flatten_reshape*
&my_cnn_block_1_statefulpartitionedcall,
(my_cnn_block_1_statefulpartitionedcall_0,
(my_cnn_block_1_statefulpartitionedcall_1,
(my_cnn_block_1_statefulpartitionedcall_2,
(my_cnn_block_1_statefulpartitionedcall_3,
(my_cnn_block_1_statefulpartitionedcall_4,
(my_cnn_block_1_statefulpartitionedcall_5,
(my_cnn_block_1_statefulpartitionedcall_6,
(my_cnn_block_1_statefulpartitionedcall_7(
$my_cnn_block_statefulpartitionedcall*
&my_cnn_block_statefulpartitionedcall_0*
&my_cnn_block_statefulpartitionedcall_1*
&my_cnn_block_statefulpartitionedcall_2*
&my_cnn_block_statefulpartitionedcall_3*
&my_cnn_block_statefulpartitionedcall_4*
&my_cnn_block_statefulpartitionedcall_5*
&my_cnn_block_statefulpartitionedcall_6??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?$my_cnn_block/StatefulPartitionedCall?&my_cnn_block_1/StatefulPartitionedCall?
$my_cnn_block/StatefulPartitionedCallStatefulPartitionedCallxmy_cnn_block_49033my_cnn_block_49035my_cnn_block_49037my_cnn_block_49039*
Tin	
2*
Tout
2	*
_collective_manager_ids
 *?
_output_shapes?
?:?????????:?????????:?????????:?????????:?????????::?????????:?????????:*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__forward_call_49841?
&my_cnn_block_1/StatefulPartitionedCallStatefulPartitionedCall-my_cnn_block/StatefulPartitionedCall:output:0my_cnn_block_1_49057my_cnn_block_1_49059my_cnn_block_1_49061my_cnn_block_1_49063*
Tin	
2*
Tout
2	*
_collective_manager_ids
 *?
_output_shapes?
?:?????????0:?????????0:?????????0:?????????0:?????????0:00:?????????0:?????????:0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__forward_call_49752^
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
 "B
dense_matmul_readvariableop#dense/MatMul/ReadVariableOp:value:0"&

dense_reludense/Relu:activations:0"+
flatten_reshapeflatten/Reshape:output:0"
identityIdentity:output:0"Y
&my_cnn_block_1_statefulpartitionedcall/my_cnn_block_1/StatefulPartitionedCall:output:0"[
(my_cnn_block_1_statefulpartitionedcall_0/my_cnn_block_1/StatefulPartitionedCall:output:1"[
(my_cnn_block_1_statefulpartitionedcall_1/my_cnn_block_1/StatefulPartitionedCall:output:2"[
(my_cnn_block_1_statefulpartitionedcall_2/my_cnn_block_1/StatefulPartitionedCall:output:3"[
(my_cnn_block_1_statefulpartitionedcall_3/my_cnn_block_1/StatefulPartitionedCall:output:4"[
(my_cnn_block_1_statefulpartitionedcall_4/my_cnn_block_1/StatefulPartitionedCall:output:5"[
(my_cnn_block_1_statefulpartitionedcall_5/my_cnn_block_1/StatefulPartitionedCall:output:6"[
(my_cnn_block_1_statefulpartitionedcall_6/my_cnn_block_1/StatefulPartitionedCall:output:7"[
(my_cnn_block_1_statefulpartitionedcall_7/my_cnn_block_1/StatefulPartitionedCall:output:8"U
$my_cnn_block_statefulpartitionedcall-my_cnn_block/StatefulPartitionedCall:output:1"W
&my_cnn_block_statefulpartitionedcall_0-my_cnn_block/StatefulPartitionedCall:output:2"W
&my_cnn_block_statefulpartitionedcall_1-my_cnn_block/StatefulPartitionedCall:output:3"W
&my_cnn_block_statefulpartitionedcall_2-my_cnn_block/StatefulPartitionedCall:output:4"W
&my_cnn_block_statefulpartitionedcall_3-my_cnn_block/StatefulPartitionedCall:output:5"W
&my_cnn_block_statefulpartitionedcall_4-my_cnn_block/StatefulPartitionedCall:output:6"W
&my_cnn_block_statefulpartitionedcall_5-my_cnn_block/StatefulPartitionedCall:output:7"W
&my_cnn_block_statefulpartitionedcall_6-my_cnn_block/StatefulPartitionedCall:output:8*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : *C
backward_function_name)'__inference___backward_call_49672_498762<
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
?
?
,__inference_my_cnn_block_layer_call_fn_51037	
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
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_47870w
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
?
__forward_call_49752	
input<
"my_cnn_normalization_layer_2_49044:00
"my_cnn_normalization_layer_2_49046:0<
"my_cnn_normalization_layer_3_49049:000
"my_cnn_normalization_layer_3_49051:0
identity8
4my_cnn_normalization_layer_3_statefulpartitionedcall
max_pooling2d_1_maxpool:
6my_cnn_normalization_layer_3_statefulpartitionedcall_0:
6my_cnn_normalization_layer_3_statefulpartitionedcall_1:
6my_cnn_normalization_layer_3_statefulpartitionedcall_28
4my_cnn_normalization_layer_2_statefulpartitionedcall:
6my_cnn_normalization_layer_2_statefulpartitionedcall_0:
6my_cnn_normalization_layer_2_statefulpartitionedcall_1??4my_cnn_normalization_layer_2/StatefulPartitionedCall?4my_cnn_normalization_layer_3/StatefulPartitionedCall?
4my_cnn_normalization_layer_2/StatefulPartitionedCallStatefulPartitionedCallinput"my_cnn_normalization_layer_2_49044"my_cnn_normalization_layer_2_49046*
Tin
2*
Tout
2*
_collective_manager_ids
 *w
_output_shapese
c:?????????0:?????????0:?????????:0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__forward_call_49736?
4my_cnn_normalization_layer_3/StatefulPartitionedCallStatefulPartitionedCall=my_cnn_normalization_layer_2/StatefulPartitionedCall:output:0"my_cnn_normalization_layer_3_49049"my_cnn_normalization_layer_3_49051*
Tin
2*
Tout
2*
_collective_manager_ids
 *w
_output_shapese
c:?????????0:?????????0:?????????0:00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__forward_call_49709?
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
identityIdentity:output:0";
max_pooling2d_1_maxpool max_pooling2d_1/MaxPool:output:0"u
4my_cnn_normalization_layer_2_statefulpartitionedcall=my_cnn_normalization_layer_2/StatefulPartitionedCall:output:1"w
6my_cnn_normalization_layer_2_statefulpartitionedcall_0=my_cnn_normalization_layer_2/StatefulPartitionedCall:output:2"w
6my_cnn_normalization_layer_2_statefulpartitionedcall_1=my_cnn_normalization_layer_2/StatefulPartitionedCall:output:3"u
4my_cnn_normalization_layer_3_statefulpartitionedcall=my_cnn_normalization_layer_3/StatefulPartitionedCall:output:0"w
6my_cnn_normalization_layer_3_statefulpartitionedcall_0=my_cnn_normalization_layer_3/StatefulPartitionedCall:output:1"w
6my_cnn_normalization_layer_3_statefulpartitionedcall_1=my_cnn_normalization_layer_3/StatefulPartitionedCall:output:2"w
6my_cnn_normalization_layer_3_statefulpartitionedcall_2=my_cnn_normalization_layer_3/StatefulPartitionedCall:output:3*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : *C
backward_function_name)'__inference___backward_call_49686_497532l
4my_cnn_normalization_layer_2/StatefulPartitionedCall4my_cnn_normalization_layer_2/StatefulPartitionedCall2l
4my_cnn_normalization_layer_3/StatefulPartitionedCall4my_cnn_normalization_layer_3/StatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_nameinput
?)
?
I__inference_my_autoencoder_layer_call_and_return_conditional_losses_50406
x&
my_cnn_50355:
my_cnn_50357:&
my_cnn_50359:
my_cnn_50361:&
my_cnn_50363:0
my_cnn_50365:0&
my_cnn_50367:00
my_cnn_50369:0
my_cnn_50371:	?

my_cnn_50373:
#
my_decoder_50376:	
?
my_decoder_50378:	?*
my_decoder_50380:0
my_decoder_50382:*
my_decoder_50384:
my_decoder_50386:
identity??/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp?my_cnn/StatefulPartitionedCall?"my_decoder/StatefulPartitionedCall?
my_cnn/StatefulPartitionedCallStatefulPartitionedCallxmy_cnn_50355my_cnn_50357my_cnn_50359my_cnn_50361my_cnn_50363my_cnn_50365my_cnn_50367my_cnn_50369my_cnn_50371my_cnn_50373*
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
__inference_call_47687?
"my_decoder/StatefulPartitionedCallStatefulPartitionedCall'my_cnn/StatefulPartitionedCall:output:0my_decoder_50376my_decoder_50378my_decoder_50380my_decoder_50382my_decoder_50384my_decoder_50386*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_47766?
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_50355*&
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
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_50359*&
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
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_50363*&
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
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_50367*&
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
: ?
IdentityIdentity+my_decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp0^conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp^my_cnn/StatefulPartitionedCall#^my_decoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 2b
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp2@
my_cnn/StatefulPartitionedCallmy_cnn/StatefulPartitionedCall2H
"my_decoder/StatefulPartitionedCall"my_decoder/StatefulPartitionedCall:R N
/
_output_shapes
:?????????

_user_specified_namex
?
?
0__inference_conv2d_transpose_layer_call_fn_51246

inputs!
unknown:0
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_48382?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????0: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
@broadcast_weights_1_assert_broadcastable_AssertGuard_false_49366?
|broadcast_weights_1_assert_broadcastable_assertguard_assert_broadcast_weights_1_assert_broadcastable_is_valid_shape_identity
v
rbroadcast_weights_1_assert_broadcastable_assertguard_assert_broadcast_weights_1_assert_broadcastable_weights_shapeu
qbroadcast_weights_1_assert_broadcastable_assertguard_assert_broadcast_weights_1_assert_broadcastable_values_shaper
nbroadcast_weights_1_assert_broadcastable_assertguard_assert_broadcast_weights_1_assert_broadcastable_is_scalar
C
?broadcast_weights_1_assert_broadcastable_assertguard_identity_1
??;broadcast_weights_1/assert_broadcastable/AssertGuard/Assert?
Bbroadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*8
value/B- B'weights can not be broadcast to values.?
Bbroadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*
valueB Bweights.shape=?
Bbroadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*9
value0B. B(my_autoencoder/StatefulPartitionedCall:0?
Bbroadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*
valueB Bvalues.shape=?
Bbroadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5Const*
_output_shapes
: *
dtype0*
valueB Bdata/1:0?
Bbroadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7Const*
_output_shapes
: *
dtype0*
valueB B
is_scalar=?
;broadcast_weights_1/assert_broadcastable/AssertGuard/AssertAssert|broadcast_weights_1_assert_broadcastable_assertguard_assert_broadcast_weights_1_assert_broadcastable_is_valid_shape_identityKbroadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0:output:0Kbroadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1:output:0Kbroadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2:output:0rbroadcast_weights_1_assert_broadcastable_assertguard_assert_broadcast_weights_1_assert_broadcastable_weights_shapeKbroadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4:output:0Kbroadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5:output:0qbroadcast_weights_1_assert_broadcastable_assertguard_assert_broadcast_weights_1_assert_broadcastable_values_shapeKbroadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7:output:0nbroadcast_weights_1_assert_broadcastable_assertguard_assert_broadcast_weights_1_assert_broadcastable_is_scalar*
T
2	
*
_output_shapes
 ?
=broadcast_weights_1/assert_broadcastable/AssertGuard/IdentityIdentity|broadcast_weights_1_assert_broadcastable_assertguard_assert_broadcast_weights_1_assert_broadcastable_is_valid_shape_identity<^broadcast_weights_1/assert_broadcastable/AssertGuard/Assert*
T0
*
_output_shapes
: ?
?broadcast_weights_1/assert_broadcastable/AssertGuard/Identity_1IdentityFbroadcast_weights_1/assert_broadcastable/AssertGuard/Identity:output:0:^broadcast_weights_1/assert_broadcastable/AssertGuard/NoOp*
T0
*
_output_shapes
: ?
9broadcast_weights_1/assert_broadcastable/AssertGuard/NoOpNoOp<^broadcast_weights_1/assert_broadcastable/AssertGuard/Assert*"
_acd_function_control_output(*
_output_shapes
 "?
?broadcast_weights_1_assert_broadcastable_assertguard_identity_1Hbroadcast_weights_1/assert_broadcastable/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: ::: 2z
;broadcast_weights_1/assert_broadcastable/AssertGuard/Assert;broadcast_weights_1/assert_broadcastable/AssertGuard/Assert: 

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
? 
?
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_51321

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
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
??
?
__inference_test_step_49416

data_0

data_1.
my_autoencoder_49212:"
my_autoencoder_49214:.
my_autoencoder_49216:"
my_autoencoder_49218:.
my_autoencoder_49220:0"
my_autoencoder_49222:0.
my_autoencoder_49224:00"
my_autoencoder_49226:0'
my_autoencoder_49228:	?
"
my_autoencoder_49230:
'
my_autoencoder_49232:	
?#
my_autoencoder_49234:	?.
my_autoencoder_49236:0"
my_autoencoder_49238:.
my_autoencoder_49240:"
my_autoencoder_49242:&
assignaddvariableop_resource: (
assignaddvariableop_1_resource: (
assignaddvariableop_4_resource: (
assignaddvariableop_5_resource: .
$div_no_nan_2_readvariableop_resource: 0
&div_no_nan_2_readvariableop_1_resource: 

identity_3

identity_4??AssignAddVariableOp?AssignAddVariableOp_1?AssignAddVariableOp_2?AssignAddVariableOp_3?AssignAddVariableOp_4?AssignAddVariableOp_5?4broadcast_weights_1/assert_broadcastable/AssertGuard?/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp?div_no_nan/ReadVariableOp?div_no_nan/ReadVariableOp_1?div_no_nan_1/ReadVariableOp?div_no_nan_1/ReadVariableOp_1?div_no_nan_2/ReadVariableOp?div_no_nan_2/ReadVariableOp_1?&my_autoencoder/StatefulPartitionedCall?
&my_autoencoder/StatefulPartitionedCallStatefulPartitionedCalldata_0my_autoencoder_49212my_autoencoder_49214my_autoencoder_49216my_autoencoder_49218my_autoencoder_49220my_autoencoder_49222my_autoencoder_49224my_autoencoder_49226my_autoencoder_49228my_autoencoder_49230my_autoencoder_49232my_autoencoder_49234my_autoencoder_49236my_autoencoder_49238my_autoencoder_49240my_autoencoder_49242*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_47781?
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_autoencoder_49212*&
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
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_autoencoder_49216*&
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
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_autoencoder_49220*&
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
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_autoencoder_49224*&
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
: ?
$mean_squared_error/SquaredDifferenceSquaredDifference/my_autoencoder/StatefulPartitionedCall:output:0data_1*
T0*/
_output_shapes
:?????????t
)mean_squared_error/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
mean_squared_error/MeanMean(mean_squared_error/SquaredDifference:z:02mean_squared_error/Mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????k
&mean_squared_error/weighted_loss/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
$mean_squared_error/weighted_loss/MulMul mean_squared_error/Mean:output:0/mean_squared_error/weighted_loss/Const:output:0*
T0*+
_output_shapes
:?????????}
(mean_squared_error/weighted_loss/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          ?
$mean_squared_error/weighted_loss/SumSum(mean_squared_error/weighted_loss/Mul:z:01mean_squared_error/weighted_loss/Const_1:output:0*
T0*
_output_shapes
: ?
-mean_squared_error/weighted_loss/num_elementsSize(mean_squared_error/weighted_loss/Mul:z:0*
T0*
_output_shapes
: ?
2mean_squared_error/weighted_loss/num_elements/CastCast6mean_squared_error/weighted_loss/num_elements:output:0*

DstT0*

SrcT0*
_output_shapes
: g
%mean_squared_error/weighted_loss/RankConst*
_output_shapes
: *
dtype0*
value	B : n
,mean_squared_error/weighted_loss/range/startConst*
_output_shapes
: *
dtype0*
value	B : n
,mean_squared_error/weighted_loss/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
&mean_squared_error/weighted_loss/rangeRange5mean_squared_error/weighted_loss/range/start:output:0.mean_squared_error/weighted_loss/Rank:output:05mean_squared_error/weighted_loss/range/delta:output:0*
_output_shapes
: ?
&mean_squared_error/weighted_loss/Sum_1Sum-mean_squared_error/weighted_loss/Sum:output:0/mean_squared_error/weighted_loss/range:output:0*
T0*
_output_shapes
: ?
&mean_squared_error/weighted_loss/valueDivNoNan/mean_squared_error/weighted_loss/Sum_1:output:06mean_squared_error/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: ;
ShapeShapedata_1*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
AddNAddN!conv2d/kernel/Regularizer/mul:z:0#conv2d_1/kernel/Regularizer/mul:z:0#conv2d_2/kernel/Regularizer/mul:z:0#conv2d_3/kernel/Regularizer/mul:z:0*
N*
T0*
_output_shapes
: p
AddN_1AddN*mean_squared_error/weighted_loss/value:z:0
AddN:sum:0*
N*
T0*
_output_shapes
: T
CastCaststrided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: C
MulMulAddN_1:sum:0Cast:y:0*
T0*
_output_shapes
: F
RankConst*
_output_shapes
: *
dtype0*
value	B : M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :c
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
: D
SumSumMul:z:0range:output:0*
T0*
_output_shapes
: y
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceSum:output:0*
_output_shapes
 *
dtype0H
Rank_1Const*
_output_shapes
: *
dtype0*
value	B : O
range_1/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :k
range_1Rangerange_1/start:output:0Rank_1:output:0range_1/delta:output:0*
_output_shapes
: I
Sum_1SumCast:y:0range_1:output:0*
T0*
_output_shapes
: ?
AssignAddVariableOp_1AssignAddVariableOpassignaddvariableop_1_resourceSum_1:output:0^AssignAddVariableOp*
_output_shapes
 *
dtype0p
AddN_2AddN*mean_squared_error/weighted_loss/value:z:0
AddN:sum:0*
N*
T0*
_output_shapes
: H
Rank_2Const*
_output_shapes
: *
dtype0*
value	B : O
range_2/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :k
range_2Rangerange_2/start:output:0Rank_2:output:0range_2/delta:output:0*
_output_shapes
: M
Sum_2SumAddN_2:sum:0range_2:output:0*
T0*
_output_shapes
: ?
AssignAddVariableOp_2AssignAddVariableOpassignaddvariableop_resourceSum_2:output:0^AssignAddVariableOp*
_output_shapes
 *
dtype0F
SizeConst*
_output_shapes
: *
dtype0*
value	B :M
Cast_1CastSize:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
AssignAddVariableOp_3AssignAddVariableOpassignaddvariableop_1_resource
Cast_1:y:0^AssignAddVariableOp_1^AssignAddVariableOp_2*
_output_shapes
 *
dtype0?
6broadcast_weights_1/assert_broadcastable/weights/shapeShape/my_autoencoder/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:w
5broadcast_weights_1/assert_broadcastable/weights/rankConst*
_output_shapes
: *
dtype0*
value	B :k
5broadcast_weights_1/assert_broadcastable/values/shapeShapedata_1*
T0*
_output_shapes
:v
4broadcast_weights_1/assert_broadcastable/values/rankConst*
_output_shapes
: *
dtype0*
value	B :v
4broadcast_weights_1/assert_broadcastable/is_scalar/xConst*
_output_shapes
: *
dtype0*
value	B : ?
2broadcast_weights_1/assert_broadcastable/is_scalarEqual=broadcast_weights_1/assert_broadcastable/is_scalar/x:output:0>broadcast_weights_1/assert_broadcastable/weights/rank:output:0*
T0*
_output_shapes
: ?
7broadcast_weights_1/assert_broadcastable/is_valid_shapeStatelessIf6broadcast_weights_1/assert_broadcastable/is_scalar:z:06broadcast_weights_1/assert_broadcastable/is_scalar:z:0=broadcast_weights_1/assert_broadcastable/values/rank:output:0>broadcast_weights_1/assert_broadcastable/weights/rank:output:0>broadcast_weights_1/assert_broadcastable/values/shape:output:0?broadcast_weights_1/assert_broadcastable/weights/shape:output:0*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *V
else_branchGRE
Cbroadcast_weights_1_assert_broadcastable_is_valid_shape_false_49312*
output_shapes
: *U
then_branchFRD
Bbroadcast_weights_1_assert_broadcastable_is_valid_shape_true_49311?
@broadcast_weights_1/assert_broadcastable/is_valid_shape/IdentityIdentity@broadcast_weights_1/assert_broadcastable/is_valid_shape:output:0*
T0
*
_output_shapes
: ?
.broadcast_weights_1/assert_broadcastable/ConstConst*
_output_shapes
: *
dtype0*8
value/B- B'weights can not be broadcast to values.
0broadcast_weights_1/assert_broadcastable/Const_1Const*
_output_shapes
: *
dtype0*
valueB Bweights.shape=?
0broadcast_weights_1/assert_broadcastable/Const_2Const*
_output_shapes
: *
dtype0*9
value0B. B(my_autoencoder/StatefulPartitionedCall:0~
0broadcast_weights_1/assert_broadcastable/Const_3Const*
_output_shapes
: *
dtype0*
valueB Bvalues.shape=y
0broadcast_weights_1/assert_broadcastable/Const_4Const*
_output_shapes
: *
dtype0*
valueB Bdata/1:0{
0broadcast_weights_1/assert_broadcastable/Const_5Const*
_output_shapes
: *
dtype0*
valueB B
is_scalar=?
4broadcast_weights_1/assert_broadcastable/AssertGuardIfIbroadcast_weights_1/assert_broadcastable/is_valid_shape/Identity:output:0Ibroadcast_weights_1/assert_broadcastable/is_valid_shape/Identity:output:0?broadcast_weights_1/assert_broadcastable/weights/shape:output:0>broadcast_weights_1/assert_broadcastable/values/shape:output:06broadcast_weights_1/assert_broadcastable/is_scalar:z:0*
Tcond0
*
Tin
2

*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *S
else_branchDRB
@broadcast_weights_1_assert_broadcastable_AssertGuard_false_49366*
output_shapes
: *R
then_branchCRA
?broadcast_weights_1_assert_broadcastable_AssertGuard_true_49365?
=broadcast_weights_1/assert_broadcastable/AssertGuard/IdentityIdentity=broadcast_weights_1/assert_broadcastable/AssertGuard:output:0*
T0
*
_output_shapes
: ?
#broadcast_weights_1/ones_like/ShapeShapedata_1>^broadcast_weights_1/assert_broadcastable/AssertGuard/Identity*
T0*
_output_shapes
:?
#broadcast_weights_1/ones_like/ConstConst>^broadcast_weights_1/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ???
broadcast_weights_1/ones_likeFill,broadcast_weights_1/ones_like/Shape:output:0,broadcast_weights_1/ones_like/Const:output:0*
T0*/
_output_shapes
:??????????
broadcast_weights_1Mul/my_autoencoder/StatefulPartitionedCall:output:0&broadcast_weights_1/ones_like:output:0*
T0*/
_output_shapes
:?????????g
Mul_1Muldata_1broadcast_weights_1:z:0*
T0*/
_output_shapes
:?????????^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             H
Sum_3Sum	Mul_1:z:0Const:output:0*
T0*
_output_shapes
: 
AssignAddVariableOp_4AssignAddVariableOpassignaddvariableop_4_resourceSum_3:output:0*
_output_shapes
 *
dtype0`
Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             X
Sum_4Sumbroadcast_weights_1:z:0Const_1:output:0*
T0*
_output_shapes
: ?
AssignAddVariableOp_5AssignAddVariableOpassignaddvariableop_5_resourceSum_4:output:0^AssignAddVariableOp_4*
_output_shapes
 *
dtype0?
div_no_nan/ReadVariableOpReadVariableOpassignaddvariableop_resource^AssignAddVariableOp_2*
_output_shapes
: *
dtype0?
div_no_nan/ReadVariableOp_1ReadVariableOpassignaddvariableop_1_resource^AssignAddVariableOp_3*
_output_shapes
: *
dtype0

div_no_nanDivNoNan!div_no_nan/ReadVariableOp:value:0#div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: E
IdentityIdentitydiv_no_nan:z:0*
T0*
_output_shapes
: ?
div_no_nan_1/ReadVariableOpReadVariableOpassignaddvariableop_4_resource^AssignAddVariableOp_4*
_output_shapes
: *
dtype0?
div_no_nan_1/ReadVariableOp_1ReadVariableOpassignaddvariableop_5_resource^AssignAddVariableOp_5*
_output_shapes
: *
dtype0?
div_no_nan_1DivNoNan#div_no_nan_1/ReadVariableOp:value:0%div_no_nan_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
: I

Identity_1Identitydiv_no_nan_1:z:0*
T0*
_output_shapes
: x
div_no_nan_2/ReadVariableOpReadVariableOp$div_no_nan_2_readvariableop_resource*
_output_shapes
: *
dtype0|
div_no_nan_2/ReadVariableOp_1ReadVariableOp&div_no_nan_2_readvariableop_1_resource*
_output_shapes
: *
dtype0?
div_no_nan_2DivNoNan#div_no_nan_2/ReadVariableOp:value:0%div_no_nan_2/ReadVariableOp_1:value:0*
T0*
_output_shapes
: I

Identity_2Identitydiv_no_nan_2:z:0*
T0*
_output_shapes
: S

Identity_3IdentityIdentity_2:output:0^NoOp*
T0*
_output_shapes
: S

Identity_4IdentityIdentity_1:output:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp^AssignAddVariableOp^AssignAddVariableOp_1^AssignAddVariableOp_2^AssignAddVariableOp_3^AssignAddVariableOp_4^AssignAddVariableOp_55^broadcast_weights_1/assert_broadcastable/AssertGuard0^conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp^div_no_nan/ReadVariableOp^div_no_nan/ReadVariableOp_1^div_no_nan_1/ReadVariableOp^div_no_nan_1/ReadVariableOp_1^div_no_nan_2/ReadVariableOp^div_no_nan_2/ReadVariableOp_1'^my_autoencoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : 2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_12.
AssignAddVariableOp_2AssignAddVariableOp_22.
AssignAddVariableOp_3AssignAddVariableOp_32.
AssignAddVariableOp_4AssignAddVariableOp_42.
AssignAddVariableOp_5AssignAddVariableOp_52l
4broadcast_weights_1/assert_broadcastable/AssertGuard4broadcast_weights_1/assert_broadcastable/AssertGuard2b
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp26
div_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp2:
div_no_nan/ReadVariableOp_1div_no_nan/ReadVariableOp_12:
div_no_nan_1/ReadVariableOpdiv_no_nan_1/ReadVariableOp2>
div_no_nan_1/ReadVariableOp_1div_no_nan_1/ReadVariableOp_12:
div_no_nan_2/ReadVariableOpdiv_no_nan_2/ReadVariableOp2>
div_no_nan_2/ReadVariableOp_1div_no_nan_2/ReadVariableOp_12P
&my_autoencoder/StatefulPartitionedCall&my_autoencoder/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_namedata/0:WS
/
_output_shapes
:?????????
 
_user_specified_namedata/1
??
?#
__inference_train_step_50239

data_0

data_1.
my_autoencoder_49458:"
my_autoencoder_49460:.
my_autoencoder_49462:"
my_autoencoder_49464:.
my_autoencoder_49466:0"
my_autoencoder_49468:0.
my_autoencoder_49470:00"
my_autoencoder_49472:0'
my_autoencoder_49474:	?
"
my_autoencoder_49476:
'
my_autoencoder_49478:	
?#
my_autoencoder_49480:	?.
my_autoencoder_49482:0"
my_autoencoder_49484:.
my_autoencoder_49486:"
my_autoencoder_49488:&
assignaddvariableop_resource: (
assignaddvariableop_1_resource: +
!adam_cast_readvariableop_resource: &
adam_readvariableop_resource:	 -
#adam_cast_2_readvariableop_resource: -
#adam_cast_3_readvariableop_resource: >
$adam_adam_update_resourceapplyadam_m:>
$adam_adam_update_resourceapplyadam_v:4
&adam_adam_update_1_resourceapplyadam_m:4
&adam_adam_update_1_resourceapplyadam_v:@
&adam_adam_update_2_resourceapplyadam_m:@
&adam_adam_update_2_resourceapplyadam_v:4
&adam_adam_update_3_resourceapplyadam_m:4
&adam_adam_update_3_resourceapplyadam_v:@
&adam_adam_update_4_resourceapplyadam_m:0@
&adam_adam_update_4_resourceapplyadam_v:04
&adam_adam_update_5_resourceapplyadam_m:04
&adam_adam_update_5_resourceapplyadam_v:0@
&adam_adam_update_6_resourceapplyadam_m:00@
&adam_adam_update_6_resourceapplyadam_v:004
&adam_adam_update_7_resourceapplyadam_m:04
&adam_adam_update_7_resourceapplyadam_v:09
&adam_adam_update_8_resourceapplyadam_m:	?
9
&adam_adam_update_8_resourceapplyadam_v:	?
4
&adam_adam_update_9_resourceapplyadam_m:
4
&adam_adam_update_9_resourceapplyadam_v:
:
'adam_adam_update_10_resourceapplyadam_m:	
?:
'adam_adam_update_10_resourceapplyadam_v:	
?6
'adam_adam_update_11_resourceapplyadam_m:	?6
'adam_adam_update_11_resourceapplyadam_v:	?A
'adam_adam_update_12_resourceapplyadam_m:0A
'adam_adam_update_12_resourceapplyadam_v:05
'adam_adam_update_13_resourceapplyadam_m:5
'adam_adam_update_13_resourceapplyadam_v:A
'adam_adam_update_14_resourceapplyadam_m:A
'adam_adam_update_14_resourceapplyadam_v:5
'adam_adam_update_15_resourceapplyadam_m:5
'adam_adam_update_15_resourceapplyadam_v:(
assignaddvariableop_4_resource: (
assignaddvariableop_5_resource: .
$div_no_nan_2_readvariableop_resource: 0
&div_no_nan_2_readvariableop_1_resource: 

identity_3

identity_4??Adam/Adam/AssignAddVariableOp?"Adam/Adam/update/ResourceApplyAdam?$Adam/Adam/update_1/ResourceApplyAdam?%Adam/Adam/update_10/ResourceApplyAdam?%Adam/Adam/update_11/ResourceApplyAdam?%Adam/Adam/update_12/ResourceApplyAdam?%Adam/Adam/update_13/ResourceApplyAdam?%Adam/Adam/update_14/ResourceApplyAdam?%Adam/Adam/update_15/ResourceApplyAdam?$Adam/Adam/update_2/ResourceApplyAdam?$Adam/Adam/update_3/ResourceApplyAdam?$Adam/Adam/update_4/ResourceApplyAdam?$Adam/Adam/update_5/ResourceApplyAdam?$Adam/Adam/update_6/ResourceApplyAdam?$Adam/Adam/update_7/ResourceApplyAdam?$Adam/Adam/update_8/ResourceApplyAdam?$Adam/Adam/update_9/ResourceApplyAdam?Adam/Cast/ReadVariableOp?Adam/Cast_2/ReadVariableOp?Adam/Cast_3/ReadVariableOp?Adam/ReadVariableOp?AssignAddVariableOp?AssignAddVariableOp_1?AssignAddVariableOp_2?AssignAddVariableOp_3?AssignAddVariableOp_4?AssignAddVariableOp_5?4broadcast_weights_1/assert_broadcastable/AssertGuard?/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp?div_no_nan/ReadVariableOp?div_no_nan/ReadVariableOp_1?div_no_nan_1/ReadVariableOp?div_no_nan_1/ReadVariableOp_1?div_no_nan_2/ReadVariableOp?div_no_nan_2/ReadVariableOp_1?&my_autoencoder/StatefulPartitionedCall?

&my_autoencoder/StatefulPartitionedCallStatefulPartitionedCalldata_0my_autoencoder_49458my_autoencoder_49460my_autoencoder_49462my_autoencoder_49464my_autoencoder_49466my_autoencoder_49468my_autoencoder_49470my_autoencoder_49472my_autoencoder_49474my_autoencoder_49476my_autoencoder_49478my_autoencoder_49480my_autoencoder_49482my_autoencoder_49484my_autoencoder_49486my_autoencoder_49488*
Tin
2*(
Tout 
2*
_collective_manager_ids
 *?
_output_shapes?
?:?????????:?????????::?????????0:0:??????????:	
?:?????????
:?????????
:	?
:??????????:?????????0:?????????0:?????????0:?????????0:?????????0:00:?????????0:?????????:0:?????????:?????????:?????????:?????????::?????????:?????????:*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *
fR
__forward_call_49945?
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_autoencoder_49458*&
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
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_autoencoder_49462*&
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
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_autoencoder_49466*&
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
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_autoencoder_49470*&
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
: ?
$mean_squared_error/SquaredDifferenceSquaredDifference/my_autoencoder/StatefulPartitionedCall:output:0data_1*
T0*/
_output_shapes
:?????????t
)mean_squared_error/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
mean_squared_error/MeanMean(mean_squared_error/SquaredDifference:z:02mean_squared_error/Mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????k
&mean_squared_error/weighted_loss/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
$mean_squared_error/weighted_loss/MulMul mean_squared_error/Mean:output:0/mean_squared_error/weighted_loss/Const:output:0*
T0*+
_output_shapes
:?????????}
(mean_squared_error/weighted_loss/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          ?
$mean_squared_error/weighted_loss/SumSum(mean_squared_error/weighted_loss/Mul:z:01mean_squared_error/weighted_loss/Const_1:output:0*
T0*
_output_shapes
: ?
-mean_squared_error/weighted_loss/num_elementsSize(mean_squared_error/weighted_loss/Mul:z:0*
T0*
_output_shapes
: ?
2mean_squared_error/weighted_loss/num_elements/CastCast6mean_squared_error/weighted_loss/num_elements:output:0*

DstT0*

SrcT0*
_output_shapes
: g
%mean_squared_error/weighted_loss/RankConst*
_output_shapes
: *
dtype0*
value	B : n
,mean_squared_error/weighted_loss/range/startConst*
_output_shapes
: *
dtype0*
value	B : n
,mean_squared_error/weighted_loss/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
&mean_squared_error/weighted_loss/rangeRange5mean_squared_error/weighted_loss/range/start:output:0.mean_squared_error/weighted_loss/Rank:output:05mean_squared_error/weighted_loss/range/delta:output:0*
_output_shapes
: ?
&mean_squared_error/weighted_loss/Sum_1Sum-mean_squared_error/weighted_loss/Sum:output:0/mean_squared_error/weighted_loss/range:output:0*
T0*
_output_shapes
: ?
&mean_squared_error/weighted_loss/valueDivNoNan/mean_squared_error/weighted_loss/Sum_1:output:06mean_squared_error/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: ;
ShapeShapedata_1*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
AddNAddN!conv2d/kernel/Regularizer/mul:z:0#conv2d_1/kernel/Regularizer/mul:z:0#conv2d_2/kernel/Regularizer/mul:z:0#conv2d_3/kernel/Regularizer/mul:z:0*
N*
T0*
_output_shapes
: p
AddN_1AddN*mean_squared_error/weighted_loss/value:z:0
AddN:sum:0*
N*
T0*
_output_shapes
: T
CastCaststrided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: C
MulMulAddN_1:sum:0Cast:y:0*
T0*
_output_shapes
: F
RankConst*
_output_shapes
: *
dtype0*
value	B : M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :c
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
: D
SumSumMul:z:0range:output:0*
T0*
_output_shapes
: y
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceSum:output:0*
_output_shapes
 *
dtype0H
Rank_1Const*
_output_shapes
: *
dtype0*
value	B : O
range_1/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :k
range_1Rangerange_1/start:output:0Rank_1:output:0range_1/delta:output:0*
_output_shapes
: I
Sum_1SumCast:y:0range_1:output:0*
T0*
_output_shapes
: ?
AssignAddVariableOp_1AssignAddVariableOpassignaddvariableop_1_resourceSum_1:output:0^AssignAddVariableOp*
_output_shapes
 *
dtype0p
AddN_2AddN*mean_squared_error/weighted_loss/value:z:0
AddN:sum:0*
N*
T0*
_output_shapes
: I
onesConst*
_output_shapes
: *
dtype0*
valueB
 *  ??}
:gradient_tape/mean_squared_error/weighted_loss/value/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
<gradient_tape/mean_squared_error/weighted_loss/value/Shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
Jgradient_tape/mean_squared_error/weighted_loss/value/BroadcastGradientArgsBroadcastGradientArgsCgradient_tape/mean_squared_error/weighted_loss/value/Shape:output:0Egradient_tape/mean_squared_error/weighted_loss/value/Shape_1:output:0*2
_output_shapes 
:?????????:??????????
?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nanDivNoNanones:output:06mean_squared_error/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: ?
8gradient_tape/mean_squared_error/weighted_loss/value/SumSumCgradient_tape/mean_squared_error/weighted_loss/value/div_no_nan:z:0Ogradient_tape/mean_squared_error/weighted_loss/value/BroadcastGradientArgs:r0:0*
T0*
_output_shapes
: ?
<gradient_tape/mean_squared_error/weighted_loss/value/ReshapeReshapeAgradient_tape/mean_squared_error/weighted_loss/value/Sum:output:0Cgradient_tape/mean_squared_error/weighted_loss/value/Shape:output:0*
T0*
_output_shapes
: ?
8gradient_tape/mean_squared_error/weighted_loss/value/NegNeg/mean_squared_error/weighted_loss/Sum_1:output:0*
T0*
_output_shapes
: ?
Agradient_tape/mean_squared_error/weighted_loss/value/div_no_nan_1DivNoNan<gradient_tape/mean_squared_error/weighted_loss/value/Neg:y:06mean_squared_error/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: ?
Agradient_tape/mean_squared_error/weighted_loss/value/div_no_nan_2DivNoNanEgradient_tape/mean_squared_error/weighted_loss/value/div_no_nan_1:z:06mean_squared_error/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: ?
8gradient_tape/mean_squared_error/weighted_loss/value/mulMulones:output:0Egradient_tape/mean_squared_error/weighted_loss/value/div_no_nan_2:z:0*
T0*
_output_shapes
: ?
:gradient_tape/mean_squared_error/weighted_loss/value/Sum_1Sum<gradient_tape/mean_squared_error/weighted_loss/value/mul:z:0Ogradient_tape/mean_squared_error/weighted_loss/value/BroadcastGradientArgs:r1:0*
T0*
_output_shapes
: ?
>gradient_tape/mean_squared_error/weighted_loss/value/Reshape_1ReshapeCgradient_tape/mean_squared_error/weighted_loss/value/Sum_1:output:0Egradient_tape/mean_squared_error/weighted_loss/value/Shape_1:output:0*
T0*
_output_shapes
: 
<gradient_tape/mean_squared_error/weighted_loss/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
>gradient_tape/mean_squared_error/weighted_loss/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
6gradient_tape/mean_squared_error/weighted_loss/ReshapeReshapeEgradient_tape/mean_squared_error/weighted_loss/value/Reshape:output:0Ggradient_tape/mean_squared_error/weighted_loss/Reshape/shape_1:output:0*
T0*
_output_shapes
: w
4gradient_tape/mean_squared_error/weighted_loss/ConstConst*
_output_shapes
: *
dtype0*
valueB ?
3gradient_tape/mean_squared_error/weighted_loss/TileTile?gradient_tape/mean_squared_error/weighted_loss/Reshape:output:0=gradient_tape/mean_squared_error/weighted_loss/Const:output:0*
T0*
_output_shapes
: ?
/gradient_tape/conv2d/kernel/Regularizer/mul/MulMulones:output:0)conv2d/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ?
1gradient_tape/conv2d/kernel/Regularizer/mul/Mul_1Mulones:output:0(conv2d/kernel/Regularizer/mul/x:output:0*
T0*
_output_shapes
: ?
1gradient_tape/conv2d_1/kernel/Regularizer/mul/MulMulones:output:0+conv2d_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ?
3gradient_tape/conv2d_1/kernel/Regularizer/mul/Mul_1Mulones:output:0*conv2d_1/kernel/Regularizer/mul/x:output:0*
T0*
_output_shapes
: ?
1gradient_tape/conv2d_2/kernel/Regularizer/mul/MulMulones:output:0+conv2d_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ?
3gradient_tape/conv2d_2/kernel/Regularizer/mul/Mul_1Mulones:output:0*conv2d_2/kernel/Regularizer/mul/x:output:0*
T0*
_output_shapes
: ?
1gradient_tape/conv2d_3/kernel/Regularizer/mul/MulMulones:output:0+conv2d_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ?
3gradient_tape/conv2d_3/kernel/Regularizer/mul/Mul_1Mulones:output:0*conv2d_3/kernel/Regularizer/mul/x:output:0*
T0*
_output_shapes
: ?
>gradient_tape/mean_squared_error/weighted_loss/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
8gradient_tape/mean_squared_error/weighted_loss/Reshape_1Reshape<gradient_tape/mean_squared_error/weighted_loss/Tile:output:0Ggradient_tape/mean_squared_error/weighted_loss/Reshape_1/shape:output:0*
T0*"
_output_shapes
:?
4gradient_tape/mean_squared_error/weighted_loss/ShapeShape(mean_squared_error/weighted_loss/Mul:z:0*
T0*
_output_shapes
:?
5gradient_tape/mean_squared_error/weighted_loss/Tile_1TileAgradient_tape/mean_squared_error/weighted_loss/Reshape_1:output:0=gradient_tape/mean_squared_error/weighted_loss/Shape:output:0*
T0*+
_output_shapes
:??????????
+gradient_tape/conv2d/kernel/Regularizer/mulMul7conv2d/kernel/Regularizer/L2Loss/ReadVariableOp:value:05gradient_tape/conv2d/kernel/Regularizer/mul/Mul_1:z:0*
T0*&
_output_shapes
:?
-gradient_tape/conv2d_1/kernel/Regularizer/mulMul9conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:07gradient_tape/conv2d_1/kernel/Regularizer/mul/Mul_1:z:0*
T0*&
_output_shapes
:?
-gradient_tape/conv2d_2/kernel/Regularizer/mulMul9conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:07gradient_tape/conv2d_2/kernel/Regularizer/mul/Mul_1:z:0*
T0*&
_output_shapes
:0?
-gradient_tape/conv2d_3/kernel/Regularizer/mulMul9conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:07gradient_tape/conv2d_3/kernel/Regularizer/mul/Mul_1:z:0*
T0*&
_output_shapes
:00?
2gradient_tape/mean_squared_error/weighted_loss/MulMul>gradient_tape/mean_squared_error/weighted_loss/Tile_1:output:0/mean_squared_error/weighted_loss/Const:output:0*
T0*+
_output_shapes
:?????????~
&gradient_tape/mean_squared_error/ShapeShape(mean_squared_error/SquaredDifference:z:0*
T0*
_output_shapes
:?
%gradient_tape/mean_squared_error/SizeConst*9
_class/
-+loc:@gradient_tape/mean_squared_error/Shape*
_output_shapes
: *
dtype0*
value	B :?
$gradient_tape/mean_squared_error/addAddV22mean_squared_error/Mean/reduction_indices:output:0.gradient_tape/mean_squared_error/Size:output:0*
T0*9
_class/
-+loc:@gradient_tape/mean_squared_error/Shape*
_output_shapes
: ?
$gradient_tape/mean_squared_error/modFloorMod(gradient_tape/mean_squared_error/add:z:0.gradient_tape/mean_squared_error/Size:output:0*
T0*9
_class/
-+loc:@gradient_tape/mean_squared_error/Shape*
_output_shapes
: ?
(gradient_tape/mean_squared_error/Shape_1Const*9
_class/
-+loc:@gradient_tape/mean_squared_error/Shape*
_output_shapes
: *
dtype0*
valueB ?
,gradient_tape/mean_squared_error/range/startConst*9
_class/
-+loc:@gradient_tape/mean_squared_error/Shape*
_output_shapes
: *
dtype0*
value	B : ?
,gradient_tape/mean_squared_error/range/deltaConst*9
_class/
-+loc:@gradient_tape/mean_squared_error/Shape*
_output_shapes
: *
dtype0*
value	B :?
&gradient_tape/mean_squared_error/rangeRange5gradient_tape/mean_squared_error/range/start:output:0.gradient_tape/mean_squared_error/Size:output:05gradient_tape/mean_squared_error/range/delta:output:0*9
_class/
-+loc:@gradient_tape/mean_squared_error/Shape*
_output_shapes
:?
+gradient_tape/mean_squared_error/ones/ConstConst*9
_class/
-+loc:@gradient_tape/mean_squared_error/Shape*
_output_shapes
: *
dtype0*
value	B :?
%gradient_tape/mean_squared_error/onesFill1gradient_tape/mean_squared_error/Shape_1:output:04gradient_tape/mean_squared_error/ones/Const:output:0*
T0*9
_class/
-+loc:@gradient_tape/mean_squared_error/Shape*
_output_shapes
: ?
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch/gradient_tape/mean_squared_error/range:output:0(gradient_tape/mean_squared_error/mod:z:0/gradient_tape/mean_squared_error/Shape:output:0.gradient_tape/mean_squared_error/ones:output:0*
N*
T0*9
_class/
-+loc:@gradient_tape/mean_squared_error/Shape*
_output_shapes
:?
(gradient_tape/mean_squared_error/ReshapeReshape6gradient_tape/mean_squared_error/weighted_loss/Mul:z:07gradient_tape/mean_squared_error/DynamicStitch:merged:0*
T0*J
_output_shapes8
6:4?????????????????????????????????????
,gradient_tape/mean_squared_error/BroadcastToBroadcastTo1gradient_tape/mean_squared_error/Reshape:output:0/gradient_tape/mean_squared_error/Shape:output:0*
T0*/
_output_shapes
:??????????
(gradient_tape/mean_squared_error/Shape_2Shape(mean_squared_error/SquaredDifference:z:0*
T0*
_output_shapes
:x
(gradient_tape/mean_squared_error/Shape_3Shape mean_squared_error/Mean:output:0*
T0*
_output_shapes
:p
&gradient_tape/mean_squared_error/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
%gradient_tape/mean_squared_error/ProdProd1gradient_tape/mean_squared_error/Shape_2:output:0/gradient_tape/mean_squared_error/Const:output:0*
T0*
_output_shapes
: r
(gradient_tape/mean_squared_error/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
'gradient_tape/mean_squared_error/Prod_1Prod1gradient_tape/mean_squared_error/Shape_3:output:01gradient_tape/mean_squared_error/Const_1:output:0*
T0*
_output_shapes
: l
*gradient_tape/mean_squared_error/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :?
(gradient_tape/mean_squared_error/MaximumMaximum0gradient_tape/mean_squared_error/Prod_1:output:03gradient_tape/mean_squared_error/Maximum/y:output:0*
T0*
_output_shapes
: ?
)gradient_tape/mean_squared_error/floordivFloorDiv.gradient_tape/mean_squared_error/Prod:output:0,gradient_tape/mean_squared_error/Maximum:z:0*
T0*
_output_shapes
: ?
%gradient_tape/mean_squared_error/CastCast-gradient_tape/mean_squared_error/floordiv:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
(gradient_tape/mean_squared_error/truedivRealDiv5gradient_tape/mean_squared_error/BroadcastTo:output:0)gradient_tape/mean_squared_error/Cast:y:0*
T0*/
_output_shapes
:??????????
'gradient_tape/mean_squared_error/scalarConst)^gradient_tape/mean_squared_error/truediv*
_output_shapes
: *
dtype0*
valueB
 *   @?
$gradient_tape/mean_squared_error/MulMul0gradient_tape/mean_squared_error/scalar:output:0,gradient_tape/mean_squared_error/truediv:z:0*
T0*/
_output_shapes
:??????????
$gradient_tape/mean_squared_error/subSub/my_autoencoder/StatefulPartitionedCall:output:0data_1)^gradient_tape/mean_squared_error/truediv*
T0*/
_output_shapes
:??????????
&gradient_tape/mean_squared_error/mul_1Mul(gradient_tape/mean_squared_error/Mul:z:0(gradient_tape/mean_squared_error/sub:z:0*
T0*/
_output_shapes
:??????????
(gradient_tape/mean_squared_error/Shape_4Shape/my_autoencoder/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:^
(gradient_tape/mean_squared_error/Shape_5Shapedata_1*
T0*
_output_shapes
:?
6gradient_tape/mean_squared_error/BroadcastGradientArgsBroadcastGradientArgs1gradient_tape/mean_squared_error/Shape_4:output:01gradient_tape/mean_squared_error/Shape_5:output:0*2
_output_shapes 
:?????????:??????????
$gradient_tape/mean_squared_error/SumSum*gradient_tape/mean_squared_error/mul_1:z:0;gradient_tape/mean_squared_error/BroadcastGradientArgs:r0:0*
T0*
_output_shapes
:?
*gradient_tape/mean_squared_error/Reshape_1Reshape-gradient_tape/mean_squared_error/Sum:output:01gradient_tape/mean_squared_error/Shape_4:output:0*
T0*/
_output_shapes
:??????????
PartitionedCallPartitionedCall3gradient_tape/mean_squared_error/Reshape_1:output:0/my_autoencoder/StatefulPartitionedCall:output:1/my_autoencoder/StatefulPartitionedCall:output:2/my_autoencoder/StatefulPartitionedCall:output:3/my_autoencoder/StatefulPartitionedCall:output:4/my_autoencoder/StatefulPartitionedCall:output:5/my_autoencoder/StatefulPartitionedCall:output:6/my_autoencoder/StatefulPartitionedCall:output:7/my_autoencoder/StatefulPartitionedCall:output:8/my_autoencoder/StatefulPartitionedCall:output:90my_autoencoder/StatefulPartitionedCall:output:100my_autoencoder/StatefulPartitionedCall:output:110my_autoencoder/StatefulPartitionedCall:output:120my_autoencoder/StatefulPartitionedCall:output:130my_autoencoder/StatefulPartitionedCall:output:140my_autoencoder/StatefulPartitionedCall:output:150my_autoencoder/StatefulPartitionedCall:output:160my_autoencoder/StatefulPartitionedCall:output:170my_autoencoder/StatefulPartitionedCall:output:180my_autoencoder/StatefulPartitionedCall:output:190my_autoencoder/StatefulPartitionedCall:output:200my_autoencoder/StatefulPartitionedCall:output:210my_autoencoder/StatefulPartitionedCall:output:220my_autoencoder/StatefulPartitionedCall:output:230my_autoencoder/StatefulPartitionedCall:output:240my_autoencoder/StatefulPartitionedCall:output:250my_autoencoder/StatefulPartitionedCall:output:260my_autoencoder/StatefulPartitionedCall:output:27*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *?
_output_shapes?
?:?????????:::::0:0:00:0:	?
:
:	
?:?:0:::* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference___backward_call_49614_49946?
AddN_3AddN/gradient_tape/conv2d/kernel/Regularizer/mul:z:0PartitionedCall:output:1*
N*
T0*&
_output_shapes
:?
AddN_4AddN1gradient_tape/conv2d_1/kernel/Regularizer/mul:z:0PartitionedCall:output:3*
N*
T0*&
_output_shapes
:?
AddN_5AddN1gradient_tape/conv2d_2/kernel/Regularizer/mul:z:0PartitionedCall:output:5*
N*
T0*&
_output_shapes
:0?
AddN_6AddN1gradient_tape/conv2d_3/kernel/Regularizer/mul:z:0PartitionedCall:output:7*
N*
T0*&
_output_shapes
:00r
Adam/Cast/ReadVariableOpReadVariableOp!adam_cast_readvariableop_resource*
_output_shapes
: *
dtype0?
Adam/IdentityIdentity Adam/Cast/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: h
Adam/ReadVariableOpReadVariableOpadam_readvariableop_resource*
_output_shapes
: *
dtype0	z

Adam/add/yConst",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0	*
value	B	 R?
Adam/addAddV2Adam/ReadVariableOp:value:0Adam/add/y:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0	*
_output_shapes
: 
Adam/Cast_1CastAdam/add:z:0",/job:localhost/replica:0/task:0/device:CPU:0*

DstT0*

SrcT0	*
_output_shapes
: v
Adam/Cast_2/ReadVariableOpReadVariableOp#adam_cast_2_readvariableop_resource*
_output_shapes
: *
dtype0?
Adam/Identity_1Identity"Adam/Cast_2/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: v
Adam/Cast_3/ReadVariableOpReadVariableOp#adam_cast_3_readvariableop_resource*
_output_shapes
: *
dtype0?
Adam/Identity_2Identity"Adam/Cast_3/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: ?
Adam/PowPowAdam/Identity_1:output:0Adam/Cast_1:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: ?

Adam/Pow_1PowAdam/Identity_2:output:0Adam/Cast_1:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: }

Adam/sub/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ???
Adam/subSubAdam/sub/x:output:0Adam/Pow_1:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: n
	Adam/SqrtSqrtAdam/sub:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: 
Adam/sub_1/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ???

Adam/sub_1SubAdam/sub_1/x:output:0Adam/Pow:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: ?
Adam/truedivRealDivAdam/Sqrt:y:0Adam/sub_1:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: ?
Adam/mulMulAdam/Identity:output:0Adam/truediv:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: }

Adam/ConstConst",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *???3
Adam/sub_2/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ???

Adam/sub_2SubAdam/sub_2/x:output:0Adam/Identity_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: 
Adam/sub_3/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ???

Adam/sub_3SubAdam/sub_3/x:output:0Adam/Identity_2:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: Z
Adam/Identity_3IdentityAddN_3:sum:0*
T0*&
_output_shapes
:Z
Adam/Identity_4IdentityPartitionedCall:output:2*
T0*
_output_shapes
:Z
Adam/Identity_5IdentityAddN_4:sum:0*
T0*&
_output_shapes
:Z
Adam/Identity_6IdentityPartitionedCall:output:4*
T0*
_output_shapes
:Z
Adam/Identity_7IdentityAddN_5:sum:0*
T0*&
_output_shapes
:0Z
Adam/Identity_8IdentityPartitionedCall:output:6*
T0*
_output_shapes
:0Z
Adam/Identity_9IdentityAddN_6:sum:0*
T0*&
_output_shapes
:00[
Adam/Identity_10IdentityPartitionedCall:output:8*
T0*
_output_shapes
:0`
Adam/Identity_11IdentityPartitionedCall:output:9*
T0*
_output_shapes
:	?
\
Adam/Identity_12IdentityPartitionedCall:output:10*
T0*
_output_shapes
:
a
Adam/Identity_13IdentityPartitionedCall:output:11*
T0*
_output_shapes
:	
?]
Adam/Identity_14IdentityPartitionedCall:output:12*
T0*
_output_shapes	
:?h
Adam/Identity_15IdentityPartitionedCall:output:13*
T0*&
_output_shapes
:0\
Adam/Identity_16IdentityPartitionedCall:output:14*
T0*
_output_shapes
:h
Adam/Identity_17IdentityPartitionedCall:output:15*
T0*&
_output_shapes
:\
Adam/Identity_18IdentityPartitionedCall:output:16*
T0*
_output_shapes
:?	
Adam/IdentityN	IdentityNAddN_3:sum:0PartitionedCall:output:2AddN_4:sum:0PartitionedCall:output:4AddN_5:sum:0PartitionedCall:output:6AddN_6:sum:0PartitionedCall:output:8PartitionedCall:output:9PartitionedCall:output:10PartitionedCall:output:11PartitionedCall:output:12PartitionedCall:output:13PartitionedCall:output:14PartitionedCall:output:15PartitionedCall:output:16AddN_3:sum:0PartitionedCall:output:2AddN_4:sum:0PartitionedCall:output:4AddN_5:sum:0PartitionedCall:output:6AddN_6:sum:0PartitionedCall:output:8PartitionedCall:output:9PartitionedCall:output:10PartitionedCall:output:11PartitionedCall:output:12PartitionedCall:output:13PartitionedCall:output:14PartitionedCall:output:15PartitionedCall:output:16*)
T$
"2 *+
_gradient_op_typeCustomGradient-50023*?
_output_shapes?
?:::::0:0:00:0:	?
:
:	
?:?:0::::::::0:0:00:0:	?
:
:	
?:?:0:::?
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdammy_autoencoder_49458$adam_adam_update_resourceapplyadam_m$adam_adam_update_resourceapplyadam_vAdam/Pow:z:0Adam/Pow_1:z:0Adam/Identity:output:0Adam/Identity_1:output:0Adam/Identity_2:output:0Adam/Const:output:0Adam/IdentityN:output:00^conv2d/kernel/Regularizer/L2Loss/ReadVariableOp'^my_autoencoder/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@my_autoencoder/49458*
_output_shapes
 *
use_locking(?
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdammy_autoencoder_49460&adam_adam_update_1_resourceapplyadam_m&adam_adam_update_1_resourceapplyadam_vAdam/Pow:z:0Adam/Pow_1:z:0Adam/Identity:output:0Adam/Identity_1:output:0Adam/Identity_2:output:0Adam/Const:output:0Adam/IdentityN:output:1'^my_autoencoder/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@my_autoencoder/49460*
_output_shapes
 *
use_locking(?
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdammy_autoencoder_49462&adam_adam_update_2_resourceapplyadam_m&adam_adam_update_2_resourceapplyadam_vAdam/Pow:z:0Adam/Pow_1:z:0Adam/Identity:output:0Adam/Identity_1:output:0Adam/Identity_2:output:0Adam/Const:output:0Adam/IdentityN:output:22^conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp'^my_autoencoder/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@my_autoencoder/49462*
_output_shapes
 *
use_locking(?
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdammy_autoencoder_49464&adam_adam_update_3_resourceapplyadam_m&adam_adam_update_3_resourceapplyadam_vAdam/Pow:z:0Adam/Pow_1:z:0Adam/Identity:output:0Adam/Identity_1:output:0Adam/Identity_2:output:0Adam/Const:output:0Adam/IdentityN:output:3'^my_autoencoder/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@my_autoencoder/49464*
_output_shapes
 *
use_locking(?
$Adam/Adam/update_4/ResourceApplyAdamResourceApplyAdammy_autoencoder_49466&adam_adam_update_4_resourceapplyadam_m&adam_adam_update_4_resourceapplyadam_vAdam/Pow:z:0Adam/Pow_1:z:0Adam/Identity:output:0Adam/Identity_1:output:0Adam/Identity_2:output:0Adam/Const:output:0Adam/IdentityN:output:42^conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp'^my_autoencoder/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@my_autoencoder/49466*
_output_shapes
 *
use_locking(?
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdammy_autoencoder_49468&adam_adam_update_5_resourceapplyadam_m&adam_adam_update_5_resourceapplyadam_vAdam/Pow:z:0Adam/Pow_1:z:0Adam/Identity:output:0Adam/Identity_1:output:0Adam/Identity_2:output:0Adam/Const:output:0Adam/IdentityN:output:5'^my_autoencoder/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@my_autoencoder/49468*
_output_shapes
 *
use_locking(?
$Adam/Adam/update_6/ResourceApplyAdamResourceApplyAdammy_autoencoder_49470&adam_adam_update_6_resourceapplyadam_m&adam_adam_update_6_resourceapplyadam_vAdam/Pow:z:0Adam/Pow_1:z:0Adam/Identity:output:0Adam/Identity_1:output:0Adam/Identity_2:output:0Adam/Const:output:0Adam/IdentityN:output:62^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp'^my_autoencoder/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@my_autoencoder/49470*
_output_shapes
 *
use_locking(?
$Adam/Adam/update_7/ResourceApplyAdamResourceApplyAdammy_autoencoder_49472&adam_adam_update_7_resourceapplyadam_m&adam_adam_update_7_resourceapplyadam_vAdam/Pow:z:0Adam/Pow_1:z:0Adam/Identity:output:0Adam/Identity_1:output:0Adam/Identity_2:output:0Adam/Const:output:0Adam/IdentityN:output:7'^my_autoencoder/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@my_autoencoder/49472*
_output_shapes
 *
use_locking(?
$Adam/Adam/update_8/ResourceApplyAdamResourceApplyAdammy_autoencoder_49474&adam_adam_update_8_resourceapplyadam_m&adam_adam_update_8_resourceapplyadam_vAdam/Pow:z:0Adam/Pow_1:z:0Adam/Identity:output:0Adam/Identity_1:output:0Adam/Identity_2:output:0Adam/Const:output:0Adam/IdentityN:output:8'^my_autoencoder/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@my_autoencoder/49474*
_output_shapes
 *
use_locking(?
$Adam/Adam/update_9/ResourceApplyAdamResourceApplyAdammy_autoencoder_49476&adam_adam_update_9_resourceapplyadam_m&adam_adam_update_9_resourceapplyadam_vAdam/Pow:z:0Adam/Pow_1:z:0Adam/Identity:output:0Adam/Identity_1:output:0Adam/Identity_2:output:0Adam/Const:output:0Adam/IdentityN:output:9'^my_autoencoder/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@my_autoencoder/49476*
_output_shapes
 *
use_locking(?
%Adam/Adam/update_10/ResourceApplyAdamResourceApplyAdammy_autoencoder_49478'adam_adam_update_10_resourceapplyadam_m'adam_adam_update_10_resourceapplyadam_vAdam/Pow:z:0Adam/Pow_1:z:0Adam/Identity:output:0Adam/Identity_1:output:0Adam/Identity_2:output:0Adam/Const:output:0Adam/IdentityN:output:10'^my_autoencoder/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@my_autoencoder/49478*
_output_shapes
 *
use_locking(?
%Adam/Adam/update_11/ResourceApplyAdamResourceApplyAdammy_autoencoder_49480'adam_adam_update_11_resourceapplyadam_m'adam_adam_update_11_resourceapplyadam_vAdam/Pow:z:0Adam/Pow_1:z:0Adam/Identity:output:0Adam/Identity_1:output:0Adam/Identity_2:output:0Adam/Const:output:0Adam/IdentityN:output:11'^my_autoencoder/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@my_autoencoder/49480*
_output_shapes
 *
use_locking(?
%Adam/Adam/update_12/ResourceApplyAdamResourceApplyAdammy_autoencoder_49482'adam_adam_update_12_resourceapplyadam_m'adam_adam_update_12_resourceapplyadam_vAdam/Pow:z:0Adam/Pow_1:z:0Adam/Identity:output:0Adam/Identity_1:output:0Adam/Identity_2:output:0Adam/Const:output:0Adam/IdentityN:output:12'^my_autoencoder/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@my_autoencoder/49482*
_output_shapes
 *
use_locking(?
%Adam/Adam/update_13/ResourceApplyAdamResourceApplyAdammy_autoencoder_49484'adam_adam_update_13_resourceapplyadam_m'adam_adam_update_13_resourceapplyadam_vAdam/Pow:z:0Adam/Pow_1:z:0Adam/Identity:output:0Adam/Identity_1:output:0Adam/Identity_2:output:0Adam/Const:output:0Adam/IdentityN:output:13'^my_autoencoder/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@my_autoencoder/49484*
_output_shapes
 *
use_locking(?
%Adam/Adam/update_14/ResourceApplyAdamResourceApplyAdammy_autoencoder_49486'adam_adam_update_14_resourceapplyadam_m'adam_adam_update_14_resourceapplyadam_vAdam/Pow:z:0Adam/Pow_1:z:0Adam/Identity:output:0Adam/Identity_1:output:0Adam/Identity_2:output:0Adam/Const:output:0Adam/IdentityN:output:14'^my_autoencoder/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@my_autoencoder/49486*
_output_shapes
 *
use_locking(?
%Adam/Adam/update_15/ResourceApplyAdamResourceApplyAdammy_autoencoder_49488'adam_adam_update_15_resourceapplyadam_m'adam_adam_update_15_resourceapplyadam_vAdam/Pow:z:0Adam/Pow_1:z:0Adam/Identity:output:0Adam/Identity_1:output:0Adam/Identity_2:output:0Adam/Const:output:0Adam/IdentityN:output:15'^my_autoencoder/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@my_autoencoder/49488*
_output_shapes
 *
use_locking(?
Adam/Adam/group_depsNoOp#^Adam/Adam/update/ResourceApplyAdam%^Adam/Adam/update_1/ResourceApplyAdam&^Adam/Adam/update_10/ResourceApplyAdam&^Adam/Adam/update_11/ResourceApplyAdam&^Adam/Adam/update_12/ResourceApplyAdam&^Adam/Adam/update_13/ResourceApplyAdam&^Adam/Adam/update_14/ResourceApplyAdam&^Adam/Adam/update_15/ResourceApplyAdam%^Adam/Adam/update_2/ResourceApplyAdam%^Adam/Adam/update_3/ResourceApplyAdam%^Adam/Adam/update_4/ResourceApplyAdam%^Adam/Adam/update_5/ResourceApplyAdam%^Adam/Adam/update_6/ResourceApplyAdam%^Adam/Adam/update_7/ResourceApplyAdam%^Adam/Adam/update_8/ResourceApplyAdam%^Adam/Adam/update_9/ResourceApplyAdam",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 h
Adam/Adam/ConstConst^Adam/Adam/group_deps*
_output_shapes
: *
dtype0	*
value	B	 R?
Adam/Adam/AssignAddVariableOpAssignAddVariableOpadam_readvariableop_resourceAdam/Adam/Const:output:0^Adam/ReadVariableOp*
_output_shapes
 *
dtype0	H
Rank_2Const*
_output_shapes
: *
dtype0*
value	B : O
range_2/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :k
range_2Rangerange_2/start:output:0Rank_2:output:0range_2/delta:output:0*
_output_shapes
: M
Sum_2SumAddN_2:sum:0range_2:output:0*
T0*
_output_shapes
: ?
AssignAddVariableOp_2AssignAddVariableOpassignaddvariableop_resourceSum_2:output:0^AssignAddVariableOp*
_output_shapes
 *
dtype0F
SizeConst*
_output_shapes
: *
dtype0*
value	B :M
Cast_1CastSize:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
AssignAddVariableOp_3AssignAddVariableOpassignaddvariableop_1_resource
Cast_1:y:0^AssignAddVariableOp_1^AssignAddVariableOp_2*
_output_shapes
 *
dtype0?
6broadcast_weights_1/assert_broadcastable/weights/shapeShape/my_autoencoder/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:w
5broadcast_weights_1/assert_broadcastable/weights/rankConst*
_output_shapes
: *
dtype0*
value	B :k
5broadcast_weights_1/assert_broadcastable/values/shapeShapedata_1*
T0*
_output_shapes
:v
4broadcast_weights_1/assert_broadcastable/values/rankConst*
_output_shapes
: *
dtype0*
value	B :v
4broadcast_weights_1/assert_broadcastable/is_scalar/xConst*
_output_shapes
: *
dtype0*
value	B : ?
2broadcast_weights_1/assert_broadcastable/is_scalarEqual=broadcast_weights_1/assert_broadcastable/is_scalar/x:output:0>broadcast_weights_1/assert_broadcastable/weights/rank:output:0*
T0*
_output_shapes
: ?
7broadcast_weights_1/assert_broadcastable/is_valid_shapeStatelessIf6broadcast_weights_1/assert_broadcastable/is_scalar:z:06broadcast_weights_1/assert_broadcastable/is_scalar:z:0=broadcast_weights_1/assert_broadcastable/values/rank:output:0>broadcast_weights_1/assert_broadcastable/weights/rank:output:0>broadcast_weights_1/assert_broadcastable/values/shape:output:0?broadcast_weights_1/assert_broadcastable/weights/shape:output:0*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *V
else_branchGRE
Cbroadcast_weights_1_assert_broadcastable_is_valid_shape_false_50135*
output_shapes
: *U
then_branchFRD
Bbroadcast_weights_1_assert_broadcastable_is_valid_shape_true_50134?
@broadcast_weights_1/assert_broadcastable/is_valid_shape/IdentityIdentity@broadcast_weights_1/assert_broadcastable/is_valid_shape:output:0*
T0
*
_output_shapes
: ?
.broadcast_weights_1/assert_broadcastable/ConstConst*
_output_shapes
: *
dtype0*8
value/B- B'weights can not be broadcast to values.
0broadcast_weights_1/assert_broadcastable/Const_1Const*
_output_shapes
: *
dtype0*
valueB Bweights.shape=?
0broadcast_weights_1/assert_broadcastable/Const_2Const*
_output_shapes
: *
dtype0*9
value0B. B(my_autoencoder/StatefulPartitionedCall:0~
0broadcast_weights_1/assert_broadcastable/Const_3Const*
_output_shapes
: *
dtype0*
valueB Bvalues.shape=y
0broadcast_weights_1/assert_broadcastable/Const_4Const*
_output_shapes
: *
dtype0*
valueB Bdata/1:0{
0broadcast_weights_1/assert_broadcastable/Const_5Const*
_output_shapes
: *
dtype0*
valueB B
is_scalar=?
4broadcast_weights_1/assert_broadcastable/AssertGuardIfIbroadcast_weights_1/assert_broadcastable/is_valid_shape/Identity:output:0Ibroadcast_weights_1/assert_broadcastable/is_valid_shape/Identity:output:0?broadcast_weights_1/assert_broadcastable/weights/shape:output:0>broadcast_weights_1/assert_broadcastable/values/shape:output:06broadcast_weights_1/assert_broadcastable/is_scalar:z:0*
Tcond0
*
Tin
2

*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *S
else_branchDRB
@broadcast_weights_1_assert_broadcastable_AssertGuard_false_50189*
output_shapes
: *R
then_branchCRA
?broadcast_weights_1_assert_broadcastable_AssertGuard_true_50188?
=broadcast_weights_1/assert_broadcastable/AssertGuard/IdentityIdentity=broadcast_weights_1/assert_broadcastable/AssertGuard:output:0*
T0
*
_output_shapes
: ?
#broadcast_weights_1/ones_like/ShapeShapedata_1>^broadcast_weights_1/assert_broadcastable/AssertGuard/Identity*
T0*
_output_shapes
:?
#broadcast_weights_1/ones_like/ConstConst>^broadcast_weights_1/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ???
broadcast_weights_1/ones_likeFill,broadcast_weights_1/ones_like/Shape:output:0,broadcast_weights_1/ones_like/Const:output:0*
T0*/
_output_shapes
:??????????
broadcast_weights_1Mul/my_autoencoder/StatefulPartitionedCall:output:0&broadcast_weights_1/ones_like:output:0*
T0*/
_output_shapes
:?????????g
Mul_1Muldata_1broadcast_weights_1:z:0*
T0*/
_output_shapes
:?????????^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             H
Sum_3Sum	Mul_1:z:0Const:output:0*
T0*
_output_shapes
: 
AssignAddVariableOp_4AssignAddVariableOpassignaddvariableop_4_resourceSum_3:output:0*
_output_shapes
 *
dtype0`
Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             X
Sum_4Sumbroadcast_weights_1:z:0Const_1:output:0*
T0*
_output_shapes
: ?
AssignAddVariableOp_5AssignAddVariableOpassignaddvariableop_5_resourceSum_4:output:0^AssignAddVariableOp_4*
_output_shapes
 *
dtype0?
div_no_nan/ReadVariableOpReadVariableOpassignaddvariableop_resource^AssignAddVariableOp_2*
_output_shapes
: *
dtype0?
div_no_nan/ReadVariableOp_1ReadVariableOpassignaddvariableop_1_resource^AssignAddVariableOp_3*
_output_shapes
: *
dtype0

div_no_nanDivNoNan!div_no_nan/ReadVariableOp:value:0#div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: E
IdentityIdentitydiv_no_nan:z:0*
T0*
_output_shapes
: ?
div_no_nan_1/ReadVariableOpReadVariableOpassignaddvariableop_4_resource^AssignAddVariableOp_4*
_output_shapes
: *
dtype0?
div_no_nan_1/ReadVariableOp_1ReadVariableOpassignaddvariableop_5_resource^AssignAddVariableOp_5*
_output_shapes
: *
dtype0?
div_no_nan_1DivNoNan#div_no_nan_1/ReadVariableOp:value:0%div_no_nan_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
: I

Identity_1Identitydiv_no_nan_1:z:0*
T0*
_output_shapes
: x
div_no_nan_2/ReadVariableOpReadVariableOp$div_no_nan_2_readvariableop_resource*
_output_shapes
: *
dtype0|
div_no_nan_2/ReadVariableOp_1ReadVariableOp&div_no_nan_2_readvariableop_1_resource*
_output_shapes
: *
dtype0?
div_no_nan_2DivNoNan#div_no_nan_2/ReadVariableOp:value:0%div_no_nan_2/ReadVariableOp_1:value:0*
T0*
_output_shapes
: I

Identity_2Identitydiv_no_nan_2:z:0*
T0*
_output_shapes
: S

Identity_3IdentityIdentity_2:output:0^NoOp*
T0*
_output_shapes
: S

Identity_4IdentityIdentity_1:output:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp^Adam/Adam/AssignAddVariableOp#^Adam/Adam/update/ResourceApplyAdam%^Adam/Adam/update_1/ResourceApplyAdam&^Adam/Adam/update_10/ResourceApplyAdam&^Adam/Adam/update_11/ResourceApplyAdam&^Adam/Adam/update_12/ResourceApplyAdam&^Adam/Adam/update_13/ResourceApplyAdam&^Adam/Adam/update_14/ResourceApplyAdam&^Adam/Adam/update_15/ResourceApplyAdam%^Adam/Adam/update_2/ResourceApplyAdam%^Adam/Adam/update_3/ResourceApplyAdam%^Adam/Adam/update_4/ResourceApplyAdam%^Adam/Adam/update_5/ResourceApplyAdam%^Adam/Adam/update_6/ResourceApplyAdam%^Adam/Adam/update_7/ResourceApplyAdam%^Adam/Adam/update_8/ResourceApplyAdam%^Adam/Adam/update_9/ResourceApplyAdam^Adam/Cast/ReadVariableOp^Adam/Cast_2/ReadVariableOp^Adam/Cast_3/ReadVariableOp^Adam/ReadVariableOp^AssignAddVariableOp^AssignAddVariableOp_1^AssignAddVariableOp_2^AssignAddVariableOp_3^AssignAddVariableOp_4^AssignAddVariableOp_55^broadcast_weights_1/assert_broadcastable/AssertGuard0^conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp^div_no_nan/ReadVariableOp^div_no_nan/ReadVariableOp_1^div_no_nan_1/ReadVariableOp^div_no_nan_1/ReadVariableOp_1^div_no_nan_2/ReadVariableOp^div_no_nan_2/ReadVariableOp_1'^my_autoencoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
Adam/Adam/AssignAddVariableOpAdam/Adam/AssignAddVariableOp2H
"Adam/Adam/update/ResourceApplyAdam"Adam/Adam/update/ResourceApplyAdam2L
$Adam/Adam/update_1/ResourceApplyAdam$Adam/Adam/update_1/ResourceApplyAdam2N
%Adam/Adam/update_10/ResourceApplyAdam%Adam/Adam/update_10/ResourceApplyAdam2N
%Adam/Adam/update_11/ResourceApplyAdam%Adam/Adam/update_11/ResourceApplyAdam2N
%Adam/Adam/update_12/ResourceApplyAdam%Adam/Adam/update_12/ResourceApplyAdam2N
%Adam/Adam/update_13/ResourceApplyAdam%Adam/Adam/update_13/ResourceApplyAdam2N
%Adam/Adam/update_14/ResourceApplyAdam%Adam/Adam/update_14/ResourceApplyAdam2N
%Adam/Adam/update_15/ResourceApplyAdam%Adam/Adam/update_15/ResourceApplyAdam2L
$Adam/Adam/update_2/ResourceApplyAdam$Adam/Adam/update_2/ResourceApplyAdam2L
$Adam/Adam/update_3/ResourceApplyAdam$Adam/Adam/update_3/ResourceApplyAdam2L
$Adam/Adam/update_4/ResourceApplyAdam$Adam/Adam/update_4/ResourceApplyAdam2L
$Adam/Adam/update_5/ResourceApplyAdam$Adam/Adam/update_5/ResourceApplyAdam2L
$Adam/Adam/update_6/ResourceApplyAdam$Adam/Adam/update_6/ResourceApplyAdam2L
$Adam/Adam/update_7/ResourceApplyAdam$Adam/Adam/update_7/ResourceApplyAdam2L
$Adam/Adam/update_8/ResourceApplyAdam$Adam/Adam/update_8/ResourceApplyAdam2L
$Adam/Adam/update_9/ResourceApplyAdam$Adam/Adam/update_9/ResourceApplyAdam24
Adam/Cast/ReadVariableOpAdam/Cast/ReadVariableOp28
Adam/Cast_2/ReadVariableOpAdam/Cast_2/ReadVariableOp28
Adam/Cast_3/ReadVariableOpAdam/Cast_3/ReadVariableOp2*
Adam/ReadVariableOpAdam/ReadVariableOp2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_12.
AssignAddVariableOp_2AssignAddVariableOp_22.
AssignAddVariableOp_3AssignAddVariableOp_32.
AssignAddVariableOp_4AssignAddVariableOp_42.
AssignAddVariableOp_5AssignAddVariableOp_52l
4broadcast_weights_1/assert_broadcastable/AssertGuard4broadcast_weights_1/assert_broadcastable/AssertGuard2b
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp26
div_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp2:
div_no_nan/ReadVariableOp_1div_no_nan/ReadVariableOp_12:
div_no_nan_1/ReadVariableOpdiv_no_nan_1/ReadVariableOp2>
div_no_nan_1/ReadVariableOp_1div_no_nan_1/ReadVariableOp_12:
div_no_nan_2/ReadVariableOpdiv_no_nan_2/ReadVariableOp2>
div_no_nan_2/ReadVariableOp_1div_no_nan_2/ReadVariableOp_12P
&my_autoencoder/StatefulPartitionedCall&my_autoencoder/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_namedata/0:WS
/
_output_shapes
:?????????
 
_user_specified_namedata/1:-)
'
_class
loc:@my_autoencoder/49458:-)
'
_class
loc:@my_autoencoder/49458:-)
'
_class
loc:@my_autoencoder/49460:-)
'
_class
loc:@my_autoencoder/49460:-)
'
_class
loc:@my_autoencoder/49462:-)
'
_class
loc:@my_autoencoder/49462:-)
'
_class
loc:@my_autoencoder/49464:-)
'
_class
loc:@my_autoencoder/49464:- )
'
_class
loc:@my_autoencoder/49466:-!)
'
_class
loc:@my_autoencoder/49466:-")
'
_class
loc:@my_autoencoder/49468:-#)
'
_class
loc:@my_autoencoder/49468:-$)
'
_class
loc:@my_autoencoder/49470:-%)
'
_class
loc:@my_autoencoder/49470:-&)
'
_class
loc:@my_autoencoder/49472:-')
'
_class
loc:@my_autoencoder/49472:-()
'
_class
loc:@my_autoencoder/49474:-))
'
_class
loc:@my_autoencoder/49474:-*)
'
_class
loc:@my_autoencoder/49476:-+)
'
_class
loc:@my_autoencoder/49476:-,)
'
_class
loc:@my_autoencoder/49478:--)
'
_class
loc:@my_autoencoder/49478:-.)
'
_class
loc:@my_autoencoder/49480:-/)
'
_class
loc:@my_autoencoder/49480:-0)
'
_class
loc:@my_autoencoder/49482:-1)
'
_class
loc:@my_autoencoder/49482:-2)
'
_class
loc:@my_autoencoder/49484:-3)
'
_class
loc:@my_autoencoder/49484:-4)
'
_class
loc:@my_autoencoder/49486:-5)
'
_class
loc:@my_autoencoder/49486:-6)
'
_class
loc:@my_autoencoder/49488:-7)
'
_class
loc:@my_autoencoder/49488
?
?
E__inference_my_decoder_layer_call_and_return_conditional_losses_48483
x 
dense_1_48451:	
?
dense_1_48453:	?0
conv2d_transpose_48472:0$
conv2d_transpose_48474:2
conv2d_transpose_1_48477:&
conv2d_transpose_1_48479:
identity??(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCallxdense_1_48451dense_1_48453*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_48450?
reshape/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_48470?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_transpose_48472conv2d_transpose_48474*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_48382?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_48477conv2d_transpose_1_48479*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_48426?
IdentityIdentity3conv2d_transpose_1/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????
: : : : : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:J F
'
_output_shapes
:?????????


_user_specified_namex
?
?
.__inference_my_autoencoder_layer_call_fn_48718
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

	unknown_9:	
?

unknown_10:	?$

unknown_11:0

unknown_12:$

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_my_autoencoder_layer_call_and_return_conditional_losses_48683w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
.__inference_my_autoencoder_layer_call_fn_50352
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

	unknown_9:	
?

unknown_10:	?$

unknown_11:0

unknown_12:$

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_my_autoencoder_layer_call_and_return_conditional_losses_48811w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:?????????

_user_specified_namex
?
^
B__inference_reshape_layer_call_and_return_conditional_losses_51237

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :0?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????0`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
?broadcast_weights_1_assert_broadcastable_AssertGuard_true_49365?
~broadcast_weights_1_assert_broadcastable_assertguard_identity_broadcast_weights_1_assert_broadcastable_is_valid_shape_identity
D
@broadcast_weights_1_assert_broadcastable_assertguard_placeholderF
Bbroadcast_weights_1_assert_broadcastable_assertguard_placeholder_1F
Bbroadcast_weights_1_assert_broadcastable_assertguard_placeholder_2
C
?broadcast_weights_1_assert_broadcastable_assertguard_identity_1
W
9broadcast_weights_1/assert_broadcastable/AssertGuard/NoOpNoOp*
_output_shapes
 ?
=broadcast_weights_1/assert_broadcastable/AssertGuard/IdentityIdentity~broadcast_weights_1_assert_broadcastable_assertguard_identity_broadcast_weights_1_assert_broadcastable_is_valid_shape_identity:^broadcast_weights_1/assert_broadcastable/AssertGuard/NoOp*
T0
*
_output_shapes
: ?
?broadcast_weights_1/assert_broadcastable/AssertGuard/Identity_1IdentityFbroadcast_weights_1/assert_broadcastable/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "?
?broadcast_weights_1_assert_broadcastable_assertguard_identity_1Hbroadcast_weights_1/assert_broadcastable/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: ::: : 

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?/
?
__forward_call_49945
x&
my_cnn_49422:
my_cnn_49424:&
my_cnn_49426:
my_cnn_49428:&
my_cnn_49430:0
my_cnn_49432:0&
my_cnn_49434:00
my_cnn_49436:0
my_cnn_49438:	?

my_cnn_49440:
#
my_decoder_49443:	
?
my_decoder_49445:	?*
my_decoder_49447:0
my_decoder_49449:*
my_decoder_49451:
my_decoder_49453:
identity&
"my_decoder_statefulpartitionedcall(
$my_decoder_statefulpartitionedcall_0(
$my_decoder_statefulpartitionedcall_1(
$my_decoder_statefulpartitionedcall_2(
$my_decoder_statefulpartitionedcall_3(
$my_decoder_statefulpartitionedcall_4(
$my_decoder_statefulpartitionedcall_5"
my_cnn_statefulpartitionedcall$
 my_cnn_statefulpartitionedcall_0$
 my_cnn_statefulpartitionedcall_1$
 my_cnn_statefulpartitionedcall_2$
 my_cnn_statefulpartitionedcall_3$
 my_cnn_statefulpartitionedcall_4$
 my_cnn_statefulpartitionedcall_5$
 my_cnn_statefulpartitionedcall_6$
 my_cnn_statefulpartitionedcall_7$
 my_cnn_statefulpartitionedcall_8$
 my_cnn_statefulpartitionedcall_9%
!my_cnn_statefulpartitionedcall_10%
!my_cnn_statefulpartitionedcall_11%
!my_cnn_statefulpartitionedcall_12%
!my_cnn_statefulpartitionedcall_13%
!my_cnn_statefulpartitionedcall_14%
!my_cnn_statefulpartitionedcall_15%
!my_cnn_statefulpartitionedcall_16%
!my_cnn_statefulpartitionedcall_17%
!my_cnn_statefulpartitionedcall_18??my_cnn/StatefulPartitionedCall?"my_decoder/StatefulPartitionedCall?
my_cnn/StatefulPartitionedCallStatefulPartitionedCallxmy_cnn_49422my_cnn_49424my_cnn_49426my_cnn_49428my_cnn_49430my_cnn_49432my_cnn_49434my_cnn_49436my_cnn_49438my_cnn_49440*
Tin
2*!
Tout
2*
_collective_manager_ids
 *?
_output_shapes?
?:?????????
:?????????
:	?
:??????????:?????????0:?????????0:?????????0:?????????0:?????????0:00:?????????0:?????????:0:?????????:?????????:?????????:?????????::?????????:?????????:*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *
fR
__forward_call_49875?
"my_decoder/StatefulPartitionedCallStatefulPartitionedCall'my_cnn/StatefulPartitionedCall:output:0my_decoder_49443my_decoder_49445my_decoder_49447my_decoder_49449my_decoder_49451my_decoder_49453*
Tin
	2*
Tout

2*
_collective_manager_ids
 *?
_output_shapes?
?:?????????:?????????::?????????0:0:??????????:	
?:?????????
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *
fR
__forward_call_49649?
IdentityIdentity+my_decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp^my_cnn/StatefulPartitionedCall#^my_decoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"I
my_cnn_statefulpartitionedcall'my_cnn/StatefulPartitionedCall:output:1"K
 my_cnn_statefulpartitionedcall_0'my_cnn/StatefulPartitionedCall:output:2"K
 my_cnn_statefulpartitionedcall_1'my_cnn/StatefulPartitionedCall:output:3"M
!my_cnn_statefulpartitionedcall_10(my_cnn/StatefulPartitionedCall:output:12"M
!my_cnn_statefulpartitionedcall_11(my_cnn/StatefulPartitionedCall:output:13"M
!my_cnn_statefulpartitionedcall_12(my_cnn/StatefulPartitionedCall:output:14"M
!my_cnn_statefulpartitionedcall_13(my_cnn/StatefulPartitionedCall:output:15"M
!my_cnn_statefulpartitionedcall_14(my_cnn/StatefulPartitionedCall:output:16"M
!my_cnn_statefulpartitionedcall_15(my_cnn/StatefulPartitionedCall:output:17"M
!my_cnn_statefulpartitionedcall_16(my_cnn/StatefulPartitionedCall:output:18"M
!my_cnn_statefulpartitionedcall_17(my_cnn/StatefulPartitionedCall:output:19"M
!my_cnn_statefulpartitionedcall_18(my_cnn/StatefulPartitionedCall:output:20"K
 my_cnn_statefulpartitionedcall_2'my_cnn/StatefulPartitionedCall:output:4"K
 my_cnn_statefulpartitionedcall_3'my_cnn/StatefulPartitionedCall:output:5"K
 my_cnn_statefulpartitionedcall_4'my_cnn/StatefulPartitionedCall:output:6"K
 my_cnn_statefulpartitionedcall_5'my_cnn/StatefulPartitionedCall:output:7"K
 my_cnn_statefulpartitionedcall_6'my_cnn/StatefulPartitionedCall:output:8"K
 my_cnn_statefulpartitionedcall_7'my_cnn/StatefulPartitionedCall:output:9"L
 my_cnn_statefulpartitionedcall_8(my_cnn/StatefulPartitionedCall:output:10"L
 my_cnn_statefulpartitionedcall_9(my_cnn/StatefulPartitionedCall:output:11"Q
"my_decoder_statefulpartitionedcall+my_decoder/StatefulPartitionedCall:output:1"S
$my_decoder_statefulpartitionedcall_0+my_decoder/StatefulPartitionedCall:output:2"S
$my_decoder_statefulpartitionedcall_1+my_decoder/StatefulPartitionedCall:output:3"S
$my_decoder_statefulpartitionedcall_2+my_decoder/StatefulPartitionedCall:output:4"S
$my_decoder_statefulpartitionedcall_3+my_decoder/StatefulPartitionedCall:output:5"S
$my_decoder_statefulpartitionedcall_4+my_decoder/StatefulPartitionedCall:output:6"S
$my_decoder_statefulpartitionedcall_5+my_decoder/StatefulPartitionedCall:output:7*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : *C
backward_function_name)'__inference___backward_call_49614_499462@
my_cnn/StatefulPartitionedCallmy_cnn/StatefulPartitionedCall2H
"my_decoder/StatefulPartitionedCall"my_decoder/StatefulPartitionedCall:R N
/
_output_shapes
:?????????

_user_specified_namex
?&
?
'__inference___backward_call_49775_49842
placeholderi
egradients_max_pooling2d_maxpool_grad_maxpoolgrad_my_cnn_normalization_layer_1_statefulpartitionedcallJ
Fgradients_max_pooling2d_maxpool_grad_maxpoolgrad_max_pooling2d_maxpool|
xgradients_my_cnn_normalization_layer_1_statefulpartitionedcall_grad_my_cnn_normalization_layer_1_statefulpartitionedcall~
zgradients_my_cnn_normalization_layer_1_statefulpartitionedcall_grad_my_cnn_normalization_layer_1_statefulpartitionedcall_1~
zgradients_my_cnn_normalization_layer_1_statefulpartitionedcall_grad_my_cnn_normalization_layer_1_statefulpartitionedcall_2x
tgradients_my_cnn_normalization_layer_statefulpartitionedcall_grad_my_cnn_normalization_layer_statefulpartitionedcallz
vgradients_my_cnn_normalization_layer_statefulpartitionedcall_grad_my_cnn_normalization_layer_statefulpartitionedcall_1z
vgradients_my_cnn_normalization_layer_statefulpartitionedcall_grad_my_cnn_normalization_layer_statefulpartitionedcall_2
identity

identity_1

identity_2

identity_3

identity_4f
gradients/grad_ys_0Identityplaceholder*
T0*/
_output_shapes
:??????????
0gradients/max_pooling2d/MaxPool_grad/MaxPoolGradMaxPoolGradegradients_max_pooling2d_maxpool_grad_maxpoolgrad_my_cnn_normalization_layer_1_statefulpartitionedcallFgradients_max_pooling2d_maxpool_grad_maxpoolgrad_max_pooling2d_maxpoolgradients/grad_ys_0:output:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
?
Sgradients/my_cnn_normalization_layer_1/StatefulPartitionedCall_grad/PartitionedCallPartitionedCall9gradients/max_pooling2d/MaxPool_grad/MaxPoolGrad:output:0xgradients_my_cnn_normalization_layer_1_statefulpartitionedcall_grad_my_cnn_normalization_layer_1_statefulpartitionedcallzgradients_my_cnn_normalization_layer_1_statefulpartitionedcall_grad_my_cnn_normalization_layer_1_statefulpartitionedcall_1zgradients_my_cnn_normalization_layer_1_statefulpartitionedcall_grad_my_cnn_normalization_layer_1_statefulpartitionedcall_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *G
_output_shapes5
3:?????????::* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference___backward_call_49782_49799?
Qgradients/my_cnn_normalization_layer/StatefulPartitionedCall_grad/PartitionedCallPartitionedCall\gradients/my_cnn_normalization_layer_1/StatefulPartitionedCall_grad/PartitionedCall:output:0tgradients_my_cnn_normalization_layer_statefulpartitionedcall_grad_my_cnn_normalization_layer_statefulpartitionedcallvgradients_my_cnn_normalization_layer_statefulpartitionedcall_grad_my_cnn_normalization_layer_statefulpartitionedcall_1vgradients_my_cnn_normalization_layer_statefulpartitionedcall_grad_my_cnn_normalization_layer_statefulpartitionedcall_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *G
_output_shapes5
3:?????????::* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference___backward_call_49809_49826?
IdentityIdentityZgradients/my_cnn_normalization_layer/StatefulPartitionedCall_grad/PartitionedCall:output:0*
T0*/
_output_shapes
:??????????

Identity_1IdentityZgradients/my_cnn_normalization_layer/StatefulPartitionedCall_grad/PartitionedCall:output:1*
T0*&
_output_shapes
:?

Identity_2IdentityZgradients/my_cnn_normalization_layer/StatefulPartitionedCall_grad/PartitionedCall:output:2*
T0*
_output_shapes
:?

Identity_3Identity\gradients/my_cnn_normalization_layer_1/StatefulPartitionedCall_grad/PartitionedCall:output:1*
T0*&
_output_shapes
:?

Identity_4Identity\gradients/my_cnn_normalization_layer_1/StatefulPartitionedCall_grad/PartitionedCall:output:2*
T0*
_output_shapes
:"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????::?????????:?????????:*/
forward_function_name__forward_call_49841:5 1
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:,(
&
_output_shapes
::51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:,(
&
_output_shapes
:
?

?
@__inference_dense_layer_call_and_return_conditional_losses_47932

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
?
?
__inference_call_49457
x&
my_cnn_49422:
my_cnn_49424:&
my_cnn_49426:
my_cnn_49428:&
my_cnn_49430:0
my_cnn_49432:0&
my_cnn_49434:00
my_cnn_49436:0
my_cnn_49438:	?

my_cnn_49440:
#
my_decoder_49443:	
?
my_decoder_49445:	?*
my_decoder_49447:0
my_decoder_49449:*
my_decoder_49451:
my_decoder_49453:
identity??my_cnn/StatefulPartitionedCall?"my_decoder/StatefulPartitionedCall?
my_cnn/StatefulPartitionedCallStatefulPartitionedCallxmy_cnn_49422my_cnn_49424my_cnn_49426my_cnn_49428my_cnn_49430my_cnn_49432my_cnn_49434my_cnn_49436my_cnn_49438my_cnn_49440*
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
__inference_call_49076?
"my_decoder/StatefulPartitionedCallStatefulPartitionedCall'my_cnn/StatefulPartitionedCall:output:0my_decoder_49443my_decoder_49445my_decoder_49447my_decoder_49449my_decoder_49451my_decoder_49453*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_49155?
IdentityIdentity+my_decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp^my_cnn/StatefulPartitionedCall#^my_decoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 2@
my_cnn/StatefulPartitionedCallmy_cnn/StatefulPartitionedCall2H
"my_decoder/StatefulPartitionedCall"my_decoder/StatefulPartitionedCall:R N
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
?
?
Cbroadcast_weights_1_assert_broadcastable_is_valid_shape_false_50135G
Cbroadcast_weights_1_assert_broadcastable_is_valid_shape_placeholder
?
?broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_broadcast_weights_1_assert_broadcastable_values_rank?
?broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_broadcast_weights_1_assert_broadcastable_weights_rank?
?broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_broadcast_weights_1_assert_broadcastable_values_shape?
?broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_broadcast_weights_1_assert_broadcastable_weights_shapeD
@broadcast_weights_1_assert_broadcastable_is_valid_shape_identity
?
^broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqual?broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_broadcast_weights_1_assert_broadcastable_values_rank?broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_broadcast_weights_1_assert_broadcastable_weights_rank*
T0*
_output_shapes
: ?
Qbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shapeStatelessIfbbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:z:0?broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_broadcast_weights_1_assert_broadcastable_values_shape?broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_broadcast_weights_1_assert_broadcastable_weights_shapebbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *p
else_branchaR_
]broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_false_50144*
output_shapes
: *o
then_branch`R^
\broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_true_50143?
Zbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/IdentityIdentityZbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape:output:0*
T0
*
_output_shapes
: ?
@broadcast_weights_1/assert_broadcastable/is_valid_shape/IdentityIdentitycbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Identity:output:0*
T0
*
_output_shapes
: "?
@broadcast_weights_1_assert_broadcastable_is_valid_shape_identityIbroadcast_weights_1/assert_broadcastable/is_valid_shape/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:
?@
?
"__inference_internal_grad_fn_51718
result_grads_0
result_grads_1
result_grads_2
result_grads_3
result_grads_4
result_grads_5
result_grads_6
result_grads_7
result_grads_8
result_grads_9
result_grads_10
result_grads_11
result_grads_12
result_grads_13
result_grads_14
result_grads_15
result_grads_16
result_grads_17
result_grads_18
result_grads_19
result_grads_20
result_grads_21
result_grads_22
result_grads_23
result_grads_24
result_grads_25
result_grads_26
result_grads_27
result_grads_28
result_grads_29
result_grads_30
result_grads_31
identity_16
identity_17
identity_18
identity_19
identity_20
identity_21
identity_22
identity_23
identity_24
identity_25
identity_26
identity_27
identity_28
identity_29
identity_30
identity_31U
IdentityIdentityresult_grads_0*
T0*&
_output_shapes
:K

Identity_1Identityresult_grads_1*
T0*
_output_shapes
:W

Identity_2Identityresult_grads_2*
T0*&
_output_shapes
:K

Identity_3Identityresult_grads_3*
T0*
_output_shapes
:W

Identity_4Identityresult_grads_4*
T0*&
_output_shapes
:0K

Identity_5Identityresult_grads_5*
T0*
_output_shapes
:0W

Identity_6Identityresult_grads_6*
T0*&
_output_shapes
:00K

Identity_7Identityresult_grads_7*
T0*
_output_shapes
:0P

Identity_8Identityresult_grads_8*
T0*
_output_shapes
:	?
K

Identity_9Identityresult_grads_9*
T0*
_output_shapes
:
R
Identity_10Identityresult_grads_10*
T0*
_output_shapes
:	
?N
Identity_11Identityresult_grads_11*
T0*
_output_shapes	
:?Y
Identity_12Identityresult_grads_12*
T0*&
_output_shapes
:0M
Identity_13Identityresult_grads_13*
T0*
_output_shapes
:Y
Identity_14Identityresult_grads_14*
T0*&
_output_shapes
:M
Identity_15Identityresult_grads_15*
T0*
_output_shapes
:?
	IdentityN	IdentityNresult_grads_0result_grads_1result_grads_2result_grads_3result_grads_4result_grads_5result_grads_6result_grads_7result_grads_8result_grads_9result_grads_10result_grads_11result_grads_12result_grads_13result_grads_14result_grads_15result_grads_0result_grads_1result_grads_2result_grads_3result_grads_4result_grads_5result_grads_6result_grads_7result_grads_8result_grads_9result_grads_10result_grads_11result_grads_12result_grads_13result_grads_14result_grads_15*)
T$
"2 *+
_gradient_op_typeCustomGradient-51653*?
_output_shapes?
?:::::0:0:00:0:	?
:
:	
?:?:0::::::::0:0:00:0:	?
:
:	
?:?:0:::\
Identity_16IdentityIdentityN:output:0*
T0*&
_output_shapes
:P
Identity_17IdentityIdentityN:output:1*
T0*
_output_shapes
:\
Identity_18IdentityIdentityN:output:2*
T0*&
_output_shapes
:P
Identity_19IdentityIdentityN:output:3*
T0*
_output_shapes
:\
Identity_20IdentityIdentityN:output:4*
T0*&
_output_shapes
:0P
Identity_21IdentityIdentityN:output:5*
T0*
_output_shapes
:0\
Identity_22IdentityIdentityN:output:6*
T0*&
_output_shapes
:00P
Identity_23IdentityIdentityN:output:7*
T0*
_output_shapes
:0U
Identity_24IdentityIdentityN:output:8*
T0*
_output_shapes
:	?
P
Identity_25IdentityIdentityN:output:9*
T0*
_output_shapes
:
V
Identity_26IdentityIdentityN:output:10*
T0*
_output_shapes
:	
?R
Identity_27IdentityIdentityN:output:11*
T0*
_output_shapes	
:?]
Identity_28IdentityIdentityN:output:12*
T0*&
_output_shapes
:0Q
Identity_29IdentityIdentityN:output:13*
T0*
_output_shapes
:]
Identity_30IdentityIdentityN:output:14*
T0*&
_output_shapes
:Q
Identity_31IdentityIdentityN:output:15*
T0*
_output_shapes
:"#
identity_16Identity_16:output:0"#
identity_17Identity_17:output:0"#
identity_18Identity_18:output:0"#
identity_19Identity_19:output:0"#
identity_20Identity_20:output:0"#
identity_21Identity_21:output:0"#
identity_22Identity_22:output:0"#
identity_23Identity_23:output:0"#
identity_24Identity_24:output:0"#
identity_25Identity_25:output:0"#
identity_26Identity_26:output:0"#
identity_27Identity_27:output:0"#
identity_28Identity_28:output:0"#
identity_29Identity_29:output:0"#
identity_30Identity_30:output:0"#
identity_31Identity_31:output:0*?
_input_shapes?
?:::::0:0:00:0:	?
:
:	
?:?:0::::::::0:0:00:0:	?
:
:	
?:?:0::::V R
&
_output_shapes
:
(
_user_specified_nameresult_grads_0:JF

_output_shapes
:
(
_user_specified_nameresult_grads_1:VR
&
_output_shapes
:
(
_user_specified_nameresult_grads_2:JF

_output_shapes
:
(
_user_specified_nameresult_grads_3:VR
&
_output_shapes
:0
(
_user_specified_nameresult_grads_4:JF

_output_shapes
:0
(
_user_specified_nameresult_grads_5:VR
&
_output_shapes
:00
(
_user_specified_nameresult_grads_6:JF

_output_shapes
:0
(
_user_specified_nameresult_grads_7:OK

_output_shapes
:	?

(
_user_specified_nameresult_grads_8:J	F

_output_shapes
:

(
_user_specified_nameresult_grads_9:P
L

_output_shapes
:	
?
)
_user_specified_nameresult_grads_10:LH

_output_shapes	
:?
)
_user_specified_nameresult_grads_11:WS
&
_output_shapes
:0
)
_user_specified_nameresult_grads_12:KG

_output_shapes
:
)
_user_specified_nameresult_grads_13:WS
&
_output_shapes
:
)
_user_specified_nameresult_grads_14:KG

_output_shapes
:
)
_user_specified_nameresult_grads_15:WS
&
_output_shapes
:
)
_user_specified_nameresult_grads_16:KG

_output_shapes
:
)
_user_specified_nameresult_grads_17:WS
&
_output_shapes
:
)
_user_specified_nameresult_grads_18:KG

_output_shapes
:
)
_user_specified_nameresult_grads_19:WS
&
_output_shapes
:0
)
_user_specified_nameresult_grads_20:KG

_output_shapes
:0
)
_user_specified_nameresult_grads_21:WS
&
_output_shapes
:00
)
_user_specified_nameresult_grads_22:KG

_output_shapes
:0
)
_user_specified_nameresult_grads_23:PL

_output_shapes
:	?

)
_user_specified_nameresult_grads_24:KG

_output_shapes
:

)
_user_specified_nameresult_grads_25:PL

_output_shapes
:	
?
)
_user_specified_nameresult_grads_26:LH

_output_shapes	
:?
)
_user_specified_nameresult_grads_27:WS
&
_output_shapes
:0
)
_user_specified_nameresult_grads_28:KG

_output_shapes
:
)
_user_specified_nameresult_grads_29:WS
&
_output_shapes
:
)
_user_specified_nameresult_grads_30:KG

_output_shapes
:
)
_user_specified_nameresult_grads_31
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
?
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_51073	
input:
 my_cnn_normalization_layer_51053:.
 my_cnn_normalization_layer_51055:<
"my_cnn_normalization_layer_1_51058:0
"my_cnn_normalization_layer_1_51060:
identity??/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp?2my_cnn_normalization_layer/StatefulPartitionedCall?4my_cnn_normalization_layer_1/StatefulPartitionedCall?
2my_cnn_normalization_layer/StatefulPartitionedCallStatefulPartitionedCallinput my_cnn_normalization_layer_51053 my_cnn_normalization_layer_51055*
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
4my_cnn_normalization_layer_1/StatefulPartitionedCallStatefulPartitionedCall;my_cnn_normalization_layer/StatefulPartitionedCall:output:0"my_cnn_normalization_layer_1_51058"my_cnn_normalization_layer_1_51060*
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
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp my_cnn_normalization_layer_51053*&
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
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp"my_cnn_normalization_layer_1_51058*&
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
Bbroadcast_weights_1_assert_broadcastable_is_valid_shape_true_50134w
sbroadcast_weights_1_assert_broadcastable_is_valid_shape_identity_broadcast_weights_1_assert_broadcastable_is_scalar
G
Cbroadcast_weights_1_assert_broadcastable_is_valid_shape_placeholderI
Ebroadcast_weights_1_assert_broadcastable_is_valid_shape_placeholder_1I
Ebroadcast_weights_1_assert_broadcastable_is_valid_shape_placeholder_2I
Ebroadcast_weights_1_assert_broadcastable_is_valid_shape_placeholder_3D
@broadcast_weights_1_assert_broadcastable_is_valid_shape_identity
?
@broadcast_weights_1/assert_broadcastable/is_valid_shape/IdentityIdentitysbroadcast_weights_1_assert_broadcastable_is_valid_shape_identity_broadcast_weights_1_assert_broadcastable_is_scalar*
T0
*
_output_shapes
: "?
@broadcast_weights_1_assert_broadcastable_is_valid_shape_identityIbroadcast_weights_1/assert_broadcastable/is_valid_shape/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:
?
?
,__inference_my_cnn_block_layer_call_fn_51050	
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
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_48127w
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
?
?
'__inference___backward_call_49693_49710
placeholder?
;gradients_activation_3_relu_grad_relugrad_activation_3_relu+
'gradients_conv2d_3_conv2d_grad_shapen_xH
Dgradients_conv2d_3_conv2d_grad_shapen_conv2d_3_conv2d_readvariableop
identity

identity_1

identity_2f
gradients/grad_ys_0Identityplaceholder*
T0*/
_output_shapes
:?????????0?
)gradients/activation_3/Relu_grad/ReluGradReluGradgradients/grad_ys_0:output:0;gradients_activation_3_relu_grad_relugrad_activation_3_relu*
T0*/
_output_shapes
:?????????0?
+gradients/conv2d_3/BiasAdd_grad/BiasAddGradBiasAddGrad5gradients/activation_3/Relu_grad/ReluGrad:backprops:0*
T0*
_output_shapes
:0?
%gradients/conv2d_3/Conv2D_grad/ShapeNShapeN'gradients_conv2d_3_conv2d_grad_shapen_xDgradients_conv2d_3_conv2d_grad_shapen_conv2d_3_conv2d_readvariableop*
N*
T0* 
_output_shapes
::?
2gradients/conv2d_3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput.gradients/conv2d_3/Conv2D_grad/ShapeN:output:0Dgradients_conv2d_3_conv2d_grad_shapen_conv2d_3_conv2d_readvariableop5gradients/activation_3/Relu_grad/ReluGrad:backprops:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
?
3gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter'gradients_conv2d_3_conv2d_grad_shapen_x.gradients/conv2d_3/Conv2D_grad/ShapeN:output:15gradients/activation_3/Relu_grad/ReluGrad:backprops:0*
T0*&
_output_shapes
:00*
paddingSAME*
strides
?
IdentityIdentity;gradients/conv2d_3/Conv2D_grad/Conv2DBackpropInput:output:0*
T0*/
_output_shapes
:?????????0?

Identity_1Identity<gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilter:output:0*
T0*&
_output_shapes
:00q

Identity_2Identity4gradients/conv2d_3/BiasAdd_grad/BiasAddGrad:output:0*
T0*
_output_shapes
:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:?????????0:?????????0:?????????0:00*/
forward_function_name__forward_call_49709:5 1
/
_output_shapes
:?????????0:51
/
_output_shapes
:?????????0:51
/
_output_shapes
:?????????0:,(
&
_output_shapes
:00
?
?
@broadcast_weights_1_assert_broadcastable_AssertGuard_false_50189?
|broadcast_weights_1_assert_broadcastable_assertguard_assert_broadcast_weights_1_assert_broadcastable_is_valid_shape_identity
v
rbroadcast_weights_1_assert_broadcastable_assertguard_assert_broadcast_weights_1_assert_broadcastable_weights_shapeu
qbroadcast_weights_1_assert_broadcastable_assertguard_assert_broadcast_weights_1_assert_broadcastable_values_shaper
nbroadcast_weights_1_assert_broadcastable_assertguard_assert_broadcast_weights_1_assert_broadcastable_is_scalar
C
?broadcast_weights_1_assert_broadcastable_assertguard_identity_1
??;broadcast_weights_1/assert_broadcastable/AssertGuard/Assert?
Bbroadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*8
value/B- B'weights can not be broadcast to values.?
Bbroadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*
valueB Bweights.shape=?
Bbroadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*9
value0B. B(my_autoencoder/StatefulPartitionedCall:0?
Bbroadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*
valueB Bvalues.shape=?
Bbroadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5Const*
_output_shapes
: *
dtype0*
valueB Bdata/1:0?
Bbroadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7Const*
_output_shapes
: *
dtype0*
valueB B
is_scalar=?
;broadcast_weights_1/assert_broadcastable/AssertGuard/AssertAssert|broadcast_weights_1_assert_broadcastable_assertguard_assert_broadcast_weights_1_assert_broadcastable_is_valid_shape_identityKbroadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0:output:0Kbroadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1:output:0Kbroadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2:output:0rbroadcast_weights_1_assert_broadcastable_assertguard_assert_broadcast_weights_1_assert_broadcastable_weights_shapeKbroadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4:output:0Kbroadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5:output:0qbroadcast_weights_1_assert_broadcastable_assertguard_assert_broadcast_weights_1_assert_broadcastable_values_shapeKbroadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7:output:0nbroadcast_weights_1_assert_broadcastable_assertguard_assert_broadcast_weights_1_assert_broadcastable_is_scalar*
T
2	
*
_output_shapes
 ?
=broadcast_weights_1/assert_broadcastable/AssertGuard/IdentityIdentity|broadcast_weights_1_assert_broadcastable_assertguard_assert_broadcast_weights_1_assert_broadcastable_is_valid_shape_identity<^broadcast_weights_1/assert_broadcastable/AssertGuard/Assert*
T0
*
_output_shapes
: ?
?broadcast_weights_1/assert_broadcastable/AssertGuard/Identity_1IdentityFbroadcast_weights_1/assert_broadcastable/AssertGuard/Identity:output:0:^broadcast_weights_1/assert_broadcastable/AssertGuard/NoOp*
T0
*
_output_shapes
: ?
9broadcast_weights_1/assert_broadcastable/AssertGuard/NoOpNoOp<^broadcast_weights_1/assert_broadcastable/AssertGuard/Assert*"
_acd_function_control_output(*
_output_shapes
 "?
?broadcast_weights_1_assert_broadcastable_assertguard_identity_1Hbroadcast_weights_1/assert_broadcastable/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: ::: 2z
;broadcast_weights_1/assert_broadcastable/AssertGuard/Assert;broadcast_weights_1/assert_broadcastable/AssertGuard/Assert: 

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?/
?
A__inference_my_cnn_layer_call_and_return_conditional_losses_48301
input_1,
my_cnn_block_48260: 
my_cnn_block_48262:,
my_cnn_block_48264: 
my_cnn_block_48266:.
my_cnn_block_1_48269:0"
my_cnn_block_1_48271:0.
my_cnn_block_1_48273:00"
my_cnn_block_1_48275:0
dense_48279:	?

dense_48281:

identity??/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp?dense/StatefulPartitionedCall?$my_cnn_block/StatefulPartitionedCall?&my_cnn_block_1/StatefulPartitionedCall?
$my_cnn_block/StatefulPartitionedCallStatefulPartitionedCallinput_1my_cnn_block_48260my_cnn_block_48262my_cnn_block_48264my_cnn_block_48266*
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
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_47870?
&my_cnn_block_1/StatefulPartitionedCallStatefulPartitionedCall-my_cnn_block/StatefulPartitionedCall:output:0my_cnn_block_1_48269my_cnn_block_1_48271my_cnn_block_1_48273my_cnn_block_1_48275*
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
I__inference_my_cnn_block_1_layer_call_and_return_conditional_losses_47903?
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
B__inference_flatten_layer_call_and_return_conditional_losses_47919?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_48279dense_48281*
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
@__inference_dense_layer_call_and_return_conditional_losses_47932?
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_48260*&
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
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_48264*&
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
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_1_48269*&
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
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_1_48273*&
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
?W
?
'__inference___backward_call_49614_49946
placeholderX
Tgradients_my_decoder_statefulpartitionedcall_grad_my_decoder_statefulpartitionedcallZ
Vgradients_my_decoder_statefulpartitionedcall_grad_my_decoder_statefulpartitionedcall_1Z
Vgradients_my_decoder_statefulpartitionedcall_grad_my_decoder_statefulpartitionedcall_2Z
Vgradients_my_decoder_statefulpartitionedcall_grad_my_decoder_statefulpartitionedcall_3Z
Vgradients_my_decoder_statefulpartitionedcall_grad_my_decoder_statefulpartitionedcall_4Z
Vgradients_my_decoder_statefulpartitionedcall_grad_my_decoder_statefulpartitionedcall_5Z
Vgradients_my_decoder_statefulpartitionedcall_grad_my_decoder_statefulpartitionedcall_6P
Lgradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcallR
Ngradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_1R
Ngradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_2R
Ngradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_3R
Ngradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_4R
Ngradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_5R
Ngradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_6R
Ngradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_7R
Ngradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_8R
Ngradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_9S
Ogradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_10S
Ogradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_11S
Ogradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_12S
Ogradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_13S
Ogradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_14S
Ogradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_15S
Ogradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_16S
Ogradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_17S
Ogradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_18S
Ogradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_19
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11
identity_12
identity_13
identity_14
identity_15
identity_16f
gradients/grad_ys_0Identityplaceholder*
T0*/
_output_shapes
:??????????
Agradients/my_decoder/StatefulPartitionedCall_grad/PartitionedCallPartitionedCallgradients/grad_ys_0:output:0Tgradients_my_decoder_statefulpartitionedcall_grad_my_decoder_statefulpartitionedcallVgradients_my_decoder_statefulpartitionedcall_grad_my_decoder_statefulpartitionedcall_1Vgradients_my_decoder_statefulpartitionedcall_grad_my_decoder_statefulpartitionedcall_2Vgradients_my_decoder_statefulpartitionedcall_grad_my_decoder_statefulpartitionedcall_3Vgradients_my_decoder_statefulpartitionedcall_grad_my_decoder_statefulpartitionedcall_4Vgradients_my_decoder_statefulpartitionedcall_grad_my_decoder_statefulpartitionedcall_5Vgradients_my_decoder_statefulpartitionedcall_grad_my_decoder_statefulpartitionedcall_6*
Tin

2*
Tout
	2*
_collective_manager_ids
 *i
_output_shapesW
U:?????????
:	
?:?:0:::* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference___backward_call_49618_49650?
=gradients/my_cnn/StatefulPartitionedCall_grad/PartitionedCallPartitionedCallJgradients/my_decoder/StatefulPartitionedCall_grad/PartitionedCall:output:0Lgradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcallNgradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_1Ngradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_2Ngradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_3Ngradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_4Ngradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_5Ngradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_6Ngradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_7Ngradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_8Ngradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_9Ogradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_10Ogradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_11Ogradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_12Ogradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_13Ogradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_14Ogradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_15Ogradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_16Ogradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_17Ogradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_18Ogradients_my_cnn_statefulpartitionedcall_grad_my_cnn_statefulpartitionedcall_19* 
Tin
2*
Tout
2*
_collective_manager_ids
 *?
_output_shapes?
?:?????????:::::0:0:00:0:	?
:
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference___backward_call_49672_49876?
IdentityIdentityFgradients/my_cnn/StatefulPartitionedCall_grad/PartitionedCall:output:0*
T0*/
_output_shapes
:??????????

Identity_1IdentityFgradients/my_cnn/StatefulPartitionedCall_grad/PartitionedCall:output:1*
T0*&
_output_shapes
:?

Identity_2IdentityFgradients/my_cnn/StatefulPartitionedCall_grad/PartitionedCall:output:2*
T0*
_output_shapes
:?

Identity_3IdentityFgradients/my_cnn/StatefulPartitionedCall_grad/PartitionedCall:output:3*
T0*&
_output_shapes
:?

Identity_4IdentityFgradients/my_cnn/StatefulPartitionedCall_grad/PartitionedCall:output:4*
T0*
_output_shapes
:?

Identity_5IdentityFgradients/my_cnn/StatefulPartitionedCall_grad/PartitionedCall:output:5*
T0*&
_output_shapes
:0?

Identity_6IdentityFgradients/my_cnn/StatefulPartitionedCall_grad/PartitionedCall:output:6*
T0*
_output_shapes
:0?

Identity_7IdentityFgradients/my_cnn/StatefulPartitionedCall_grad/PartitionedCall:output:7*
T0*&
_output_shapes
:00?

Identity_8IdentityFgradients/my_cnn/StatefulPartitionedCall_grad/PartitionedCall:output:8*
T0*
_output_shapes
:0?

Identity_9IdentityFgradients/my_cnn/StatefulPartitionedCall_grad/PartitionedCall:output:9*
T0*
_output_shapes
:	?
?
Identity_10IdentityGgradients/my_cnn/StatefulPartitionedCall_grad/PartitionedCall:output:10*
T0*
_output_shapes
:
?
Identity_11IdentityJgradients/my_decoder/StatefulPartitionedCall_grad/PartitionedCall:output:1*
T0*
_output_shapes
:	
??
Identity_12IdentityJgradients/my_decoder/StatefulPartitionedCall_grad/PartitionedCall:output:2*
T0*
_output_shapes	
:??
Identity_13IdentityJgradients/my_decoder/StatefulPartitionedCall_grad/PartitionedCall:output:3*
T0*&
_output_shapes
:0?
Identity_14IdentityJgradients/my_decoder/StatefulPartitionedCall_grad/PartitionedCall:output:4*
T0*
_output_shapes
:?
Identity_15IdentityJgradients/my_decoder/StatefulPartitionedCall_grad/PartitionedCall:output:5*
T0*&
_output_shapes
:?
Identity_16IdentityJgradients/my_decoder/StatefulPartitionedCall_grad/PartitionedCall:output:6*
T0*
_output_shapes
:"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"#
identity_12Identity_12:output:0"#
identity_13Identity_13:output:0"#
identity_14Identity_14:output:0"#
identity_15Identity_15:output:0"#
identity_16Identity_16:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????::?????????0:0:??????????:	
?:?????????
:?????????
:	?
:??????????:?????????0:?????????0:?????????0:?????????0:?????????0:00:?????????0:?????????:0:?????????:?????????:?????????:?????????::?????????:?????????:*/
forward_function_name__forward_call_49945:5 1
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:,(
&
_output_shapes
::51
/
_output_shapes
:?????????0:,(
&
_output_shapes
:0:.*
(
_output_shapes
:??????????:%!

_output_shapes
:	
?:-)
'
_output_shapes
:?????????
:-)
'
_output_shapes
:?????????
:%	!

_output_shapes
:	?
:.
*
(
_output_shapes
:??????????:51
/
_output_shapes
:?????????0:51
/
_output_shapes
:?????????0:51
/
_output_shapes
:?????????0:51
/
_output_shapes
:?????????0:51
/
_output_shapes
:?????????0:,(
&
_output_shapes
:00:51
/
_output_shapes
:?????????0:51
/
_output_shapes
:?????????:,(
&
_output_shapes
:0:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:,(
&
_output_shapes
::51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:,(
&
_output_shapes
:
?	
?
Bbroadcast_weights_1_assert_broadcastable_is_valid_shape_true_49311w
sbroadcast_weights_1_assert_broadcastable_is_valid_shape_identity_broadcast_weights_1_assert_broadcastable_is_scalar
G
Cbroadcast_weights_1_assert_broadcastable_is_valid_shape_placeholderI
Ebroadcast_weights_1_assert_broadcastable_is_valid_shape_placeholder_1I
Ebroadcast_weights_1_assert_broadcastable_is_valid_shape_placeholder_2I
Ebroadcast_weights_1_assert_broadcastable_is_valid_shape_placeholder_3D
@broadcast_weights_1_assert_broadcastable_is_valid_shape_identity
?
@broadcast_weights_1/assert_broadcastable/is_valid_shape/IdentityIdentitysbroadcast_weights_1_assert_broadcastable_is_valid_shape_identity_broadcast_weights_1_assert_broadcastable_is_scalar*
T0
*
_output_shapes
: "?
@broadcast_weights_1_assert_broadcastable_is_valid_shape_identityIbroadcast_weights_1/assert_broadcastable/is_valid_shape/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:
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
?
?
__forward_call_49841	
input:
 my_cnn_normalization_layer_49020:.
 my_cnn_normalization_layer_49022:<
"my_cnn_normalization_layer_1_49025:0
"my_cnn_normalization_layer_1_49027:
identity8
4my_cnn_normalization_layer_1_statefulpartitionedcall
max_pooling2d_maxpool:
6my_cnn_normalization_layer_1_statefulpartitionedcall_0:
6my_cnn_normalization_layer_1_statefulpartitionedcall_1:
6my_cnn_normalization_layer_1_statefulpartitionedcall_26
2my_cnn_normalization_layer_statefulpartitionedcall8
4my_cnn_normalization_layer_statefulpartitionedcall_08
4my_cnn_normalization_layer_statefulpartitionedcall_1??2my_cnn_normalization_layer/StatefulPartitionedCall?4my_cnn_normalization_layer_1/StatefulPartitionedCall?
2my_cnn_normalization_layer/StatefulPartitionedCallStatefulPartitionedCallinput my_cnn_normalization_layer_49020 my_cnn_normalization_layer_49022*
Tin
2*
Tout
2*
_collective_manager_ids
 *w
_output_shapese
c:?????????:?????????:?????????:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__forward_call_49825?
4my_cnn_normalization_layer_1/StatefulPartitionedCallStatefulPartitionedCall;my_cnn_normalization_layer/StatefulPartitionedCall:output:0"my_cnn_normalization_layer_1_49025"my_cnn_normalization_layer_1_49027*
Tin
2*
Tout
2*
_collective_manager_ids
 *w
_output_shapese
c:?????????:?????????:?????????:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__forward_call_49798?
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
identityIdentity:output:0"7
max_pooling2d_maxpoolmax_pooling2d/MaxPool:output:0"u
4my_cnn_normalization_layer_1_statefulpartitionedcall=my_cnn_normalization_layer_1/StatefulPartitionedCall:output:0"w
6my_cnn_normalization_layer_1_statefulpartitionedcall_0=my_cnn_normalization_layer_1/StatefulPartitionedCall:output:1"w
6my_cnn_normalization_layer_1_statefulpartitionedcall_1=my_cnn_normalization_layer_1/StatefulPartitionedCall:output:2"w
6my_cnn_normalization_layer_1_statefulpartitionedcall_2=my_cnn_normalization_layer_1/StatefulPartitionedCall:output:3"q
2my_cnn_normalization_layer_statefulpartitionedcall;my_cnn_normalization_layer/StatefulPartitionedCall:output:1"s
4my_cnn_normalization_layer_statefulpartitionedcall_0;my_cnn_normalization_layer/StatefulPartitionedCall:output:2"s
4my_cnn_normalization_layer_statefulpartitionedcall_1;my_cnn_normalization_layer/StatefulPartitionedCall:output:3*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : *C
backward_function_name)'__inference___backward_call_49775_498422h
2my_cnn_normalization_layer/StatefulPartitionedCall2my_cnn_normalization_layer/StatefulPartitionedCall2l
4my_cnn_normalization_layer_1/StatefulPartitionedCall4my_cnn_normalization_layer_1/StatefulPartitionedCall:V R
/
_output_shapes
:?????????

_user_specified_nameinput
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
?	
?
*__inference_my_decoder_layer_call_fn_48498
input_1
unknown:	
?
	unknown_0:	?#
	unknown_1:0
	unknown_2:#
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_my_decoder_layer_call_and_return_conditional_losses_48483w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????
: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????

!
_user_specified_name	input_1
?

?
]broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_false_49321a
]broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_placeholderc
_broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_placeholder_1?
?broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity_broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank
^
Zbroadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity
?
Zbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/IdentityIdentity?broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity_broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank*
T0
*
_output_shapes
: "?
Zbroadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identitycbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::: :  

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
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
?	
?
__inference_loss_fn_1_51479T
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
__inference_loss_fn_2_51488T
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
?)
?
I__inference_my_autoencoder_layer_call_and_return_conditional_losses_48991
input_1&
my_cnn_48940:
my_cnn_48942:&
my_cnn_48944:
my_cnn_48946:&
my_cnn_48948:0
my_cnn_48950:0&
my_cnn_48952:00
my_cnn_48954:0
my_cnn_48956:	?

my_cnn_48958:
#
my_decoder_48961:	
?
my_decoder_48963:	?*
my_decoder_48965:0
my_decoder_48967:*
my_decoder_48969:
my_decoder_48971:
identity??/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp?my_cnn/StatefulPartitionedCall?"my_decoder/StatefulPartitionedCall?
my_cnn/StatefulPartitionedCallStatefulPartitionedCallinput_1my_cnn_48940my_cnn_48942my_cnn_48944my_cnn_48946my_cnn_48948my_cnn_48950my_cnn_48952my_cnn_48954my_cnn_48956my_cnn_48958*
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
A__inference_my_cnn_layer_call_and_return_conditional_losses_48209?
"my_decoder/StatefulPartitionedCallStatefulPartitionedCall'my_cnn/StatefulPartitionedCall:output:0my_decoder_48961my_decoder_48963my_decoder_48965my_decoder_48967my_decoder_48969my_decoder_48971*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_my_decoder_layer_call_and_return_conditional_losses_48553?
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_48940*&
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
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_48944*&
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
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_48948*&
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
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_48952*&
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
: ?
IdentityIdentity+my_decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp0^conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp^my_cnn/StatefulPartitionedCall#^my_decoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 2b
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp2@
my_cnn/StatefulPartitionedCallmy_cnn/StatefulPartitionedCall2H
"my_decoder/StatefulPartitionedCall"my_decoder/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
C
'__inference_flatten_layer_call_fn_51173

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
B__inference_flatten_layer_call_and_return_conditional_losses_47919a
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
?
?
E__inference_my_decoder_layer_call_and_return_conditional_losses_48553
x 
dense_1_48536:	
?
dense_1_48538:	?0
conv2d_transpose_48542:0$
conv2d_transpose_48544:2
conv2d_transpose_1_48547:&
conv2d_transpose_1_48549:
identity??(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCallxdense_1_48536dense_1_48538*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_48450?
reshape/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_48470?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_transpose_48542conv2d_transpose_48544*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_48382?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_48547conv2d_transpose_1_48549*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_48426?
IdentityIdentity3conv2d_transpose_1/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????
: : : : : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:J F
'
_output_shapes
:?????????

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
?
?
'__inference_dense_1_layer_call_fn_51208

inputs
unknown:	
?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_48450p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
*__inference_my_decoder_layer_call_fn_50891
x
unknown:	
?
	unknown_0:	?#
	unknown_1:0
	unknown_2:#
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_my_decoder_layer_call_and_return_conditional_losses_48483w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????
: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:?????????


_user_specified_namex
?

?
&__inference_my_cnn_layer_call_fn_48257
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
A__inference_my_cnn_layer_call_and_return_conditional_losses_48209o
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
?
?
'__inference___backward_call_49782_49799
placeholder?
;gradients_activation_1_relu_grad_relugrad_activation_1_relu+
'gradients_conv2d_1_conv2d_grad_shapen_xH
Dgradients_conv2d_1_conv2d_grad_shapen_conv2d_1_conv2d_readvariableop
identity

identity_1

identity_2f
gradients/grad_ys_0Identityplaceholder*
T0*/
_output_shapes
:??????????
)gradients/activation_1/Relu_grad/ReluGradReluGradgradients/grad_ys_0:output:0;gradients_activation_1_relu_grad_relugrad_activation_1_relu*
T0*/
_output_shapes
:??????????
+gradients/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGrad5gradients/activation_1/Relu_grad/ReluGrad:backprops:0*
T0*
_output_shapes
:?
%gradients/conv2d_1/Conv2D_grad/ShapeNShapeN'gradients_conv2d_1_conv2d_grad_shapen_xDgradients_conv2d_1_conv2d_grad_shapen_conv2d_1_conv2d_readvariableop*
N*
T0* 
_output_shapes
::?
2gradients/conv2d_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput.gradients/conv2d_1/Conv2D_grad/ShapeN:output:0Dgradients_conv2d_1_conv2d_grad_shapen_conv2d_1_conv2d_readvariableop5gradients/activation_1/Relu_grad/ReluGrad:backprops:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
3gradients/conv2d_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter'gradients_conv2d_1_conv2d_grad_shapen_x.gradients/conv2d_1/Conv2D_grad/ShapeN:output:15gradients/activation_1/Relu_grad/ReluGrad:backprops:0*
T0*&
_output_shapes
:*
paddingSAME*
strides
?
IdentityIdentity;gradients/conv2d_1/Conv2D_grad/Conv2DBackpropInput:output:0*
T0*/
_output_shapes
:??????????

Identity_1Identity<gradients/conv2d_1/Conv2D_grad/Conv2DBackpropFilter:output:0*
T0*&
_output_shapes
:q

Identity_2Identity4gradients/conv2d_1/BiasAdd_grad/BiasAddGrad:output:0*
T0*
_output_shapes
:"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:?????????:?????????:?????????:*/
forward_function_name__forward_call_49798:5 1
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:,(
&
_output_shapes
:
?
?
__inference_call_47781
x&
my_cnn_47688:
my_cnn_47690:&
my_cnn_47692:
my_cnn_47694:&
my_cnn_47696:0
my_cnn_47698:0&
my_cnn_47700:00
my_cnn_47702:0
my_cnn_47704:	?

my_cnn_47706:
#
my_decoder_47767:	
?
my_decoder_47769:	?*
my_decoder_47771:0
my_decoder_47773:*
my_decoder_47775:
my_decoder_47777:
identity??my_cnn/StatefulPartitionedCall?"my_decoder/StatefulPartitionedCall?
my_cnn/StatefulPartitionedCallStatefulPartitionedCallxmy_cnn_47688my_cnn_47690my_cnn_47692my_cnn_47694my_cnn_47696my_cnn_47698my_cnn_47700my_cnn_47702my_cnn_47704my_cnn_47706*
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
__inference_call_47687?
"my_decoder/StatefulPartitionedCallStatefulPartitionedCall'my_cnn/StatefulPartitionedCall:output:0my_decoder_47767my_decoder_47769my_decoder_47771my_decoder_47773my_decoder_47775my_decoder_47777*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_47766?
IdentityIdentity+my_decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp^my_cnn/StatefulPartitionedCall#^my_decoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 2@
my_cnn/StatefulPartitionedCallmy_cnn/StatefulPartitionedCall2H
"my_decoder/StatefulPartitionedCall"my_decoder/StatefulPartitionedCall:R N
/
_output_shapes
:?????????

_user_specified_namex
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

?
&__inference_my_cnn_layer_call_fn_50679
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
A__inference_my_cnn_layer_call_and_return_conditional_losses_47955o
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
?
I__inference_my_cnn_block_1_layer_call_and_return_conditional_losses_48055	
input<
"my_cnn_normalization_layer_2_48024:00
"my_cnn_normalization_layer_2_48026:0<
"my_cnn_normalization_layer_3_48040:000
"my_cnn_normalization_layer_3_48042:0
identity??1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp?4my_cnn_normalization_layer_2/StatefulPartitionedCall?4my_cnn_normalization_layer_3/StatefulPartitionedCall?
4my_cnn_normalization_layer_2/StatefulPartitionedCallStatefulPartitionedCallinput"my_cnn_normalization_layer_2_48024"my_cnn_normalization_layer_2_48026*
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
4my_cnn_normalization_layer_3/StatefulPartitionedCallStatefulPartitionedCall=my_cnn_normalization_layer_2/StatefulPartitionedCall:output:0"my_cnn_normalization_layer_3_48040"my_cnn_normalization_layer_3_48042*
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
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp"my_cnn_normalization_layer_2_48024*&
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
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp"my_cnn_normalization_layer_3_48040*&
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
?
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_47825

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
?
I
-__inference_max_pooling2d_layer_call_fn_51378

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
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_47825?
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
?G
?
__inference_call_49155
x9
&dense_1_matmul_readvariableop_resource:	
?6
'dense_1_biasadd_readvariableop_resource:	?S
9conv2d_transpose_conv2d_transpose_readvariableop_resource:0>
0conv2d_transpose_biasadd_readvariableop_resource:U
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_1_biasadd_readvariableop_resource:
identity??'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	
?*
dtype0u
dense_1/MatMulMatMulx%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????U
reshape/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :0?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
reshape/ReshapeReshapedense_1/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0^
conv2d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0*
dtype0?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????i
conv2d_transpose_1/ShapeShape!conv2d_transpose/BiasAdd:output:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????z
IdentityIdentity#conv2d_transpose_1/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????
: : : : : : 2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????


_user_specified_namex
?G
?
__inference_call_50596
x9
&dense_1_matmul_readvariableop_resource:	
?6
'dense_1_biasadd_readvariableop_resource:	?S
9conv2d_transpose_conv2d_transpose_readvariableop_resource:0>
0conv2d_transpose_biasadd_readvariableop_resource:U
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_1_biasadd_readvariableop_resource:
identity??'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	
?*
dtype0u
dense_1/MatMulMatMulx%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????U
reshape/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :0?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
reshape/ReshapeReshapedense_1/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0^
conv2d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0*
dtype0?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????i
conv2d_transpose_1/ShapeShape!conv2d_transpose/BiasAdd:output:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????z
IdentityIdentity#conv2d_transpose_1/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????
: : : : : : 2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????


_user_specified_namex
?
?
E__inference_my_decoder_layer_call_and_return_conditional_losses_48625
input_1 
dense_1_48608:	
?
dense_1_48610:	?0
conv2d_transpose_48614:0$
conv2d_transpose_48616:2
conv2d_transpose_1_48619:&
conv2d_transpose_1_48621:
identity??(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_1_48608dense_1_48610*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_48450?
reshape/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_48470?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_transpose_48614conv2d_transpose_48616*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_48382?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_48619conv2d_transpose_1_48621*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_48426?
IdentityIdentity3conv2d_transpose_1/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????
: : : : : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:P L
'
_output_shapes
:?????????

!
_user_specified_name	input_1
??
?%
!__inference__traced_restore_51984
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
5
"assignvariableop_10_dense_1_kernel:	
?/
 assignvariableop_11_dense_1_bias:	?E
+assignvariableop_12_conv2d_transpose_kernel:07
)assignvariableop_13_conv2d_transpose_bias:G
-assignvariableop_14_conv2d_transpose_1_kernel:9
+assignvariableop_15_conv2d_transpose_1_bias:%
assignvariableop_16_total_2: %
assignvariableop_17_count_2: %
assignvariableop_18_total_1: %
assignvariableop_19_count_1: '
assignvariableop_20_adam_iter:	 )
assignvariableop_21_adam_beta_1: )
assignvariableop_22_adam_beta_2: (
assignvariableop_23_adam_decay: 0
&assignvariableop_24_adam_learning_rate: #
assignvariableop_25_total: #
assignvariableop_26_count: B
(assignvariableop_27_adam_conv2d_kernel_m:4
&assignvariableop_28_adam_conv2d_bias_m:D
*assignvariableop_29_adam_conv2d_1_kernel_m:6
(assignvariableop_30_adam_conv2d_1_bias_m:D
*assignvariableop_31_adam_conv2d_2_kernel_m:06
(assignvariableop_32_adam_conv2d_2_bias_m:0D
*assignvariableop_33_adam_conv2d_3_kernel_m:006
(assignvariableop_34_adam_conv2d_3_bias_m:0:
'assignvariableop_35_adam_dense_kernel_m:	?
3
%assignvariableop_36_adam_dense_bias_m:
<
)assignvariableop_37_adam_dense_1_kernel_m:	
?6
'assignvariableop_38_adam_dense_1_bias_m:	?L
2assignvariableop_39_adam_conv2d_transpose_kernel_m:0>
0assignvariableop_40_adam_conv2d_transpose_bias_m:N
4assignvariableop_41_adam_conv2d_transpose_1_kernel_m:@
2assignvariableop_42_adam_conv2d_transpose_1_bias_m:B
(assignvariableop_43_adam_conv2d_kernel_v:4
&assignvariableop_44_adam_conv2d_bias_v:D
*assignvariableop_45_adam_conv2d_1_kernel_v:6
(assignvariableop_46_adam_conv2d_1_bias_v:D
*assignvariableop_47_adam_conv2d_2_kernel_v:06
(assignvariableop_48_adam_conv2d_2_bias_v:0D
*assignvariableop_49_adam_conv2d_3_kernel_v:006
(assignvariableop_50_adam_conv2d_3_bias_v:0:
'assignvariableop_51_adam_dense_kernel_v:	?
3
%assignvariableop_52_adam_dense_bias_v:
<
)assignvariableop_53_adam_dense_1_kernel_v:	
?6
'assignvariableop_54_adam_dense_1_bias_v:	?L
2assignvariableop_55_adam_conv2d_transpose_kernel_v:0>
0assignvariableop_56_adam_conv2d_transpose_bias_v:N
4assignvariableop_57_adam_conv2d_transpose_1_kernel_v:@
2assignvariableop_58_adam_conv2d_transpose_1_bias_v:
identity_60??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*?
value?B?<B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*?
value?B?<B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*J
dtypes@
>2<	[
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
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp+assignvariableop_12_conv2d_transpose_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp)assignvariableop_13_conv2d_transpose_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp-assignvariableop_14_conv2d_transpose_1_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp+assignvariableop_15_conv2d_transpose_1_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpassignvariableop_16_total_2Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpassignvariableop_17_count_2Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpassignvariableop_18_total_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpassignvariableop_19_count_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_iterIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_beta_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_beta_2Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_decayIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_learning_rateIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOpassignvariableop_25_totalIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpassignvariableop_26_countIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_conv2d_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_conv2d_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_conv2d_1_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_conv2d_1_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_conv2d_2_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_conv2d_2_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_conv2d_3_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_conv2d_3_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp'assignvariableop_35_adam_dense_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp%assignvariableop_36_adam_dense_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_dense_1_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp'assignvariableop_38_adam_dense_1_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp2assignvariableop_39_adam_conv2d_transpose_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp0assignvariableop_40_adam_conv2d_transpose_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp4assignvariableop_41_adam_conv2d_transpose_1_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp2assignvariableop_42_adam_conv2d_transpose_1_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp(assignvariableop_43_adam_conv2d_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp&assignvariableop_44_adam_conv2d_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_conv2d_1_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_conv2d_1_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_conv2d_2_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_conv2d_2_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_conv2d_3_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_conv2d_3_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOp'assignvariableop_51_adam_dense_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp%assignvariableop_52_adam_dense_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp)assignvariableop_53_adam_dense_1_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp'assignvariableop_54_adam_dense_1_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOp2assignvariableop_55_adam_conv2d_transpose_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOp0assignvariableop_56_adam_conv2d_transpose_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOp4assignvariableop_57_adam_conv2d_transpose_1_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOp2assignvariableop_58_adam_conv2d_transpose_1_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?

Identity_59Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_60IdentityIdentity_59:output:0^NoOp_1*
T0*
_output_shapes
: ?

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_60Identity_60:output:0*?
_input_shapesz
x: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
^
B__inference_reshape_layer_call_and_return_conditional_losses_48470

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :0?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????0`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
I__inference_my_cnn_block_1_layer_call_and_return_conditional_losses_51168	
input<
"my_cnn_normalization_layer_2_51148:00
"my_cnn_normalization_layer_2_51150:0<
"my_cnn_normalization_layer_3_51153:000
"my_cnn_normalization_layer_3_51155:0
identity??1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp?4my_cnn_normalization_layer_2/StatefulPartitionedCall?4my_cnn_normalization_layer_3/StatefulPartitionedCall?
4my_cnn_normalization_layer_2/StatefulPartitionedCallStatefulPartitionedCallinput"my_cnn_normalization_layer_2_51148"my_cnn_normalization_layer_2_51150*
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
4my_cnn_normalization_layer_3/StatefulPartitionedCallStatefulPartitionedCall=my_cnn_normalization_layer_2/StatefulPartitionedCall:output:0"my_cnn_normalization_layer_3_51153"my_cnn_normalization_layer_3_51155*
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
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp"my_cnn_normalization_layer_2_51148*&
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
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp"my_cnn_normalization_layer_3_51153*&
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
?
K
/__inference_max_pooling2d_1_layer_call_fn_51440

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
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_47837?
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
?
?
E__inference_my_decoder_layer_call_and_return_conditional_losses_48605
input_1 
dense_1_48588:	
?
dense_1_48590:	?0
conv2d_transpose_48594:0$
conv2d_transpose_48596:2
conv2d_transpose_1_48599:&
conv2d_transpose_1_48601:
identity??(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_1_48588dense_1_48590*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_48450?
reshape/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_48470?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_transpose_48594conv2d_transpose_48596*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_48382?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_48599conv2d_transpose_1_48601*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_48426?
IdentityIdentity3conv2d_transpose_1/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????
: : : : : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:P L
'
_output_shapes
:?????????

!
_user_specified_name	input_1
?0
?
A__inference_my_cnn_layer_call_and_return_conditional_losses_50798
x,
my_cnn_block_50754: 
my_cnn_block_50756:,
my_cnn_block_50758: 
my_cnn_block_50760:.
my_cnn_block_1_50763:0"
my_cnn_block_1_50765:0.
my_cnn_block_1_50767:00"
my_cnn_block_1_50769:07
$dense_matmul_readvariableop_resource:	?
3
%dense_biasadd_readvariableop_resource:

identity??/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?$my_cnn_block/StatefulPartitionedCall?&my_cnn_block_1/StatefulPartitionedCall?
$my_cnn_block/StatefulPartitionedCallStatefulPartitionedCallxmy_cnn_block_50754my_cnn_block_50756my_cnn_block_50758my_cnn_block_50760*
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
&my_cnn_block_1/StatefulPartitionedCallStatefulPartitionedCall-my_cnn_block/StatefulPartitionedCall:output:0my_cnn_block_1_50763my_cnn_block_1_50765my_cnn_block_1_50767my_cnn_block_1_50769*
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
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_50754*&
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
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_50758*&
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
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_1_50763*&
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
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_1_50767*&
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
?.
?
A__inference_my_cnn_layer_call_and_return_conditional_losses_48209
x,
my_cnn_block_48168: 
my_cnn_block_48170:,
my_cnn_block_48172: 
my_cnn_block_48174:.
my_cnn_block_1_48177:0"
my_cnn_block_1_48179:0.
my_cnn_block_1_48181:00"
my_cnn_block_1_48183:0
dense_48187:	?

dense_48189:

identity??/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp?dense/StatefulPartitionedCall?$my_cnn_block/StatefulPartitionedCall?&my_cnn_block_1/StatefulPartitionedCall?
$my_cnn_block/StatefulPartitionedCallStatefulPartitionedCallxmy_cnn_block_48168my_cnn_block_48170my_cnn_block_48172my_cnn_block_48174*
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
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_48127?
&my_cnn_block_1/StatefulPartitionedCallStatefulPartitionedCall-my_cnn_block/StatefulPartitionedCall:output:0my_cnn_block_1_48177my_cnn_block_1_48179my_cnn_block_1_48181my_cnn_block_1_48183*
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
I__inference_my_cnn_block_1_layer_call_and_return_conditional_losses_48055?
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
B__inference_flatten_layer_call_and_return_conditional_losses_47919?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_48187dense_48189*
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
@__inference_dense_layer_call_and_return_conditional_losses_47932?
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_48168*&
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
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_48172*&
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
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_1_48177*&
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
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_1_48181*&
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
?
?
 __inference__wrapped_model_47816
input_1.
my_autoencoder_47782:"
my_autoencoder_47784:.
my_autoencoder_47786:"
my_autoencoder_47788:.
my_autoencoder_47790:0"
my_autoencoder_47792:0.
my_autoencoder_47794:00"
my_autoencoder_47796:0'
my_autoencoder_47798:	?
"
my_autoencoder_47800:
'
my_autoencoder_47802:	
?#
my_autoencoder_47804:	?.
my_autoencoder_47806:0"
my_autoencoder_47808:.
my_autoencoder_47810:"
my_autoencoder_47812:
identity??&my_autoencoder/StatefulPartitionedCall?
&my_autoencoder/StatefulPartitionedCallStatefulPartitionedCallinput_1my_autoencoder_47782my_autoencoder_47784my_autoencoder_47786my_autoencoder_47788my_autoencoder_47790my_autoencoder_47792my_autoencoder_47794my_autoencoder_47796my_autoencoder_47798my_autoencoder_47800my_autoencoder_47802my_autoencoder_47804my_autoencoder_47806my_autoencoder_47808my_autoencoder_47810my_autoencoder_47812*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_47781?
IdentityIdentity/my_autoencoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????o
NoOpNoOp'^my_autoencoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 2P
&my_autoencoder/StatefulPartitionedCall&my_autoencoder/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
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
?
?
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_51096	
input:
 my_cnn_normalization_layer_51076:.
 my_cnn_normalization_layer_51078:<
"my_cnn_normalization_layer_1_51081:0
"my_cnn_normalization_layer_1_51083:
identity??/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp?2my_cnn_normalization_layer/StatefulPartitionedCall?4my_cnn_normalization_layer_1/StatefulPartitionedCall?
2my_cnn_normalization_layer/StatefulPartitionedCallStatefulPartitionedCallinput my_cnn_normalization_layer_51076 my_cnn_normalization_layer_51078*
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
4my_cnn_normalization_layer_1/StatefulPartitionedCallStatefulPartitionedCall;my_cnn_normalization_layer/StatefulPartitionedCall:output:0"my_cnn_normalization_layer_1_51081"my_cnn_normalization_layer_1_51083*
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
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp my_cnn_normalization_layer_51076*&
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
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp"my_cnn_normalization_layer_1_51081*&
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
@__inference_dense_layer_call_and_return_conditional_losses_51199

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
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_51179

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
.__inference_my_cnn_block_1_layer_call_fn_51122	
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
I__inference_my_cnn_block_1_layer_call_and_return_conditional_losses_48055w
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
?G
?
E__inference_my_decoder_layer_call_and_return_conditional_losses_50966
x9
&dense_1_matmul_readvariableop_resource:	
?6
'dense_1_biasadd_readvariableop_resource:	?S
9conv2d_transpose_conv2d_transpose_readvariableop_resource:0>
0conv2d_transpose_biasadd_readvariableop_resource:U
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_1_biasadd_readvariableop_resource:
identity??'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	
?*
dtype0u
dense_1/MatMulMatMulx%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????U
reshape/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :0?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
reshape/ReshapeReshapedense_1/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0^
conv2d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0*
dtype0?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????i
conv2d_transpose_1/ShapeShape!conv2d_transpose/BiasAdd:output:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????z
IdentityIdentity#conv2d_transpose_1/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????
: : : : : : 2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????


_user_specified_namex
? 
?
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_48382

inputsB
(conv2d_transpose_readvariableop_resource:0-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:0*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????0
 
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
?
?
__inference_call_49170
x&
my_cnn_49077:
my_cnn_49079:&
my_cnn_49081:
my_cnn_49083:&
my_cnn_49085:0
my_cnn_49087:0&
my_cnn_49089:00
my_cnn_49091:0
my_cnn_49093:	?

my_cnn_49095:
#
my_decoder_49156:	
?
my_decoder_49158:	?*
my_decoder_49160:0
my_decoder_49162:*
my_decoder_49164:
my_decoder_49166:
identity??my_cnn/StatefulPartitionedCall?"my_decoder/StatefulPartitionedCall?
my_cnn/StatefulPartitionedCallStatefulPartitionedCallxmy_cnn_49077my_cnn_49079my_cnn_49081my_cnn_49083my_cnn_49085my_cnn_49087my_cnn_49089my_cnn_49091my_cnn_49093my_cnn_49095*
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
__inference_call_49076?
"my_decoder/StatefulPartitionedCallStatefulPartitionedCall'my_cnn/StatefulPartitionedCall:output:0my_decoder_49156my_decoder_49158my_decoder_49160my_decoder_49162my_decoder_49164my_decoder_49166*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_49155?
IdentityIdentity+my_decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp^my_cnn/StatefulPartitionedCall#^my_decoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 2@
my_cnn/StatefulPartitionedCallmy_cnn/StatefulPartitionedCall2H
"my_decoder/StatefulPartitionedCall"my_decoder/StatefulPartitionedCall:R N
/
_output_shapes
:?????????

_user_specified_namex
?
?
2__inference_conv2d_transpose_1_layer_call_fn_51288

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_48426?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?)
?
I__inference_my_autoencoder_layer_call_and_return_conditional_losses_48811
x&
my_cnn_48760:
my_cnn_48762:&
my_cnn_48764:
my_cnn_48766:&
my_cnn_48768:0
my_cnn_48770:0&
my_cnn_48772:00
my_cnn_48774:0
my_cnn_48776:	?

my_cnn_48778:
#
my_decoder_48781:	
?
my_decoder_48783:	?*
my_decoder_48785:0
my_decoder_48787:*
my_decoder_48789:
my_decoder_48791:
identity??/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp?my_cnn/StatefulPartitionedCall?"my_decoder/StatefulPartitionedCall?
my_cnn/StatefulPartitionedCallStatefulPartitionedCallxmy_cnn_48760my_cnn_48762my_cnn_48764my_cnn_48766my_cnn_48768my_cnn_48770my_cnn_48772my_cnn_48774my_cnn_48776my_cnn_48778*
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
A__inference_my_cnn_layer_call_and_return_conditional_losses_48209?
"my_decoder/StatefulPartitionedCallStatefulPartitionedCall'my_cnn/StatefulPartitionedCall:output:0my_decoder_48781my_decoder_48783my_decoder_48785my_decoder_48787my_decoder_48789my_decoder_48791*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_my_decoder_layer_call_and_return_conditional_losses_48553?
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_48760*&
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
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_48764*&
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
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_48768*&
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
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_48772*&
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
: ?
IdentityIdentity+my_decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp0^conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp^my_cnn/StatefulPartitionedCall#^my_decoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 2b
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp2@
my_cnn/StatefulPartitionedCallmy_cnn/StatefulPartitionedCall2H
"my_decoder/StatefulPartitionedCall"my_decoder/StatefulPartitionedCall:R N
/
_output_shapes
:?????????

_user_specified_namex
? 
?
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_51279

inputsB
(conv2d_transpose_readvariableop_resource:0-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:0*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
I__inference_my_cnn_block_1_layer_call_and_return_conditional_losses_51145	
input<
"my_cnn_normalization_layer_2_51125:00
"my_cnn_normalization_layer_2_51127:0<
"my_cnn_normalization_layer_3_51130:000
"my_cnn_normalization_layer_3_51132:0
identity??1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp?4my_cnn_normalization_layer_2/StatefulPartitionedCall?4my_cnn_normalization_layer_3/StatefulPartitionedCall?
4my_cnn_normalization_layer_2/StatefulPartitionedCallStatefulPartitionedCallinput"my_cnn_normalization_layer_2_51125"my_cnn_normalization_layer_2_51127*
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
4my_cnn_normalization_layer_3/StatefulPartitionedCallStatefulPartitionedCall=my_cnn_normalization_layer_2/StatefulPartitionedCall:output:0"my_cnn_normalization_layer_3_51130"my_cnn_normalization_layer_3_51132*
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
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp"my_cnn_normalization_layer_2_51125*&
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
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp"my_cnn_normalization_layer_3_51130*&
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
?
__inference_loss_fn_3_51497T
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
?
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_51445

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
I__inference_my_cnn_block_1_layer_call_and_return_conditional_losses_47903	
input<
"my_cnn_normalization_layer_2_47883:00
"my_cnn_normalization_layer_2_47885:0<
"my_cnn_normalization_layer_3_47888:000
"my_cnn_normalization_layer_3_47890:0
identity??1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp?4my_cnn_normalization_layer_2/StatefulPartitionedCall?4my_cnn_normalization_layer_3/StatefulPartitionedCall?
4my_cnn_normalization_layer_2/StatefulPartitionedCallStatefulPartitionedCallinput"my_cnn_normalization_layer_2_47883"my_cnn_normalization_layer_2_47885*
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
4my_cnn_normalization_layer_3/StatefulPartitionedCallStatefulPartitionedCall=my_cnn_normalization_layer_2/StatefulPartitionedCall:output:0"my_cnn_normalization_layer_3_47888"my_cnn_normalization_layer_3_47890*
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
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp"my_cnn_normalization_layer_2_47883*&
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
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp"my_cnn_normalization_layer_3_47888*&
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
?G
?
__inference_call_50654
x9
&dense_1_matmul_readvariableop_resource:	
?6
'dense_1_biasadd_readvariableop_resource:	?S
9conv2d_transpose_conv2d_transpose_readvariableop_resource:0>
0conv2d_transpose_biasadd_readvariableop_resource:U
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_1_biasadd_readvariableop_resource:
identity??'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	
?*
dtype0u
dense_1/MatMulMatMulx%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????U
reshape/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :0?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
reshape/ReshapeReshapedense_1/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0^
conv2d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0*
dtype0?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????i
conv2d_transpose_1/ShapeShape!conv2d_transpose/BiasAdd:output:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????z
IdentityIdentity#conv2d_transpose_1/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????
: : : : : : 2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????


_user_specified_namex
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_47919

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
?
#__inference_signature_wrapper_50278
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

	unknown_9:	
?

unknown_10:	?$

unknown_11:0

unknown_12:$

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_47816w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
B__inference_dense_1_layer_call_and_return_conditional_losses_48450

inputs1
matmul_readvariableop_resource:	
?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
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

?
&__inference_my_cnn_layer_call_fn_50704
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
A__inference_my_cnn_layer_call_and_return_conditional_losses_48209o
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
?
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_51383

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
?G
?
E__inference_my_decoder_layer_call_and_return_conditional_losses_51024
x9
&dense_1_matmul_readvariableop_resource:	
?6
'dense_1_biasadd_readvariableop_resource:	?S
9conv2d_transpose_conv2d_transpose_readvariableop_resource:0>
0conv2d_transpose_biasadd_readvariableop_resource:U
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_1_biasadd_readvariableop_resource:
identity??'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	
?*
dtype0u
dense_1/MatMulMatMulx%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????U
reshape/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :0?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
reshape/ReshapeReshapedense_1/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0^
conv2d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0*
dtype0?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????i
conv2d_transpose_1/ShapeShape!conv2d_transpose/BiasAdd:output:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????z
IdentityIdentity#conv2d_transpose_1/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????
: : : : : : 2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????


_user_specified_namex
?G
?
__inference_call_47766
x9
&dense_1_matmul_readvariableop_resource:	
?6
'dense_1_biasadd_readvariableop_resource:	?S
9conv2d_transpose_conv2d_transpose_readvariableop_resource:0>
0conv2d_transpose_biasadd_readvariableop_resource:U
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_1_biasadd_readvariableop_resource:
identity??'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	
?*
dtype0u
dense_1/MatMulMatMulx%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????U
reshape/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :0?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
reshape/ReshapeReshapedense_1/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0^
conv2d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0*
dtype0?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????i
conv2d_transpose_1/ShapeShape!conv2d_transpose/BiasAdd:output:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????z
IdentityIdentity#conv2d_transpose_1/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????
: : : : : : 2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????


_user_specified_namex
?'
?
'__inference___backward_call_49686_49753
placeholderk
ggradients_max_pooling2d_1_maxpool_grad_maxpoolgrad_my_cnn_normalization_layer_3_statefulpartitionedcallN
Jgradients_max_pooling2d_1_maxpool_grad_maxpoolgrad_max_pooling2d_1_maxpool|
xgradients_my_cnn_normalization_layer_3_statefulpartitionedcall_grad_my_cnn_normalization_layer_3_statefulpartitionedcall~
zgradients_my_cnn_normalization_layer_3_statefulpartitionedcall_grad_my_cnn_normalization_layer_3_statefulpartitionedcall_1~
zgradients_my_cnn_normalization_layer_3_statefulpartitionedcall_grad_my_cnn_normalization_layer_3_statefulpartitionedcall_2|
xgradients_my_cnn_normalization_layer_2_statefulpartitionedcall_grad_my_cnn_normalization_layer_2_statefulpartitionedcall~
zgradients_my_cnn_normalization_layer_2_statefulpartitionedcall_grad_my_cnn_normalization_layer_2_statefulpartitionedcall_1~
zgradients_my_cnn_normalization_layer_2_statefulpartitionedcall_grad_my_cnn_normalization_layer_2_statefulpartitionedcall_2
identity

identity_1

identity_2

identity_3

identity_4f
gradients/grad_ys_0Identityplaceholder*
T0*/
_output_shapes
:?????????0?
2gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGradMaxPoolGradggradients_max_pooling2d_1_maxpool_grad_maxpoolgrad_my_cnn_normalization_layer_3_statefulpartitionedcallJgradients_max_pooling2d_1_maxpool_grad_maxpoolgrad_max_pooling2d_1_maxpoolgradients/grad_ys_0:output:0*/
_output_shapes
:?????????0*
ksize
*
paddingVALID*
strides
?
Sgradients/my_cnn_normalization_layer_3/StatefulPartitionedCall_grad/PartitionedCallPartitionedCall;gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGrad:output:0xgradients_my_cnn_normalization_layer_3_statefulpartitionedcall_grad_my_cnn_normalization_layer_3_statefulpartitionedcallzgradients_my_cnn_normalization_layer_3_statefulpartitionedcall_grad_my_cnn_normalization_layer_3_statefulpartitionedcall_1zgradients_my_cnn_normalization_layer_3_statefulpartitionedcall_grad_my_cnn_normalization_layer_3_statefulpartitionedcall_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *G
_output_shapes5
3:?????????0:00:0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference___backward_call_49693_49710?
Sgradients/my_cnn_normalization_layer_2/StatefulPartitionedCall_grad/PartitionedCallPartitionedCall\gradients/my_cnn_normalization_layer_3/StatefulPartitionedCall_grad/PartitionedCall:output:0xgradients_my_cnn_normalization_layer_2_statefulpartitionedcall_grad_my_cnn_normalization_layer_2_statefulpartitionedcallzgradients_my_cnn_normalization_layer_2_statefulpartitionedcall_grad_my_cnn_normalization_layer_2_statefulpartitionedcall_1zgradients_my_cnn_normalization_layer_2_statefulpartitionedcall_grad_my_cnn_normalization_layer_2_statefulpartitionedcall_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *G
_output_shapes5
3:?????????:0:0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference___backward_call_49720_49737?
IdentityIdentity\gradients/my_cnn_normalization_layer_2/StatefulPartitionedCall_grad/PartitionedCall:output:0*
T0*/
_output_shapes
:??????????

Identity_1Identity\gradients/my_cnn_normalization_layer_2/StatefulPartitionedCall_grad/PartitionedCall:output:1*
T0*&
_output_shapes
:0?

Identity_2Identity\gradients/my_cnn_normalization_layer_2/StatefulPartitionedCall_grad/PartitionedCall:output:2*
T0*
_output_shapes
:0?

Identity_3Identity\gradients/my_cnn_normalization_layer_3/StatefulPartitionedCall_grad/PartitionedCall:output:1*
T0*&
_output_shapes
:00?

Identity_4Identity\gradients/my_cnn_normalization_layer_3/StatefulPartitionedCall_grad/PartitionedCall:output:2*
T0*
_output_shapes
:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????0:?????????0:?????????0:?????????0:?????????0:00:?????????0:?????????:0*/
forward_function_name__forward_call_49752:5 1
/
_output_shapes
:?????????0:51
/
_output_shapes
:?????????0:51
/
_output_shapes
:?????????0:51
/
_output_shapes
:?????????0:51
/
_output_shapes
:?????????0:,(
&
_output_shapes
:00:51
/
_output_shapes
:?????????0:51
/
_output_shapes
:?????????:,(
&
_output_shapes
:0
?
?
__inference_call_49208
x&
my_cnn_49173:
my_cnn_49175:&
my_cnn_49177:
my_cnn_49179:&
my_cnn_49181:0
my_cnn_49183:0&
my_cnn_49185:00
my_cnn_49187:0
my_cnn_49189:	?

my_cnn_49191:
#
my_decoder_49194:	
?
my_decoder_49196:	?*
my_decoder_49198:0
my_decoder_49200:*
my_decoder_49202:
my_decoder_49204:
identity??my_cnn/StatefulPartitionedCall?"my_decoder/StatefulPartitionedCall?
my_cnn/StatefulPartitionedCallStatefulPartitionedCallxmy_cnn_49173my_cnn_49175my_cnn_49177my_cnn_49179my_cnn_49181my_cnn_49183my_cnn_49185my_cnn_49187my_cnn_49189my_cnn_49191*
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
__inference_call_47687?
"my_decoder/StatefulPartitionedCallStatefulPartitionedCall'my_cnn/StatefulPartitionedCall:output:0my_decoder_49194my_decoder_49196my_decoder_49198my_decoder_49200my_decoder_49202my_decoder_49204*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_47766?
IdentityIdentity+my_decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp^my_cnn/StatefulPartitionedCall#^my_decoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 2@
my_cnn/StatefulPartitionedCallmy_cnn/StatefulPartitionedCall2H
"my_decoder/StatefulPartitionedCall"my_decoder/StatefulPartitionedCall:R N
/
_output_shapes
:?????????

_user_specified_namex
?
?
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_47870	
input:
 my_cnn_normalization_layer_47850:.
 my_cnn_normalization_layer_47852:<
"my_cnn_normalization_layer_1_47855:0
"my_cnn_normalization_layer_1_47857:
identity??/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp?2my_cnn_normalization_layer/StatefulPartitionedCall?4my_cnn_normalization_layer_1/StatefulPartitionedCall?
2my_cnn_normalization_layer/StatefulPartitionedCallStatefulPartitionedCallinput my_cnn_normalization_layer_47850 my_cnn_normalization_layer_47852*
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
4my_cnn_normalization_layer_1/StatefulPartitionedCallStatefulPartitionedCall;my_cnn_normalization_layer/StatefulPartitionedCall:output:0"my_cnn_normalization_layer_1_47855"my_cnn_normalization_layer_1_47857*
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
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp my_cnn_normalization_layer_47850*&
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
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp"my_cnn_normalization_layer_1_47855*&
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
?,
?
'__inference___backward_call_49618_49650
placeholderd
`gradients_conv2d_transpose_1_conv2d_transpose_grad_conv2dbackpropfilter_conv2d_transpose_biasaddp
lgradients_conv2d_transpose_1_conv2d_transpose_grad_conv2d_conv2d_transpose_1_conv2d_transpose_readvariableopY
Ugradients_conv2d_transpose_conv2d_transpose_grad_conv2dbackpropfilter_reshape_reshapel
hgradients_conv2d_transpose_conv2d_transpose_grad_conv2d_conv2d_transpose_conv2d_transpose_readvariableop8
4gradients_reshape_reshape_grad_shape_dense_1_biasaddF
Bgradients_dense_1_matmul_grad_matmul_dense_1_matmul_readvariableop,
(gradients_dense_1_matmul_grad_matmul_1_x
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6f
gradients/grad_ys_0Identityplaceholder*
T0*/
_output_shapes
:??????????
5gradients/conv2d_transpose_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients/grad_ys_0:output:0*
T0*
_output_shapes
:?
8gradients/conv2d_transpose_1/conv2d_transpose_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            ?
Ggradients/conv2d_transpose_1/conv2d_transpose_grad/Conv2DBackpropFilterConv2DBackpropFiltergradients/grad_ys_0:output:0Agradients/conv2d_transpose_1/conv2d_transpose_grad/Shape:output:0`gradients_conv2d_transpose_1_conv2d_transpose_grad_conv2dbackpropfilter_conv2d_transpose_biasadd*
T0*&
_output_shapes
:*
paddingSAME*
strides
?
9gradients/conv2d_transpose_1/conv2d_transpose_grad/Conv2DConv2Dgradients/grad_ys_0:output:0lgradients_conv2d_transpose_1_conv2d_transpose_grad_conv2d_conv2d_transpose_1_conv2d_transpose_readvariableop*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
3gradients/conv2d_transpose/BiasAdd_grad/BiasAddGradBiasAddGradBgradients/conv2d_transpose_1/conv2d_transpose_grad/Conv2D:output:0*
T0*
_output_shapes
:?
6gradients/conv2d_transpose/conv2d_transpose_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         0   ?
Egradients/conv2d_transpose/conv2d_transpose_grad/Conv2DBackpropFilterConv2DBackpropFilterBgradients/conv2d_transpose_1/conv2d_transpose_grad/Conv2D:output:0?gradients/conv2d_transpose/conv2d_transpose_grad/Shape:output:0Ugradients_conv2d_transpose_conv2d_transpose_grad_conv2dbackpropfilter_reshape_reshape*
T0*&
_output_shapes
:0*
paddingSAME*
strides
?
7gradients/conv2d_transpose/conv2d_transpose_grad/Conv2DConv2DBgradients/conv2d_transpose_1/conv2d_transpose_grad/Conv2D:output:0hgradients_conv2d_transpose_conv2d_transpose_grad_conv2d_conv2d_transpose_conv2d_transpose_readvariableop*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
?
$gradients/reshape/Reshape_grad/ShapeShape4gradients_reshape_reshape_grad_shape_dense_1_biasadd*
T0*
_output_shapes
:?
&gradients/reshape/Reshape_grad/ReshapeReshape@gradients/conv2d_transpose/conv2d_transpose_grad/Conv2D:output:0-gradients/reshape/Reshape_grad/Shape:output:0*
T0*(
_output_shapes
:???????????
*gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad/gradients/reshape/Reshape_grad/Reshape:output:0*
T0*
_output_shapes	
:??
$gradients/dense_1/MatMul_grad/MatMulMatMul/gradients/reshape/Reshape_grad/Reshape:output:0Bgradients_dense_1_matmul_grad_matmul_dense_1_matmul_readvariableop*
T0*'
_output_shapes
:?????????
*
transpose_b(?
&gradients/dense_1/MatMul_grad/MatMul_1MatMul(gradients_dense_1_matmul_grad_matmul_1_x/gradients/reshape/Reshape_grad/Reshape:output:0*
T0*
_output_shapes
:	
?*
transpose_a(v
IdentityIdentity.gradients/dense_1/MatMul_grad/MatMul:product:0*
T0*'
_output_shapes
:?????????
r

Identity_1Identity0gradients/dense_1/MatMul_grad/MatMul_1:product:0*
T0*
_output_shapes
:	
?q

Identity_2Identity3gradients/dense_1/BiasAdd_grad/BiasAddGrad:output:0*
T0*
_output_shapes	
:??

Identity_3IdentityNgradients/conv2d_transpose/conv2d_transpose_grad/Conv2DBackpropFilter:output:0*
T0*&
_output_shapes
:0y

Identity_4Identity<gradients/conv2d_transpose/BiasAdd_grad/BiasAddGrad:output:0*
T0*
_output_shapes
:?

Identity_5IdentityPgradients/conv2d_transpose_1/conv2d_transpose_grad/Conv2DBackpropFilter:output:0*
T0*&
_output_shapes
:{

Identity_6Identity>gradients/conv2d_transpose_1/BiasAdd_grad/BiasAddGrad:output:0*
T0*
_output_shapes
:"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????::?????????0:0:??????????:	
?:?????????
*/
forward_function_name__forward_call_49649:5 1
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:,(
&
_output_shapes
::51
/
_output_shapes
:?????????0:,(
&
_output_shapes
:0:.*
(
_output_shapes
:??????????:%!

_output_shapes
:	
?:-)
'
_output_shapes
:?????????

?*
?
\broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_true_50143?
?broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_broadcast_weights_1_assert_broadcastable_values_shape?
?broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_1_broadcast_weights_1_assert_broadcastable_weights_shapea
]broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_placeholder
^
Zbroadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity
?
qbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
mbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDims?broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_broadcast_weights_1_assert_broadcastable_values_shapezbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:output:0*
T0*
_output_shapes

:?
?broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"      ?
rbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :?
lbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFill?broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape/shape_as_tensor:output:0{broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:output:0*
T0*
_output_shapes

:?
nbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
ibroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2vbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:output:0ubroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:output:0wbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:output:0*
N*
T0*
_output_shapes

:?
sbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
obroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDims?broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_1_broadcast_weights_1_assert_broadcastable_weights_shape|broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:output:0*
T0*
_output_shapes

:?
{broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperationxbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:output:0rbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:output:0*
T0*<
_output_shapes*
(:?????????:?????????:*
set_operationa-b?
sbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSize?broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:result_values:0*
T0*
_output_shapes
: ?
dbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConst*
_output_shapes
: *
dtype0*
value	B : ?
bbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualmbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:output:0|broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:output:0*
T0*
_output_shapes
: ?
Zbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/IdentityIdentityfbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:z:0*
T0
*
_output_shapes
: "?
Zbroadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identitycbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::: :  

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
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
?/
?
A__inference_my_cnn_layer_call_and_return_conditional_losses_48345
input_1,
my_cnn_block_48304: 
my_cnn_block_48306:,
my_cnn_block_48308: 
my_cnn_block_48310:.
my_cnn_block_1_48313:0"
my_cnn_block_1_48315:0.
my_cnn_block_1_48317:00"
my_cnn_block_1_48319:0
dense_48323:	?

dense_48325:

identity??/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp?dense/StatefulPartitionedCall?$my_cnn_block/StatefulPartitionedCall?&my_cnn_block_1/StatefulPartitionedCall?
$my_cnn_block/StatefulPartitionedCallStatefulPartitionedCallinput_1my_cnn_block_48304my_cnn_block_48306my_cnn_block_48308my_cnn_block_48310*
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
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_48127?
&my_cnn_block_1/StatefulPartitionedCallStatefulPartitionedCall-my_cnn_block/StatefulPartitionedCall:output:0my_cnn_block_1_48313my_cnn_block_1_48315my_cnn_block_1_48317my_cnn_block_1_48319*
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
I__inference_my_cnn_block_1_layer_call_and_return_conditional_losses_48055?
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
B__inference_flatten_layer_call_and_return_conditional_losses_47919?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_48323dense_48325*
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
@__inference_dense_layer_call_and_return_conditional_losses_47932?
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_48304*&
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
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_48308*&
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
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_1_48313*&
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
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_1_48317*&
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
?
?
*__inference_my_decoder_layer_call_fn_50908
x
unknown:	
?
	unknown_0:	?#
	unknown_1:0
	unknown_2:#
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_my_decoder_layer_call_and_return_conditional_losses_48553w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????
: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:?????????


_user_specified_namex
?
?
Cbroadcast_weights_1_assert_broadcastable_is_valid_shape_false_49312G
Cbroadcast_weights_1_assert_broadcastable_is_valid_shape_placeholder
?
?broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_broadcast_weights_1_assert_broadcastable_values_rank?
?broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_broadcast_weights_1_assert_broadcastable_weights_rank?
?broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_broadcast_weights_1_assert_broadcastable_values_shape?
?broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_broadcast_weights_1_assert_broadcastable_weights_shapeD
@broadcast_weights_1_assert_broadcastable_is_valid_shape_identity
?
^broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqual?broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_broadcast_weights_1_assert_broadcastable_values_rank?broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_broadcast_weights_1_assert_broadcastable_weights_rank*
T0*
_output_shapes
: ?
Qbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shapeStatelessIfbbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:z:0?broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_broadcast_weights_1_assert_broadcastable_values_shape?broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_broadcast_weights_1_assert_broadcastable_weights_shapebbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *p
else_branchaR_
]broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_false_49321*
output_shapes
: *o
then_branch`R^
\broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_true_49320?
Zbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/IdentityIdentityZbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape:output:0*
T0
*
_output_shapes
: ?
@broadcast_weights_1/assert_broadcastable/is_valid_shape/IdentityIdentitycbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Identity:output:0*
T0
*
_output_shapes
: "?
@broadcast_weights_1_assert_broadcastable_is_valid_shape_identityIbroadcast_weights_1/assert_broadcastable/is_valid_shape/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:
?
?
'__inference___backward_call_49720_49737
placeholder?
;gradients_activation_2_relu_grad_relugrad_activation_2_relu+
'gradients_conv2d_2_conv2d_grad_shapen_xH
Dgradients_conv2d_2_conv2d_grad_shapen_conv2d_2_conv2d_readvariableop
identity

identity_1

identity_2f
gradients/grad_ys_0Identityplaceholder*
T0*/
_output_shapes
:?????????0?
)gradients/activation_2/Relu_grad/ReluGradReluGradgradients/grad_ys_0:output:0;gradients_activation_2_relu_grad_relugrad_activation_2_relu*
T0*/
_output_shapes
:?????????0?
+gradients/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGrad5gradients/activation_2/Relu_grad/ReluGrad:backprops:0*
T0*
_output_shapes
:0?
%gradients/conv2d_2/Conv2D_grad/ShapeNShapeN'gradients_conv2d_2_conv2d_grad_shapen_xDgradients_conv2d_2_conv2d_grad_shapen_conv2d_2_conv2d_readvariableop*
N*
T0* 
_output_shapes
::?
2gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput.gradients/conv2d_2/Conv2D_grad/ShapeN:output:0Dgradients_conv2d_2_conv2d_grad_shapen_conv2d_2_conv2d_readvariableop5gradients/activation_2/Relu_grad/ReluGrad:backprops:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
3gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter'gradients_conv2d_2_conv2d_grad_shapen_x.gradients/conv2d_2/Conv2D_grad/ShapeN:output:15gradients/activation_2/Relu_grad/ReluGrad:backprops:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
?
IdentityIdentity;gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInput:output:0*
T0*/
_output_shapes
:??????????

Identity_1Identity<gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilter:output:0*
T0*&
_output_shapes
:0q

Identity_2Identity4gradients/conv2d_2/BiasAdd_grad/BiasAddGrad:output:0*
T0*
_output_shapes
:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:?????????0:?????????0:?????????:0*/
forward_function_name__forward_call_49736:5 1
/
_output_shapes
:?????????0:51
/
_output_shapes
:?????????0:51
/
_output_shapes
:?????????:,(
&
_output_shapes
:0
?	
?
B__inference_dense_1_layer_call_and_return_conditional_losses_51218

inputs1
matmul_readvariableop_resource:	
?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_47837

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
?.
?
A__inference_my_cnn_layer_call_and_return_conditional_losses_47955
x,
my_cnn_block_47871: 
my_cnn_block_47873:,
my_cnn_block_47875: 
my_cnn_block_47877:.
my_cnn_block_1_47904:0"
my_cnn_block_1_47906:0.
my_cnn_block_1_47908:00"
my_cnn_block_1_47910:0
dense_47933:	?

dense_47935:

identity??/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp?dense/StatefulPartitionedCall?$my_cnn_block/StatefulPartitionedCall?&my_cnn_block_1/StatefulPartitionedCall?
$my_cnn_block/StatefulPartitionedCallStatefulPartitionedCallxmy_cnn_block_47871my_cnn_block_47873my_cnn_block_47875my_cnn_block_47877*
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
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_47870?
&my_cnn_block_1/StatefulPartitionedCallStatefulPartitionedCall-my_cnn_block/StatefulPartitionedCall:output:0my_cnn_block_1_47904my_cnn_block_1_47906my_cnn_block_1_47908my_cnn_block_1_47910*
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
I__inference_my_cnn_block_1_layer_call_and_return_conditional_losses_47903?
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
B__inference_flatten_layer_call_and_return_conditional_losses_47919?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_47933dense_47935*
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
@__inference_dense_layer_call_and_return_conditional_losses_47932?
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_47871*&
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
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_47875*&
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
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_1_47904*&
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
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_1_47908*&
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
?
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_48127	
input:
 my_cnn_normalization_layer_48096:.
 my_cnn_normalization_layer_48098:<
"my_cnn_normalization_layer_1_48112:0
"my_cnn_normalization_layer_1_48114:
identity??/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp?2my_cnn_normalization_layer/StatefulPartitionedCall?4my_cnn_normalization_layer_1/StatefulPartitionedCall?
2my_cnn_normalization_layer/StatefulPartitionedCallStatefulPartitionedCallinput my_cnn_normalization_layer_48096 my_cnn_normalization_layer_48098*
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
4my_cnn_normalization_layer_1/StatefulPartitionedCallStatefulPartitionedCall;my_cnn_normalization_layer/StatefulPartitionedCall:output:0"my_cnn_normalization_layer_1_48112"my_cnn_normalization_layer_1_48114*
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
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp my_cnn_normalization_layer_48096*&
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
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp"my_cnn_normalization_layer_1_48112*&
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
?
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_48426

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
.__inference_my_autoencoder_layer_call_fn_48883
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

	unknown_9:	
?

unknown_10:	?$

unknown_11:0

unknown_12:$

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_my_autoencoder_layer_call_and_return_conditional_losses_48811w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
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
?)
?
I__inference_my_autoencoder_layer_call_and_return_conditional_losses_48937
input_1&
my_cnn_48886:
my_cnn_48888:&
my_cnn_48890:
my_cnn_48892:&
my_cnn_48894:0
my_cnn_48896:0&
my_cnn_48898:00
my_cnn_48900:0
my_cnn_48902:	?

my_cnn_48904:
#
my_decoder_48907:	
?
my_decoder_48909:	?*
my_decoder_48911:0
my_decoder_48913:*
my_decoder_48915:
my_decoder_48917:
identity??/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp?my_cnn/StatefulPartitionedCall?"my_decoder/StatefulPartitionedCall?
my_cnn/StatefulPartitionedCallStatefulPartitionedCallinput_1my_cnn_48886my_cnn_48888my_cnn_48890my_cnn_48892my_cnn_48894my_cnn_48896my_cnn_48898my_cnn_48900my_cnn_48902my_cnn_48904*
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
A__inference_my_cnn_layer_call_and_return_conditional_losses_47955?
"my_decoder/StatefulPartitionedCallStatefulPartitionedCall'my_cnn/StatefulPartitionedCall:output:0my_decoder_48907my_decoder_48909my_decoder_48911my_decoder_48913my_decoder_48915my_decoder_48917*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_my_decoder_layer_call_and_return_conditional_losses_48483?
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_48886*&
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
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_48890*&
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
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_48894*&
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
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_48898*&
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
: ?
IdentityIdentity+my_decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp0^conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp^my_cnn/StatefulPartitionedCall#^my_decoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 2b
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp2@
my_cnn/StatefulPartitionedCallmy_cnn/StatefulPartitionedCall2H
"my_decoder/StatefulPartitionedCall"my_decoder/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
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
?
?
__inference_call_49076
x,
my_cnn_block_49033: 
my_cnn_block_49035:,
my_cnn_block_49037: 
my_cnn_block_49039:.
my_cnn_block_1_49057:0"
my_cnn_block_1_49059:0.
my_cnn_block_1_49061:00"
my_cnn_block_1_49063:07
$dense_matmul_readvariableop_resource:	?
3
%dense_biasadd_readvariableop_resource:

identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?$my_cnn_block/StatefulPartitionedCall?&my_cnn_block_1/StatefulPartitionedCall?
$my_cnn_block/StatefulPartitionedCallStatefulPartitionedCallxmy_cnn_block_49033my_cnn_block_49035my_cnn_block_49037my_cnn_block_49039*
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
&my_cnn_block_1/StatefulPartitionedCallStatefulPartitionedCall-my_cnn_block/StatefulPartitionedCall:output:0my_cnn_block_1_49057my_cnn_block_1_49059my_cnn_block_1_49061my_cnn_block_1_49063*
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
?

?
&__inference_my_cnn_layer_call_fn_47978
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
A__inference_my_cnn_layer_call_and_return_conditional_losses_47955o
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
?H
?
'__inference___backward_call_49672_49876
placeholder1
-gradients_dense_relu_grad_relugrad_dense_reluB
>gradients_dense_matmul_grad_matmul_dense_matmul_readvariableop8
4gradients_dense_matmul_grad_matmul_1_flatten_reshapeO
Kgradients_flatten_reshape_grad_shape_my_cnn_block_1_statefulpartitionedcall`
\gradients_my_cnn_block_1_statefulpartitionedcall_grad_my_cnn_block_1_statefulpartitionedcallb
^gradients_my_cnn_block_1_statefulpartitionedcall_grad_my_cnn_block_1_statefulpartitionedcall_1b
^gradients_my_cnn_block_1_statefulpartitionedcall_grad_my_cnn_block_1_statefulpartitionedcall_2b
^gradients_my_cnn_block_1_statefulpartitionedcall_grad_my_cnn_block_1_statefulpartitionedcall_3b
^gradients_my_cnn_block_1_statefulpartitionedcall_grad_my_cnn_block_1_statefulpartitionedcall_4b
^gradients_my_cnn_block_1_statefulpartitionedcall_grad_my_cnn_block_1_statefulpartitionedcall_5b
^gradients_my_cnn_block_1_statefulpartitionedcall_grad_my_cnn_block_1_statefulpartitionedcall_6b
^gradients_my_cnn_block_1_statefulpartitionedcall_grad_my_cnn_block_1_statefulpartitionedcall_7\
Xgradients_my_cnn_block_statefulpartitionedcall_grad_my_cnn_block_statefulpartitionedcall^
Zgradients_my_cnn_block_statefulpartitionedcall_grad_my_cnn_block_statefulpartitionedcall_1^
Zgradients_my_cnn_block_statefulpartitionedcall_grad_my_cnn_block_statefulpartitionedcall_2^
Zgradients_my_cnn_block_statefulpartitionedcall_grad_my_cnn_block_statefulpartitionedcall_3^
Zgradients_my_cnn_block_statefulpartitionedcall_grad_my_cnn_block_statefulpartitionedcall_4^
Zgradients_my_cnn_block_statefulpartitionedcall_grad_my_cnn_block_statefulpartitionedcall_5^
Zgradients_my_cnn_block_statefulpartitionedcall_grad_my_cnn_block_statefulpartitionedcall_6^
Zgradients_my_cnn_block_statefulpartitionedcall_grad_my_cnn_block_statefulpartitionedcall_7
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:?????????
?
"gradients/dense/Relu_grad/ReluGradReluGradgradients/grad_ys_0:output:0-gradients_dense_relu_grad_relugrad_dense_relu*
T0*'
_output_shapes
:?????????
?
(gradients/dense/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/dense/Relu_grad/ReluGrad:backprops:0*
T0*
_output_shapes
:
?
"gradients/dense/MatMul_grad/MatMulMatMul.gradients/dense/Relu_grad/ReluGrad:backprops:0>gradients_dense_matmul_grad_matmul_dense_matmul_readvariableop*
T0*(
_output_shapes
:??????????*
transpose_b(?
$gradients/dense/MatMul_grad/MatMul_1MatMul4gradients_dense_matmul_grad_matmul_1_flatten_reshape.gradients/dense/Relu_grad/ReluGrad:backprops:0*
T0*
_output_shapes
:	?
*
transpose_a(?
$gradients/flatten/Reshape_grad/ShapeShapeKgradients_flatten_reshape_grad_shape_my_cnn_block_1_statefulpartitionedcall*
T0*
_output_shapes
:?
&gradients/flatten/Reshape_grad/ReshapeReshape,gradients/dense/MatMul_grad/MatMul:product:0-gradients/flatten/Reshape_grad/Shape:output:0*
T0*/
_output_shapes
:?????????0?	
Egradients/my_cnn_block_1/StatefulPartitionedCall_grad/PartitionedCallPartitionedCall/gradients/flatten/Reshape_grad/Reshape:output:0\gradients_my_cnn_block_1_statefulpartitionedcall_grad_my_cnn_block_1_statefulpartitionedcall^gradients_my_cnn_block_1_statefulpartitionedcall_grad_my_cnn_block_1_statefulpartitionedcall_1^gradients_my_cnn_block_1_statefulpartitionedcall_grad_my_cnn_block_1_statefulpartitionedcall_2^gradients_my_cnn_block_1_statefulpartitionedcall_grad_my_cnn_block_1_statefulpartitionedcall_3^gradients_my_cnn_block_1_statefulpartitionedcall_grad_my_cnn_block_1_statefulpartitionedcall_4^gradients_my_cnn_block_1_statefulpartitionedcall_grad_my_cnn_block_1_statefulpartitionedcall_5^gradients_my_cnn_block_1_statefulpartitionedcall_grad_my_cnn_block_1_statefulpartitionedcall_6^gradients_my_cnn_block_1_statefulpartitionedcall_grad_my_cnn_block_1_statefulpartitionedcall_7*
Tin
2	*
Tout	
2*
_collective_manager_ids
 *_
_output_shapesM
K:?????????:0:0:00:0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference___backward_call_49686_49753?	
Cgradients/my_cnn_block/StatefulPartitionedCall_grad/PartitionedCallPartitionedCallNgradients/my_cnn_block_1/StatefulPartitionedCall_grad/PartitionedCall:output:0Xgradients_my_cnn_block_statefulpartitionedcall_grad_my_cnn_block_statefulpartitionedcallZgradients_my_cnn_block_statefulpartitionedcall_grad_my_cnn_block_statefulpartitionedcall_1Zgradients_my_cnn_block_statefulpartitionedcall_grad_my_cnn_block_statefulpartitionedcall_2Zgradients_my_cnn_block_statefulpartitionedcall_grad_my_cnn_block_statefulpartitionedcall_3Zgradients_my_cnn_block_statefulpartitionedcall_grad_my_cnn_block_statefulpartitionedcall_4Zgradients_my_cnn_block_statefulpartitionedcall_grad_my_cnn_block_statefulpartitionedcall_5Zgradients_my_cnn_block_statefulpartitionedcall_grad_my_cnn_block_statefulpartitionedcall_6Zgradients_my_cnn_block_statefulpartitionedcall_grad_my_cnn_block_statefulpartitionedcall_7*
Tin
2	*
Tout	
2*
_collective_manager_ids
 *_
_output_shapesM
K:?????????::::* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference___backward_call_49775_49842?
IdentityIdentityLgradients/my_cnn_block/StatefulPartitionedCall_grad/PartitionedCall:output:0*
T0*/
_output_shapes
:??????????

Identity_1IdentityLgradients/my_cnn_block/StatefulPartitionedCall_grad/PartitionedCall:output:1*
T0*&
_output_shapes
:?

Identity_2IdentityLgradients/my_cnn_block/StatefulPartitionedCall_grad/PartitionedCall:output:2*
T0*
_output_shapes
:?

Identity_3IdentityLgradients/my_cnn_block/StatefulPartitionedCall_grad/PartitionedCall:output:3*
T0*&
_output_shapes
:?

Identity_4IdentityLgradients/my_cnn_block/StatefulPartitionedCall_grad/PartitionedCall:output:4*
T0*
_output_shapes
:?

Identity_5IdentityNgradients/my_cnn_block_1/StatefulPartitionedCall_grad/PartitionedCall:output:1*
T0*&
_output_shapes
:0?

Identity_6IdentityNgradients/my_cnn_block_1/StatefulPartitionedCall_grad/PartitionedCall:output:2*
T0*
_output_shapes
:0?

Identity_7IdentityNgradients/my_cnn_block_1/StatefulPartitionedCall_grad/PartitionedCall:output:3*
T0*&
_output_shapes
:00?

Identity_8IdentityNgradients/my_cnn_block_1/StatefulPartitionedCall_grad/PartitionedCall:output:4*
T0*
_output_shapes
:0p

Identity_9Identity.gradients/dense/MatMul_grad/MatMul_1:product:0*
T0*
_output_shapes
:	?
o
Identity_10Identity1gradients/dense/BiasAdd_grad/BiasAddGrad:output:0*
T0*
_output_shapes
:
"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????
:?????????
:	?
:??????????:?????????0:?????????0:?????????0:?????????0:?????????0:00:?????????0:?????????:0:?????????:?????????:?????????:?????????::?????????:?????????:*/
forward_function_name__forward_call_49875:- )
'
_output_shapes
:?????????
:-)
'
_output_shapes
:?????????
:%!

_output_shapes
:	?
:.*
(
_output_shapes
:??????????:51
/
_output_shapes
:?????????0:51
/
_output_shapes
:?????????0:51
/
_output_shapes
:?????????0:51
/
_output_shapes
:?????????0:51
/
_output_shapes
:?????????0:,	(
&
_output_shapes
:00:5
1
/
_output_shapes
:?????????0:51
/
_output_shapes
:?????????:,(
&
_output_shapes
:0:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:,(
&
_output_shapes
::51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:,(
&
_output_shapes
:
?)
?
I__inference_my_autoencoder_layer_call_and_return_conditional_losses_48683
x&
my_cnn_48632:
my_cnn_48634:&
my_cnn_48636:
my_cnn_48638:&
my_cnn_48640:0
my_cnn_48642:0&
my_cnn_48644:00
my_cnn_48646:0
my_cnn_48648:	?

my_cnn_48650:
#
my_decoder_48653:	
?
my_decoder_48655:	?*
my_decoder_48657:0
my_decoder_48659:*
my_decoder_48661:
my_decoder_48663:
identity??/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp?my_cnn/StatefulPartitionedCall?"my_decoder/StatefulPartitionedCall?
my_cnn/StatefulPartitionedCallStatefulPartitionedCallxmy_cnn_48632my_cnn_48634my_cnn_48636my_cnn_48638my_cnn_48640my_cnn_48642my_cnn_48644my_cnn_48646my_cnn_48648my_cnn_48650*
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
A__inference_my_cnn_layer_call_and_return_conditional_losses_47955?
"my_decoder/StatefulPartitionedCallStatefulPartitionedCall'my_cnn/StatefulPartitionedCall:output:0my_decoder_48653my_decoder_48655my_decoder_48657my_decoder_48659my_decoder_48661my_decoder_48663*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_my_decoder_layer_call_and_return_conditional_losses_48483?
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_48632*&
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
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_48636*&
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
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_48640*&
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
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_48644*&
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
: ?
IdentityIdentity+my_decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp0^conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp^my_cnn/StatefulPartitionedCall#^my_decoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 2b
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp2@
my_cnn/StatefulPartitionedCallmy_cnn/StatefulPartitionedCall2H
"my_decoder/StatefulPartitionedCall"my_decoder/StatefulPartitionedCall:R N
/
_output_shapes
:?????????

_user_specified_namex
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
?
?
%__inference_dense_layer_call_fn_51188

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
@__inference_dense_layer_call_and_return_conditional_losses_47932o
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
?)
?
I__inference_my_autoencoder_layer_call_and_return_conditional_losses_50460
x&
my_cnn_50409:
my_cnn_50411:&
my_cnn_50413:
my_cnn_50415:&
my_cnn_50417:0
my_cnn_50419:0&
my_cnn_50421:00
my_cnn_50423:0
my_cnn_50425:	?

my_cnn_50427:
#
my_decoder_50430:	
?
my_decoder_50432:	?*
my_decoder_50434:0
my_decoder_50436:*
my_decoder_50438:
my_decoder_50440:
identity??/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp?my_cnn/StatefulPartitionedCall?"my_decoder/StatefulPartitionedCall?
my_cnn/StatefulPartitionedCallStatefulPartitionedCallxmy_cnn_50409my_cnn_50411my_cnn_50413my_cnn_50415my_cnn_50417my_cnn_50419my_cnn_50421my_cnn_50423my_cnn_50425my_cnn_50427*
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
__inference_call_49076?
"my_decoder/StatefulPartitionedCallStatefulPartitionedCall'my_cnn/StatefulPartitionedCall:output:0my_decoder_50430my_decoder_50432my_decoder_50434my_decoder_50436my_decoder_50438my_decoder_50440*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_49155?
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_50409*&
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
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_50413*&
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
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_50417*&
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
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_50421*&
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
: ?
IdentityIdentity+my_decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp0^conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2^conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp^my_cnn/StatefulPartitionedCall#^my_decoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 2b
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp2@
my_cnn/StatefulPartitionedCallmy_cnn/StatefulPartitionedCall2H
"my_decoder/StatefulPartitionedCall"my_decoder/StatefulPartitionedCall:R N
/
_output_shapes
:?????????

_user_specified_namex
?

?
?broadcast_weights_1_assert_broadcastable_AssertGuard_true_50188?
~broadcast_weights_1_assert_broadcastable_assertguard_identity_broadcast_weights_1_assert_broadcastable_is_valid_shape_identity
D
@broadcast_weights_1_assert_broadcastable_assertguard_placeholderF
Bbroadcast_weights_1_assert_broadcastable_assertguard_placeholder_1F
Bbroadcast_weights_1_assert_broadcastable_assertguard_placeholder_2
C
?broadcast_weights_1_assert_broadcastable_assertguard_identity_1
W
9broadcast_weights_1/assert_broadcastable/AssertGuard/NoOpNoOp*
_output_shapes
 ?
=broadcast_weights_1/assert_broadcastable/AssertGuard/IdentityIdentity~broadcast_weights_1_assert_broadcastable_assertguard_identity_broadcast_weights_1_assert_broadcastable_is_valid_shape_identity:^broadcast_weights_1/assert_broadcastable/AssertGuard/NoOp*
T0
*
_output_shapes
: ?
?broadcast_weights_1/assert_broadcastable/AssertGuard/Identity_1IdentityFbroadcast_weights_1/assert_broadcastable/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "?
?broadcast_weights_1_assert_broadcastable_assertguard_identity_1Hbroadcast_weights_1/assert_broadcastable/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: ::: : 

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?0
?
A__inference_my_cnn_layer_call_and_return_conditional_losses_50751
x,
my_cnn_block_50707: 
my_cnn_block_50709:,
my_cnn_block_50711: 
my_cnn_block_50713:.
my_cnn_block_1_50716:0"
my_cnn_block_1_50718:0.
my_cnn_block_1_50720:00"
my_cnn_block_1_50722:07
$dense_matmul_readvariableop_resource:	?
3
%dense_biasadd_readvariableop_resource:

identity??/conv2d/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOp?1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?$my_cnn_block/StatefulPartitionedCall?&my_cnn_block_1/StatefulPartitionedCall?
$my_cnn_block/StatefulPartitionedCallStatefulPartitionedCallxmy_cnn_block_50707my_cnn_block_50709my_cnn_block_50711my_cnn_block_50713*
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
&my_cnn_block_1/StatefulPartitionedCallStatefulPartitionedCall-my_cnn_block/StatefulPartitionedCall:output:0my_cnn_block_1_50716my_cnn_block_1_50718my_cnn_block_1_50720my_cnn_block_1_50722*
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
/conv2d/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_50707*&
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
1conv2d_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_50711*&
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
1conv2d_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_1_50716*&
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
1conv2d_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmy_cnn_block_1_50720*&
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
?
?
'__inference___backward_call_49809_49826
placeholder;
7gradients_activation_relu_grad_relugrad_activation_relu)
%gradients_conv2d_conv2d_grad_shapen_xD
@gradients_conv2d_conv2d_grad_shapen_conv2d_conv2d_readvariableop
identity

identity_1

identity_2f
gradients/grad_ys_0Identityplaceholder*
T0*/
_output_shapes
:??????????
'gradients/activation/Relu_grad/ReluGradReluGradgradients/grad_ys_0:output:07gradients_activation_relu_grad_relugrad_activation_relu*
T0*/
_output_shapes
:??????????
)gradients/conv2d/BiasAdd_grad/BiasAddGradBiasAddGrad3gradients/activation/Relu_grad/ReluGrad:backprops:0*
T0*
_output_shapes
:?
#gradients/conv2d/Conv2D_grad/ShapeNShapeN%gradients_conv2d_conv2d_grad_shapen_x@gradients_conv2d_conv2d_grad_shapen_conv2d_conv2d_readvariableop*
N*
T0* 
_output_shapes
::?
0gradients/conv2d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput,gradients/conv2d/Conv2D_grad/ShapeN:output:0@gradients_conv2d_conv2d_grad_shapen_conv2d_conv2d_readvariableop3gradients/activation/Relu_grad/ReluGrad:backprops:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
1gradients/conv2d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter%gradients_conv2d_conv2d_grad_shapen_x,gradients/conv2d/Conv2D_grad/ShapeN:output:13gradients/activation/Relu_grad/ReluGrad:backprops:0*
T0*&
_output_shapes
:*
paddingSAME*
strides
?
IdentityIdentity9gradients/conv2d/Conv2D_grad/Conv2DBackpropInput:output:0*
T0*/
_output_shapes
:??????????

Identity_1Identity:gradients/conv2d/Conv2D_grad/Conv2DBackpropFilter:output:0*
T0*&
_output_shapes
:o

Identity_2Identity2gradients/conv2d/BiasAdd_grad/BiasAddGrad:output:0*
T0*
_output_shapes
:"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:?????????:?????????:?????????:*/
forward_function_name__forward_call_49825:5 1
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:,(
&
_output_shapes
:
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
?
?
.__inference_my_cnn_block_1_layer_call_fn_51109	
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
I__inference_my_cnn_block_1_layer_call_and_return_conditional_losses_47903w
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
?

?
]broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_false_50144a
]broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_placeholderc
_broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_placeholder_1?
?broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity_broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank
^
Zbroadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity
?
Zbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/IdentityIdentity?broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity_broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank*
T0
*
_output_shapes
: "?
Zbroadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identitycbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::: :  

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?*
?
\broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_true_49320?
?broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_broadcast_weights_1_assert_broadcastable_values_shape?
?broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_1_broadcast_weights_1_assert_broadcastable_weights_shapea
]broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_placeholder
^
Zbroadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity
?
qbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
mbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDims?broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_broadcast_weights_1_assert_broadcastable_values_shapezbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:output:0*
T0*
_output_shapes

:?
?broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"      ?
rbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :?
lbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFill?broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape/shape_as_tensor:output:0{broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:output:0*
T0*
_output_shapes

:?
nbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
ibroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2vbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:output:0ubroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:output:0wbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:output:0*
N*
T0*
_output_shapes

:?
sbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
obroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDims?broadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_1_broadcast_weights_1_assert_broadcastable_weights_shape|broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:output:0*
T0*
_output_shapes

:?
{broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperationxbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:output:0rbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:output:0*
T0*<
_output_shapes*
(:?????????:?????????:*
set_operationa-b?
sbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSize?broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:result_values:0*
T0*
_output_shapes
: ?
dbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConst*
_output_shapes
: *
dtype0*
value	B : ?
bbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualmbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:output:0|broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:output:0*
T0*
_output_shapes
: ?
Zbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/IdentityIdentityfbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:z:0*
T0
*
_output_shapes
: "?
Zbroadcast_weights_1_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identitycbroadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::: :  

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?
C
'__inference_reshape_layer_call_fn_51223

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_48470h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
*__inference_my_decoder_layer_call_fn_48585
input_1
unknown:	
?
	unknown_0:	?#
	unknown_1:0
	unknown_2:#
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_my_decoder_layer_call_and_return_conditional_losses_48553w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????
: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????

!
_user_specified_name	input_1
?M
?
__forward_call_49649
x_09
&dense_1_matmul_readvariableop_resource:	
?6
'dense_1_biasadd_readvariableop_resource:	?S
9conv2d_transpose_conv2d_transpose_readvariableop_resource:0>
0conv2d_transpose_biasadd_readvariableop_resource:U
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_1_biasadd_readvariableop_resource:
identity
conv2d_transpose_biasadd6
2conv2d_transpose_1_conv2d_transpose_readvariableop
reshape_reshape4
0conv2d_transpose_conv2d_transpose_readvariableop
dense_1_biasadd!
dense_1_matmul_readvariableop
x??'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	
?*
dtype0w
dense_1/MatMulMatMulx_0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????U
reshape/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :0?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
reshape/ReshapeReshapedense_1/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????0^
conv2d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0*
dtype0?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????i
conv2d_transpose_1/ShapeShape!conv2d_transpose/BiasAdd:output:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????z
IdentityIdentity#conv2d_transpose_1/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "p
2conv2d_transpose_1_conv2d_transpose_readvariableop:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0"=
conv2d_transpose_biasadd!conv2d_transpose/BiasAdd:output:0"l
0conv2d_transpose_conv2d_transpose_readvariableop8conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0"+
dense_1_biasadddense_1/BiasAdd:output:0"F
dense_1_matmul_readvariableop%dense_1/MatMul/ReadVariableOp:value:0"
identityIdentity:output:0"+
reshape_reshapereshape/Reshape:output:0"
xx_0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????
: : : : : : *C
backward_function_name)'__inference___backward_call_49618_496502R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????


_user_specified_namex:
"__inference_internal_grad_fn_51718CustomGradient-50023"?	L
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
serving_default_input_1:0?????????D
output_18
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
encoder
	decoder

metrices
	optimizer
call
	test_step

train_step

signatures"
_tf_keras_model
?
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
 16
!17
"18
#19"
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15"
trackable_list_wrapper
 "
trackable_list_wrapper
?
$non_trainable_variables

%layers
&metrics
'layer_regularization_losses
(layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
)trace_0
*trace_1
+trace_2
,trace_32?
.__inference_my_autoencoder_layer_call_fn_48718
.__inference_my_autoencoder_layer_call_fn_50315
.__inference_my_autoencoder_layer_call_fn_50352
.__inference_my_autoencoder_layer_call_fn_48883?
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
 z)trace_0z*trace_1z+trace_2z,trace_3
?
-trace_0
.trace_1
/trace_2
0trace_32?
I__inference_my_autoencoder_layer_call_and_return_conditional_losses_50406
I__inference_my_autoencoder_layer_call_and_return_conditional_losses_50460
I__inference_my_autoencoder_layer_call_and_return_conditional_losses_48937
I__inference_my_autoencoder_layer_call_and_return_conditional_losses_48991?
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
 z-trace_0z.trace_1z/trace_2z0trace_3
?B?
 __inference__wrapped_model_47816input_1"?
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
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

7block1

8block2
9flatten
:out
;call"
_tf_keras_model
?
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses

Bdense1
Creshape

Dtrans1

Etrans2
Foutput_layer
Gcall"
_tf_keras_model
.
H0
I1"
trackable_list_wrapper
?
Jiter

Kbeta_1

Lbeta_2
	Mdecay
Nlearning_ratem?m?m?m?m?m?m?m?m?m?m?m?m?m?m?m?v?v?v?v?v?v?v?v?v?v?v?v?v?v?v?v?"
	optimizer
?
Otrace_0
Ptrace_12?
__inference_call_49170
__inference_call_49208?
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
 zOtrace_0zPtrace_1
?
Qtrace_02?
__inference_test_step_49416?
???
FullArgSpec
args?
jself
jdata
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
 zQtrace_0
?
Rtrace_02?
__inference_train_step_50239?
???
FullArgSpec
args?
jself
jdata
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
 zRtrace_0
,
Sserving_default"
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
!:	
?2dense_1/kernel
:?2dense_1/bias
1:/02conv2d_transpose/kernel
#:!2conv2d_transpose/bias
3:12conv2d_transpose_1/kernel
%:#2conv2d_transpose_1/bias
:  (2total
:  (2count
:  (2total
:  (2count
<
 0
!1
"2
#3"
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
5
T0
H1
I2"
trackable_list_wrapper
 "
trackable_list_wrapper
8
Hloss
Iaccuracy"
trackable_dict_wrapper
?B?
.__inference_my_autoencoder_layer_call_fn_48718input_1"?
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
.__inference_my_autoencoder_layer_call_fn_50315x"?
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
.__inference_my_autoencoder_layer_call_fn_50352x"?
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
.__inference_my_autoencoder_layer_call_fn_48883input_1"?
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
?B?
I__inference_my_autoencoder_layer_call_and_return_conditional_losses_50406x"?
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
?B?
I__inference_my_autoencoder_layer_call_and_return_conditional_losses_50460x"?
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
?B?
I__inference_my_autoencoder_layer_call_and_return_conditional_losses_48937input_1"?
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
?B?
I__inference_my_autoencoder_layer_call_and_return_conditional_losses_48991input_1"?
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
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
?
Ztrace_0
[trace_1
\trace_2
]trace_32?
&__inference_my_cnn_layer_call_fn_47978
&__inference_my_cnn_layer_call_fn_50679
&__inference_my_cnn_layer_call_fn_50704
&__inference_my_cnn_layer_call_fn_48257?
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
 zZtrace_0z[trace_1z\trace_2z]trace_3
?
^trace_0
_trace_1
`trace_2
atrace_32?
A__inference_my_cnn_layer_call_and_return_conditional_losses_50751
A__inference_my_cnn_layer_call_and_return_conditional_losses_50798
A__inference_my_cnn_layer_call_and_return_conditional_losses_48301
A__inference_my_cnn_layer_call_and_return_conditional_losses_48345?
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
 z^trace_0z_trace_1z`trace_2zatrace_3
?
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses
hconv_layers
ipool
jcall"
_tf_keras_layer
?
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses
qconv_layers
rpool
scall"
_tf_keras_layer
?
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses"
_tf_keras_layer
?
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
?
?trace_0
?trace_12?
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
 z?trace_0z?trace_1
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_1
?trace_2
?trace_32?
*__inference_my_decoder_layer_call_fn_48498
*__inference_my_decoder_layer_call_fn_50891
*__inference_my_decoder_layer_call_fn_50908
*__inference_my_decoder_layer_call_fn_48585?
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
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?trace_0
?trace_1
?trace_2
?trace_32?
E__inference_my_decoder_layer_call_and_return_conditional_losses_50966
E__inference_my_decoder_layer_call_and_return_conditional_losses_51024
E__inference_my_decoder_layer_call_and_return_conditional_losses_48605
E__inference_my_decoder_layer_call_and_return_conditional_losses_48625?
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
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias
!?_jit_compiled_convolution_op"
_tf_keras_layer
L
?	keras_api
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?trace_0
?trace_12?
__inference_call_50596
__inference_call_50654?
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
 z?trace_0z?trace_1
P
?	variables
?	keras_api
	 total
	!count"
_tf_keras_metric
a
?	variables
?	keras_api
	"total
	#count
?
_fn_kwargs"
_tf_keras_metric
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?B?
__inference_call_49170x"?
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
__inference_call_49208x"?
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
__inference_test_step_49416data/0data/1"?
???
FullArgSpec
args?
jself
jdata
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
__inference_train_step_50239data/0data/1"?
???
FullArgSpec
args?
jself
jdata
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
#__inference_signature_wrapper_50278input_1"?
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
R
?	variables
?	keras_api

?total

?count"
_tf_keras_metric
 "
trackable_list_wrapper
<
70
81
92
:3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
&__inference_my_cnn_layer_call_fn_47978input_1"?
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
&__inference_my_cnn_layer_call_fn_50679x"?
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
&__inference_my_cnn_layer_call_fn_50704x"?
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
&__inference_my_cnn_layer_call_fn_48257input_1"?
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
A__inference_my_cnn_layer_call_and_return_conditional_losses_50751x"?
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
A__inference_my_cnn_layer_call_and_return_conditional_losses_50798x"?
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
A__inference_my_cnn_layer_call_and_return_conditional_losses_48301input_1"?
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
A__inference_my_cnn_layer_call_and_return_conditional_losses_48345input_1"?
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
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
,__inference_my_cnn_block_layer_call_fn_51037
,__inference_my_cnn_block_layer_call_fn_51050?
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
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_51073
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_51096?
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
 z?trace_0z?trace_1
0
?0
?1"
trackable_list_wrapper
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
 z?trace_0z?trace_1
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
.__inference_my_cnn_block_1_layer_call_fn_51109
.__inference_my_cnn_block_1_layer_call_fn_51122?
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
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
I__inference_my_cnn_block_1_layer_call_and_return_conditional_losses_51145
I__inference_my_cnn_block_1_layer_call_and_return_conditional_losses_51168?
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
 z?trace_0z?trace_1
0
?0
?1"
trackable_list_wrapper
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
 z?trace_0z?trace_1
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
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
'__inference_flatten_layer_call_fn_51173?
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
B__inference_flatten_layer_call_and_return_conditional_losses_51179?
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
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
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
?
?trace_02?
%__inference_dense_layer_call_fn_51188?
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
@__inference_dense_layer_call_and_return_conditional_losses_51199?
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
 "
trackable_list_wrapper
C
B0
C1
D2
E3
F4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
*__inference_my_decoder_layer_call_fn_48498input_1"?
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
*__inference_my_decoder_layer_call_fn_50891x"?
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
*__inference_my_decoder_layer_call_fn_50908x"?
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
*__inference_my_decoder_layer_call_fn_48585input_1"?
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
E__inference_my_decoder_layer_call_and_return_conditional_losses_50966x"?
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
E__inference_my_decoder_layer_call_and_return_conditional_losses_51024x"?
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
?B?
E__inference_my_decoder_layer_call_and_return_conditional_losses_48605input_1"?
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
?B?
E__inference_my_decoder_layer_call_and_return_conditional_losses_48625input_1"?
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
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
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
?
?trace_02?
'__inference_dense_1_layer_call_fn_51208?
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
B__inference_dense_1_layer_call_and_return_conditional_losses_51218?
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
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
'__inference_reshape_layer_call_fn_51223?
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
B__inference_reshape_layer_call_and_return_conditional_losses_51237?
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
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
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
?
?trace_02?
0__inference_conv2d_transpose_layer_call_fn_51246?
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
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_51279?
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
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
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
?
?trace_02?
2__inference_conv2d_transpose_1_layer_call_fn_51288?
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
 z?trace_0
?
?trace_02?
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_51321?
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
 z?trace_0
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
_generic_user_object
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
?B?
__inference_call_50596x"?
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
__inference_call_50654x"?
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
.
 0
!1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
.
"0
#1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_list_wrapper
7
?0
?1
i2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
,__inference_my_cnn_block_layer_call_fn_51037input"?
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
,__inference_my_cnn_block_layer_call_fn_51050input"?
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
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_51073input"?
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
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_51096input"?
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
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?
conv_layer
?
activation
	?call"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?
conv_layer
?
activation
	?call"
_tf_keras_layer
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
?
?trace_02?
-__inference_max_pooling2d_layer_call_fn_51378?
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
 z?trace_0
?
?trace_02?
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_51383?
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
 z?trace_0
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
7
?0
?1
r2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
.__inference_my_cnn_block_1_layer_call_fn_51109input"?
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
.__inference_my_cnn_block_1_layer_call_fn_51122input"?
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
I__inference_my_cnn_block_1_layer_call_and_return_conditional_losses_51145input"?
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
I__inference_my_cnn_block_1_layer_call_and_return_conditional_losses_51168input"?
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
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?
conv_layer
?
activation
	?call"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?
conv_layer
?
activation
	?call"
_tf_keras_layer
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
?
?trace_02?
/__inference_max_pooling2d_1_layer_call_fn_51440?
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
 z?trace_0
?
?trace_02?
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_51445?
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
 z?trace_0
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
'__inference_flatten_layer_call_fn_51173inputs"?
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
B__inference_flatten_layer_call_and_return_conditional_losses_51179inputs"?
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
%__inference_dense_layer_call_fn_51188inputs"?
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
@__inference_dense_layer_call_and_return_conditional_losses_51199inputs"?
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
'__inference_dense_1_layer_call_fn_51208inputs"?
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
B__inference_dense_1_layer_call_and_return_conditional_losses_51218inputs"?
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
'__inference_reshape_layer_call_fn_51223inputs"?
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
B__inference_reshape_layer_call_and_return_conditional_losses_51237inputs"?
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
0__inference_conv2d_transpose_layer_call_fn_51246inputs"?
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
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_51279inputs"?
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
2__inference_conv2d_transpose_1_layer_call_fn_51288inputs"?
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
?B?
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_51321inputs"?
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
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?trace_0
?trace_12?
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
 z?trace_0z?trace_1
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?trace_0
?trace_12?
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
 z?trace_0z?trace_1
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
-__inference_max_pooling2d_layer_call_fn_51378inputs"?
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
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_51383inputs"?
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
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?trace_0
?trace_12?
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
 z?trace_0z?trace_1
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?trace_0
?trace_12?
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
 z?trace_0z?trace_1
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
/__inference_max_pooling2d_1_layer_call_fn_51440inputs"?
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
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_51445inputs"?
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
?trace_02?
__inference_loss_fn_0_51470?
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
annotations? *? z?trace_0
 "
trackable_list_wrapper
0
?0
?1"
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
?0"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?trace_02?
__inference_loss_fn_1_51479?
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
annotations? *? z?trace_0
 "
trackable_list_wrapper
0
?0
?1"
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
?0"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?trace_02?
__inference_loss_fn_2_51488?
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
annotations? *? z?trace_0
 "
trackable_list_wrapper
0
?0
?1"
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
?0"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?trace_02?
__inference_loss_fn_3_51497?
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
annotations? *? z?trace_0
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
__inference_loss_fn_0_51470"?
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
?0"
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
__inference_loss_fn_1_51479"?
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
?0"
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
__inference_loss_fn_2_51488"?
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
?0"
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
__inference_loss_fn_3_51497"?
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
?0"
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
,:*2Adam/conv2d/kernel/m
:2Adam/conv2d/bias/m
.:,2Adam/conv2d_1/kernel/m
 :2Adam/conv2d_1/bias/m
.:,02Adam/conv2d_2/kernel/m
 :02Adam/conv2d_2/bias/m
.:,002Adam/conv2d_3/kernel/m
 :02Adam/conv2d_3/bias/m
$:"	?
2Adam/dense/kernel/m
:
2Adam/dense/bias/m
&:$	
?2Adam/dense_1/kernel/m
 :?2Adam/dense_1/bias/m
6:402Adam/conv2d_transpose/kernel/m
(:&2Adam/conv2d_transpose/bias/m
8:62 Adam/conv2d_transpose_1/kernel/m
*:(2Adam/conv2d_transpose_1/bias/m
,:*2Adam/conv2d/kernel/v
:2Adam/conv2d/bias/v
.:,2Adam/conv2d_1/kernel/v
 :2Adam/conv2d_1/bias/v
.:,02Adam/conv2d_2/kernel/v
 :02Adam/conv2d_2/bias/v
.:,002Adam/conv2d_3/kernel/v
 :02Adam/conv2d_3/bias/v
$:"	?
2Adam/dense/kernel/v
:
2Adam/dense/bias/v
&:$	
?2Adam/dense_1/kernel/v
 :?2Adam/dense_1/bias/v
6:402Adam/conv2d_transpose/kernel/v
(:&2Adam/conv2d_transpose/bias/v
8:62 Adam/conv2d_transpose_1/kernel/v
*:(2Adam/conv2d_transpose_1/bias/v?
 __inference__wrapped_model_47816?8?5
.?+
)?&
input_1?????????
? ";?8
6
output_1*?'
output_1??????????
__inference_call_49170l6?3
,?)
#? 
x?????????
p
? " ???????????
__inference_call_49208l6?3
,?)
#? 
x?????????
p 
? " ??????????x
__inference_call_50507^
6?3
,?)
#? 
x?????????
p
? "??????????
x
__inference_call_50538^
6?3
,?)
#? 
x?????????
p 
? "??????????
t
__inference_call_50596Z.?+
$?!
?
x?????????

p
? " ??????????t
__inference_call_50654Z.?+
$?!
?
x?????????

p 
? " ??????????~
__inference_call_50821d:?7
0?-
'?$
input?????????
p
? " ??????????~
__inference_call_50836d:?7
0?-
'?$
input?????????
p 
? " ??????????~
__inference_call_50859d:?7
0?-
'?$
input?????????
p
? " ??????????0~
__inference_call_50874d:?7
0?-
'?$
input?????????
p 
? " ??????????0x
__inference_call_51336^6?3
,?)
#? 
x?????????
p
? " ??????????x
__inference_call_51347^6?3
,?)
#? 
x?????????
p 
? " ??????????x
__inference_call_51362^6?3
,?)
#? 
x?????????
p
? " ??????????x
__inference_call_51373^6?3
,?)
#? 
x?????????
p 
? " ??????????x
__inference_call_51398^6?3
,?)
#? 
x?????????
p
? " ??????????0x
__inference_call_51409^6?3
,?)
#? 
x?????????
p 
? " ??????????0x
__inference_call_51424^6?3
,?)
#? 
x?????????0
p
? " ??????????0x
__inference_call_51435^6?3
,?)
#? 
x?????????0
p 
? " ??????????0?
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_51321?I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
2__inference_conv2d_transpose_1_layer_call_fn_51288?I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_51279?I?F
??<
:?7
inputs+???????????????????????????0
? "??<
5?2
0+???????????????????????????
? ?
0__inference_conv2d_transpose_layer_call_fn_51246?I?F
??<
:?7
inputs+???????????????????????????0
? "2?/+????????????????????????????
B__inference_dense_1_layer_call_and_return_conditional_losses_51218]/?,
%?"
 ?
inputs?????????

? "&?#
?
0??????????
? {
'__inference_dense_1_layer_call_fn_51208P/?,
%?"
 ?
inputs?????????

? "????????????
@__inference_dense_layer_call_and_return_conditional_losses_51199]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????

? y
%__inference_dense_layer_call_fn_51188P0?-
&?#
!?
inputs??????????
? "??????????
?
B__inference_flatten_layer_call_and_return_conditional_losses_51179a7?4
-?*
(?%
inputs?????????0
? "&?#
?
0??????????
? 
'__inference_flatten_layer_call_fn_51173T7?4
-?*
(?%
inputs?????????0
? "????????????
"__inference_internal_grad_fn_51718????
???

 
'?$
result_grads_0
?
result_grads_1
'?$
result_grads_2
?
result_grads_3
'?$
result_grads_40
?
result_grads_50
'?$
result_grads_600
?
result_grads_70
 ?
result_grads_8	?

?
result_grads_9

!?
result_grads_10	
?
?
result_grads_11?
(?%
result_grads_120
?
result_grads_13
(?%
result_grads_14
?
result_grads_15
(?%
result_grads_16
?
result_grads_17
(?%
result_grads_18
?
result_grads_19
(?%
result_grads_200
?
result_grads_210
(?%
result_grads_2200
?
result_grads_230
!?
result_grads_24	?

?
result_grads_25

!?
result_grads_26	
?
?
result_grads_27?
(?%
result_grads_280
?
result_grads_29
(?%
result_grads_30
?
result_grads_31
? "???

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 
?
16
?
17
?
18
?
19
?
200
?
210
?
2200
?
230
?
24	?

?
25

?
26	
?
?
27?
?
280
?
29
?
30
?
31:
__inference_loss_fn_0_51470?

? 
? "? :
__inference_loss_fn_1_51479?

? 
? "? :
__inference_loss_fn_2_51488?

? 
? "? :
__inference_loss_fn_3_51497?

? 
? "? ?
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_51445?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_1_layer_call_fn_51440?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_51383?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
-__inference_max_pooling2d_layer_call_fn_51378?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
I__inference_my_autoencoder_layer_call_and_return_conditional_losses_48937<?9
2?/
)?&
input_1?????????
p 
? "-?*
#? 
0?????????
? ?
I__inference_my_autoencoder_layer_call_and_return_conditional_losses_48991<?9
2?/
)?&
input_1?????????
p
? "-?*
#? 
0?????????
? ?
I__inference_my_autoencoder_layer_call_and_return_conditional_losses_50406y6?3
,?)
#? 
x?????????
p 
? "-?*
#? 
0?????????
? ?
I__inference_my_autoencoder_layer_call_and_return_conditional_losses_50460y6?3
,?)
#? 
x?????????
p
? "-?*
#? 
0?????????
? ?
.__inference_my_autoencoder_layer_call_fn_48718r<?9
2?/
)?&
input_1?????????
p 
? " ???????????
.__inference_my_autoencoder_layer_call_fn_48883r<?9
2?/
)?&
input_1?????????
p
? " ???????????
.__inference_my_autoencoder_layer_call_fn_50315l6?3
,?)
#? 
x?????????
p 
? " ???????????
.__inference_my_autoencoder_layer_call_fn_50352l6?3
,?)
#? 
x?????????
p
? " ???????????
I__inference_my_cnn_block_1_layer_call_and_return_conditional_losses_51145q:?7
0?-
'?$
input?????????
p 
? "-?*
#? 
0?????????0
? ?
I__inference_my_cnn_block_1_layer_call_and_return_conditional_losses_51168q:?7
0?-
'?$
input?????????
p
? "-?*
#? 
0?????????0
? ?
.__inference_my_cnn_block_1_layer_call_fn_51109d:?7
0?-
'?$
input?????????
p 
? " ??????????0?
.__inference_my_cnn_block_1_layer_call_fn_51122d:?7
0?-
'?$
input?????????
p
? " ??????????0?
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_51073q:?7
0?-
'?$
input?????????
p 
? "-?*
#? 
0?????????
? ?
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_51096q:?7
0?-
'?$
input?????????
p
? "-?*
#? 
0?????????
? ?
,__inference_my_cnn_block_layer_call_fn_51037d:?7
0?-
'?$
input?????????
p 
? " ???????????
,__inference_my_cnn_block_layer_call_fn_51050d:?7
0?-
'?$
input?????????
p
? " ???????????
A__inference_my_cnn_layer_call_and_return_conditional_losses_48301q
<?9
2?/
)?&
input_1?????????
p 
? "%?"
?
0?????????

? ?
A__inference_my_cnn_layer_call_and_return_conditional_losses_48345q
<?9
2?/
)?&
input_1?????????
p
? "%?"
?
0?????????

? ?
A__inference_my_cnn_layer_call_and_return_conditional_losses_50751k
6?3
,?)
#? 
x?????????
p 
? "%?"
?
0?????????

? ?
A__inference_my_cnn_layer_call_and_return_conditional_losses_50798k
6?3
,?)
#? 
x?????????
p
? "%?"
?
0?????????

? ?
&__inference_my_cnn_layer_call_fn_47978d
<?9
2?/
)?&
input_1?????????
p 
? "??????????
?
&__inference_my_cnn_layer_call_fn_48257d
<?9
2?/
)?&
input_1?????????
p
? "??????????
?
&__inference_my_cnn_layer_call_fn_50679^
6?3
,?)
#? 
x?????????
p 
? "??????????
?
&__inference_my_cnn_layer_call_fn_50704^
6?3
,?)
#? 
x?????????
p
? "??????????
?
E__inference_my_decoder_layer_call_and_return_conditional_losses_48605m4?1
*?'
!?
input_1?????????

p 
? "-?*
#? 
0?????????
? ?
E__inference_my_decoder_layer_call_and_return_conditional_losses_48625m4?1
*?'
!?
input_1?????????

p
? "-?*
#? 
0?????????
? ?
E__inference_my_decoder_layer_call_and_return_conditional_losses_50966g.?+
$?!
?
x?????????

p 
? "-?*
#? 
0?????????
? ?
E__inference_my_decoder_layer_call_and_return_conditional_losses_51024g.?+
$?!
?
x?????????

p
? "-?*
#? 
0?????????
? ?
*__inference_my_decoder_layer_call_fn_48498`4?1
*?'
!?
input_1?????????

p 
? " ???????????
*__inference_my_decoder_layer_call_fn_48585`4?1
*?'
!?
input_1?????????

p
? " ???????????
*__inference_my_decoder_layer_call_fn_50891Z.?+
$?!
?
x?????????

p 
? " ???????????
*__inference_my_decoder_layer_call_fn_50908Z.?+
$?!
?
x?????????

p
? " ???????????
B__inference_reshape_layer_call_and_return_conditional_losses_51237a0?-
&?#
!?
inputs??????????
? "-?*
#? 
0?????????0
? 
'__inference_reshape_layer_call_fn_51223T0?-
&?#
!?
inputs??????????
? " ??????????0?
#__inference_signature_wrapper_50278?C?@
? 
9?6
4
input_1)?&
input_1?????????";?8
6
output_1*?'
output_1??????????
__inference_test_step_49416??? !"#f?c
\?Y
W?T
(?%
data/0?????????
(?%
data/1?????????
? "9?6

accuracy?
accuracy 

loss?

loss ?
__inference_train_step_50239?\??NJKL???????????????????????????????? !"#f?c
\?Y
W?T
(?%
data/0?????????
(?%
data/1?????????
? "9?6

accuracy?
accuracy 

loss?

loss 