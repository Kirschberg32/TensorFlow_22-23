û¬%
­
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
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

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
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
®
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
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

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
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

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
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
0
Sigmoid
x"T
y"T"
Ttype:

2
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
÷
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
-
Tanh
x"T
y"T"
Ttype:

2
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.10.02unknown8Þ·"

Adam/dense_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_14/bias/v
y
(Adam/dense_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/v*
_output_shapes
:*
dtype0

Adam/dense_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_14/kernel/v

*Adam/dense_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/v*
_output_shapes

:*
dtype0
ª
)Adam/rnn_2/my_lstm_cell_2/dense_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)Adam/rnn_2/my_lstm_cell_2/dense_13/bias/v
£
=Adam/rnn_2/my_lstm_cell_2/dense_13/bias/v/Read/ReadVariableOpReadVariableOp)Adam/rnn_2/my_lstm_cell_2/dense_13/bias/v*
_output_shapes
:*
dtype0
²
+Adam/rnn_2/my_lstm_cell_2/dense_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:'*<
shared_name-+Adam/rnn_2/my_lstm_cell_2/dense_13/kernel/v
«
?Adam/rnn_2/my_lstm_cell_2/dense_13/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/rnn_2/my_lstm_cell_2/dense_13/kernel/v*
_output_shapes

:'*
dtype0
ª
)Adam/rnn_2/my_lstm_cell_2/dense_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)Adam/rnn_2/my_lstm_cell_2/dense_12/bias/v
£
=Adam/rnn_2/my_lstm_cell_2/dense_12/bias/v/Read/ReadVariableOpReadVariableOp)Adam/rnn_2/my_lstm_cell_2/dense_12/bias/v*
_output_shapes
:*
dtype0
²
+Adam/rnn_2/my_lstm_cell_2/dense_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:'*<
shared_name-+Adam/rnn_2/my_lstm_cell_2/dense_12/kernel/v
«
?Adam/rnn_2/my_lstm_cell_2/dense_12/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/rnn_2/my_lstm_cell_2/dense_12/kernel/v*
_output_shapes

:'*
dtype0
ª
)Adam/rnn_2/my_lstm_cell_2/dense_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)Adam/rnn_2/my_lstm_cell_2/dense_11/bias/v
£
=Adam/rnn_2/my_lstm_cell_2/dense_11/bias/v/Read/ReadVariableOpReadVariableOp)Adam/rnn_2/my_lstm_cell_2/dense_11/bias/v*
_output_shapes
:*
dtype0
²
+Adam/rnn_2/my_lstm_cell_2/dense_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:'*<
shared_name-+Adam/rnn_2/my_lstm_cell_2/dense_11/kernel/v
«
?Adam/rnn_2/my_lstm_cell_2/dense_11/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/rnn_2/my_lstm_cell_2/dense_11/kernel/v*
_output_shapes

:'*
dtype0
ª
)Adam/rnn_2/my_lstm_cell_2/dense_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)Adam/rnn_2/my_lstm_cell_2/dense_10/bias/v
£
=Adam/rnn_2/my_lstm_cell_2/dense_10/bias/v/Read/ReadVariableOpReadVariableOp)Adam/rnn_2/my_lstm_cell_2/dense_10/bias/v*
_output_shapes
:*
dtype0
²
+Adam/rnn_2/my_lstm_cell_2/dense_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:'*<
shared_name-+Adam/rnn_2/my_lstm_cell_2/dense_10/kernel/v
«
?Adam/rnn_2/my_lstm_cell_2/dense_10/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/rnn_2/my_lstm_cell_2/dense_10/kernel/v*
_output_shapes

:'*
dtype0

Adam/conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_5/bias/v
y
(Adam/conv2d_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_5/kernel/v

*Adam/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_4/bias/v
y
(Adam/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_4/kernel/v

*Adam/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/v*&
_output_shapes
:*
dtype0

Adam/dense_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_14/bias/m
y
(Adam/dense_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/m*
_output_shapes
:*
dtype0

Adam/dense_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_14/kernel/m

*Adam/dense_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/m*
_output_shapes

:*
dtype0
ª
)Adam/rnn_2/my_lstm_cell_2/dense_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)Adam/rnn_2/my_lstm_cell_2/dense_13/bias/m
£
=Adam/rnn_2/my_lstm_cell_2/dense_13/bias/m/Read/ReadVariableOpReadVariableOp)Adam/rnn_2/my_lstm_cell_2/dense_13/bias/m*
_output_shapes
:*
dtype0
²
+Adam/rnn_2/my_lstm_cell_2/dense_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:'*<
shared_name-+Adam/rnn_2/my_lstm_cell_2/dense_13/kernel/m
«
?Adam/rnn_2/my_lstm_cell_2/dense_13/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/rnn_2/my_lstm_cell_2/dense_13/kernel/m*
_output_shapes

:'*
dtype0
ª
)Adam/rnn_2/my_lstm_cell_2/dense_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)Adam/rnn_2/my_lstm_cell_2/dense_12/bias/m
£
=Adam/rnn_2/my_lstm_cell_2/dense_12/bias/m/Read/ReadVariableOpReadVariableOp)Adam/rnn_2/my_lstm_cell_2/dense_12/bias/m*
_output_shapes
:*
dtype0
²
+Adam/rnn_2/my_lstm_cell_2/dense_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:'*<
shared_name-+Adam/rnn_2/my_lstm_cell_2/dense_12/kernel/m
«
?Adam/rnn_2/my_lstm_cell_2/dense_12/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/rnn_2/my_lstm_cell_2/dense_12/kernel/m*
_output_shapes

:'*
dtype0
ª
)Adam/rnn_2/my_lstm_cell_2/dense_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)Adam/rnn_2/my_lstm_cell_2/dense_11/bias/m
£
=Adam/rnn_2/my_lstm_cell_2/dense_11/bias/m/Read/ReadVariableOpReadVariableOp)Adam/rnn_2/my_lstm_cell_2/dense_11/bias/m*
_output_shapes
:*
dtype0
²
+Adam/rnn_2/my_lstm_cell_2/dense_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:'*<
shared_name-+Adam/rnn_2/my_lstm_cell_2/dense_11/kernel/m
«
?Adam/rnn_2/my_lstm_cell_2/dense_11/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/rnn_2/my_lstm_cell_2/dense_11/kernel/m*
_output_shapes

:'*
dtype0
ª
)Adam/rnn_2/my_lstm_cell_2/dense_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)Adam/rnn_2/my_lstm_cell_2/dense_10/bias/m
£
=Adam/rnn_2/my_lstm_cell_2/dense_10/bias/m/Read/ReadVariableOpReadVariableOp)Adam/rnn_2/my_lstm_cell_2/dense_10/bias/m*
_output_shapes
:*
dtype0
²
+Adam/rnn_2/my_lstm_cell_2/dense_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:'*<
shared_name-+Adam/rnn_2/my_lstm_cell_2/dense_10/kernel/m
«
?Adam/rnn_2/my_lstm_cell_2/dense_10/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/rnn_2/my_lstm_cell_2/dense_10/kernel/m*
_output_shapes

:'*
dtype0

Adam/conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_5/bias/m
y
(Adam/conv2d_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_5/kernel/m

*Adam/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_4/bias/m
y
(Adam/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_4/kernel/m

*Adam/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/m*&
_output_shapes
:*
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
r
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_14/bias
k
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
_output_shapes
:*
dtype0
z
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_14/kernel
s
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel*
_output_shapes

:*
dtype0

"rnn_2/my_lstm_cell_2/dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"rnn_2/my_lstm_cell_2/dense_13/bias

6rnn_2/my_lstm_cell_2/dense_13/bias/Read/ReadVariableOpReadVariableOp"rnn_2/my_lstm_cell_2/dense_13/bias*
_output_shapes
:*
dtype0
¤
$rnn_2/my_lstm_cell_2/dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:'*5
shared_name&$rnn_2/my_lstm_cell_2/dense_13/kernel

8rnn_2/my_lstm_cell_2/dense_13/kernel/Read/ReadVariableOpReadVariableOp$rnn_2/my_lstm_cell_2/dense_13/kernel*
_output_shapes

:'*
dtype0

"rnn_2/my_lstm_cell_2/dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"rnn_2/my_lstm_cell_2/dense_12/bias

6rnn_2/my_lstm_cell_2/dense_12/bias/Read/ReadVariableOpReadVariableOp"rnn_2/my_lstm_cell_2/dense_12/bias*
_output_shapes
:*
dtype0
¤
$rnn_2/my_lstm_cell_2/dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:'*5
shared_name&$rnn_2/my_lstm_cell_2/dense_12/kernel

8rnn_2/my_lstm_cell_2/dense_12/kernel/Read/ReadVariableOpReadVariableOp$rnn_2/my_lstm_cell_2/dense_12/kernel*
_output_shapes

:'*
dtype0

"rnn_2/my_lstm_cell_2/dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"rnn_2/my_lstm_cell_2/dense_11/bias

6rnn_2/my_lstm_cell_2/dense_11/bias/Read/ReadVariableOpReadVariableOp"rnn_2/my_lstm_cell_2/dense_11/bias*
_output_shapes
:*
dtype0
¤
$rnn_2/my_lstm_cell_2/dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:'*5
shared_name&$rnn_2/my_lstm_cell_2/dense_11/kernel

8rnn_2/my_lstm_cell_2/dense_11/kernel/Read/ReadVariableOpReadVariableOp$rnn_2/my_lstm_cell_2/dense_11/kernel*
_output_shapes

:'*
dtype0

"rnn_2/my_lstm_cell_2/dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"rnn_2/my_lstm_cell_2/dense_10/bias

6rnn_2/my_lstm_cell_2/dense_10/bias/Read/ReadVariableOpReadVariableOp"rnn_2/my_lstm_cell_2/dense_10/bias*
_output_shapes
:*
dtype0
¤
$rnn_2/my_lstm_cell_2/dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:'*5
shared_name&$rnn_2/my_lstm_cell_2/dense_10/kernel

8rnn_2/my_lstm_cell_2/dense_10/kernel/Read/ReadVariableOpReadVariableOp$rnn_2/my_lstm_cell_2/dense_10/kernel*
_output_shapes

:'*
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
:*
dtype0

conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
:*
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
:*
dtype0

conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:*
dtype0

serving_default_input_1Placeholder*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ*
dtype0*(
shape:ÿÿÿÿÿÿÿÿÿ
Ù
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/bias$rnn_2/my_lstm_cell_2/dense_10/kernel"rnn_2/my_lstm_cell_2/dense_10/bias$rnn_2/my_lstm_cell_2/dense_11/kernel"rnn_2/my_lstm_cell_2/dense_11/bias$rnn_2/my_lstm_cell_2/dense_12/kernel"rnn_2/my_lstm_cell_2/dense_12/bias$rnn_2/my_lstm_cell_2/dense_13/kernel"rnn_2/my_lstm_cell_2/dense_13/biasdense_14/kerneldense_14/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_52680

NoOpNoOp

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ö
valueËBÇ B¿
Ó
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
conv_block1
	global_pooling

timedistributed
	lstm_cell

rnn_buffer
output_layer
metrics_list
	optimizer
call

signatures*

0
1
2
3
4
5
6
7
8
9
10
11
12
13
 14
!15
"16
#17*
j
0
1
2
3
4
5
6
7
8
9
10
11
12
13*
* 
°
$non_trainable_variables

%layers
metrics
&layer_regularization_losses
'layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
(trace_0
)trace_1
*trace_2
+trace_3* 
6
,trace_0
-trace_1
.trace_2
/trace_3* 
* 
«
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6conv_layers
7call*

8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses* 

>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses
		layer* 
¤
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses
JinputConcat
Ksigmoid1_layer
L
multiplier
Msigmoid2_layer
N
tanh_layer
	Oadder
Psigmoid3_layer
Qtanh
Rlayer_norm_2*
ª
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses
cell
Y
state_spec*
¦
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses

kernel
bias*

`0
a1*
Ü
biter

cbeta_1

dbeta_2
	edecay
flearning_ratemmmmmmmm m¡m¢m£m¤m¥m¦v§v¨v©vªv«v¬v­v®v¯v°v±v²v³v´*

gtrace_0
htrace_1* 

iserving_default* 
OI
VARIABLE_VALUEconv2d_4/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_4/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_5/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_5/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE$rnn_2/my_lstm_cell_2/dense_10/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"rnn_2/my_lstm_cell_2/dense_10/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE$rnn_2/my_lstm_cell_2/dense_11/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"rnn_2/my_lstm_cell_2/dense_11/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE$rnn_2/my_lstm_cell_2/dense_12/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"rnn_2/my_lstm_cell_2/dense_12/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE$rnn_2/my_lstm_cell_2/dense_13/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"rnn_2/my_lstm_cell_2/dense_13/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_14/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_14/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
HB
VARIABLE_VALUEtotal_1'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
HB
VARIABLE_VALUEcount_1'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
F@
VARIABLE_VALUEtotal'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
F@
VARIABLE_VALUEcount'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
 
 0
!1
"2
#3*
.
0
	1

2
3
4
5*
* 

`loss
aaccuracy*
* 
* 
* 
* 
* 
* 
* 
* 
 
0
1
2
3*
 
0
1
2
3*
* 

jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

otrace_0
ptrace_1* 

qtrace_0
rtrace_1* 

s0
t1*

utrace_0
vtrace_1* 
* 
* 
* 

wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses* 

|trace_0* 

}trace_0* 
* 
* 
* 

~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 
<
0
1
2
3
4
5
6
7*
<
0
1
2
3
4
5
6
7*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses*

trace_0* 

trace_0* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
¬
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

kernel
bias*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
¬
 	variables
¡trainable_variables
¢regularization_losses
£	keras_api
¤__call__
+¥&call_and_return_all_conditional_losses

kernel
bias*
¬
¦	variables
§trainable_variables
¨regularization_losses
©	keras_api
ª__call__
+«&call_and_return_all_conditional_losses

kernel
bias*

¬	variables
­trainable_variables
®regularization_losses
¯	keras_api
°__call__
+±&call_and_return_all_conditional_losses* 
¬
²	variables
³trainable_variables
´regularization_losses
µ	keras_api
¶__call__
+·&call_and_return_all_conditional_losses

kernel
bias*

¸	variables
¹trainable_variables
ºregularization_losses
»	keras_api
¼__call__
+½&call_and_return_all_conditional_losses* 

¾	keras_api* 
<
0
1
2
3
4
5
6
7*
<
0
1
2
3
4
5
6
7*
* 
¥
¿states
Ànon_trainable_variables
Álayers
Âmetrics
 Ãlayer_regularization_losses
Älayer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*
:
Åtrace_0
Ætrace_1
Çtrace_2
Ètrace_3* 
:
Étrace_0
Êtrace_1
Ëtrace_2
Ìtrace_3* 
* 

0
1*

0
1*
* 

Ínon_trainable_variables
Îlayers
Ïmetrics
 Ðlayer_regularization_losses
Ñlayer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses*

Òtrace_0* 

Ótrace_0* 
:
Ô	variables
Õ	keras_api
	 total
	!count*
K
Ö	variables
×	keras_api
	"total
	#count
Ø
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

s0
t1*
* 
* 
* 
* 
* 
* 
* 
Ï
Ù	variables
Útrainable_variables
Ûregularization_losses
Ü	keras_api
Ý__call__
+Þ&call_and_return_all_conditional_losses

kernel
bias
!ß_jit_compiled_convolution_op*
Ï
à	variables
átrainable_variables
âregularization_losses
ã	keras_api
ä__call__
+å&call_and_return_all_conditional_losses

kernel
bias
!æ_jit_compiled_convolution_op*
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
	
	0* 
* 
* 
* 
* 
* 
* 
* 
* 
C
J0
K1
L2
M3
N4
O5
P6
Q7
R8*
* 
* 
* 
* 
* 
* 
* 
* 

çnon_trainable_variables
èlayers
émetrics
 êlayer_regularization_losses
ëlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 

0
1*

0
1*
* 

ìnon_trainable_variables
ílayers
îmetrics
 ïlayer_regularization_losses
ðlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ñnon_trainable_variables
òlayers
ómetrics
 ôlayer_regularization_losses
õlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 

0
1*

0
1*
* 

önon_trainable_variables
÷layers
ømetrics
 ùlayer_regularization_losses
úlayer_metrics
 	variables
¡trainable_variables
¢regularization_losses
¤__call__
+¥&call_and_return_all_conditional_losses
'¥"call_and_return_conditional_losses*
* 
* 

0
1*

0
1*
* 

ûnon_trainable_variables
ülayers
ýmetrics
 þlayer_regularization_losses
ÿlayer_metrics
¦	variables
§trainable_variables
¨regularization_losses
ª__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¬	variables
­trainable_variables
®regularization_losses
°__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses* 
* 
* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
²	variables
³trainable_variables
´regularization_losses
¶__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¸	variables
¹trainable_variables
ºregularization_losses
¼__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

0*
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
 0
!1*

Ô	variables*

"0
#1*

Ö	variables*
* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ù	variables
Útrainable_variables
Ûregularization_losses
Ý__call__
+Þ&call_and_return_all_conditional_losses
'Þ"call_and_return_conditional_losses*
* 
* 
* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
à	variables
átrainable_variables
âregularization_losses
ä__call__
+å&call_and_return_all_conditional_losses
'å"call_and_return_conditional_losses*
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
* 
* 
* 
* 
* 
* 
* 
* 
* 
rl
VARIABLE_VALUEAdam/conv2d_4/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_4/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_5/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_5/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/rnn_2/my_lstm_cell_2/dense_10/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/rnn_2/my_lstm_cell_2/dense_10/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/rnn_2/my_lstm_cell_2/dense_11/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/rnn_2/my_lstm_cell_2/dense_11/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/rnn_2/my_lstm_cell_2/dense_12/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/rnn_2/my_lstm_cell_2/dense_12/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/rnn_2/my_lstm_cell_2/dense_13/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/rnn_2/my_lstm_cell_2/dense_13/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/dense_14/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_14/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_4/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_4/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_5/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_5/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/rnn_2/my_lstm_cell_2/dense_10/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/rnn_2/my_lstm_cell_2/dense_10/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/rnn_2/my_lstm_cell_2/dense_11/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/rnn_2/my_lstm_cell_2/dense_11/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/rnn_2/my_lstm_cell_2/dense_12/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/rnn_2/my_lstm_cell_2/dense_12/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/rnn_2/my_lstm_cell_2/dense_13/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/rnn_2/my_lstm_cell_2/dense_13/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/dense_14/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_14/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp8rnn_2/my_lstm_cell_2/dense_10/kernel/Read/ReadVariableOp6rnn_2/my_lstm_cell_2/dense_10/bias/Read/ReadVariableOp8rnn_2/my_lstm_cell_2/dense_11/kernel/Read/ReadVariableOp6rnn_2/my_lstm_cell_2/dense_11/bias/Read/ReadVariableOp8rnn_2/my_lstm_cell_2/dense_12/kernel/Read/ReadVariableOp6rnn_2/my_lstm_cell_2/dense_12/bias/Read/ReadVariableOp8rnn_2/my_lstm_cell_2/dense_13/kernel/Read/ReadVariableOp6rnn_2/my_lstm_cell_2/dense_13/bias/Read/ReadVariableOp#dense_14/kernel/Read/ReadVariableOp!dense_14/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp*Adam/conv2d_4/kernel/m/Read/ReadVariableOp(Adam/conv2d_4/bias/m/Read/ReadVariableOp*Adam/conv2d_5/kernel/m/Read/ReadVariableOp(Adam/conv2d_5/bias/m/Read/ReadVariableOp?Adam/rnn_2/my_lstm_cell_2/dense_10/kernel/m/Read/ReadVariableOp=Adam/rnn_2/my_lstm_cell_2/dense_10/bias/m/Read/ReadVariableOp?Adam/rnn_2/my_lstm_cell_2/dense_11/kernel/m/Read/ReadVariableOp=Adam/rnn_2/my_lstm_cell_2/dense_11/bias/m/Read/ReadVariableOp?Adam/rnn_2/my_lstm_cell_2/dense_12/kernel/m/Read/ReadVariableOp=Adam/rnn_2/my_lstm_cell_2/dense_12/bias/m/Read/ReadVariableOp?Adam/rnn_2/my_lstm_cell_2/dense_13/kernel/m/Read/ReadVariableOp=Adam/rnn_2/my_lstm_cell_2/dense_13/bias/m/Read/ReadVariableOp*Adam/dense_14/kernel/m/Read/ReadVariableOp(Adam/dense_14/bias/m/Read/ReadVariableOp*Adam/conv2d_4/kernel/v/Read/ReadVariableOp(Adam/conv2d_4/bias/v/Read/ReadVariableOp*Adam/conv2d_5/kernel/v/Read/ReadVariableOp(Adam/conv2d_5/bias/v/Read/ReadVariableOp?Adam/rnn_2/my_lstm_cell_2/dense_10/kernel/v/Read/ReadVariableOp=Adam/rnn_2/my_lstm_cell_2/dense_10/bias/v/Read/ReadVariableOp?Adam/rnn_2/my_lstm_cell_2/dense_11/kernel/v/Read/ReadVariableOp=Adam/rnn_2/my_lstm_cell_2/dense_11/bias/v/Read/ReadVariableOp?Adam/rnn_2/my_lstm_cell_2/dense_12/kernel/v/Read/ReadVariableOp=Adam/rnn_2/my_lstm_cell_2/dense_12/bias/v/Read/ReadVariableOp?Adam/rnn_2/my_lstm_cell_2/dense_13/kernel/v/Read/ReadVariableOp=Adam/rnn_2/my_lstm_cell_2/dense_13/bias/v/Read/ReadVariableOp*Adam/dense_14/kernel/v/Read/ReadVariableOp(Adam/dense_14/bias/v/Read/ReadVariableOpConst*@
Tin9
725	*
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
GPU 2J 8 *'
f"R 
__inference__traced_save_54566

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/bias$rnn_2/my_lstm_cell_2/dense_10/kernel"rnn_2/my_lstm_cell_2/dense_10/bias$rnn_2/my_lstm_cell_2/dense_11/kernel"rnn_2/my_lstm_cell_2/dense_11/bias$rnn_2/my_lstm_cell_2/dense_12/kernel"rnn_2/my_lstm_cell_2/dense_12/bias$rnn_2/my_lstm_cell_2/dense_13/kernel"rnn_2/my_lstm_cell_2/dense_13/biasdense_14/kerneldense_14/biastotal_1count_1totalcount	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateAdam/conv2d_4/kernel/mAdam/conv2d_4/bias/mAdam/conv2d_5/kernel/mAdam/conv2d_5/bias/m+Adam/rnn_2/my_lstm_cell_2/dense_10/kernel/m)Adam/rnn_2/my_lstm_cell_2/dense_10/bias/m+Adam/rnn_2/my_lstm_cell_2/dense_11/kernel/m)Adam/rnn_2/my_lstm_cell_2/dense_11/bias/m+Adam/rnn_2/my_lstm_cell_2/dense_12/kernel/m)Adam/rnn_2/my_lstm_cell_2/dense_12/bias/m+Adam/rnn_2/my_lstm_cell_2/dense_13/kernel/m)Adam/rnn_2/my_lstm_cell_2/dense_13/bias/mAdam/dense_14/kernel/mAdam/dense_14/bias/mAdam/conv2d_4/kernel/vAdam/conv2d_4/bias/vAdam/conv2d_5/kernel/vAdam/conv2d_5/bias/v+Adam/rnn_2/my_lstm_cell_2/dense_10/kernel/v)Adam/rnn_2/my_lstm_cell_2/dense_10/bias/v+Adam/rnn_2/my_lstm_cell_2/dense_11/kernel/v)Adam/rnn_2/my_lstm_cell_2/dense_11/bias/v+Adam/rnn_2/my_lstm_cell_2/dense_12/kernel/v)Adam/rnn_2/my_lstm_cell_2/dense_12/bias/v+Adam/rnn_2/my_lstm_cell_2/dense_13/kernel/v)Adam/rnn_2/my_lstm_cell_2/dense_13/bias/vAdam/dense_14/kernel/vAdam/dense_14/bias/v*?
Tin8
624*
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
GPU 2J 8 **
f%R#
!__inference__traced_restore_54729ã 
ôg
¯
@__inference_rnn_2_layer_call_and_return_conditional_losses_54177

inputsH
6my_lstm_cell_2_dense_10_matmul_readvariableop_resource:'E
7my_lstm_cell_2_dense_10_biasadd_readvariableop_resource:H
6my_lstm_cell_2_dense_11_matmul_readvariableop_resource:'E
7my_lstm_cell_2_dense_11_biasadd_readvariableop_resource:H
6my_lstm_cell_2_dense_12_matmul_readvariableop_resource:'E
7my_lstm_cell_2_dense_12_biasadd_readvariableop_resource:H
6my_lstm_cell_2_dense_13_matmul_readvariableop_resource:'E
7my_lstm_cell_2_dense_13_biasadd_readvariableop_resource:
identity¢.my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp¢-my_lstm_cell_2/dense_10/MatMul/ReadVariableOp¢.my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp¢-my_lstm_cell_2/dense_11/MatMul/ReadVariableOp¢.my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp¢-my_lstm_cell_2/dense_12/MatMul/ReadVariableOp¢.my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp¢-my_lstm_cell_2/dense_13/MatMul/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskj
(my_lstm_cell_2/concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ç
#my_lstm_cell_2/concatenate_5/concatConcatV2strided_slice_2:output:0zeros:output:01my_lstm_cell_2/concatenate_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'¤
-my_lstm_cell_2/dense_10/MatMul/ReadVariableOpReadVariableOp6my_lstm_cell_2_dense_10_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0¿
my_lstm_cell_2/dense_10/MatMulMatMul,my_lstm_cell_2/concatenate_5/concat:output:05my_lstm_cell_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOpReadVariableOp7my_lstm_cell_2_dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
my_lstm_cell_2/dense_10/BiasAddBiasAdd(my_lstm_cell_2/dense_10/MatMul:product:06my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
my_lstm_cell_2/dense_10/SigmoidSigmoid(my_lstm_cell_2/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
my_lstm_cell_2/multiply_2/mulMulzeros_1:output:0#my_lstm_cell_2/dense_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-my_lstm_cell_2/dense_11/MatMul/ReadVariableOpReadVariableOp6my_lstm_cell_2_dense_11_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0¿
my_lstm_cell_2/dense_11/MatMulMatMul,my_lstm_cell_2/concatenate_5/concat:output:05my_lstm_cell_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOpReadVariableOp7my_lstm_cell_2_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
my_lstm_cell_2/dense_11/BiasAddBiasAdd(my_lstm_cell_2/dense_11/MatMul:product:06my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
my_lstm_cell_2/dense_11/SigmoidSigmoid(my_lstm_cell_2/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-my_lstm_cell_2/dense_12/MatMul/ReadVariableOpReadVariableOp6my_lstm_cell_2_dense_12_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0¿
my_lstm_cell_2/dense_12/MatMulMatMul,my_lstm_cell_2/concatenate_5/concat:output:05my_lstm_cell_2/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOpReadVariableOp7my_lstm_cell_2_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
my_lstm_cell_2/dense_12/BiasAddBiasAdd(my_lstm_cell_2/dense_12/MatMul:product:06my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
my_lstm_cell_2/dense_12/TanhTanh(my_lstm_cell_2/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
my_lstm_cell_2/multiply_2/mul_1Mul#my_lstm_cell_2/dense_11/Sigmoid:y:0 my_lstm_cell_2/dense_12/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
my_lstm_cell_2/add_5/addAddV2!my_lstm_cell_2/multiply_2/mul:z:0#my_lstm_cell_2/multiply_2/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-my_lstm_cell_2/dense_13/MatMul/ReadVariableOpReadVariableOp6my_lstm_cell_2_dense_13_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0¿
my_lstm_cell_2/dense_13/MatMulMatMul,my_lstm_cell_2/concatenate_5/concat:output:05my_lstm_cell_2/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOpReadVariableOp7my_lstm_cell_2_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
my_lstm_cell_2/dense_13/BiasAddBiasAdd(my_lstm_cell_2/dense_13/MatMul:product:06my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
my_lstm_cell_2/dense_13/SigmoidSigmoid(my_lstm_cell_2/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
 my_lstm_cell_2/activation_2/TanhTanhmy_lstm_cell_2/add_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
my_lstm_cell_2/multiply_2/mul_2Mul$my_lstm_cell_2/activation_2/Tanh:y:0#my_lstm_cell_2/dense_13/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ø
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:06my_lstm_cell_2_dense_10_matmul_readvariableop_resource7my_lstm_cell_2_dense_10_biasadd_readvariableop_resource6my_lstm_cell_2_dense_11_matmul_readvariableop_resource7my_lstm_cell_2_dense_11_biasadd_readvariableop_resource6my_lstm_cell_2_dense_12_matmul_readvariableop_resource7my_lstm_cell_2_dense_12_biasadd_readvariableop_resource6my_lstm_cell_2_dense_13_matmul_readvariableop_resource7my_lstm_cell_2_dense_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( *
bodyR
while_body_54073*
condR
while_cond_54072*U
output_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
NoOpNoOp/^my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp.^my_lstm_cell_2/dense_10/MatMul/ReadVariableOp/^my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp.^my_lstm_cell_2/dense_11/MatMul/ReadVariableOp/^my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp.^my_lstm_cell_2/dense_12/MatMul/ReadVariableOp/^my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp.^my_lstm_cell_2/dense_13/MatMul/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2`
.my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp.my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp2^
-my_lstm_cell_2/dense_10/MatMul/ReadVariableOp-my_lstm_cell_2/dense_10/MatMul/ReadVariableOp2`
.my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp.my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp2^
-my_lstm_cell_2/dense_11/MatMul/ReadVariableOp-my_lstm_cell_2/dense_11/MatMul/ReadVariableOp2`
.my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp.my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp2^
-my_lstm_cell_2/dense_12/MatMul/ReadVariableOp-my_lstm_cell_2/dense_12/MatMul/ReadVariableOp2`
.my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp.my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp2^
-my_lstm_cell_2/dense_13/MatMul/ReadVariableOp-my_lstm_cell_2/dense_13/MatMul/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ	
´
%__inference_rnn_2_layer_call_fn_53634

inputs
unknown:'
	unknown_0:
	unknown_1:'
	unknown_2:
	unknown_3:'
	unknown_4:
	unknown_5:'
	unknown_6:
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_rnn_2_layer_call_and_return_conditional_losses_51565s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 
i
M__inference_time_distributed_2_layer_call_and_return_conditional_losses_50941

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿå
*global_average_pooling2d_2/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_50897\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿS
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:¢
	Reshape_1Reshape3global_average_pooling2d_2/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿg
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±
û
rnn_2_while_cond_53049(
$rnn_2_while_rnn_2_while_loop_counter.
*rnn_2_while_rnn_2_while_maximum_iterations
rnn_2_while_placeholder
rnn_2_while_placeholder_1
rnn_2_while_placeholder_2
rnn_2_while_placeholder_3*
&rnn_2_while_less_rnn_2_strided_slice_1?
;rnn_2_while_rnn_2_while_cond_53049___redundant_placeholder0?
;rnn_2_while_rnn_2_while_cond_53049___redundant_placeholder1?
;rnn_2_while_rnn_2_while_cond_53049___redundant_placeholder2?
;rnn_2_while_rnn_2_while_cond_53049___redundant_placeholder3?
;rnn_2_while_rnn_2_while_cond_53049___redundant_placeholder4?
;rnn_2_while_rnn_2_while_cond_53049___redundant_placeholder5?
;rnn_2_while_rnn_2_while_cond_53049___redundant_placeholder6?
;rnn_2_while_rnn_2_while_cond_53049___redundant_placeholder7?
;rnn_2_while_rnn_2_while_cond_53049___redundant_placeholder8
rnn_2_while_identity
z
rnn_2/while/LessLessrnn_2_while_placeholder&rnn_2_while_less_rnn_2_strided_slice_1*
T0*
_output_shapes
: W
rnn_2/while/IdentityIdentityrnn_2/while/Less:z:0*
T0
*
_output_shapes
: "5
rnn_2_while_identityrnn_2/while/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ôg
¯
@__inference_rnn_2_layer_call_and_return_conditional_losses_54351

inputsH
6my_lstm_cell_2_dense_10_matmul_readvariableop_resource:'E
7my_lstm_cell_2_dense_10_biasadd_readvariableop_resource:H
6my_lstm_cell_2_dense_11_matmul_readvariableop_resource:'E
7my_lstm_cell_2_dense_11_biasadd_readvariableop_resource:H
6my_lstm_cell_2_dense_12_matmul_readvariableop_resource:'E
7my_lstm_cell_2_dense_12_biasadd_readvariableop_resource:H
6my_lstm_cell_2_dense_13_matmul_readvariableop_resource:'E
7my_lstm_cell_2_dense_13_biasadd_readvariableop_resource:
identity¢.my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp¢-my_lstm_cell_2/dense_10/MatMul/ReadVariableOp¢.my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp¢-my_lstm_cell_2/dense_11/MatMul/ReadVariableOp¢.my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp¢-my_lstm_cell_2/dense_12/MatMul/ReadVariableOp¢.my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp¢-my_lstm_cell_2/dense_13/MatMul/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskj
(my_lstm_cell_2/concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ç
#my_lstm_cell_2/concatenate_5/concatConcatV2strided_slice_2:output:0zeros:output:01my_lstm_cell_2/concatenate_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'¤
-my_lstm_cell_2/dense_10/MatMul/ReadVariableOpReadVariableOp6my_lstm_cell_2_dense_10_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0¿
my_lstm_cell_2/dense_10/MatMulMatMul,my_lstm_cell_2/concatenate_5/concat:output:05my_lstm_cell_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOpReadVariableOp7my_lstm_cell_2_dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
my_lstm_cell_2/dense_10/BiasAddBiasAdd(my_lstm_cell_2/dense_10/MatMul:product:06my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
my_lstm_cell_2/dense_10/SigmoidSigmoid(my_lstm_cell_2/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
my_lstm_cell_2/multiply_2/mulMulzeros_1:output:0#my_lstm_cell_2/dense_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-my_lstm_cell_2/dense_11/MatMul/ReadVariableOpReadVariableOp6my_lstm_cell_2_dense_11_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0¿
my_lstm_cell_2/dense_11/MatMulMatMul,my_lstm_cell_2/concatenate_5/concat:output:05my_lstm_cell_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOpReadVariableOp7my_lstm_cell_2_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
my_lstm_cell_2/dense_11/BiasAddBiasAdd(my_lstm_cell_2/dense_11/MatMul:product:06my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
my_lstm_cell_2/dense_11/SigmoidSigmoid(my_lstm_cell_2/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-my_lstm_cell_2/dense_12/MatMul/ReadVariableOpReadVariableOp6my_lstm_cell_2_dense_12_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0¿
my_lstm_cell_2/dense_12/MatMulMatMul,my_lstm_cell_2/concatenate_5/concat:output:05my_lstm_cell_2/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOpReadVariableOp7my_lstm_cell_2_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
my_lstm_cell_2/dense_12/BiasAddBiasAdd(my_lstm_cell_2/dense_12/MatMul:product:06my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
my_lstm_cell_2/dense_12/TanhTanh(my_lstm_cell_2/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
my_lstm_cell_2/multiply_2/mul_1Mul#my_lstm_cell_2/dense_11/Sigmoid:y:0 my_lstm_cell_2/dense_12/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
my_lstm_cell_2/add_5/addAddV2!my_lstm_cell_2/multiply_2/mul:z:0#my_lstm_cell_2/multiply_2/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-my_lstm_cell_2/dense_13/MatMul/ReadVariableOpReadVariableOp6my_lstm_cell_2_dense_13_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0¿
my_lstm_cell_2/dense_13/MatMulMatMul,my_lstm_cell_2/concatenate_5/concat:output:05my_lstm_cell_2/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOpReadVariableOp7my_lstm_cell_2_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
my_lstm_cell_2/dense_13/BiasAddBiasAdd(my_lstm_cell_2/dense_13/MatMul:product:06my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
my_lstm_cell_2/dense_13/SigmoidSigmoid(my_lstm_cell_2/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
 my_lstm_cell_2/activation_2/TanhTanhmy_lstm_cell_2/add_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
my_lstm_cell_2/multiply_2/mul_2Mul$my_lstm_cell_2/activation_2/Tanh:y:0#my_lstm_cell_2/dense_13/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ø
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:06my_lstm_cell_2_dense_10_matmul_readvariableop_resource7my_lstm_cell_2_dense_10_biasadd_readvariableop_resource6my_lstm_cell_2_dense_11_matmul_readvariableop_resource7my_lstm_cell_2_dense_11_biasadd_readvariableop_resource6my_lstm_cell_2_dense_12_matmul_readvariableop_resource7my_lstm_cell_2_dense_12_biasadd_readvariableop_resource6my_lstm_cell_2_dense_13_matmul_readvariableop_resource7my_lstm_cell_2_dense_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( *
bodyR
while_body_54247*
condR
while_cond_54246*U
output_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
NoOpNoOp/^my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp.^my_lstm_cell_2/dense_10/MatMul/ReadVariableOp/^my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp.^my_lstm_cell_2/dense_11/MatMul/ReadVariableOp/^my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp.^my_lstm_cell_2/dense_12/MatMul/ReadVariableOp/^my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp.^my_lstm_cell_2/dense_13/MatMul/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2`
.my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp.my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp2^
-my_lstm_cell_2/dense_10/MatMul/ReadVariableOp-my_lstm_cell_2/dense_10/MatMul/ReadVariableOp2`
.my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp.my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp2^
-my_lstm_cell_2/dense_11/MatMul/ReadVariableOp-my_lstm_cell_2/dense_11/MatMul/ReadVariableOp2`
.my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp.my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp2^
-my_lstm_cell_2/dense_12/MatMul/ReadVariableOp-my_lstm_cell_2/dense_12/MatMul/ReadVariableOp2`
.my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp.my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp2^
-my_lstm_cell_2/dense_13/MatMul/ReadVariableOp-my_lstm_cell_2/dense_13/MatMul/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
i
M__inference_time_distributed_2_layer_call_and_return_conditional_losses_53501

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
1global_average_pooling2d_2/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      §
global_average_pooling2d_2/MeanMeanReshape:output:0:global_average_pooling2d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿS
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape(global_average_pooling2d_2/Mean:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿg
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×
ó
__inference_call_52428
x.
my_cnn_block_2_52214:"
my_cnn_block_2_52216:.
my_cnn_block_2_52218:"
my_cnn_block_2_52220:N
<rnn_2_my_lstm_cell_2_dense_10_matmul_readvariableop_resource:'K
=rnn_2_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource:N
<rnn_2_my_lstm_cell_2_dense_11_matmul_readvariableop_resource:'K
=rnn_2_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource:N
<rnn_2_my_lstm_cell_2_dense_12_matmul_readvariableop_resource:'K
=rnn_2_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource:N
<rnn_2_my_lstm_cell_2_dense_13_matmul_readvariableop_resource:'K
=rnn_2_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource:<
*dense_14_tensordot_readvariableop_resource:6
(dense_14_biasadd_readvariableop_resource:
identity¢dense_14/BiasAdd/ReadVariableOp¢!dense_14/Tensordot/ReadVariableOp¢&my_cnn_block_2/StatefulPartitionedCall¢4rnn_2/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp¢3rnn_2/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp¢4rnn_2/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp¢3rnn_2/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp¢4rnn_2/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp¢3rnn_2/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp¢4rnn_2/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp¢3rnn_2/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp¢rnn_2/while
&my_cnn_block_2/StatefulPartitionedCallStatefulPartitionedCallxmy_cnn_block_2_52214my_cnn_block_2_52216my_cnn_block_2_52218my_cnn_block_2_52220*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *
fR
__inference_call_52213y
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         »
time_distributed_2/ReshapeReshape/my_cnn_block_2/StatefulPartitionedCall:output:0)time_distributed_2/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Dtime_distributed_2/global_average_pooling2d_2/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      à
2time_distributed_2/global_average_pooling2d_2/MeanMean#time_distributed_2/Reshape:output:0Mtime_distributed_2/global_average_pooling2d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
"time_distributed_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      Ç
time_distributed_2/Reshape_1Reshape;time_distributed_2/global_average_pooling2d_2/Mean:output:0+time_distributed_2/Reshape_1/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
"time_distributed_2/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         ¿
time_distributed_2/Reshape_2Reshape/my_cnn_block_2/StatefulPartitionedCall:output:0+time_distributed_2/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
rnn_2/ShapeShape%time_distributed_2/Reshape_1:output:0*
T0*
_output_shapes
:c
rnn_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
rnn_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
rnn_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
rnn_2/strided_sliceStridedSlicernn_2/Shape:output:0"rnn_2/strided_slice/stack:output:0$rnn_2/strided_slice/stack_1:output:0$rnn_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
rnn_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
rnn_2/zeros/packedPackrnn_2/strided_slice:output:0rnn_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
rnn_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
rnn_2/zerosFillrnn_2/zeros/packed:output:0rnn_2/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
rnn_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
rnn_2/zeros_1/packedPackrnn_2/strided_slice:output:0rnn_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:X
rnn_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
rnn_2/zeros_1Fillrnn_2/zeros_1/packed:output:0rnn_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
rnn_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
rnn_2/transpose	Transpose%time_distributed_2/Reshape_1:output:0rnn_2/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
rnn_2/Shape_1Shapernn_2/transpose:y:0*
T0*
_output_shapes
:e
rnn_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: g
rnn_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
rnn_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
rnn_2/strided_slice_1StridedSlicernn_2/Shape_1:output:0$rnn_2/strided_slice_1/stack:output:0&rnn_2/strided_slice_1/stack_1:output:0&rnn_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!rnn_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
rnn_2/TensorArrayV2TensorListReserve*rnn_2/TensorArrayV2/element_shape:output:0rnn_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
;rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ò
-rnn_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn_2/transpose:y:0Drnn_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒe
rnn_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: g
rnn_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
rnn_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
rnn_2/strided_slice_2StridedSlicernn_2/transpose:y:0$rnn_2/strided_slice_2/stack:output:0&rnn_2/strided_slice_2/stack_1:output:0&rnn_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskp
.rnn_2/my_lstm_cell_2/concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ß
)rnn_2/my_lstm_cell_2/concatenate_5/concatConcatV2rnn_2/strided_slice_2:output:0rnn_2/zeros:output:07rnn_2/my_lstm_cell_2/concatenate_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'°
3rnn_2/my_lstm_cell_2/dense_10/MatMul/ReadVariableOpReadVariableOp<rnn_2_my_lstm_cell_2_dense_10_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0Ñ
$rnn_2/my_lstm_cell_2/dense_10/MatMulMatMul2rnn_2/my_lstm_cell_2/concatenate_5/concat:output:0;rnn_2/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
4rnn_2/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOpReadVariableOp=rnn_2_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ð
%rnn_2/my_lstm_cell_2/dense_10/BiasAddBiasAdd.rnn_2/my_lstm_cell_2/dense_10/MatMul:product:0<rnn_2/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%rnn_2/my_lstm_cell_2/dense_10/SigmoidSigmoid.rnn_2/my_lstm_cell_2/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#rnn_2/my_lstm_cell_2/multiply_2/mulMulrnn_2/zeros_1:output:0)rnn_2/my_lstm_cell_2/dense_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
3rnn_2/my_lstm_cell_2/dense_11/MatMul/ReadVariableOpReadVariableOp<rnn_2_my_lstm_cell_2_dense_11_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0Ñ
$rnn_2/my_lstm_cell_2/dense_11/MatMulMatMul2rnn_2/my_lstm_cell_2/concatenate_5/concat:output:0;rnn_2/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
4rnn_2/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOpReadVariableOp=rnn_2_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ð
%rnn_2/my_lstm_cell_2/dense_11/BiasAddBiasAdd.rnn_2/my_lstm_cell_2/dense_11/MatMul:product:0<rnn_2/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%rnn_2/my_lstm_cell_2/dense_11/SigmoidSigmoid.rnn_2/my_lstm_cell_2/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
3rnn_2/my_lstm_cell_2/dense_12/MatMul/ReadVariableOpReadVariableOp<rnn_2_my_lstm_cell_2_dense_12_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0Ñ
$rnn_2/my_lstm_cell_2/dense_12/MatMulMatMul2rnn_2/my_lstm_cell_2/concatenate_5/concat:output:0;rnn_2/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
4rnn_2/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOpReadVariableOp=rnn_2_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ð
%rnn_2/my_lstm_cell_2/dense_12/BiasAddBiasAdd.rnn_2/my_lstm_cell_2/dense_12/MatMul:product:0<rnn_2/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"rnn_2/my_lstm_cell_2/dense_12/TanhTanh.rnn_2/my_lstm_cell_2/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
%rnn_2/my_lstm_cell_2/multiply_2/mul_1Mul)rnn_2/my_lstm_cell_2/dense_11/Sigmoid:y:0&rnn_2/my_lstm_cell_2/dense_12/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
rnn_2/my_lstm_cell_2/add_5/addAddV2'rnn_2/my_lstm_cell_2/multiply_2/mul:z:0)rnn_2/my_lstm_cell_2/multiply_2/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
3rnn_2/my_lstm_cell_2/dense_13/MatMul/ReadVariableOpReadVariableOp<rnn_2_my_lstm_cell_2_dense_13_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0Ñ
$rnn_2/my_lstm_cell_2/dense_13/MatMulMatMul2rnn_2/my_lstm_cell_2/concatenate_5/concat:output:0;rnn_2/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
4rnn_2/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOpReadVariableOp=rnn_2_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ð
%rnn_2/my_lstm_cell_2/dense_13/BiasAddBiasAdd.rnn_2/my_lstm_cell_2/dense_13/MatMul:product:0<rnn_2/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%rnn_2/my_lstm_cell_2/dense_13/SigmoidSigmoid.rnn_2/my_lstm_cell_2/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&rnn_2/my_lstm_cell_2/activation_2/TanhTanh"rnn_2/my_lstm_cell_2/add_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
%rnn_2/my_lstm_cell_2/multiply_2/mul_2Mul*rnn_2/my_lstm_cell_2/activation_2/Tanh:y:0)rnn_2/my_lstm_cell_2/dense_13/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
#rnn_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ê
rnn_2/TensorArrayV2_1TensorListReserve,rnn_2/TensorArrayV2_1/element_shape:output:0rnn_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒL

rnn_2/timeConst*
_output_shapes
: *
dtype0*
value	B : i
rnn_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿZ
rnn_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ê	
rnn_2/whileWhile!rnn_2/while/loop_counter:output:0'rnn_2/while/maximum_iterations:output:0rnn_2/time:output:0rnn_2/TensorArrayV2_1:handle:0rnn_2/zeros:output:0rnn_2/zeros_1:output:0rnn_2/strided_slice_1:output:0=rnn_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0<rnn_2_my_lstm_cell_2_dense_10_matmul_readvariableop_resource=rnn_2_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource<rnn_2_my_lstm_cell_2_dense_11_matmul_readvariableop_resource=rnn_2_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource<rnn_2_my_lstm_cell_2_dense_12_matmul_readvariableop_resource=rnn_2_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource<rnn_2_my_lstm_cell_2_dense_13_matmul_readvariableop_resource=rnn_2_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( *"
bodyR
rnn_2_while_body_52298*"
condR
rnn_2_while_cond_52297*U
output_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : *
parallel_iterations 
6rnn_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ô
(rnn_2/TensorArrayV2Stack/TensorListStackTensorListStackrnn_2/while:output:3?rnn_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0n
rnn_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿg
rnn_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
rnn_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¥
rnn_2/strided_slice_3StridedSlice1rnn_2/TensorArrayV2Stack/TensorListStack:tensor:0$rnn_2/strided_slice_3/stack:output:0&rnn_2/strided_slice_3/stack_1:output:0&rnn_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskk
rnn_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¨
rnn_2/transpose_1	Transpose1rnn_2/TensorArrayV2Stack/TensorListStack:tensor:0rnn_2/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_14/Tensordot/ReadVariableOpReadVariableOp*dense_14_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0a
dense_14/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_14/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ]
dense_14/Tensordot/ShapeShapernn_2/transpose_1:y:0*
T0*
_output_shapes
:b
 dense_14/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_14/Tensordot/GatherV2GatherV2!dense_14/Tensordot/Shape:output:0 dense_14/Tensordot/free:output:0)dense_14/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_14/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
dense_14/Tensordot/GatherV2_1GatherV2!dense_14/Tensordot/Shape:output:0 dense_14/Tensordot/axes:output:0+dense_14/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_14/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_14/Tensordot/ProdProd$dense_14/Tensordot/GatherV2:output:0!dense_14/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_14/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_14/Tensordot/Prod_1Prod&dense_14/Tensordot/GatherV2_1:output:0#dense_14/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_14/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
dense_14/Tensordot/concatConcatV2 dense_14/Tensordot/free:output:0 dense_14/Tensordot/axes:output:0'dense_14/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_14/Tensordot/stackPack dense_14/Tensordot/Prod:output:0"dense_14/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_14/Tensordot/transpose	Transposernn_2/transpose_1:y:0"dense_14/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
dense_14/Tensordot/ReshapeReshape dense_14/Tensordot/transpose:y:0!dense_14/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
dense_14/Tensordot/MatMulMatMul#dense_14/Tensordot/Reshape:output:0)dense_14/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_14/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_14/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
dense_14/Tensordot/concat_1ConcatV2$dense_14/Tensordot/GatherV2:output:0#dense_14/Tensordot/Const_2:output:0)dense_14/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_14/TensordotReshape#dense_14/Tensordot/MatMul:product:0$dense_14/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_14/BiasAddBiasAdddense_14/Tensordot:output:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
IdentityIdentitydense_14/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ÷
NoOpNoOp ^dense_14/BiasAdd/ReadVariableOp"^dense_14/Tensordot/ReadVariableOp'^my_cnn_block_2/StatefulPartitionedCall5^rnn_2/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp4^rnn_2/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp5^rnn_2/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp4^rnn_2/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp5^rnn_2/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp4^rnn_2/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp5^rnn_2/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp4^rnn_2/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp^rnn_2/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2F
!dense_14/Tensordot/ReadVariableOp!dense_14/Tensordot/ReadVariableOp2P
&my_cnn_block_2/StatefulPartitionedCall&my_cnn_block_2/StatefulPartitionedCall2l
4rnn_2/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp4rnn_2/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp2j
3rnn_2/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp3rnn_2/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp2l
4rnn_2/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp4rnn_2/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp2j
3rnn_2/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp3rnn_2/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp2l
4rnn_2/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp4rnn_2/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp2j
3rnn_2/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp3rnn_2/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp2l
4rnn_2/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp4rnn_2/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp2j
3rnn_2/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp3rnn_2/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp2
rnn_2/whilernn_2/while:V R
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
*
¢

while_body_51738
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_my_lstm_cell_2_51762_0:'*
while_my_lstm_cell_2_51764_0:.
while_my_lstm_cell_2_51766_0:'*
while_my_lstm_cell_2_51768_0:.
while_my_lstm_cell_2_51770_0:'*
while_my_lstm_cell_2_51772_0:.
while_my_lstm_cell_2_51774_0:'*
while_my_lstm_cell_2_51776_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_my_lstm_cell_2_51762:'(
while_my_lstm_cell_2_51764:,
while_my_lstm_cell_2_51766:'(
while_my_lstm_cell_2_51768:,
while_my_lstm_cell_2_51770:'(
while_my_lstm_cell_2_51772:,
while_my_lstm_cell_2_51774:'(
while_my_lstm_cell_2_51776:¢,while/my_lstm_cell_2/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0Ù
,while/my_lstm_cell_2/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_my_lstm_cell_2_51762_0while_my_lstm_cell_2_51764_0while_my_lstm_cell_2_51766_0while_my_lstm_cell_2_51768_0while_my_lstm_cell_2_51770_0while_my_lstm_cell_2_51772_0while_my_lstm_cell_2_51774_0while_my_lstm_cell_2_51776_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_my_lstm_cell_2_layer_call_and_return_conditional_losses_51022Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder5while/my_lstm_cell_2/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity5while/my_lstm_cell_2/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity5while/my_lstm_cell_2/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{

while/NoOpNoOp-^while/my_lstm_cell_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_my_lstm_cell_2_51762while_my_lstm_cell_2_51762_0":
while_my_lstm_cell_2_51764while_my_lstm_cell_2_51764_0":
while_my_lstm_cell_2_51766while_my_lstm_cell_2_51766_0":
while_my_lstm_cell_2_51768while_my_lstm_cell_2_51768_0":
while_my_lstm_cell_2_51770while_my_lstm_cell_2_51770_0":
while_my_lstm_cell_2_51772while_my_lstm_cell_2_51772_0":
while_my_lstm_cell_2_51774while_my_lstm_cell_2_51774_0":
while_my_lstm_cell_2_51776while_my_lstm_cell_2_51776_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2\
,while/my_lstm_cell_2/StatefulPartitionedCall,while/my_lstm_cell_2/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
±
û
rnn_2_while_cond_52514(
$rnn_2_while_rnn_2_while_loop_counter.
*rnn_2_while_rnn_2_while_maximum_iterations
rnn_2_while_placeholder
rnn_2_while_placeholder_1
rnn_2_while_placeholder_2
rnn_2_while_placeholder_3*
&rnn_2_while_less_rnn_2_strided_slice_1?
;rnn_2_while_rnn_2_while_cond_52514___redundant_placeholder0?
;rnn_2_while_rnn_2_while_cond_52514___redundant_placeholder1?
;rnn_2_while_rnn_2_while_cond_52514___redundant_placeholder2?
;rnn_2_while_rnn_2_while_cond_52514___redundant_placeholder3?
;rnn_2_while_rnn_2_while_cond_52514___redundant_placeholder4?
;rnn_2_while_rnn_2_while_cond_52514___redundant_placeholder5?
;rnn_2_while_rnn_2_while_cond_52514___redundant_placeholder6?
;rnn_2_while_rnn_2_while_cond_52514___redundant_placeholder7?
;rnn_2_while_rnn_2_while_cond_52514___redundant_placeholder8
rnn_2_while_identity
z
rnn_2/while/LessLessrnn_2_while_placeholder&rnn_2_while_less_rnn_2_strided_slice_1*
T0*
_output_shapes
: W
rnn_2/while/IdentityIdentityrnn_2/while/Less:z:0*
T0
*
_output_shapes
: "5
rnn_2_while_identityrnn_2/while/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
;
À
@__inference_rnn_2_layer_call_and_return_conditional_losses_51134

inputs&
my_lstm_cell_2_51023:'"
my_lstm_cell_2_51025:&
my_lstm_cell_2_51027:'"
my_lstm_cell_2_51029:&
my_lstm_cell_2_51031:'"
my_lstm_cell_2_51033:&
my_lstm_cell_2_51035:'"
my_lstm_cell_2_51037:
identity¢&my_lstm_cell_2/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskó
&my_lstm_cell_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0my_lstm_cell_2_51023my_lstm_cell_2_51025my_lstm_cell_2_51027my_lstm_cell_2_51029my_lstm_cell_2_51031my_lstm_cell_2_51033my_lstm_cell_2_51035my_lstm_cell_2_51037*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_my_lstm_cell_2_layer_call_and_return_conditional_losses_51022n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ä
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0my_lstm_cell_2_51023my_lstm_cell_2_51025my_lstm_cell_2_51027my_lstm_cell_2_51029my_lstm_cell_2_51031my_lstm_cell_2_51033my_lstm_cell_2_51035my_lstm_cell_2_51037*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( *
bodyR
while_body_51046*
condR
while_cond_51045*U
output_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿk
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿw
NoOpNoOp'^my_lstm_cell_2/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : 2P
&my_lstm_cell_2/StatefulPartitionedCall&my_lstm_cell_2/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©i
é
rnn_2_while_body_53050(
$rnn_2_while_rnn_2_while_loop_counter.
*rnn_2_while_rnn_2_while_maximum_iterations
rnn_2_while_placeholder
rnn_2_while_placeholder_1
rnn_2_while_placeholder_2
rnn_2_while_placeholder_3'
#rnn_2_while_rnn_2_strided_slice_1_0c
_rnn_2_while_tensorarrayv2read_tensorlistgetitem_rnn_2_tensorarrayunstack_tensorlistfromtensor_0V
Drnn_2_while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource_0:'S
Ernn_2_while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource_0:V
Drnn_2_while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource_0:'S
Ernn_2_while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource_0:V
Drnn_2_while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource_0:'S
Ernn_2_while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource_0:V
Drnn_2_while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource_0:'S
Ernn_2_while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource_0:
rnn_2_while_identity
rnn_2_while_identity_1
rnn_2_while_identity_2
rnn_2_while_identity_3
rnn_2_while_identity_4
rnn_2_while_identity_5%
!rnn_2_while_rnn_2_strided_slice_1a
]rnn_2_while_tensorarrayv2read_tensorlistgetitem_rnn_2_tensorarrayunstack_tensorlistfromtensorT
Brnn_2_while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource:'Q
Crnn_2_while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource:T
Brnn_2_while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource:'Q
Crnn_2_while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource:T
Brnn_2_while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource:'Q
Crnn_2_while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource:T
Brnn_2_while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource:'Q
Crnn_2_while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource:¢:rnn_2/while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp¢9rnn_2/while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp¢:rnn_2/while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp¢9rnn_2/while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp¢:rnn_2/while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp¢9rnn_2/while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp¢:rnn_2/while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp¢9rnn_2/while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp
=rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ä
/rnn_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_rnn_2_while_tensorarrayv2read_tensorlistgetitem_rnn_2_tensorarrayunstack_tensorlistfromtensor_0rnn_2_while_placeholderFrnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0v
4rnn_2/while/my_lstm_cell_2/concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
/rnn_2/while/my_lstm_cell_2/concatenate_5/concatConcatV26rnn_2/while/TensorArrayV2Read/TensorListGetItem:item:0rnn_2_while_placeholder_2=rnn_2/while/my_lstm_cell_2/concatenate_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'¾
9rnn_2/while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOpReadVariableOpDrnn_2_while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ã
*rnn_2/while/my_lstm_cell_2/dense_10/MatMulMatMul8rnn_2/while/my_lstm_cell_2/concatenate_5/concat:output:0Arnn_2/while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
:rnn_2/while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOpReadVariableOpErnn_2_while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0â
+rnn_2/while/my_lstm_cell_2/dense_10/BiasAddBiasAdd4rnn_2/while/my_lstm_cell_2/dense_10/MatMul:product:0Brnn_2/while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+rnn_2/while/my_lstm_cell_2/dense_10/SigmoidSigmoid4rnn_2/while/my_lstm_cell_2/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
)rnn_2/while/my_lstm_cell_2/multiply_2/mulMulrnn_2_while_placeholder_3/rnn_2/while/my_lstm_cell_2/dense_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
9rnn_2/while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOpReadVariableOpDrnn_2_while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ã
*rnn_2/while/my_lstm_cell_2/dense_11/MatMulMatMul8rnn_2/while/my_lstm_cell_2/concatenate_5/concat:output:0Arnn_2/while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
:rnn_2/while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOpReadVariableOpErnn_2_while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0â
+rnn_2/while/my_lstm_cell_2/dense_11/BiasAddBiasAdd4rnn_2/while/my_lstm_cell_2/dense_11/MatMul:product:0Brnn_2/while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+rnn_2/while/my_lstm_cell_2/dense_11/SigmoidSigmoid4rnn_2/while/my_lstm_cell_2/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
9rnn_2/while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOpReadVariableOpDrnn_2_while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ã
*rnn_2/while/my_lstm_cell_2/dense_12/MatMulMatMul8rnn_2/while/my_lstm_cell_2/concatenate_5/concat:output:0Arnn_2/while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
:rnn_2/while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOpReadVariableOpErnn_2_while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0â
+rnn_2/while/my_lstm_cell_2/dense_12/BiasAddBiasAdd4rnn_2/while/my_lstm_cell_2/dense_12/MatMul:product:0Brnn_2/while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(rnn_2/while/my_lstm_cell_2/dense_12/TanhTanh4rnn_2/while/my_lstm_cell_2/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
+rnn_2/while/my_lstm_cell_2/multiply_2/mul_1Mul/rnn_2/while/my_lstm_cell_2/dense_11/Sigmoid:y:0,rnn_2/while/my_lstm_cell_2/dense_12/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
$rnn_2/while/my_lstm_cell_2/add_5/addAddV2-rnn_2/while/my_lstm_cell_2/multiply_2/mul:z:0/rnn_2/while/my_lstm_cell_2/multiply_2/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
9rnn_2/while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOpReadVariableOpDrnn_2_while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ã
*rnn_2/while/my_lstm_cell_2/dense_13/MatMulMatMul8rnn_2/while/my_lstm_cell_2/concatenate_5/concat:output:0Arnn_2/while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
:rnn_2/while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOpReadVariableOpErnn_2_while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0â
+rnn_2/while/my_lstm_cell_2/dense_13/BiasAddBiasAdd4rnn_2/while/my_lstm_cell_2/dense_13/MatMul:product:0Brnn_2/while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+rnn_2/while/my_lstm_cell_2/dense_13/SigmoidSigmoid4rnn_2/while/my_lstm_cell_2/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,rnn_2/while/my_lstm_cell_2/activation_2/TanhTanh(rnn_2/while/my_lstm_cell_2/add_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
+rnn_2/while/my_lstm_cell_2/multiply_2/mul_2Mul0rnn_2/while/my_lstm_cell_2/activation_2/Tanh:y:0/rnn_2/while/my_lstm_cell_2/dense_13/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
0rnn_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrnn_2_while_placeholder_1rnn_2_while_placeholder/rnn_2/while/my_lstm_cell_2/multiply_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒS
rnn_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :n
rnn_2/while/addAddV2rnn_2_while_placeholderrnn_2/while/add/y:output:0*
T0*
_output_shapes
: U
rnn_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
rnn_2/while/add_1AddV2$rnn_2_while_rnn_2_while_loop_counterrnn_2/while/add_1/y:output:0*
T0*
_output_shapes
: k
rnn_2/while/IdentityIdentityrnn_2/while/add_1:z:0^rnn_2/while/NoOp*
T0*
_output_shapes
: 
rnn_2/while/Identity_1Identity*rnn_2_while_rnn_2_while_maximum_iterations^rnn_2/while/NoOp*
T0*
_output_shapes
: k
rnn_2/while/Identity_2Identityrnn_2/while/add:z:0^rnn_2/while/NoOp*
T0*
_output_shapes
: 
rnn_2/while/Identity_3Identity@rnn_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^rnn_2/while/NoOp*
T0*
_output_shapes
: 
rnn_2/while/Identity_4Identity/rnn_2/while/my_lstm_cell_2/multiply_2/mul_2:z:0^rnn_2/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn_2/while/Identity_5Identity(rnn_2/while/my_lstm_cell_2/add_5/add:z:0^rnn_2/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
rnn_2/while/NoOpNoOp;^rnn_2/while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp:^rnn_2/while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp;^rnn_2/while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp:^rnn_2/while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp;^rnn_2/while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp:^rnn_2/while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp;^rnn_2/while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp:^rnn_2/while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "5
rnn_2_while_identityrnn_2/while/Identity:output:0"9
rnn_2_while_identity_1rnn_2/while/Identity_1:output:0"9
rnn_2_while_identity_2rnn_2/while/Identity_2:output:0"9
rnn_2_while_identity_3rnn_2/while/Identity_3:output:0"9
rnn_2_while_identity_4rnn_2/while/Identity_4:output:0"9
rnn_2_while_identity_5rnn_2/while/Identity_5:output:0"
Crnn_2_while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resourceErnn_2_while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource_0"
Brnn_2_while_my_lstm_cell_2_dense_10_matmul_readvariableop_resourceDrnn_2_while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource_0"
Crnn_2_while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resourceErnn_2_while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource_0"
Brnn_2_while_my_lstm_cell_2_dense_11_matmul_readvariableop_resourceDrnn_2_while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource_0"
Crnn_2_while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resourceErnn_2_while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource_0"
Brnn_2_while_my_lstm_cell_2_dense_12_matmul_readvariableop_resourceDrnn_2_while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource_0"
Crnn_2_while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resourceErnn_2_while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource_0"
Brnn_2_while_my_lstm_cell_2_dense_13_matmul_readvariableop_resourceDrnn_2_while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource_0"H
!rnn_2_while_rnn_2_strided_slice_1#rnn_2_while_rnn_2_strided_slice_1_0"À
]rnn_2_while_tensorarrayv2read_tensorlistgetitem_rnn_2_tensorarrayunstack_tensorlistfromtensor_rnn_2_while_tensorarrayv2read_tensorlistgetitem_rnn_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2x
:rnn_2/while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp:rnn_2/while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp2v
9rnn_2/while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp9rnn_2/while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp2x
:rnn_2/while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp:rnn_2/while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp2v
9rnn_2/while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp9rnn_2/while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp2x
:rnn_2/while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp:rnn_2/while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp2v
9rnn_2/while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp9rnn_2/while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp2x
:rnn_2/while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp:rnn_2/while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp2v
9rnn_2/while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp9rnn_2/while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
©i
é
rnn_2_while_body_52833(
$rnn_2_while_rnn_2_while_loop_counter.
*rnn_2_while_rnn_2_while_maximum_iterations
rnn_2_while_placeholder
rnn_2_while_placeholder_1
rnn_2_while_placeholder_2
rnn_2_while_placeholder_3'
#rnn_2_while_rnn_2_strided_slice_1_0c
_rnn_2_while_tensorarrayv2read_tensorlistgetitem_rnn_2_tensorarrayunstack_tensorlistfromtensor_0V
Drnn_2_while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource_0:'S
Ernn_2_while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource_0:V
Drnn_2_while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource_0:'S
Ernn_2_while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource_0:V
Drnn_2_while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource_0:'S
Ernn_2_while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource_0:V
Drnn_2_while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource_0:'S
Ernn_2_while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource_0:
rnn_2_while_identity
rnn_2_while_identity_1
rnn_2_while_identity_2
rnn_2_while_identity_3
rnn_2_while_identity_4
rnn_2_while_identity_5%
!rnn_2_while_rnn_2_strided_slice_1a
]rnn_2_while_tensorarrayv2read_tensorlistgetitem_rnn_2_tensorarrayunstack_tensorlistfromtensorT
Brnn_2_while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource:'Q
Crnn_2_while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource:T
Brnn_2_while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource:'Q
Crnn_2_while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource:T
Brnn_2_while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource:'Q
Crnn_2_while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource:T
Brnn_2_while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource:'Q
Crnn_2_while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource:¢:rnn_2/while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp¢9rnn_2/while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp¢:rnn_2/while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp¢9rnn_2/while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp¢:rnn_2/while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp¢9rnn_2/while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp¢:rnn_2/while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp¢9rnn_2/while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp
=rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ä
/rnn_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_rnn_2_while_tensorarrayv2read_tensorlistgetitem_rnn_2_tensorarrayunstack_tensorlistfromtensor_0rnn_2_while_placeholderFrnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0v
4rnn_2/while/my_lstm_cell_2/concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
/rnn_2/while/my_lstm_cell_2/concatenate_5/concatConcatV26rnn_2/while/TensorArrayV2Read/TensorListGetItem:item:0rnn_2_while_placeholder_2=rnn_2/while/my_lstm_cell_2/concatenate_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'¾
9rnn_2/while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOpReadVariableOpDrnn_2_while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ã
*rnn_2/while/my_lstm_cell_2/dense_10/MatMulMatMul8rnn_2/while/my_lstm_cell_2/concatenate_5/concat:output:0Arnn_2/while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
:rnn_2/while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOpReadVariableOpErnn_2_while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0â
+rnn_2/while/my_lstm_cell_2/dense_10/BiasAddBiasAdd4rnn_2/while/my_lstm_cell_2/dense_10/MatMul:product:0Brnn_2/while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+rnn_2/while/my_lstm_cell_2/dense_10/SigmoidSigmoid4rnn_2/while/my_lstm_cell_2/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
)rnn_2/while/my_lstm_cell_2/multiply_2/mulMulrnn_2_while_placeholder_3/rnn_2/while/my_lstm_cell_2/dense_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
9rnn_2/while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOpReadVariableOpDrnn_2_while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ã
*rnn_2/while/my_lstm_cell_2/dense_11/MatMulMatMul8rnn_2/while/my_lstm_cell_2/concatenate_5/concat:output:0Arnn_2/while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
:rnn_2/while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOpReadVariableOpErnn_2_while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0â
+rnn_2/while/my_lstm_cell_2/dense_11/BiasAddBiasAdd4rnn_2/while/my_lstm_cell_2/dense_11/MatMul:product:0Brnn_2/while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+rnn_2/while/my_lstm_cell_2/dense_11/SigmoidSigmoid4rnn_2/while/my_lstm_cell_2/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
9rnn_2/while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOpReadVariableOpDrnn_2_while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ã
*rnn_2/while/my_lstm_cell_2/dense_12/MatMulMatMul8rnn_2/while/my_lstm_cell_2/concatenate_5/concat:output:0Arnn_2/while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
:rnn_2/while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOpReadVariableOpErnn_2_while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0â
+rnn_2/while/my_lstm_cell_2/dense_12/BiasAddBiasAdd4rnn_2/while/my_lstm_cell_2/dense_12/MatMul:product:0Brnn_2/while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(rnn_2/while/my_lstm_cell_2/dense_12/TanhTanh4rnn_2/while/my_lstm_cell_2/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
+rnn_2/while/my_lstm_cell_2/multiply_2/mul_1Mul/rnn_2/while/my_lstm_cell_2/dense_11/Sigmoid:y:0,rnn_2/while/my_lstm_cell_2/dense_12/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
$rnn_2/while/my_lstm_cell_2/add_5/addAddV2-rnn_2/while/my_lstm_cell_2/multiply_2/mul:z:0/rnn_2/while/my_lstm_cell_2/multiply_2/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
9rnn_2/while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOpReadVariableOpDrnn_2_while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ã
*rnn_2/while/my_lstm_cell_2/dense_13/MatMulMatMul8rnn_2/while/my_lstm_cell_2/concatenate_5/concat:output:0Arnn_2/while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
:rnn_2/while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOpReadVariableOpErnn_2_while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0â
+rnn_2/while/my_lstm_cell_2/dense_13/BiasAddBiasAdd4rnn_2/while/my_lstm_cell_2/dense_13/MatMul:product:0Brnn_2/while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+rnn_2/while/my_lstm_cell_2/dense_13/SigmoidSigmoid4rnn_2/while/my_lstm_cell_2/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,rnn_2/while/my_lstm_cell_2/activation_2/TanhTanh(rnn_2/while/my_lstm_cell_2/add_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
+rnn_2/while/my_lstm_cell_2/multiply_2/mul_2Mul0rnn_2/while/my_lstm_cell_2/activation_2/Tanh:y:0/rnn_2/while/my_lstm_cell_2/dense_13/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
0rnn_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrnn_2_while_placeholder_1rnn_2_while_placeholder/rnn_2/while/my_lstm_cell_2/multiply_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒS
rnn_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :n
rnn_2/while/addAddV2rnn_2_while_placeholderrnn_2/while/add/y:output:0*
T0*
_output_shapes
: U
rnn_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
rnn_2/while/add_1AddV2$rnn_2_while_rnn_2_while_loop_counterrnn_2/while/add_1/y:output:0*
T0*
_output_shapes
: k
rnn_2/while/IdentityIdentityrnn_2/while/add_1:z:0^rnn_2/while/NoOp*
T0*
_output_shapes
: 
rnn_2/while/Identity_1Identity*rnn_2_while_rnn_2_while_maximum_iterations^rnn_2/while/NoOp*
T0*
_output_shapes
: k
rnn_2/while/Identity_2Identityrnn_2/while/add:z:0^rnn_2/while/NoOp*
T0*
_output_shapes
: 
rnn_2/while/Identity_3Identity@rnn_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^rnn_2/while/NoOp*
T0*
_output_shapes
: 
rnn_2/while/Identity_4Identity/rnn_2/while/my_lstm_cell_2/multiply_2/mul_2:z:0^rnn_2/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn_2/while/Identity_5Identity(rnn_2/while/my_lstm_cell_2/add_5/add:z:0^rnn_2/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
rnn_2/while/NoOpNoOp;^rnn_2/while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp:^rnn_2/while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp;^rnn_2/while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp:^rnn_2/while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp;^rnn_2/while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp:^rnn_2/while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp;^rnn_2/while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp:^rnn_2/while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "5
rnn_2_while_identityrnn_2/while/Identity:output:0"9
rnn_2_while_identity_1rnn_2/while/Identity_1:output:0"9
rnn_2_while_identity_2rnn_2/while/Identity_2:output:0"9
rnn_2_while_identity_3rnn_2/while/Identity_3:output:0"9
rnn_2_while_identity_4rnn_2/while/Identity_4:output:0"9
rnn_2_while_identity_5rnn_2/while/Identity_5:output:0"
Crnn_2_while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resourceErnn_2_while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource_0"
Brnn_2_while_my_lstm_cell_2_dense_10_matmul_readvariableop_resourceDrnn_2_while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource_0"
Crnn_2_while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resourceErnn_2_while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource_0"
Brnn_2_while_my_lstm_cell_2_dense_11_matmul_readvariableop_resourceDrnn_2_while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource_0"
Crnn_2_while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resourceErnn_2_while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource_0"
Brnn_2_while_my_lstm_cell_2_dense_12_matmul_readvariableop_resourceDrnn_2_while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource_0"
Crnn_2_while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resourceErnn_2_while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource_0"
Brnn_2_while_my_lstm_cell_2_dense_13_matmul_readvariableop_resourceDrnn_2_while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource_0"H
!rnn_2_while_rnn_2_strided_slice_1#rnn_2_while_rnn_2_strided_slice_1_0"À
]rnn_2_while_tensorarrayv2read_tensorlistgetitem_rnn_2_tensorarrayunstack_tensorlistfromtensor_rnn_2_while_tensorarrayv2read_tensorlistgetitem_rnn_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2x
:rnn_2/while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp:rnn_2/while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp2v
9rnn_2/while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp9rnn_2/while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp2x
:rnn_2/while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp:rnn_2/while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp2v
9rnn_2/while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp9rnn_2/while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp2x
:rnn_2/while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp:rnn_2/while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp2v
9rnn_2/while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp9rnn_2/while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp2x
:rnn_2/while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp:rnn_2/while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp2v
9rnn_2/while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp9rnn_2/while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ð	
¶
%__inference_rnn_2_layer_call_fn_53613
inputs_0
unknown:'
	unknown_0:
	unknown_1:'
	unknown_2:
	unknown_3:'
	unknown_4:
	unknown_5:'
	unknown_6:
identity¢StatefulPartitionedCall²
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_rnn_2_layer_call_and_return_conditional_losses_51325|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
«N

__inference_call_53300	
inputH
.conv2d_4_conv2d_conv2d_readvariableop_resource:I
;conv2d_4_squeeze_batch_dims_biasadd_readvariableop_resource:H
.conv2d_5_conv2d_conv2d_readvariableop_resource:I
;conv2d_5_squeeze_batch_dims_biasadd_readvariableop_resource:
identity¢%conv2d_4/Conv2D/Conv2D/ReadVariableOp¢2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp¢%conv2d_5/Conv2D/Conv2D/ReadVariableOp¢2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOpJ
conv2d_4/Conv2D/ShapeShapeinput*
T0*
_output_shapes
:m
#conv2d_4/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_4/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿo
%conv2d_4/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv2d_4/Conv2D/strided_sliceStridedSliceconv2d_4/Conv2D/Shape:output:0,conv2d_4/Conv2D/strided_slice/stack:output:0.conv2d_4/Conv2D/strided_slice/stack_1:output:0.conv2d_4/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_4/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         
conv2d_4/Conv2D/ReshapeReshapeinput&conv2d_4/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%conv2d_4/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_4_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ó
conv2d_4/Conv2D/Conv2DConv2D conv2d_4/Conv2D/Reshape:output:0-conv2d_4/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
t
conv2d_4/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         f
conv2d_4/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÈ
conv2d_4/Conv2D/concatConcatV2&conv2d_4/Conv2D/strided_slice:output:0(conv2d_4/Conv2D/concat/values_1:output:0$conv2d_4/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:¤
conv2d_4/Conv2D/Reshape_1Reshapeconv2d_4/Conv2D/Conv2D:output:0conv2d_4/Conv2D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿs
!conv2d_4/squeeze_batch_dims/ShapeShape"conv2d_4/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_4/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1conv2d_4/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ{
1conv2d_4/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
)conv2d_4/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_4/squeeze_batch_dims/Shape:output:08conv2d_4/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_4/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_4/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
)conv2d_4/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         À
#conv2d_4/squeeze_batch_dims/ReshapeReshape"conv2d_4/Conv2D/Reshape_1:output:02conv2d_4/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_4_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ò
#conv2d_4/squeeze_batch_dims/BiasAddBiasAdd,conv2d_4/squeeze_batch_dims/Reshape:output:0:conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+conv2d_4/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         r
'conv2d_4/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿø
"conv2d_4/squeeze_batch_dims/concatConcatV22conv2d_4/squeeze_batch_dims/strided_slice:output:04conv2d_4/squeeze_batch_dims/concat/values_1:output:00conv2d_4/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:É
%conv2d_4/squeeze_batch_dims/Reshape_1Reshape,conv2d_4/squeeze_batch_dims/BiasAdd:output:0+conv2d_4/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿs
conv2d_5/Conv2D/ShapeShape.conv2d_4/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:m
#conv2d_5/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_5/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿo
%conv2d_5/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv2d_5/Conv2D/strided_sliceStridedSliceconv2d_5/Conv2D/Shape:output:0,conv2d_5/Conv2D/strided_slice/stack:output:0.conv2d_5/Conv2D/strided_slice/stack_1:output:0.conv2d_5/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_5/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         ´
conv2d_5/Conv2D/ReshapeReshape.conv2d_4/squeeze_batch_dims/Reshape_1:output:0&conv2d_5/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%conv2d_5/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_5_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ó
conv2d_5/Conv2D/Conv2DConv2D conv2d_5/Conv2D/Reshape:output:0-conv2d_5/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
t
conv2d_5/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         f
conv2d_5/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÈ
conv2d_5/Conv2D/concatConcatV2&conv2d_5/Conv2D/strided_slice:output:0(conv2d_5/Conv2D/concat/values_1:output:0$conv2d_5/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:¤
conv2d_5/Conv2D/Reshape_1Reshapeconv2d_5/Conv2D/Conv2D:output:0conv2d_5/Conv2D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿs
!conv2d_5/squeeze_batch_dims/ShapeShape"conv2d_5/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_5/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1conv2d_5/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ{
1conv2d_5/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
)conv2d_5/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_5/squeeze_batch_dims/Shape:output:08conv2d_5/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_5/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_5/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
)conv2d_5/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         À
#conv2d_5/squeeze_batch_dims/ReshapeReshape"conv2d_5/Conv2D/Reshape_1:output:02conv2d_5/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_5_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ò
#conv2d_5/squeeze_batch_dims/BiasAddBiasAdd,conv2d_5/squeeze_batch_dims/Reshape:output:0:conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+conv2d_5/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         r
'conv2d_5/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿø
"conv2d_5/squeeze_batch_dims/concatConcatV22conv2d_5/squeeze_batch_dims/strided_slice:output:04conv2d_5/squeeze_batch_dims/concat/values_1:output:00conv2d_5/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:É
%conv2d_5/squeeze_batch_dims/Reshape_1Reshape,conv2d_5/squeeze_batch_dims/BiasAdd:output:0+conv2d_5/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity.conv2d_5/squeeze_batch_dims/Reshape_1:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp&^conv2d_4/Conv2D/Conv2D/ReadVariableOp3^conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_5/Conv2D/Conv2D/ReadVariableOp3^conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : 2N
%conv2d_4/Conv2D/Conv2D/ReadVariableOp%conv2d_4/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_5/Conv2D/Conv2D/ReadVariableOp%conv2d_5/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp:Z V
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
í
î
/__inference_my_lstm_model_2_layer_call_fn_52713
x!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:'
	unknown_4:
	unknown_5:'
	unknown_6:
	unknown_7:'
	unknown_8:
	unknown_9:'

unknown_10:

unknown_11:

unknown_12:
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_my_lstm_model_2_layer_call_and_return_conditional_losses_51620s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
*
¢

while_body_51046
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_my_lstm_cell_2_51070_0:'*
while_my_lstm_cell_2_51072_0:.
while_my_lstm_cell_2_51074_0:'*
while_my_lstm_cell_2_51076_0:.
while_my_lstm_cell_2_51078_0:'*
while_my_lstm_cell_2_51080_0:.
while_my_lstm_cell_2_51082_0:'*
while_my_lstm_cell_2_51084_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_my_lstm_cell_2_51070:'(
while_my_lstm_cell_2_51072:,
while_my_lstm_cell_2_51074:'(
while_my_lstm_cell_2_51076:,
while_my_lstm_cell_2_51078:'(
while_my_lstm_cell_2_51080:,
while_my_lstm_cell_2_51082:'(
while_my_lstm_cell_2_51084:¢,while/my_lstm_cell_2/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0Ù
,while/my_lstm_cell_2/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_my_lstm_cell_2_51070_0while_my_lstm_cell_2_51072_0while_my_lstm_cell_2_51074_0while_my_lstm_cell_2_51076_0while_my_lstm_cell_2_51078_0while_my_lstm_cell_2_51080_0while_my_lstm_cell_2_51082_0while_my_lstm_cell_2_51084_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_my_lstm_cell_2_layer_call_and_return_conditional_losses_51022Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder5while/my_lstm_cell_2/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity5while/my_lstm_cell_2/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity5while/my_lstm_cell_2/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{

while/NoOpNoOp-^while/my_lstm_cell_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_my_lstm_cell_2_51070while_my_lstm_cell_2_51070_0":
while_my_lstm_cell_2_51072while_my_lstm_cell_2_51072_0":
while_my_lstm_cell_2_51074while_my_lstm_cell_2_51074_0":
while_my_lstm_cell_2_51076while_my_lstm_cell_2_51076_0":
while_my_lstm_cell_2_51078while_my_lstm_cell_2_51078_0":
while_my_lstm_cell_2_51080while_my_lstm_cell_2_51080_0":
while_my_lstm_cell_2_51082while_my_lstm_cell_2_51082_0":
while_my_lstm_cell_2_51084while_my_lstm_cell_2_51084_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2\
,while/my_lstm_cell_2/StatefulPartitionedCall,while/my_lstm_cell_2/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Í

Ç
while_cond_54072
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_54072___redundant_placeholder03
/while_while_cond_54072___redundant_placeholder13
/while_while_cond_54072___redundant_placeholder23
/while_while_cond_54072___redundant_placeholder33
/while_while_cond_54072___redundant_placeholder43
/while_while_cond_54072___redundant_placeholder53
/while_while_cond_54072___redundant_placeholder63
/while_while_cond_54072___redundant_placeholder73
/while_while_cond_54072___redundant_placeholder8
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ÄÍ
ù"
!__inference__traced_restore_54729
file_prefix:
 assignvariableop_conv2d_4_kernel:.
 assignvariableop_1_conv2d_4_bias:<
"assignvariableop_2_conv2d_5_kernel:.
 assignvariableop_3_conv2d_5_bias:I
7assignvariableop_4_rnn_2_my_lstm_cell_2_dense_10_kernel:'C
5assignvariableop_5_rnn_2_my_lstm_cell_2_dense_10_bias:I
7assignvariableop_6_rnn_2_my_lstm_cell_2_dense_11_kernel:'C
5assignvariableop_7_rnn_2_my_lstm_cell_2_dense_11_bias:I
7assignvariableop_8_rnn_2_my_lstm_cell_2_dense_12_kernel:'C
5assignvariableop_9_rnn_2_my_lstm_cell_2_dense_12_bias:J
8assignvariableop_10_rnn_2_my_lstm_cell_2_dense_13_kernel:'D
6assignvariableop_11_rnn_2_my_lstm_cell_2_dense_13_bias:5
#assignvariableop_12_dense_14_kernel:/
!assignvariableop_13_dense_14_bias:%
assignvariableop_14_total_1: %
assignvariableop_15_count_1: #
assignvariableop_16_total: #
assignvariableop_17_count: '
assignvariableop_18_adam_iter:	 )
assignvariableop_19_adam_beta_1: )
assignvariableop_20_adam_beta_2: (
assignvariableop_21_adam_decay: 0
&assignvariableop_22_adam_learning_rate: D
*assignvariableop_23_adam_conv2d_4_kernel_m:6
(assignvariableop_24_adam_conv2d_4_bias_m:D
*assignvariableop_25_adam_conv2d_5_kernel_m:6
(assignvariableop_26_adam_conv2d_5_bias_m:Q
?assignvariableop_27_adam_rnn_2_my_lstm_cell_2_dense_10_kernel_m:'K
=assignvariableop_28_adam_rnn_2_my_lstm_cell_2_dense_10_bias_m:Q
?assignvariableop_29_adam_rnn_2_my_lstm_cell_2_dense_11_kernel_m:'K
=assignvariableop_30_adam_rnn_2_my_lstm_cell_2_dense_11_bias_m:Q
?assignvariableop_31_adam_rnn_2_my_lstm_cell_2_dense_12_kernel_m:'K
=assignvariableop_32_adam_rnn_2_my_lstm_cell_2_dense_12_bias_m:Q
?assignvariableop_33_adam_rnn_2_my_lstm_cell_2_dense_13_kernel_m:'K
=assignvariableop_34_adam_rnn_2_my_lstm_cell_2_dense_13_bias_m:<
*assignvariableop_35_adam_dense_14_kernel_m:6
(assignvariableop_36_adam_dense_14_bias_m:D
*assignvariableop_37_adam_conv2d_4_kernel_v:6
(assignvariableop_38_adam_conv2d_4_bias_v:D
*assignvariableop_39_adam_conv2d_5_kernel_v:6
(assignvariableop_40_adam_conv2d_5_bias_v:Q
?assignvariableop_41_adam_rnn_2_my_lstm_cell_2_dense_10_kernel_v:'K
=assignvariableop_42_adam_rnn_2_my_lstm_cell_2_dense_10_bias_v:Q
?assignvariableop_43_adam_rnn_2_my_lstm_cell_2_dense_11_kernel_v:'K
=assignvariableop_44_adam_rnn_2_my_lstm_cell_2_dense_11_bias_v:Q
?assignvariableop_45_adam_rnn_2_my_lstm_cell_2_dense_12_kernel_v:'K
=assignvariableop_46_adam_rnn_2_my_lstm_cell_2_dense_12_bias_v:Q
?assignvariableop_47_adam_rnn_2_my_lstm_cell_2_dense_13_kernel_v:'K
=assignvariableop_48_adam_rnn_2_my_lstm_cell_2_dense_13_bias_v:<
*assignvariableop_49_adam_dense_14_kernel_v:6
(assignvariableop_50_adam_dense_14_bias_v:
identity_52¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Â
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*è
valueÞBÛ4B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHØ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ¥
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*æ
_output_shapesÓ
Ð::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp assignvariableop_conv2d_4_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_4_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_5_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_5_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_4AssignVariableOp7assignvariableop_4_rnn_2_my_lstm_cell_2_dense_10_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_5AssignVariableOp5assignvariableop_5_rnn_2_my_lstm_cell_2_dense_10_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_6AssignVariableOp7assignvariableop_6_rnn_2_my_lstm_cell_2_dense_11_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_7AssignVariableOp5assignvariableop_7_rnn_2_my_lstm_cell_2_dense_11_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_8AssignVariableOp7assignvariableop_8_rnn_2_my_lstm_cell_2_dense_12_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_9AssignVariableOp5assignvariableop_9_rnn_2_my_lstm_cell_2_dense_12_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_10AssignVariableOp8assignvariableop_10_rnn_2_my_lstm_cell_2_dense_13_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_11AssignVariableOp6assignvariableop_11_rnn_2_my_lstm_cell_2_dense_13_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_14_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_14_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_iterIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_beta_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_beta_2Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_decayIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_learning_rateIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_conv2d_4_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_conv2d_4_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv2d_5_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv2d_5_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_27AssignVariableOp?assignvariableop_27_adam_rnn_2_my_lstm_cell_2_dense_10_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_28AssignVariableOp=assignvariableop_28_adam_rnn_2_my_lstm_cell_2_dense_10_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_29AssignVariableOp?assignvariableop_29_adam_rnn_2_my_lstm_cell_2_dense_11_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_30AssignVariableOp=assignvariableop_30_adam_rnn_2_my_lstm_cell_2_dense_11_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_31AssignVariableOp?assignvariableop_31_adam_rnn_2_my_lstm_cell_2_dense_12_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_32AssignVariableOp=assignvariableop_32_adam_rnn_2_my_lstm_cell_2_dense_12_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_33AssignVariableOp?assignvariableop_33_adam_rnn_2_my_lstm_cell_2_dense_13_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_34AssignVariableOp=assignvariableop_34_adam_rnn_2_my_lstm_cell_2_dense_13_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_14_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_14_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv2d_4_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_conv2d_4_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_conv2d_5_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_conv2d_5_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_41AssignVariableOp?assignvariableop_41_adam_rnn_2_my_lstm_cell_2_dense_10_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_42AssignVariableOp=assignvariableop_42_adam_rnn_2_my_lstm_cell_2_dense_10_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_43AssignVariableOp?assignvariableop_43_adam_rnn_2_my_lstm_cell_2_dense_11_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_44AssignVariableOp=assignvariableop_44_adam_rnn_2_my_lstm_cell_2_dense_11_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_45AssignVariableOp?assignvariableop_45_adam_rnn_2_my_lstm_cell_2_dense_12_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_46AssignVariableOp=assignvariableop_46_adam_rnn_2_my_lstm_cell_2_dense_12_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_47AssignVariableOp?assignvariableop_47_adam_rnn_2_my_lstm_cell_2_dense_13_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_48AssignVariableOp=assignvariableop_48_adam_rnn_2_my_lstm_cell_2_dense_13_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_dense_14_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_dense_14_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ±	
Identity_51Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_52IdentityIdentity_51:output:0^NoOp_1*
T0*
_output_shapes
: 	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_52Identity_52:output:0*{
_input_shapesj
h: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_50AssignVariableOp_502(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¥_
Ï
while_body_54247
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0P
>while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource_0:'M
?while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource_0:P
>while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource_0:'M
?while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource_0:P
>while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource_0:'M
?while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource_0:P
>while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource_0:'M
?while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorN
<while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource:'K
=while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource:N
<while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource:'K
=while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource:N
<while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource:'K
=while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource:N
<while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource:'K
=while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource:¢4while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp¢3while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp¢4while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp¢3while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp¢4while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp¢3while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp¢4while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp¢3while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0p
.while/my_lstm_cell_2/concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ð
)while/my_lstm_cell_2/concatenate_5/concatConcatV20while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_27while/my_lstm_cell_2/concatenate_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'²
3while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOpReadVariableOp>while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0Ñ
$while/my_lstm_cell_2/dense_10/MatMulMatMul2while/my_lstm_cell_2/concatenate_5/concat:output:0;while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
4while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOpReadVariableOp?while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Ð
%while/my_lstm_cell_2/dense_10/BiasAddBiasAdd.while/my_lstm_cell_2/dense_10/MatMul:product:0<while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%while/my_lstm_cell_2/dense_10/SigmoidSigmoid.while/my_lstm_cell_2/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#while/my_lstm_cell_2/multiply_2/mulMulwhile_placeholder_3)while/my_lstm_cell_2/dense_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
3while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOpReadVariableOp>while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0Ñ
$while/my_lstm_cell_2/dense_11/MatMulMatMul2while/my_lstm_cell_2/concatenate_5/concat:output:0;while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
4while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOpReadVariableOp?while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Ð
%while/my_lstm_cell_2/dense_11/BiasAddBiasAdd.while/my_lstm_cell_2/dense_11/MatMul:product:0<while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%while/my_lstm_cell_2/dense_11/SigmoidSigmoid.while/my_lstm_cell_2/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
3while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOpReadVariableOp>while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0Ñ
$while/my_lstm_cell_2/dense_12/MatMulMatMul2while/my_lstm_cell_2/concatenate_5/concat:output:0;while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
4while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOpReadVariableOp?while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Ð
%while/my_lstm_cell_2/dense_12/BiasAddBiasAdd.while/my_lstm_cell_2/dense_12/MatMul:product:0<while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"while/my_lstm_cell_2/dense_12/TanhTanh.while/my_lstm_cell_2/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
%while/my_lstm_cell_2/multiply_2/mul_1Mul)while/my_lstm_cell_2/dense_11/Sigmoid:y:0&while/my_lstm_cell_2/dense_12/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
while/my_lstm_cell_2/add_5/addAddV2'while/my_lstm_cell_2/multiply_2/mul:z:0)while/my_lstm_cell_2/multiply_2/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
3while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOpReadVariableOp>while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0Ñ
$while/my_lstm_cell_2/dense_13/MatMulMatMul2while/my_lstm_cell_2/concatenate_5/concat:output:0;while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
4while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOpReadVariableOp?while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Ð
%while/my_lstm_cell_2/dense_13/BiasAddBiasAdd.while/my_lstm_cell_2/dense_13/MatMul:product:0<while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%while/my_lstm_cell_2/dense_13/SigmoidSigmoid.while/my_lstm_cell_2/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&while/my_lstm_cell_2/activation_2/TanhTanh"while/my_lstm_cell_2/add_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
%while/my_lstm_cell_2/multiply_2/mul_2Mul*while/my_lstm_cell_2/activation_2/Tanh:y:0)while/my_lstm_cell_2/dense_13/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder)while/my_lstm_cell_2/multiply_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity)while/my_lstm_cell_2/multiply_2/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity"while/my_lstm_cell_2/add_5/add:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/NoOpNoOp5^while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp4^while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp5^while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp4^while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp5^while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp4^while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp5^while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp4^while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"
=while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource?while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource_0"~
<while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource>while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource_0"
=while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource?while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource_0"~
<while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource>while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource_0"
=while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource?while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource_0"~
<while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource>while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource_0"
=while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource?while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource_0"~
<while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource>while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2l
4while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp4while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp2j
3while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp3while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp2l
4while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp4while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp2j
3while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp3while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp2l
4while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp4while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp2j
3while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp3while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp2l
4while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp4while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp2j
3while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp3while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ý
à
.__inference_my_cnn_block_2_layer_call_fn_53313	
input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_my_cnn_block_2_layer_call_and_return_conditional_losses_51411{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
×
ó
__inference_call_52645
x.
my_cnn_block_2_52431:"
my_cnn_block_2_52433:.
my_cnn_block_2_52435:"
my_cnn_block_2_52437:N
<rnn_2_my_lstm_cell_2_dense_10_matmul_readvariableop_resource:'K
=rnn_2_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource:N
<rnn_2_my_lstm_cell_2_dense_11_matmul_readvariableop_resource:'K
=rnn_2_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource:N
<rnn_2_my_lstm_cell_2_dense_12_matmul_readvariableop_resource:'K
=rnn_2_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource:N
<rnn_2_my_lstm_cell_2_dense_13_matmul_readvariableop_resource:'K
=rnn_2_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource:<
*dense_14_tensordot_readvariableop_resource:6
(dense_14_biasadd_readvariableop_resource:
identity¢dense_14/BiasAdd/ReadVariableOp¢!dense_14/Tensordot/ReadVariableOp¢&my_cnn_block_2/StatefulPartitionedCall¢4rnn_2/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp¢3rnn_2/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp¢4rnn_2/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp¢3rnn_2/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp¢4rnn_2/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp¢3rnn_2/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp¢4rnn_2/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp¢3rnn_2/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp¢rnn_2/while
&my_cnn_block_2/StatefulPartitionedCallStatefulPartitionedCallxmy_cnn_block_2_52431my_cnn_block_2_52433my_cnn_block_2_52435my_cnn_block_2_52437*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *
fR
__inference_call_50641y
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         »
time_distributed_2/ReshapeReshape/my_cnn_block_2/StatefulPartitionedCall:output:0)time_distributed_2/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Dtime_distributed_2/global_average_pooling2d_2/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      à
2time_distributed_2/global_average_pooling2d_2/MeanMean#time_distributed_2/Reshape:output:0Mtime_distributed_2/global_average_pooling2d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
"time_distributed_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      Ç
time_distributed_2/Reshape_1Reshape;time_distributed_2/global_average_pooling2d_2/Mean:output:0+time_distributed_2/Reshape_1/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
"time_distributed_2/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         ¿
time_distributed_2/Reshape_2Reshape/my_cnn_block_2/StatefulPartitionedCall:output:0+time_distributed_2/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
rnn_2/ShapeShape%time_distributed_2/Reshape_1:output:0*
T0*
_output_shapes
:c
rnn_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
rnn_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
rnn_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
rnn_2/strided_sliceStridedSlicernn_2/Shape:output:0"rnn_2/strided_slice/stack:output:0$rnn_2/strided_slice/stack_1:output:0$rnn_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
rnn_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
rnn_2/zeros/packedPackrnn_2/strided_slice:output:0rnn_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
rnn_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
rnn_2/zerosFillrnn_2/zeros/packed:output:0rnn_2/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
rnn_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
rnn_2/zeros_1/packedPackrnn_2/strided_slice:output:0rnn_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:X
rnn_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
rnn_2/zeros_1Fillrnn_2/zeros_1/packed:output:0rnn_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
rnn_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
rnn_2/transpose	Transpose%time_distributed_2/Reshape_1:output:0rnn_2/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
rnn_2/Shape_1Shapernn_2/transpose:y:0*
T0*
_output_shapes
:e
rnn_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: g
rnn_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
rnn_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
rnn_2/strided_slice_1StridedSlicernn_2/Shape_1:output:0$rnn_2/strided_slice_1/stack:output:0&rnn_2/strided_slice_1/stack_1:output:0&rnn_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!rnn_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
rnn_2/TensorArrayV2TensorListReserve*rnn_2/TensorArrayV2/element_shape:output:0rnn_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
;rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ò
-rnn_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn_2/transpose:y:0Drnn_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒe
rnn_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: g
rnn_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
rnn_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
rnn_2/strided_slice_2StridedSlicernn_2/transpose:y:0$rnn_2/strided_slice_2/stack:output:0&rnn_2/strided_slice_2/stack_1:output:0&rnn_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskp
.rnn_2/my_lstm_cell_2/concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ß
)rnn_2/my_lstm_cell_2/concatenate_5/concatConcatV2rnn_2/strided_slice_2:output:0rnn_2/zeros:output:07rnn_2/my_lstm_cell_2/concatenate_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'°
3rnn_2/my_lstm_cell_2/dense_10/MatMul/ReadVariableOpReadVariableOp<rnn_2_my_lstm_cell_2_dense_10_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0Ñ
$rnn_2/my_lstm_cell_2/dense_10/MatMulMatMul2rnn_2/my_lstm_cell_2/concatenate_5/concat:output:0;rnn_2/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
4rnn_2/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOpReadVariableOp=rnn_2_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ð
%rnn_2/my_lstm_cell_2/dense_10/BiasAddBiasAdd.rnn_2/my_lstm_cell_2/dense_10/MatMul:product:0<rnn_2/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%rnn_2/my_lstm_cell_2/dense_10/SigmoidSigmoid.rnn_2/my_lstm_cell_2/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#rnn_2/my_lstm_cell_2/multiply_2/mulMulrnn_2/zeros_1:output:0)rnn_2/my_lstm_cell_2/dense_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
3rnn_2/my_lstm_cell_2/dense_11/MatMul/ReadVariableOpReadVariableOp<rnn_2_my_lstm_cell_2_dense_11_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0Ñ
$rnn_2/my_lstm_cell_2/dense_11/MatMulMatMul2rnn_2/my_lstm_cell_2/concatenate_5/concat:output:0;rnn_2/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
4rnn_2/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOpReadVariableOp=rnn_2_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ð
%rnn_2/my_lstm_cell_2/dense_11/BiasAddBiasAdd.rnn_2/my_lstm_cell_2/dense_11/MatMul:product:0<rnn_2/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%rnn_2/my_lstm_cell_2/dense_11/SigmoidSigmoid.rnn_2/my_lstm_cell_2/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
3rnn_2/my_lstm_cell_2/dense_12/MatMul/ReadVariableOpReadVariableOp<rnn_2_my_lstm_cell_2_dense_12_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0Ñ
$rnn_2/my_lstm_cell_2/dense_12/MatMulMatMul2rnn_2/my_lstm_cell_2/concatenate_5/concat:output:0;rnn_2/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
4rnn_2/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOpReadVariableOp=rnn_2_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ð
%rnn_2/my_lstm_cell_2/dense_12/BiasAddBiasAdd.rnn_2/my_lstm_cell_2/dense_12/MatMul:product:0<rnn_2/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"rnn_2/my_lstm_cell_2/dense_12/TanhTanh.rnn_2/my_lstm_cell_2/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
%rnn_2/my_lstm_cell_2/multiply_2/mul_1Mul)rnn_2/my_lstm_cell_2/dense_11/Sigmoid:y:0&rnn_2/my_lstm_cell_2/dense_12/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
rnn_2/my_lstm_cell_2/add_5/addAddV2'rnn_2/my_lstm_cell_2/multiply_2/mul:z:0)rnn_2/my_lstm_cell_2/multiply_2/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
3rnn_2/my_lstm_cell_2/dense_13/MatMul/ReadVariableOpReadVariableOp<rnn_2_my_lstm_cell_2_dense_13_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0Ñ
$rnn_2/my_lstm_cell_2/dense_13/MatMulMatMul2rnn_2/my_lstm_cell_2/concatenate_5/concat:output:0;rnn_2/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
4rnn_2/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOpReadVariableOp=rnn_2_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ð
%rnn_2/my_lstm_cell_2/dense_13/BiasAddBiasAdd.rnn_2/my_lstm_cell_2/dense_13/MatMul:product:0<rnn_2/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%rnn_2/my_lstm_cell_2/dense_13/SigmoidSigmoid.rnn_2/my_lstm_cell_2/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&rnn_2/my_lstm_cell_2/activation_2/TanhTanh"rnn_2/my_lstm_cell_2/add_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
%rnn_2/my_lstm_cell_2/multiply_2/mul_2Mul*rnn_2/my_lstm_cell_2/activation_2/Tanh:y:0)rnn_2/my_lstm_cell_2/dense_13/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
#rnn_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ê
rnn_2/TensorArrayV2_1TensorListReserve,rnn_2/TensorArrayV2_1/element_shape:output:0rnn_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒL

rnn_2/timeConst*
_output_shapes
: *
dtype0*
value	B : i
rnn_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿZ
rnn_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ê	
rnn_2/whileWhile!rnn_2/while/loop_counter:output:0'rnn_2/while/maximum_iterations:output:0rnn_2/time:output:0rnn_2/TensorArrayV2_1:handle:0rnn_2/zeros:output:0rnn_2/zeros_1:output:0rnn_2/strided_slice_1:output:0=rnn_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0<rnn_2_my_lstm_cell_2_dense_10_matmul_readvariableop_resource=rnn_2_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource<rnn_2_my_lstm_cell_2_dense_11_matmul_readvariableop_resource=rnn_2_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource<rnn_2_my_lstm_cell_2_dense_12_matmul_readvariableop_resource=rnn_2_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource<rnn_2_my_lstm_cell_2_dense_13_matmul_readvariableop_resource=rnn_2_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( *"
bodyR
rnn_2_while_body_52515*"
condR
rnn_2_while_cond_52514*U
output_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : *
parallel_iterations 
6rnn_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ô
(rnn_2/TensorArrayV2Stack/TensorListStackTensorListStackrnn_2/while:output:3?rnn_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0n
rnn_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿg
rnn_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
rnn_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¥
rnn_2/strided_slice_3StridedSlice1rnn_2/TensorArrayV2Stack/TensorListStack:tensor:0$rnn_2/strided_slice_3/stack:output:0&rnn_2/strided_slice_3/stack_1:output:0&rnn_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskk
rnn_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¨
rnn_2/transpose_1	Transpose1rnn_2/TensorArrayV2Stack/TensorListStack:tensor:0rnn_2/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_14/Tensordot/ReadVariableOpReadVariableOp*dense_14_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0a
dense_14/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_14/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ]
dense_14/Tensordot/ShapeShapernn_2/transpose_1:y:0*
T0*
_output_shapes
:b
 dense_14/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_14/Tensordot/GatherV2GatherV2!dense_14/Tensordot/Shape:output:0 dense_14/Tensordot/free:output:0)dense_14/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_14/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
dense_14/Tensordot/GatherV2_1GatherV2!dense_14/Tensordot/Shape:output:0 dense_14/Tensordot/axes:output:0+dense_14/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_14/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_14/Tensordot/ProdProd$dense_14/Tensordot/GatherV2:output:0!dense_14/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_14/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_14/Tensordot/Prod_1Prod&dense_14/Tensordot/GatherV2_1:output:0#dense_14/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_14/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
dense_14/Tensordot/concatConcatV2 dense_14/Tensordot/free:output:0 dense_14/Tensordot/axes:output:0'dense_14/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_14/Tensordot/stackPack dense_14/Tensordot/Prod:output:0"dense_14/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_14/Tensordot/transpose	Transposernn_2/transpose_1:y:0"dense_14/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
dense_14/Tensordot/ReshapeReshape dense_14/Tensordot/transpose:y:0!dense_14/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
dense_14/Tensordot/MatMulMatMul#dense_14/Tensordot/Reshape:output:0)dense_14/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_14/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_14/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
dense_14/Tensordot/concat_1ConcatV2$dense_14/Tensordot/GatherV2:output:0#dense_14/Tensordot/Const_2:output:0)dense_14/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_14/TensordotReshape#dense_14/Tensordot/MatMul:product:0$dense_14/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_14/BiasAddBiasAdddense_14/Tensordot:output:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
IdentityIdentitydense_14/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ÷
NoOpNoOp ^dense_14/BiasAdd/ReadVariableOp"^dense_14/Tensordot/ReadVariableOp'^my_cnn_block_2/StatefulPartitionedCall5^rnn_2/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp4^rnn_2/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp5^rnn_2/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp4^rnn_2/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp5^rnn_2/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp4^rnn_2/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp5^rnn_2/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp4^rnn_2/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp^rnn_2/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2F
!dense_14/Tensordot/ReadVariableOp!dense_14/Tensordot/ReadVariableOp2P
&my_cnn_block_2/StatefulPartitionedCall&my_cnn_block_2/StatefulPartitionedCall2l
4rnn_2/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp4rnn_2/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp2j
3rnn_2/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp3rnn_2/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp2l
4rnn_2/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp4rnn_2/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp2j
3rnn_2/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp3rnn_2/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp2l
4rnn_2/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp4rnn_2/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp2j
3rnn_2/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp3rnn_2/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp2l
4rnn_2/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp4rnn_2/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp2j
3rnn_2/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp3rnn_2/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp2
rnn_2/whilernn_2/while:V R
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
ë

 __inference__wrapped_model_50887
input_1/
my_lstm_model_2_50857:#
my_lstm_model_2_50859:/
my_lstm_model_2_50861:#
my_lstm_model_2_50863:'
my_lstm_model_2_50865:'#
my_lstm_model_2_50867:'
my_lstm_model_2_50869:'#
my_lstm_model_2_50871:'
my_lstm_model_2_50873:'#
my_lstm_model_2_50875:'
my_lstm_model_2_50877:'#
my_lstm_model_2_50879:'
my_lstm_model_2_50881:#
my_lstm_model_2_50883:
identity¢'my_lstm_model_2/StatefulPartitionedCall
'my_lstm_model_2/StatefulPartitionedCallStatefulPartitionedCallinput_1my_lstm_model_2_50857my_lstm_model_2_50859my_lstm_model_2_50861my_lstm_model_2_50863my_lstm_model_2_50865my_lstm_model_2_50867my_lstm_model_2_50869my_lstm_model_2_50871my_lstm_model_2_50873my_lstm_model_2_50875my_lstm_model_2_50877my_lstm_model_2_50879my_lstm_model_2_50881my_lstm_model_2_50883*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *
fR
__inference_call_50856
IdentityIdentity0my_lstm_model_2/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
NoOpNoOp(^my_lstm_model_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2R
'my_lstm_model_2/StatefulPartitionedCall'my_lstm_model_2/StatefulPartitionedCall:\ X
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
í
î
/__inference_my_lstm_model_2_layer_call_fn_52746
x!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:'
	unknown_4:
	unknown_5:'
	unknown_6:
	unknown_7:'
	unknown_8:
	unknown_9:'

unknown_10:

unknown_11:

unknown_12:
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_my_lstm_model_2_layer_call_and_return_conditional_losses_52005s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
*
¢

while_body_51477
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_my_lstm_cell_2_51501_0:'*
while_my_lstm_cell_2_51503_0:.
while_my_lstm_cell_2_51505_0:'*
while_my_lstm_cell_2_51507_0:.
while_my_lstm_cell_2_51509_0:'*
while_my_lstm_cell_2_51511_0:.
while_my_lstm_cell_2_51513_0:'*
while_my_lstm_cell_2_51515_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_my_lstm_cell_2_51501:'(
while_my_lstm_cell_2_51503:,
while_my_lstm_cell_2_51505:'(
while_my_lstm_cell_2_51507:,
while_my_lstm_cell_2_51509:'(
while_my_lstm_cell_2_51511:,
while_my_lstm_cell_2_51513:'(
while_my_lstm_cell_2_51515:¢,while/my_lstm_cell_2/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0Ù
,while/my_lstm_cell_2/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_my_lstm_cell_2_51501_0while_my_lstm_cell_2_51503_0while_my_lstm_cell_2_51505_0while_my_lstm_cell_2_51507_0while_my_lstm_cell_2_51509_0while_my_lstm_cell_2_51511_0while_my_lstm_cell_2_51513_0while_my_lstm_cell_2_51515_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_my_lstm_cell_2_layer_call_and_return_conditional_losses_51022Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder5while/my_lstm_cell_2/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity5while/my_lstm_cell_2/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity5while/my_lstm_cell_2/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{

while/NoOpNoOp-^while/my_lstm_cell_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_my_lstm_cell_2_51501while_my_lstm_cell_2_51501_0":
while_my_lstm_cell_2_51503while_my_lstm_cell_2_51503_0":
while_my_lstm_cell_2_51505while_my_lstm_cell_2_51505_0":
while_my_lstm_cell_2_51507while_my_lstm_cell_2_51507_0":
while_my_lstm_cell_2_51509while_my_lstm_cell_2_51509_0":
while_my_lstm_cell_2_51511while_my_lstm_cell_2_51511_0":
while_my_lstm_cell_2_51513while_my_lstm_cell_2_51513_0":
while_my_lstm_cell_2_51515while_my_lstm_cell_2_51515_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2\
,while/my_lstm_cell_2/StatefulPartitionedCall,while/my_lstm_cell_2/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Í

Ç
while_cond_51476
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_51476___redundant_placeholder03
/while_while_cond_51476___redundant_placeholder13
/while_while_cond_51476___redundant_placeholder23
/while_while_cond_51476___redundant_placeholder33
/while_while_cond_51476___redundant_placeholder43
/while_while_cond_51476___redundant_placeholder53
/while_while_cond_51476___redundant_placeholder63
/while_while_cond_51476___redundant_placeholder73
/while_while_cond_51476___redundant_placeholder8
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Õ:
À
@__inference_rnn_2_layer_call_and_return_conditional_losses_51826

inputs&
my_lstm_cell_2_51715:'"
my_lstm_cell_2_51717:&
my_lstm_cell_2_51719:'"
my_lstm_cell_2_51721:&
my_lstm_cell_2_51723:'"
my_lstm_cell_2_51725:&
my_lstm_cell_2_51727:'"
my_lstm_cell_2_51729:
identity¢&my_lstm_cell_2/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskó
&my_lstm_cell_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0my_lstm_cell_2_51715my_lstm_cell_2_51717my_lstm_cell_2_51719my_lstm_cell_2_51721my_lstm_cell_2_51723my_lstm_cell_2_51725my_lstm_cell_2_51727my_lstm_cell_2_51729*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_my_lstm_cell_2_layer_call_and_return_conditional_losses_51022n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ä
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0my_lstm_cell_2_51715my_lstm_cell_2_51717my_lstm_cell_2_51719my_lstm_cell_2_51721my_lstm_cell_2_51723my_lstm_cell_2_51725my_lstm_cell_2_51727my_lstm_cell_2_51729*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( *
bodyR
while_body_51738*
condR
while_cond_51737*U
output_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp'^my_lstm_cell_2/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2P
&my_lstm_cell_2/StatefulPartitionedCall&my_lstm_cell_2/StatefulPartitionedCall2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É
è
#__inference_signature_wrapper_52680
input_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:'
	unknown_4:
	unknown_5:'
	unknown_6:
	unknown_7:'
	unknown_8:
	unknown_9:'

unknown_10:

unknown_11:

unknown_12:
identity¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_50887s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Í

Ç
while_cond_51045
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_51045___redundant_placeholder03
/while_while_cond_51045___redundant_placeholder13
/while_while_cond_51045___redundant_placeholder23
/while_while_cond_51045___redundant_placeholder33
/while_while_cond_51045___redundant_placeholder43
/while_while_cond_51045___redundant_placeholder53
/while_while_cond_51045___redundant_placeholder63
/while_while_cond_51045___redundant_placeholder73
/while_while_cond_51045___redundant_placeholder8
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ÞN
Ë
I__inference_my_cnn_block_2_layer_call_and_return_conditional_losses_53446	
inputH
.conv2d_4_conv2d_conv2d_readvariableop_resource:I
;conv2d_4_squeeze_batch_dims_biasadd_readvariableop_resource:H
.conv2d_5_conv2d_conv2d_readvariableop_resource:I
;conv2d_5_squeeze_batch_dims_biasadd_readvariableop_resource:
identity¢%conv2d_4/Conv2D/Conv2D/ReadVariableOp¢2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp¢%conv2d_5/Conv2D/Conv2D/ReadVariableOp¢2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOpJ
conv2d_4/Conv2D/ShapeShapeinput*
T0*
_output_shapes
:m
#conv2d_4/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_4/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿo
%conv2d_4/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv2d_4/Conv2D/strided_sliceStridedSliceconv2d_4/Conv2D/Shape:output:0,conv2d_4/Conv2D/strided_slice/stack:output:0.conv2d_4/Conv2D/strided_slice/stack_1:output:0.conv2d_4/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_4/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         
conv2d_4/Conv2D/ReshapeReshapeinput&conv2d_4/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%conv2d_4/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_4_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ó
conv2d_4/Conv2D/Conv2DConv2D conv2d_4/Conv2D/Reshape:output:0-conv2d_4/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
t
conv2d_4/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         f
conv2d_4/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÈ
conv2d_4/Conv2D/concatConcatV2&conv2d_4/Conv2D/strided_slice:output:0(conv2d_4/Conv2D/concat/values_1:output:0$conv2d_4/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:¤
conv2d_4/Conv2D/Reshape_1Reshapeconv2d_4/Conv2D/Conv2D:output:0conv2d_4/Conv2D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿs
!conv2d_4/squeeze_batch_dims/ShapeShape"conv2d_4/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_4/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1conv2d_4/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ{
1conv2d_4/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
)conv2d_4/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_4/squeeze_batch_dims/Shape:output:08conv2d_4/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_4/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_4/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
)conv2d_4/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         À
#conv2d_4/squeeze_batch_dims/ReshapeReshape"conv2d_4/Conv2D/Reshape_1:output:02conv2d_4/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_4_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ò
#conv2d_4/squeeze_batch_dims/BiasAddBiasAdd,conv2d_4/squeeze_batch_dims/Reshape:output:0:conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+conv2d_4/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         r
'conv2d_4/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿø
"conv2d_4/squeeze_batch_dims/concatConcatV22conv2d_4/squeeze_batch_dims/strided_slice:output:04conv2d_4/squeeze_batch_dims/concat/values_1:output:00conv2d_4/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:É
%conv2d_4/squeeze_batch_dims/Reshape_1Reshape,conv2d_4/squeeze_batch_dims/BiasAdd:output:0+conv2d_4/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿs
conv2d_5/Conv2D/ShapeShape.conv2d_4/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:m
#conv2d_5/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_5/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿo
%conv2d_5/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv2d_5/Conv2D/strided_sliceStridedSliceconv2d_5/Conv2D/Shape:output:0,conv2d_5/Conv2D/strided_slice/stack:output:0.conv2d_5/Conv2D/strided_slice/stack_1:output:0.conv2d_5/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_5/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         ´
conv2d_5/Conv2D/ReshapeReshape.conv2d_4/squeeze_batch_dims/Reshape_1:output:0&conv2d_5/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%conv2d_5/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_5_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ó
conv2d_5/Conv2D/Conv2DConv2D conv2d_5/Conv2D/Reshape:output:0-conv2d_5/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
t
conv2d_5/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         f
conv2d_5/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÈ
conv2d_5/Conv2D/concatConcatV2&conv2d_5/Conv2D/strided_slice:output:0(conv2d_5/Conv2D/concat/values_1:output:0$conv2d_5/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:¤
conv2d_5/Conv2D/Reshape_1Reshapeconv2d_5/Conv2D/Conv2D:output:0conv2d_5/Conv2D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿs
!conv2d_5/squeeze_batch_dims/ShapeShape"conv2d_5/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_5/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1conv2d_5/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ{
1conv2d_5/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
)conv2d_5/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_5/squeeze_batch_dims/Shape:output:08conv2d_5/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_5/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_5/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
)conv2d_5/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         À
#conv2d_5/squeeze_batch_dims/ReshapeReshape"conv2d_5/Conv2D/Reshape_1:output:02conv2d_5/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_5_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ò
#conv2d_5/squeeze_batch_dims/BiasAddBiasAdd,conv2d_5/squeeze_batch_dims/Reshape:output:0:conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+conv2d_5/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         r
'conv2d_5/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿø
"conv2d_5/squeeze_batch_dims/concatConcatV22conv2d_5/squeeze_batch_dims/strided_slice:output:04conv2d_5/squeeze_batch_dims/concat/values_1:output:00conv2d_5/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:É
%conv2d_5/squeeze_batch_dims/Reshape_1Reshape,conv2d_5/squeeze_batch_dims/BiasAdd:output:0+conv2d_5/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity.conv2d_5/squeeze_batch_dims/Reshape_1:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp&^conv2d_4/Conv2D/Conv2D/ReadVariableOp3^conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_5/Conv2D/Conv2D/ReadVariableOp3^conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : 2N
%conv2d_4/Conv2D/Conv2D/ReadVariableOp%conv2d_4/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_5/Conv2D/Conv2D/ReadVariableOp%conv2d_5/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp:Z V
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
¥_
Ï
while_body_53725
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0P
>while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource_0:'M
?while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource_0:P
>while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource_0:'M
?while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource_0:P
>while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource_0:'M
?while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource_0:P
>while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource_0:'M
?while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorN
<while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource:'K
=while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource:N
<while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource:'K
=while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource:N
<while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource:'K
=while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource:N
<while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource:'K
=while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource:¢4while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp¢3while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp¢4while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp¢3while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp¢4while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp¢3while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp¢4while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp¢3while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0p
.while/my_lstm_cell_2/concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ð
)while/my_lstm_cell_2/concatenate_5/concatConcatV20while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_27while/my_lstm_cell_2/concatenate_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'²
3while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOpReadVariableOp>while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0Ñ
$while/my_lstm_cell_2/dense_10/MatMulMatMul2while/my_lstm_cell_2/concatenate_5/concat:output:0;while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
4while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOpReadVariableOp?while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Ð
%while/my_lstm_cell_2/dense_10/BiasAddBiasAdd.while/my_lstm_cell_2/dense_10/MatMul:product:0<while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%while/my_lstm_cell_2/dense_10/SigmoidSigmoid.while/my_lstm_cell_2/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#while/my_lstm_cell_2/multiply_2/mulMulwhile_placeholder_3)while/my_lstm_cell_2/dense_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
3while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOpReadVariableOp>while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0Ñ
$while/my_lstm_cell_2/dense_11/MatMulMatMul2while/my_lstm_cell_2/concatenate_5/concat:output:0;while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
4while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOpReadVariableOp?while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Ð
%while/my_lstm_cell_2/dense_11/BiasAddBiasAdd.while/my_lstm_cell_2/dense_11/MatMul:product:0<while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%while/my_lstm_cell_2/dense_11/SigmoidSigmoid.while/my_lstm_cell_2/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
3while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOpReadVariableOp>while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0Ñ
$while/my_lstm_cell_2/dense_12/MatMulMatMul2while/my_lstm_cell_2/concatenate_5/concat:output:0;while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
4while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOpReadVariableOp?while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Ð
%while/my_lstm_cell_2/dense_12/BiasAddBiasAdd.while/my_lstm_cell_2/dense_12/MatMul:product:0<while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"while/my_lstm_cell_2/dense_12/TanhTanh.while/my_lstm_cell_2/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
%while/my_lstm_cell_2/multiply_2/mul_1Mul)while/my_lstm_cell_2/dense_11/Sigmoid:y:0&while/my_lstm_cell_2/dense_12/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
while/my_lstm_cell_2/add_5/addAddV2'while/my_lstm_cell_2/multiply_2/mul:z:0)while/my_lstm_cell_2/multiply_2/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
3while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOpReadVariableOp>while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0Ñ
$while/my_lstm_cell_2/dense_13/MatMulMatMul2while/my_lstm_cell_2/concatenate_5/concat:output:0;while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
4while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOpReadVariableOp?while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Ð
%while/my_lstm_cell_2/dense_13/BiasAddBiasAdd.while/my_lstm_cell_2/dense_13/MatMul:product:0<while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%while/my_lstm_cell_2/dense_13/SigmoidSigmoid.while/my_lstm_cell_2/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&while/my_lstm_cell_2/activation_2/TanhTanh"while/my_lstm_cell_2/add_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
%while/my_lstm_cell_2/multiply_2/mul_2Mul*while/my_lstm_cell_2/activation_2/Tanh:y:0)while/my_lstm_cell_2/dense_13/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder)while/my_lstm_cell_2/multiply_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity)while/my_lstm_cell_2/multiply_2/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity"while/my_lstm_cell_2/add_5/add:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/NoOpNoOp5^while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp4^while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp5^while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp4^while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp5^while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp4^while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp5^while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp4^while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"
=while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource?while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource_0"~
<while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource>while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource_0"
=while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource?while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource_0"~
<while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource>while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource_0"
=while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource?while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource_0"~
<while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource>while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource_0"
=while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource?while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource_0"~
<while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource>while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2l
4while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp4while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp2j
3while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp3while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp2l
4while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp4while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp2j
3while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp3while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp2l
4while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp4while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp2j
3while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp3while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp2l
4while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp4while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp2j
3while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp3while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 

¦
J__inference_my_lstm_model_2_layer_call_and_return_conditional_losses_52107
input_1.
my_cnn_block_2_52072:"
my_cnn_block_2_52074:.
my_cnn_block_2_52076:"
my_cnn_block_2_52078:
rnn_2_52084:'
rnn_2_52086:
rnn_2_52088:'
rnn_2_52090:
rnn_2_52092:'
rnn_2_52094:
rnn_2_52096:'
rnn_2_52098: 
dense_14_52101:
dense_14_52103:
identity¢ dense_14/StatefulPartitionedCall¢&my_cnn_block_2/StatefulPartitionedCall¢rnn_2/StatefulPartitionedCallÂ
&my_cnn_block_2/StatefulPartitionedCallStatefulPartitionedCallinput_1my_cnn_block_2_52072my_cnn_block_2_52074my_cnn_block_2_52076my_cnn_block_2_52078*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_my_cnn_block_2_layer_call_and_return_conditional_losses_51411ø
"time_distributed_2/PartitionedCallPartitionedCall/my_cnn_block_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_time_distributed_2_layer_call_and_return_conditional_losses_50920y
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         »
time_distributed_2/ReshapeReshape/my_cnn_block_2/StatefulPartitionedCall:output:0)time_distributed_2/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿä
rnn_2/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_2/PartitionedCall:output:0rnn_2_52084rnn_2_52086rnn_2_52088rnn_2_52090rnn_2_52092rnn_2_52094rnn_2_52096rnn_2_52098*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_rnn_2_layer_call_and_return_conditional_losses_51565
 dense_14/StatefulPartitionedCallStatefulPartitionedCall&rnn_2/StatefulPartitionedCall:output:0dense_14_52101dense_14_52103*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_51613|
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
NoOpNoOp!^dense_14/StatefulPartitionedCall'^my_cnn_block_2/StatefulPartitionedCall^rnn_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2P
&my_cnn_block_2/StatefulPartitionedCall&my_cnn_block_2/StatefulPartitionedCall2>
rnn_2/StatefulPartitionedCallrnn_2/StatefulPartitionedCall:\ X
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Í

Ç
while_cond_51737
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_51737___redundant_placeholder03
/while_while_cond_51737___redundant_placeholder13
/while_while_cond_51737___redundant_placeholder23
/while_while_cond_51737___redundant_placeholder33
/while_while_cond_51737___redundant_placeholder43
/while_while_cond_51737___redundant_placeholder53
/while_while_cond_51737___redundant_placeholder63
/while_while_cond_51737___redundant_placeholder73
/while_while_cond_51737___redundant_placeholder8
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
¶
q
U__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_50897

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ:
À
@__inference_rnn_2_layer_call_and_return_conditional_losses_51565

inputs&
my_lstm_cell_2_51454:'"
my_lstm_cell_2_51456:&
my_lstm_cell_2_51458:'"
my_lstm_cell_2_51460:&
my_lstm_cell_2_51462:'"
my_lstm_cell_2_51464:&
my_lstm_cell_2_51466:'"
my_lstm_cell_2_51468:
identity¢&my_lstm_cell_2/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskó
&my_lstm_cell_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0my_lstm_cell_2_51454my_lstm_cell_2_51456my_lstm_cell_2_51458my_lstm_cell_2_51460my_lstm_cell_2_51462my_lstm_cell_2_51464my_lstm_cell_2_51466my_lstm_cell_2_51468*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_my_lstm_cell_2_layer_call_and_return_conditional_losses_51022n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ä
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0my_lstm_cell_2_51454my_lstm_cell_2_51456my_lstm_cell_2_51458my_lstm_cell_2_51460my_lstm_cell_2_51462my_lstm_cell_2_51464my_lstm_cell_2_51466my_lstm_cell_2_51468*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( *
bodyR
while_body_51477*
condR
while_cond_51476*U
output_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp'^my_lstm_cell_2/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2P
&my_lstm_cell_2/StatefulPartitionedCall&my_lstm_cell_2/StatefulPartitionedCall2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
;
À
@__inference_rnn_2_layer_call_and_return_conditional_losses_51325

inputs&
my_lstm_cell_2_51214:'"
my_lstm_cell_2_51216:&
my_lstm_cell_2_51218:'"
my_lstm_cell_2_51220:&
my_lstm_cell_2_51222:'"
my_lstm_cell_2_51224:&
my_lstm_cell_2_51226:'"
my_lstm_cell_2_51228:
identity¢&my_lstm_cell_2/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskó
&my_lstm_cell_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0my_lstm_cell_2_51214my_lstm_cell_2_51216my_lstm_cell_2_51218my_lstm_cell_2_51220my_lstm_cell_2_51222my_lstm_cell_2_51224my_lstm_cell_2_51226my_lstm_cell_2_51228*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_my_lstm_cell_2_layer_call_and_return_conditional_losses_51022n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ä
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0my_lstm_cell_2_51214my_lstm_cell_2_51216my_lstm_cell_2_51218my_lstm_cell_2_51220my_lstm_cell_2_51222my_lstm_cell_2_51224my_lstm_cell_2_51226my_lstm_cell_2_51228*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( *
bodyR
while_body_51237*
condR
while_cond_51236*U
output_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿk
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿw
NoOpNoOp'^my_lstm_cell_2/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : 2P
&my_lstm_cell_2/StatefulPartitionedCall&my_lstm_cell_2/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

 
J__inference_my_lstm_model_2_layer_call_and_return_conditional_losses_51620
x.
my_cnn_block_2_51412:"
my_cnn_block_2_51414:.
my_cnn_block_2_51416:"
my_cnn_block_2_51418:
rnn_2_51566:'
rnn_2_51568:
rnn_2_51570:'
rnn_2_51572:
rnn_2_51574:'
rnn_2_51576:
rnn_2_51578:'
rnn_2_51580: 
dense_14_51614:
dense_14_51616:
identity¢ dense_14/StatefulPartitionedCall¢&my_cnn_block_2/StatefulPartitionedCall¢rnn_2/StatefulPartitionedCall¼
&my_cnn_block_2/StatefulPartitionedCallStatefulPartitionedCallxmy_cnn_block_2_51412my_cnn_block_2_51414my_cnn_block_2_51416my_cnn_block_2_51418*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_my_cnn_block_2_layer_call_and_return_conditional_losses_51411ø
"time_distributed_2/PartitionedCallPartitionedCall/my_cnn_block_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_time_distributed_2_layer_call_and_return_conditional_losses_50920y
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         »
time_distributed_2/ReshapeReshape/my_cnn_block_2/StatefulPartitionedCall:output:0)time_distributed_2/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿä
rnn_2/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_2/PartitionedCall:output:0rnn_2_51566rnn_2_51568rnn_2_51570rnn_2_51572rnn_2_51574rnn_2_51576rnn_2_51578rnn_2_51580*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_rnn_2_layer_call_and_return_conditional_losses_51565
 dense_14/StatefulPartitionedCallStatefulPartitionedCall&rnn_2/StatefulPartitionedCall:output:0dense_14_51614dense_14_51616*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_51613|
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
NoOpNoOp!^dense_14/StatefulPartitionedCall'^my_cnn_block_2/StatefulPartitionedCall^rnn_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2P
&my_cnn_block_2/StatefulPartitionedCall&my_cnn_block_2/StatefulPartitionedCall2>
rnn_2/StatefulPartitionedCallrnn_2/StatefulPartitionedCall:V R
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex

 
J__inference_my_lstm_model_2_layer_call_and_return_conditional_losses_52005
x.
my_cnn_block_2_51970:"
my_cnn_block_2_51972:.
my_cnn_block_2_51974:"
my_cnn_block_2_51976:
rnn_2_51982:'
rnn_2_51984:
rnn_2_51986:'
rnn_2_51988:
rnn_2_51990:'
rnn_2_51992:
rnn_2_51994:'
rnn_2_51996: 
dense_14_51999:
dense_14_52001:
identity¢ dense_14/StatefulPartitionedCall¢&my_cnn_block_2/StatefulPartitionedCall¢rnn_2/StatefulPartitionedCall¼
&my_cnn_block_2/StatefulPartitionedCallStatefulPartitionedCallxmy_cnn_block_2_51970my_cnn_block_2_51972my_cnn_block_2_51974my_cnn_block_2_51976*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_my_cnn_block_2_layer_call_and_return_conditional_losses_51921ø
"time_distributed_2/PartitionedCallPartitionedCall/my_cnn_block_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_time_distributed_2_layer_call_and_return_conditional_losses_50941y
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         »
time_distributed_2/ReshapeReshape/my_cnn_block_2/StatefulPartitionedCall:output:0)time_distributed_2/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿä
rnn_2/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_2/PartitionedCall:output:0rnn_2_51982rnn_2_51984rnn_2_51986rnn_2_51988rnn_2_51990rnn_2_51992rnn_2_51994rnn_2_51996*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_rnn_2_layer_call_and_return_conditional_losses_51826
 dense_14/StatefulPartitionedCallStatefulPartitionedCall&rnn_2/StatefulPartitionedCall:output:0dense_14_51999dense_14_52001*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_51613|
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
NoOpNoOp!^dense_14/StatefulPartitionedCall'^my_cnn_block_2/StatefulPartitionedCall^rnn_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2P
&my_cnn_block_2/StatefulPartitionedCall&my_cnn_block_2/StatefulPartitionedCall2>
rnn_2/StatefulPartitionedCallrnn_2/StatefulPartitionedCall:V R
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
ÿ
ô
/__inference_my_lstm_model_2_layer_call_fn_51651
input_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:'
	unknown_4:
	unknown_5:'
	unknown_6:
	unknown_7:'
	unknown_8:
	unknown_9:'

unknown_10:

unknown_11:

unknown_12:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_my_lstm_model_2_layer_call_and_return_conditional_losses_51620s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Æ	
´
%__inference_rnn_2_layer_call_fn_53655

inputs
unknown:'
	unknown_0:
	unknown_1:'
	unknown_2:
	unknown_3:'
	unknown_4:
	unknown_5:'
	unknown_6:
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_rnn_2_layer_call_and_return_conditional_losses_51826s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 
i
M__inference_time_distributed_2_layer_call_and_return_conditional_losses_50920

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿå
*global_average_pooling2d_2/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_50897\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿS
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:¢
	Reshape_1Reshape3global_average_pooling2d_2/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿg
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó
N
2__inference_time_distributed_2_layer_call_fn_53462

inputs
identityÅ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_time_distributed_2_layer_call_and_return_conditional_losses_50920m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
i
M__inference_time_distributed_2_layer_call_and_return_conditional_losses_53484

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
1global_average_pooling2d_2/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      §
global_average_pooling2d_2/MeanMeanReshape:output:0:global_average_pooling2d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿS
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape(global_average_pooling2d_2/Mean:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿg
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÞN
Ë
I__inference_my_cnn_block_2_layer_call_and_return_conditional_losses_53386	
inputH
.conv2d_4_conv2d_conv2d_readvariableop_resource:I
;conv2d_4_squeeze_batch_dims_biasadd_readvariableop_resource:H
.conv2d_5_conv2d_conv2d_readvariableop_resource:I
;conv2d_5_squeeze_batch_dims_biasadd_readvariableop_resource:
identity¢%conv2d_4/Conv2D/Conv2D/ReadVariableOp¢2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp¢%conv2d_5/Conv2D/Conv2D/ReadVariableOp¢2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOpJ
conv2d_4/Conv2D/ShapeShapeinput*
T0*
_output_shapes
:m
#conv2d_4/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_4/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿo
%conv2d_4/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv2d_4/Conv2D/strided_sliceStridedSliceconv2d_4/Conv2D/Shape:output:0,conv2d_4/Conv2D/strided_slice/stack:output:0.conv2d_4/Conv2D/strided_slice/stack_1:output:0.conv2d_4/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_4/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         
conv2d_4/Conv2D/ReshapeReshapeinput&conv2d_4/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%conv2d_4/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_4_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ó
conv2d_4/Conv2D/Conv2DConv2D conv2d_4/Conv2D/Reshape:output:0-conv2d_4/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
t
conv2d_4/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         f
conv2d_4/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÈ
conv2d_4/Conv2D/concatConcatV2&conv2d_4/Conv2D/strided_slice:output:0(conv2d_4/Conv2D/concat/values_1:output:0$conv2d_4/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:¤
conv2d_4/Conv2D/Reshape_1Reshapeconv2d_4/Conv2D/Conv2D:output:0conv2d_4/Conv2D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿs
!conv2d_4/squeeze_batch_dims/ShapeShape"conv2d_4/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_4/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1conv2d_4/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ{
1conv2d_4/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
)conv2d_4/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_4/squeeze_batch_dims/Shape:output:08conv2d_4/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_4/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_4/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
)conv2d_4/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         À
#conv2d_4/squeeze_batch_dims/ReshapeReshape"conv2d_4/Conv2D/Reshape_1:output:02conv2d_4/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_4_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ò
#conv2d_4/squeeze_batch_dims/BiasAddBiasAdd,conv2d_4/squeeze_batch_dims/Reshape:output:0:conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+conv2d_4/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         r
'conv2d_4/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿø
"conv2d_4/squeeze_batch_dims/concatConcatV22conv2d_4/squeeze_batch_dims/strided_slice:output:04conv2d_4/squeeze_batch_dims/concat/values_1:output:00conv2d_4/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:É
%conv2d_4/squeeze_batch_dims/Reshape_1Reshape,conv2d_4/squeeze_batch_dims/BiasAdd:output:0+conv2d_4/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿs
conv2d_5/Conv2D/ShapeShape.conv2d_4/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:m
#conv2d_5/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_5/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿo
%conv2d_5/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv2d_5/Conv2D/strided_sliceStridedSliceconv2d_5/Conv2D/Shape:output:0,conv2d_5/Conv2D/strided_slice/stack:output:0.conv2d_5/Conv2D/strided_slice/stack_1:output:0.conv2d_5/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_5/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         ´
conv2d_5/Conv2D/ReshapeReshape.conv2d_4/squeeze_batch_dims/Reshape_1:output:0&conv2d_5/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%conv2d_5/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_5_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ó
conv2d_5/Conv2D/Conv2DConv2D conv2d_5/Conv2D/Reshape:output:0-conv2d_5/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
t
conv2d_5/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         f
conv2d_5/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÈ
conv2d_5/Conv2D/concatConcatV2&conv2d_5/Conv2D/strided_slice:output:0(conv2d_5/Conv2D/concat/values_1:output:0$conv2d_5/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:¤
conv2d_5/Conv2D/Reshape_1Reshapeconv2d_5/Conv2D/Conv2D:output:0conv2d_5/Conv2D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿs
!conv2d_5/squeeze_batch_dims/ShapeShape"conv2d_5/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_5/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1conv2d_5/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ{
1conv2d_5/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
)conv2d_5/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_5/squeeze_batch_dims/Shape:output:08conv2d_5/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_5/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_5/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
)conv2d_5/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         À
#conv2d_5/squeeze_batch_dims/ReshapeReshape"conv2d_5/Conv2D/Reshape_1:output:02conv2d_5/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_5_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ò
#conv2d_5/squeeze_batch_dims/BiasAddBiasAdd,conv2d_5/squeeze_batch_dims/Reshape:output:0:conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+conv2d_5/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         r
'conv2d_5/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿø
"conv2d_5/squeeze_batch_dims/concatConcatV22conv2d_5/squeeze_batch_dims/strided_slice:output:04conv2d_5/squeeze_batch_dims/concat/values_1:output:00conv2d_5/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:É
%conv2d_5/squeeze_batch_dims/Reshape_1Reshape,conv2d_5/squeeze_batch_dims/BiasAdd:output:0+conv2d_5/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity.conv2d_5/squeeze_batch_dims/Reshape_1:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp&^conv2d_4/Conv2D/Conv2D/ReadVariableOp3^conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_5/Conv2D/Conv2D/ReadVariableOp3^conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : 2N
%conv2d_4/Conv2D/Conv2D/ReadVariableOp%conv2d_4/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_5/Conv2D/Conv2D/ReadVariableOp%conv2d_5/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp:Z V
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
©i
é
rnn_2_while_body_52298(
$rnn_2_while_rnn_2_while_loop_counter.
*rnn_2_while_rnn_2_while_maximum_iterations
rnn_2_while_placeholder
rnn_2_while_placeholder_1
rnn_2_while_placeholder_2
rnn_2_while_placeholder_3'
#rnn_2_while_rnn_2_strided_slice_1_0c
_rnn_2_while_tensorarrayv2read_tensorlistgetitem_rnn_2_tensorarrayunstack_tensorlistfromtensor_0V
Drnn_2_while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource_0:'S
Ernn_2_while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource_0:V
Drnn_2_while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource_0:'S
Ernn_2_while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource_0:V
Drnn_2_while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource_0:'S
Ernn_2_while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource_0:V
Drnn_2_while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource_0:'S
Ernn_2_while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource_0:
rnn_2_while_identity
rnn_2_while_identity_1
rnn_2_while_identity_2
rnn_2_while_identity_3
rnn_2_while_identity_4
rnn_2_while_identity_5%
!rnn_2_while_rnn_2_strided_slice_1a
]rnn_2_while_tensorarrayv2read_tensorlistgetitem_rnn_2_tensorarrayunstack_tensorlistfromtensorT
Brnn_2_while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource:'Q
Crnn_2_while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource:T
Brnn_2_while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource:'Q
Crnn_2_while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource:T
Brnn_2_while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource:'Q
Crnn_2_while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource:T
Brnn_2_while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource:'Q
Crnn_2_while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource:¢:rnn_2/while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp¢9rnn_2/while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp¢:rnn_2/while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp¢9rnn_2/while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp¢:rnn_2/while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp¢9rnn_2/while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp¢:rnn_2/while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp¢9rnn_2/while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp
=rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ä
/rnn_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_rnn_2_while_tensorarrayv2read_tensorlistgetitem_rnn_2_tensorarrayunstack_tensorlistfromtensor_0rnn_2_while_placeholderFrnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0v
4rnn_2/while/my_lstm_cell_2/concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
/rnn_2/while/my_lstm_cell_2/concatenate_5/concatConcatV26rnn_2/while/TensorArrayV2Read/TensorListGetItem:item:0rnn_2_while_placeholder_2=rnn_2/while/my_lstm_cell_2/concatenate_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'¾
9rnn_2/while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOpReadVariableOpDrnn_2_while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ã
*rnn_2/while/my_lstm_cell_2/dense_10/MatMulMatMul8rnn_2/while/my_lstm_cell_2/concatenate_5/concat:output:0Arnn_2/while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
:rnn_2/while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOpReadVariableOpErnn_2_while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0â
+rnn_2/while/my_lstm_cell_2/dense_10/BiasAddBiasAdd4rnn_2/while/my_lstm_cell_2/dense_10/MatMul:product:0Brnn_2/while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+rnn_2/while/my_lstm_cell_2/dense_10/SigmoidSigmoid4rnn_2/while/my_lstm_cell_2/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
)rnn_2/while/my_lstm_cell_2/multiply_2/mulMulrnn_2_while_placeholder_3/rnn_2/while/my_lstm_cell_2/dense_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
9rnn_2/while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOpReadVariableOpDrnn_2_while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ã
*rnn_2/while/my_lstm_cell_2/dense_11/MatMulMatMul8rnn_2/while/my_lstm_cell_2/concatenate_5/concat:output:0Arnn_2/while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
:rnn_2/while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOpReadVariableOpErnn_2_while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0â
+rnn_2/while/my_lstm_cell_2/dense_11/BiasAddBiasAdd4rnn_2/while/my_lstm_cell_2/dense_11/MatMul:product:0Brnn_2/while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+rnn_2/while/my_lstm_cell_2/dense_11/SigmoidSigmoid4rnn_2/while/my_lstm_cell_2/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
9rnn_2/while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOpReadVariableOpDrnn_2_while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ã
*rnn_2/while/my_lstm_cell_2/dense_12/MatMulMatMul8rnn_2/while/my_lstm_cell_2/concatenate_5/concat:output:0Arnn_2/while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
:rnn_2/while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOpReadVariableOpErnn_2_while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0â
+rnn_2/while/my_lstm_cell_2/dense_12/BiasAddBiasAdd4rnn_2/while/my_lstm_cell_2/dense_12/MatMul:product:0Brnn_2/while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(rnn_2/while/my_lstm_cell_2/dense_12/TanhTanh4rnn_2/while/my_lstm_cell_2/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
+rnn_2/while/my_lstm_cell_2/multiply_2/mul_1Mul/rnn_2/while/my_lstm_cell_2/dense_11/Sigmoid:y:0,rnn_2/while/my_lstm_cell_2/dense_12/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
$rnn_2/while/my_lstm_cell_2/add_5/addAddV2-rnn_2/while/my_lstm_cell_2/multiply_2/mul:z:0/rnn_2/while/my_lstm_cell_2/multiply_2/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
9rnn_2/while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOpReadVariableOpDrnn_2_while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ã
*rnn_2/while/my_lstm_cell_2/dense_13/MatMulMatMul8rnn_2/while/my_lstm_cell_2/concatenate_5/concat:output:0Arnn_2/while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
:rnn_2/while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOpReadVariableOpErnn_2_while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0â
+rnn_2/while/my_lstm_cell_2/dense_13/BiasAddBiasAdd4rnn_2/while/my_lstm_cell_2/dense_13/MatMul:product:0Brnn_2/while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+rnn_2/while/my_lstm_cell_2/dense_13/SigmoidSigmoid4rnn_2/while/my_lstm_cell_2/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,rnn_2/while/my_lstm_cell_2/activation_2/TanhTanh(rnn_2/while/my_lstm_cell_2/add_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
+rnn_2/while/my_lstm_cell_2/multiply_2/mul_2Mul0rnn_2/while/my_lstm_cell_2/activation_2/Tanh:y:0/rnn_2/while/my_lstm_cell_2/dense_13/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
0rnn_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrnn_2_while_placeholder_1rnn_2_while_placeholder/rnn_2/while/my_lstm_cell_2/multiply_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒS
rnn_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :n
rnn_2/while/addAddV2rnn_2_while_placeholderrnn_2/while/add/y:output:0*
T0*
_output_shapes
: U
rnn_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
rnn_2/while/add_1AddV2$rnn_2_while_rnn_2_while_loop_counterrnn_2/while/add_1/y:output:0*
T0*
_output_shapes
: k
rnn_2/while/IdentityIdentityrnn_2/while/add_1:z:0^rnn_2/while/NoOp*
T0*
_output_shapes
: 
rnn_2/while/Identity_1Identity*rnn_2_while_rnn_2_while_maximum_iterations^rnn_2/while/NoOp*
T0*
_output_shapes
: k
rnn_2/while/Identity_2Identityrnn_2/while/add:z:0^rnn_2/while/NoOp*
T0*
_output_shapes
: 
rnn_2/while/Identity_3Identity@rnn_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^rnn_2/while/NoOp*
T0*
_output_shapes
: 
rnn_2/while/Identity_4Identity/rnn_2/while/my_lstm_cell_2/multiply_2/mul_2:z:0^rnn_2/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn_2/while/Identity_5Identity(rnn_2/while/my_lstm_cell_2/add_5/add:z:0^rnn_2/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
rnn_2/while/NoOpNoOp;^rnn_2/while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp:^rnn_2/while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp;^rnn_2/while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp:^rnn_2/while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp;^rnn_2/while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp:^rnn_2/while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp;^rnn_2/while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp:^rnn_2/while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "5
rnn_2_while_identityrnn_2/while/Identity:output:0"9
rnn_2_while_identity_1rnn_2/while/Identity_1:output:0"9
rnn_2_while_identity_2rnn_2/while/Identity_2:output:0"9
rnn_2_while_identity_3rnn_2/while/Identity_3:output:0"9
rnn_2_while_identity_4rnn_2/while/Identity_4:output:0"9
rnn_2_while_identity_5rnn_2/while/Identity_5:output:0"
Crnn_2_while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resourceErnn_2_while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource_0"
Brnn_2_while_my_lstm_cell_2_dense_10_matmul_readvariableop_resourceDrnn_2_while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource_0"
Crnn_2_while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resourceErnn_2_while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource_0"
Brnn_2_while_my_lstm_cell_2_dense_11_matmul_readvariableop_resourceDrnn_2_while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource_0"
Crnn_2_while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resourceErnn_2_while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource_0"
Brnn_2_while_my_lstm_cell_2_dense_12_matmul_readvariableop_resourceDrnn_2_while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource_0"
Crnn_2_while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resourceErnn_2_while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource_0"
Brnn_2_while_my_lstm_cell_2_dense_13_matmul_readvariableop_resourceDrnn_2_while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource_0"H
!rnn_2_while_rnn_2_strided_slice_1#rnn_2_while_rnn_2_strided_slice_1_0"À
]rnn_2_while_tensorarrayv2read_tensorlistgetitem_rnn_2_tensorarrayunstack_tensorlistfromtensor_rnn_2_while_tensorarrayv2read_tensorlistgetitem_rnn_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2x
:rnn_2/while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp:rnn_2/while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp2v
9rnn_2/while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp9rnn_2/while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp2x
:rnn_2/while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp:rnn_2/while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp2v
9rnn_2/while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp9rnn_2/while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp2x
:rnn_2/while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp:rnn_2/while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp2v
9rnn_2/while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp9rnn_2/while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp2x
:rnn_2/while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp:rnn_2/while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp2v
9rnn_2/while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp9rnn_2/while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
²h
±
@__inference_rnn_2_layer_call_and_return_conditional_losses_54003
inputs_0H
6my_lstm_cell_2_dense_10_matmul_readvariableop_resource:'E
7my_lstm_cell_2_dense_10_biasadd_readvariableop_resource:H
6my_lstm_cell_2_dense_11_matmul_readvariableop_resource:'E
7my_lstm_cell_2_dense_11_biasadd_readvariableop_resource:H
6my_lstm_cell_2_dense_12_matmul_readvariableop_resource:'E
7my_lstm_cell_2_dense_12_biasadd_readvariableop_resource:H
6my_lstm_cell_2_dense_13_matmul_readvariableop_resource:'E
7my_lstm_cell_2_dense_13_biasadd_readvariableop_resource:
identity¢.my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp¢-my_lstm_cell_2/dense_10/MatMul/ReadVariableOp¢.my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp¢-my_lstm_cell_2/dense_11/MatMul/ReadVariableOp¢.my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp¢-my_lstm_cell_2/dense_12/MatMul/ReadVariableOp¢.my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp¢-my_lstm_cell_2/dense_13/MatMul/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
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
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskj
(my_lstm_cell_2/concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ç
#my_lstm_cell_2/concatenate_5/concatConcatV2strided_slice_2:output:0zeros:output:01my_lstm_cell_2/concatenate_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'¤
-my_lstm_cell_2/dense_10/MatMul/ReadVariableOpReadVariableOp6my_lstm_cell_2_dense_10_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0¿
my_lstm_cell_2/dense_10/MatMulMatMul,my_lstm_cell_2/concatenate_5/concat:output:05my_lstm_cell_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOpReadVariableOp7my_lstm_cell_2_dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
my_lstm_cell_2/dense_10/BiasAddBiasAdd(my_lstm_cell_2/dense_10/MatMul:product:06my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
my_lstm_cell_2/dense_10/SigmoidSigmoid(my_lstm_cell_2/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
my_lstm_cell_2/multiply_2/mulMulzeros_1:output:0#my_lstm_cell_2/dense_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-my_lstm_cell_2/dense_11/MatMul/ReadVariableOpReadVariableOp6my_lstm_cell_2_dense_11_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0¿
my_lstm_cell_2/dense_11/MatMulMatMul,my_lstm_cell_2/concatenate_5/concat:output:05my_lstm_cell_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOpReadVariableOp7my_lstm_cell_2_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
my_lstm_cell_2/dense_11/BiasAddBiasAdd(my_lstm_cell_2/dense_11/MatMul:product:06my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
my_lstm_cell_2/dense_11/SigmoidSigmoid(my_lstm_cell_2/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-my_lstm_cell_2/dense_12/MatMul/ReadVariableOpReadVariableOp6my_lstm_cell_2_dense_12_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0¿
my_lstm_cell_2/dense_12/MatMulMatMul,my_lstm_cell_2/concatenate_5/concat:output:05my_lstm_cell_2/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOpReadVariableOp7my_lstm_cell_2_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
my_lstm_cell_2/dense_12/BiasAddBiasAdd(my_lstm_cell_2/dense_12/MatMul:product:06my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
my_lstm_cell_2/dense_12/TanhTanh(my_lstm_cell_2/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
my_lstm_cell_2/multiply_2/mul_1Mul#my_lstm_cell_2/dense_11/Sigmoid:y:0 my_lstm_cell_2/dense_12/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
my_lstm_cell_2/add_5/addAddV2!my_lstm_cell_2/multiply_2/mul:z:0#my_lstm_cell_2/multiply_2/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-my_lstm_cell_2/dense_13/MatMul/ReadVariableOpReadVariableOp6my_lstm_cell_2_dense_13_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0¿
my_lstm_cell_2/dense_13/MatMulMatMul,my_lstm_cell_2/concatenate_5/concat:output:05my_lstm_cell_2/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOpReadVariableOp7my_lstm_cell_2_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
my_lstm_cell_2/dense_13/BiasAddBiasAdd(my_lstm_cell_2/dense_13/MatMul:product:06my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
my_lstm_cell_2/dense_13/SigmoidSigmoid(my_lstm_cell_2/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
 my_lstm_cell_2/activation_2/TanhTanhmy_lstm_cell_2/add_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
my_lstm_cell_2/multiply_2/mul_2Mul$my_lstm_cell_2/activation_2/Tanh:y:0#my_lstm_cell_2/dense_13/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ø
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:06my_lstm_cell_2_dense_10_matmul_readvariableop_resource7my_lstm_cell_2_dense_10_biasadd_readvariableop_resource6my_lstm_cell_2_dense_11_matmul_readvariableop_resource7my_lstm_cell_2_dense_11_biasadd_readvariableop_resource6my_lstm_cell_2_dense_12_matmul_readvariableop_resource7my_lstm_cell_2_dense_12_biasadd_readvariableop_resource6my_lstm_cell_2_dense_13_matmul_readvariableop_resource7my_lstm_cell_2_dense_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( *
bodyR
while_body_53899*
condR
while_cond_53898*U
output_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿk
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÒ
NoOpNoOp/^my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp.^my_lstm_cell_2/dense_10/MatMul/ReadVariableOp/^my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp.^my_lstm_cell_2/dense_11/MatMul/ReadVariableOp/^my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp.^my_lstm_cell_2/dense_12/MatMul/ReadVariableOp/^my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp.^my_lstm_cell_2/dense_13/MatMul/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : 2`
.my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp.my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp2^
-my_lstm_cell_2/dense_10/MatMul/ReadVariableOp-my_lstm_cell_2/dense_10/MatMul/ReadVariableOp2`
.my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp.my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp2^
-my_lstm_cell_2/dense_11/MatMul/ReadVariableOp-my_lstm_cell_2/dense_11/MatMul/ReadVariableOp2`
.my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp.my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp2^
-my_lstm_cell_2/dense_12/MatMul/ReadVariableOp-my_lstm_cell_2/dense_12/MatMul/ReadVariableOp2`
.my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp.my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp2^
-my_lstm_cell_2/dense_13/MatMul/ReadVariableOp-my_lstm_cell_2/dense_13/MatMul/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ð	
¶
%__inference_rnn_2_layer_call_fn_53592
inputs_0
unknown:'
	unknown_0:
	unknown_1:'
	unknown_2:
	unknown_3:'
	unknown_4:
	unknown_5:'
	unknown_6:
identity¢StatefulPartitionedCall²
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_rnn_2_layer_call_and_return_conditional_losses_51134|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Ê
ú
C__inference_dense_14_layer_call_and_return_conditional_losses_54390

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÞN
Ë
I__inference_my_cnn_block_2_layer_call_and_return_conditional_losses_51411	
inputH
.conv2d_4_conv2d_conv2d_readvariableop_resource:I
;conv2d_4_squeeze_batch_dims_biasadd_readvariableop_resource:H
.conv2d_5_conv2d_conv2d_readvariableop_resource:I
;conv2d_5_squeeze_batch_dims_biasadd_readvariableop_resource:
identity¢%conv2d_4/Conv2D/Conv2D/ReadVariableOp¢2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp¢%conv2d_5/Conv2D/Conv2D/ReadVariableOp¢2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOpJ
conv2d_4/Conv2D/ShapeShapeinput*
T0*
_output_shapes
:m
#conv2d_4/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_4/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿo
%conv2d_4/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv2d_4/Conv2D/strided_sliceStridedSliceconv2d_4/Conv2D/Shape:output:0,conv2d_4/Conv2D/strided_slice/stack:output:0.conv2d_4/Conv2D/strided_slice/stack_1:output:0.conv2d_4/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_4/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         
conv2d_4/Conv2D/ReshapeReshapeinput&conv2d_4/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%conv2d_4/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_4_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ó
conv2d_4/Conv2D/Conv2DConv2D conv2d_4/Conv2D/Reshape:output:0-conv2d_4/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
t
conv2d_4/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         f
conv2d_4/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÈ
conv2d_4/Conv2D/concatConcatV2&conv2d_4/Conv2D/strided_slice:output:0(conv2d_4/Conv2D/concat/values_1:output:0$conv2d_4/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:¤
conv2d_4/Conv2D/Reshape_1Reshapeconv2d_4/Conv2D/Conv2D:output:0conv2d_4/Conv2D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿs
!conv2d_4/squeeze_batch_dims/ShapeShape"conv2d_4/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_4/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1conv2d_4/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ{
1conv2d_4/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
)conv2d_4/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_4/squeeze_batch_dims/Shape:output:08conv2d_4/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_4/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_4/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
)conv2d_4/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         À
#conv2d_4/squeeze_batch_dims/ReshapeReshape"conv2d_4/Conv2D/Reshape_1:output:02conv2d_4/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_4_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ò
#conv2d_4/squeeze_batch_dims/BiasAddBiasAdd,conv2d_4/squeeze_batch_dims/Reshape:output:0:conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+conv2d_4/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         r
'conv2d_4/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿø
"conv2d_4/squeeze_batch_dims/concatConcatV22conv2d_4/squeeze_batch_dims/strided_slice:output:04conv2d_4/squeeze_batch_dims/concat/values_1:output:00conv2d_4/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:É
%conv2d_4/squeeze_batch_dims/Reshape_1Reshape,conv2d_4/squeeze_batch_dims/BiasAdd:output:0+conv2d_4/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿs
conv2d_5/Conv2D/ShapeShape.conv2d_4/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:m
#conv2d_5/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_5/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿo
%conv2d_5/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv2d_5/Conv2D/strided_sliceStridedSliceconv2d_5/Conv2D/Shape:output:0,conv2d_5/Conv2D/strided_slice/stack:output:0.conv2d_5/Conv2D/strided_slice/stack_1:output:0.conv2d_5/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_5/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         ´
conv2d_5/Conv2D/ReshapeReshape.conv2d_4/squeeze_batch_dims/Reshape_1:output:0&conv2d_5/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%conv2d_5/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_5_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ó
conv2d_5/Conv2D/Conv2DConv2D conv2d_5/Conv2D/Reshape:output:0-conv2d_5/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
t
conv2d_5/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         f
conv2d_5/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÈ
conv2d_5/Conv2D/concatConcatV2&conv2d_5/Conv2D/strided_slice:output:0(conv2d_5/Conv2D/concat/values_1:output:0$conv2d_5/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:¤
conv2d_5/Conv2D/Reshape_1Reshapeconv2d_5/Conv2D/Conv2D:output:0conv2d_5/Conv2D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿs
!conv2d_5/squeeze_batch_dims/ShapeShape"conv2d_5/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_5/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1conv2d_5/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ{
1conv2d_5/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
)conv2d_5/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_5/squeeze_batch_dims/Shape:output:08conv2d_5/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_5/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_5/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
)conv2d_5/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         À
#conv2d_5/squeeze_batch_dims/ReshapeReshape"conv2d_5/Conv2D/Reshape_1:output:02conv2d_5/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_5_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ò
#conv2d_5/squeeze_batch_dims/BiasAddBiasAdd,conv2d_5/squeeze_batch_dims/Reshape:output:0:conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+conv2d_5/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         r
'conv2d_5/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿø
"conv2d_5/squeeze_batch_dims/concatConcatV22conv2d_5/squeeze_batch_dims/strided_slice:output:04conv2d_5/squeeze_batch_dims/concat/values_1:output:00conv2d_5/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:É
%conv2d_5/squeeze_batch_dims/Reshape_1Reshape,conv2d_5/squeeze_batch_dims/BiasAdd:output:0+conv2d_5/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity.conv2d_5/squeeze_batch_dims/Reshape_1:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp&^conv2d_4/Conv2D/Conv2D/ReadVariableOp3^conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_5/Conv2D/Conv2D/ReadVariableOp3^conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : 2N
%conv2d_4/Conv2D/Conv2D/ReadVariableOp%conv2d_4/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_5/Conv2D/Conv2D/ReadVariableOp%conv2d_5/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp:Z V
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
º
ù
.__inference_my_lstm_cell_2_layer_call_fn_53528

inputs
states_0
states_1
unknown:'
	unknown_0:
	unknown_1:'
	unknown_2:
	unknown_3:'
	unknown_4:
	unknown_5:'
	unknown_6:
identity

identity_1

identity_2¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_my_lstm_cell_2_layer_call_and_return_conditional_losses_51022o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
ÞN
Ë
I__inference_my_cnn_block_2_layer_call_and_return_conditional_losses_51921	
inputH
.conv2d_4_conv2d_conv2d_readvariableop_resource:I
;conv2d_4_squeeze_batch_dims_biasadd_readvariableop_resource:H
.conv2d_5_conv2d_conv2d_readvariableop_resource:I
;conv2d_5_squeeze_batch_dims_biasadd_readvariableop_resource:
identity¢%conv2d_4/Conv2D/Conv2D/ReadVariableOp¢2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp¢%conv2d_5/Conv2D/Conv2D/ReadVariableOp¢2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOpJ
conv2d_4/Conv2D/ShapeShapeinput*
T0*
_output_shapes
:m
#conv2d_4/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_4/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿo
%conv2d_4/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv2d_4/Conv2D/strided_sliceStridedSliceconv2d_4/Conv2D/Shape:output:0,conv2d_4/Conv2D/strided_slice/stack:output:0.conv2d_4/Conv2D/strided_slice/stack_1:output:0.conv2d_4/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_4/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         
conv2d_4/Conv2D/ReshapeReshapeinput&conv2d_4/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%conv2d_4/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_4_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ó
conv2d_4/Conv2D/Conv2DConv2D conv2d_4/Conv2D/Reshape:output:0-conv2d_4/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
t
conv2d_4/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         f
conv2d_4/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÈ
conv2d_4/Conv2D/concatConcatV2&conv2d_4/Conv2D/strided_slice:output:0(conv2d_4/Conv2D/concat/values_1:output:0$conv2d_4/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:¤
conv2d_4/Conv2D/Reshape_1Reshapeconv2d_4/Conv2D/Conv2D:output:0conv2d_4/Conv2D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿs
!conv2d_4/squeeze_batch_dims/ShapeShape"conv2d_4/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_4/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1conv2d_4/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ{
1conv2d_4/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
)conv2d_4/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_4/squeeze_batch_dims/Shape:output:08conv2d_4/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_4/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_4/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
)conv2d_4/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         À
#conv2d_4/squeeze_batch_dims/ReshapeReshape"conv2d_4/Conv2D/Reshape_1:output:02conv2d_4/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_4_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ò
#conv2d_4/squeeze_batch_dims/BiasAddBiasAdd,conv2d_4/squeeze_batch_dims/Reshape:output:0:conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+conv2d_4/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         r
'conv2d_4/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿø
"conv2d_4/squeeze_batch_dims/concatConcatV22conv2d_4/squeeze_batch_dims/strided_slice:output:04conv2d_4/squeeze_batch_dims/concat/values_1:output:00conv2d_4/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:É
%conv2d_4/squeeze_batch_dims/Reshape_1Reshape,conv2d_4/squeeze_batch_dims/BiasAdd:output:0+conv2d_4/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿs
conv2d_5/Conv2D/ShapeShape.conv2d_4/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:m
#conv2d_5/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_5/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿo
%conv2d_5/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv2d_5/Conv2D/strided_sliceStridedSliceconv2d_5/Conv2D/Shape:output:0,conv2d_5/Conv2D/strided_slice/stack:output:0.conv2d_5/Conv2D/strided_slice/stack_1:output:0.conv2d_5/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_5/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         ´
conv2d_5/Conv2D/ReshapeReshape.conv2d_4/squeeze_batch_dims/Reshape_1:output:0&conv2d_5/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%conv2d_5/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_5_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ó
conv2d_5/Conv2D/Conv2DConv2D conv2d_5/Conv2D/Reshape:output:0-conv2d_5/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
t
conv2d_5/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         f
conv2d_5/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÈ
conv2d_5/Conv2D/concatConcatV2&conv2d_5/Conv2D/strided_slice:output:0(conv2d_5/Conv2D/concat/values_1:output:0$conv2d_5/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:¤
conv2d_5/Conv2D/Reshape_1Reshapeconv2d_5/Conv2D/Conv2D:output:0conv2d_5/Conv2D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿs
!conv2d_5/squeeze_batch_dims/ShapeShape"conv2d_5/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_5/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1conv2d_5/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ{
1conv2d_5/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
)conv2d_5/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_5/squeeze_batch_dims/Shape:output:08conv2d_5/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_5/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_5/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
)conv2d_5/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         À
#conv2d_5/squeeze_batch_dims/ReshapeReshape"conv2d_5/Conv2D/Reshape_1:output:02conv2d_5/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_5_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ò
#conv2d_5/squeeze_batch_dims/BiasAddBiasAdd,conv2d_5/squeeze_batch_dims/Reshape:output:0:conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+conv2d_5/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         r
'conv2d_5/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿø
"conv2d_5/squeeze_batch_dims/concatConcatV22conv2d_5/squeeze_batch_dims/strided_slice:output:04conv2d_5/squeeze_batch_dims/concat/values_1:output:00conv2d_5/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:É
%conv2d_5/squeeze_batch_dims/Reshape_1Reshape,conv2d_5/squeeze_batch_dims/BiasAdd:output:0+conv2d_5/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity.conv2d_5/squeeze_batch_dims/Reshape_1:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp&^conv2d_4/Conv2D/Conv2D/ReadVariableOp3^conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_5/Conv2D/Conv2D/ReadVariableOp3^conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : 2N
%conv2d_4/Conv2D/Conv2D/ReadVariableOp%conv2d_4/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_5/Conv2D/Conv2D/ReadVariableOp%conv2d_5/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp:Z V
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
¶
q
U__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_53457

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
/
ü
I__inference_my_lstm_cell_2_layer_call_and_return_conditional_losses_53571

inputs
states_0
states_19
'dense_10_matmul_readvariableop_resource:'6
(dense_10_biasadd_readvariableop_resource:9
'dense_11_matmul_readvariableop_resource:'6
(dense_11_biasadd_readvariableop_resource:9
'dense_12_matmul_readvariableop_resource:'6
(dense_12_biasadd_readvariableop_resource:9
'dense_13_matmul_readvariableop_resource:'6
(dense_13_biasadd_readvariableop_resource:
identity

identity_1

identity_2¢dense_10/BiasAdd/ReadVariableOp¢dense_10/MatMul/ReadVariableOp¢dense_11/BiasAdd/ReadVariableOp¢dense_11/MatMul/ReadVariableOp¢dense_12/BiasAdd/ReadVariableOp¢dense_12/MatMul/ReadVariableOp¢dense_13/BiasAdd/ReadVariableOp¢dense_13/MatMul/ReadVariableOp[
concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatenate_5/concatConcatV2inputsstates_0"concatenate_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0
dense_10/MatMulMatMulconcatenate_5/concat:output:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_10/SigmoidSigmoiddense_10/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
multiply_2/mulMulstates_1dense_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0
dense_11/MatMulMatMulconcatenate_5/concat:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_11/SigmoidSigmoiddense_11/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0
dense_12/MatMulMatMulconcatenate_5/concat:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dense_12/TanhTanhdense_12/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
multiply_2/mul_1Muldense_11/Sigmoid:y:0dense_12/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
	add_5/addAddV2multiply_2/mul:z:0multiply_2/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0
dense_13/MatMulMatMulconcatenate_5/concat:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_13/SigmoidSigmoiddense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
activation_2/TanhTanhadd_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
multiply_2/mul_2Mulactivation_2/Tanh:y:0dense_13/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitymultiply_2/mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe

Identity_1Identitymultiply_2/mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^

Identity_2Identityadd_5/add:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
©i
é
rnn_2_while_body_52515(
$rnn_2_while_rnn_2_while_loop_counter.
*rnn_2_while_rnn_2_while_maximum_iterations
rnn_2_while_placeholder
rnn_2_while_placeholder_1
rnn_2_while_placeholder_2
rnn_2_while_placeholder_3'
#rnn_2_while_rnn_2_strided_slice_1_0c
_rnn_2_while_tensorarrayv2read_tensorlistgetitem_rnn_2_tensorarrayunstack_tensorlistfromtensor_0V
Drnn_2_while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource_0:'S
Ernn_2_while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource_0:V
Drnn_2_while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource_0:'S
Ernn_2_while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource_0:V
Drnn_2_while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource_0:'S
Ernn_2_while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource_0:V
Drnn_2_while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource_0:'S
Ernn_2_while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource_0:
rnn_2_while_identity
rnn_2_while_identity_1
rnn_2_while_identity_2
rnn_2_while_identity_3
rnn_2_while_identity_4
rnn_2_while_identity_5%
!rnn_2_while_rnn_2_strided_slice_1a
]rnn_2_while_tensorarrayv2read_tensorlistgetitem_rnn_2_tensorarrayunstack_tensorlistfromtensorT
Brnn_2_while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource:'Q
Crnn_2_while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource:T
Brnn_2_while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource:'Q
Crnn_2_while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource:T
Brnn_2_while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource:'Q
Crnn_2_while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource:T
Brnn_2_while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource:'Q
Crnn_2_while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource:¢:rnn_2/while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp¢9rnn_2/while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp¢:rnn_2/while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp¢9rnn_2/while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp¢:rnn_2/while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp¢9rnn_2/while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp¢:rnn_2/while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp¢9rnn_2/while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp
=rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ä
/rnn_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_rnn_2_while_tensorarrayv2read_tensorlistgetitem_rnn_2_tensorarrayunstack_tensorlistfromtensor_0rnn_2_while_placeholderFrnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0v
4rnn_2/while/my_lstm_cell_2/concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
/rnn_2/while/my_lstm_cell_2/concatenate_5/concatConcatV26rnn_2/while/TensorArrayV2Read/TensorListGetItem:item:0rnn_2_while_placeholder_2=rnn_2/while/my_lstm_cell_2/concatenate_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'¾
9rnn_2/while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOpReadVariableOpDrnn_2_while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ã
*rnn_2/while/my_lstm_cell_2/dense_10/MatMulMatMul8rnn_2/while/my_lstm_cell_2/concatenate_5/concat:output:0Arnn_2/while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
:rnn_2/while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOpReadVariableOpErnn_2_while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0â
+rnn_2/while/my_lstm_cell_2/dense_10/BiasAddBiasAdd4rnn_2/while/my_lstm_cell_2/dense_10/MatMul:product:0Brnn_2/while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+rnn_2/while/my_lstm_cell_2/dense_10/SigmoidSigmoid4rnn_2/while/my_lstm_cell_2/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
)rnn_2/while/my_lstm_cell_2/multiply_2/mulMulrnn_2_while_placeholder_3/rnn_2/while/my_lstm_cell_2/dense_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
9rnn_2/while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOpReadVariableOpDrnn_2_while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ã
*rnn_2/while/my_lstm_cell_2/dense_11/MatMulMatMul8rnn_2/while/my_lstm_cell_2/concatenate_5/concat:output:0Arnn_2/while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
:rnn_2/while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOpReadVariableOpErnn_2_while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0â
+rnn_2/while/my_lstm_cell_2/dense_11/BiasAddBiasAdd4rnn_2/while/my_lstm_cell_2/dense_11/MatMul:product:0Brnn_2/while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+rnn_2/while/my_lstm_cell_2/dense_11/SigmoidSigmoid4rnn_2/while/my_lstm_cell_2/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
9rnn_2/while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOpReadVariableOpDrnn_2_while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ã
*rnn_2/while/my_lstm_cell_2/dense_12/MatMulMatMul8rnn_2/while/my_lstm_cell_2/concatenate_5/concat:output:0Arnn_2/while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
:rnn_2/while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOpReadVariableOpErnn_2_while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0â
+rnn_2/while/my_lstm_cell_2/dense_12/BiasAddBiasAdd4rnn_2/while/my_lstm_cell_2/dense_12/MatMul:product:0Brnn_2/while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(rnn_2/while/my_lstm_cell_2/dense_12/TanhTanh4rnn_2/while/my_lstm_cell_2/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
+rnn_2/while/my_lstm_cell_2/multiply_2/mul_1Mul/rnn_2/while/my_lstm_cell_2/dense_11/Sigmoid:y:0,rnn_2/while/my_lstm_cell_2/dense_12/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
$rnn_2/while/my_lstm_cell_2/add_5/addAddV2-rnn_2/while/my_lstm_cell_2/multiply_2/mul:z:0/rnn_2/while/my_lstm_cell_2/multiply_2/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
9rnn_2/while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOpReadVariableOpDrnn_2_while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ã
*rnn_2/while/my_lstm_cell_2/dense_13/MatMulMatMul8rnn_2/while/my_lstm_cell_2/concatenate_5/concat:output:0Arnn_2/while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
:rnn_2/while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOpReadVariableOpErnn_2_while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0â
+rnn_2/while/my_lstm_cell_2/dense_13/BiasAddBiasAdd4rnn_2/while/my_lstm_cell_2/dense_13/MatMul:product:0Brnn_2/while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+rnn_2/while/my_lstm_cell_2/dense_13/SigmoidSigmoid4rnn_2/while/my_lstm_cell_2/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,rnn_2/while/my_lstm_cell_2/activation_2/TanhTanh(rnn_2/while/my_lstm_cell_2/add_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
+rnn_2/while/my_lstm_cell_2/multiply_2/mul_2Mul0rnn_2/while/my_lstm_cell_2/activation_2/Tanh:y:0/rnn_2/while/my_lstm_cell_2/dense_13/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
0rnn_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrnn_2_while_placeholder_1rnn_2_while_placeholder/rnn_2/while/my_lstm_cell_2/multiply_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒS
rnn_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :n
rnn_2/while/addAddV2rnn_2_while_placeholderrnn_2/while/add/y:output:0*
T0*
_output_shapes
: U
rnn_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
rnn_2/while/add_1AddV2$rnn_2_while_rnn_2_while_loop_counterrnn_2/while/add_1/y:output:0*
T0*
_output_shapes
: k
rnn_2/while/IdentityIdentityrnn_2/while/add_1:z:0^rnn_2/while/NoOp*
T0*
_output_shapes
: 
rnn_2/while/Identity_1Identity*rnn_2_while_rnn_2_while_maximum_iterations^rnn_2/while/NoOp*
T0*
_output_shapes
: k
rnn_2/while/Identity_2Identityrnn_2/while/add:z:0^rnn_2/while/NoOp*
T0*
_output_shapes
: 
rnn_2/while/Identity_3Identity@rnn_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^rnn_2/while/NoOp*
T0*
_output_shapes
: 
rnn_2/while/Identity_4Identity/rnn_2/while/my_lstm_cell_2/multiply_2/mul_2:z:0^rnn_2/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn_2/while/Identity_5Identity(rnn_2/while/my_lstm_cell_2/add_5/add:z:0^rnn_2/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
rnn_2/while/NoOpNoOp;^rnn_2/while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp:^rnn_2/while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp;^rnn_2/while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp:^rnn_2/while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp;^rnn_2/while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp:^rnn_2/while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp;^rnn_2/while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp:^rnn_2/while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "5
rnn_2_while_identityrnn_2/while/Identity:output:0"9
rnn_2_while_identity_1rnn_2/while/Identity_1:output:0"9
rnn_2_while_identity_2rnn_2/while/Identity_2:output:0"9
rnn_2_while_identity_3rnn_2/while/Identity_3:output:0"9
rnn_2_while_identity_4rnn_2/while/Identity_4:output:0"9
rnn_2_while_identity_5rnn_2/while/Identity_5:output:0"
Crnn_2_while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resourceErnn_2_while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource_0"
Brnn_2_while_my_lstm_cell_2_dense_10_matmul_readvariableop_resourceDrnn_2_while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource_0"
Crnn_2_while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resourceErnn_2_while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource_0"
Brnn_2_while_my_lstm_cell_2_dense_11_matmul_readvariableop_resourceDrnn_2_while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource_0"
Crnn_2_while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resourceErnn_2_while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource_0"
Brnn_2_while_my_lstm_cell_2_dense_12_matmul_readvariableop_resourceDrnn_2_while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource_0"
Crnn_2_while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resourceErnn_2_while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource_0"
Brnn_2_while_my_lstm_cell_2_dense_13_matmul_readvariableop_resourceDrnn_2_while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource_0"H
!rnn_2_while_rnn_2_strided_slice_1#rnn_2_while_rnn_2_strided_slice_1_0"À
]rnn_2_while_tensorarrayv2read_tensorlistgetitem_rnn_2_tensorarrayunstack_tensorlistfromtensor_rnn_2_while_tensorarrayv2read_tensorlistgetitem_rnn_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2x
:rnn_2/while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp:rnn_2/while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp2v
9rnn_2/while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp9rnn_2/while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp2x
:rnn_2/while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp:rnn_2/while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp2v
9rnn_2/while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp9rnn_2/while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp2x
:rnn_2/while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp:rnn_2/while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp2v
9rnn_2/while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp9rnn_2/while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp2x
:rnn_2/while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp:rnn_2/while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp2v
9rnn_2/while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp9rnn_2/while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Í

Ç
while_cond_53898
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_53898___redundant_placeholder03
/while_while_cond_53898___redundant_placeholder13
/while_while_cond_53898___redundant_placeholder23
/while_while_cond_53898___redundant_placeholder33
/while_while_cond_53898___redundant_placeholder43
/while_while_cond_53898___redundant_placeholder53
/while_while_cond_53898___redundant_placeholder63
/while_while_cond_53898___redundant_placeholder73
/while_while_cond_53898___redundant_placeholder8
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
*
¢

while_body_51237
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_my_lstm_cell_2_51261_0:'*
while_my_lstm_cell_2_51263_0:.
while_my_lstm_cell_2_51265_0:'*
while_my_lstm_cell_2_51267_0:.
while_my_lstm_cell_2_51269_0:'*
while_my_lstm_cell_2_51271_0:.
while_my_lstm_cell_2_51273_0:'*
while_my_lstm_cell_2_51275_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_my_lstm_cell_2_51261:'(
while_my_lstm_cell_2_51263:,
while_my_lstm_cell_2_51265:'(
while_my_lstm_cell_2_51267:,
while_my_lstm_cell_2_51269:'(
while_my_lstm_cell_2_51271:,
while_my_lstm_cell_2_51273:'(
while_my_lstm_cell_2_51275:¢,while/my_lstm_cell_2/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0Ù
,while/my_lstm_cell_2/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_my_lstm_cell_2_51261_0while_my_lstm_cell_2_51263_0while_my_lstm_cell_2_51265_0while_my_lstm_cell_2_51267_0while_my_lstm_cell_2_51269_0while_my_lstm_cell_2_51271_0while_my_lstm_cell_2_51273_0while_my_lstm_cell_2_51275_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_my_lstm_cell_2_layer_call_and_return_conditional_losses_51022Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder5while/my_lstm_cell_2/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity5while/my_lstm_cell_2/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity5while/my_lstm_cell_2/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{

while/NoOpNoOp-^while/my_lstm_cell_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_my_lstm_cell_2_51261while_my_lstm_cell_2_51261_0":
while_my_lstm_cell_2_51263while_my_lstm_cell_2_51263_0":
while_my_lstm_cell_2_51265while_my_lstm_cell_2_51265_0":
while_my_lstm_cell_2_51267while_my_lstm_cell_2_51267_0":
while_my_lstm_cell_2_51269while_my_lstm_cell_2_51269_0":
while_my_lstm_cell_2_51271while_my_lstm_cell_2_51271_0":
while_my_lstm_cell_2_51273while_my_lstm_cell_2_51273_0":
while_my_lstm_cell_2_51275while_my_lstm_cell_2_51275_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2\
,while/my_lstm_cell_2/StatefulPartitionedCall,while/my_lstm_cell_2/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
«N

__inference_call_53240	
inputH
.conv2d_4_conv2d_conv2d_readvariableop_resource:I
;conv2d_4_squeeze_batch_dims_biasadd_readvariableop_resource:H
.conv2d_5_conv2d_conv2d_readvariableop_resource:I
;conv2d_5_squeeze_batch_dims_biasadd_readvariableop_resource:
identity¢%conv2d_4/Conv2D/Conv2D/ReadVariableOp¢2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp¢%conv2d_5/Conv2D/Conv2D/ReadVariableOp¢2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOpJ
conv2d_4/Conv2D/ShapeShapeinput*
T0*
_output_shapes
:m
#conv2d_4/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_4/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿo
%conv2d_4/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv2d_4/Conv2D/strided_sliceStridedSliceconv2d_4/Conv2D/Shape:output:0,conv2d_4/Conv2D/strided_slice/stack:output:0.conv2d_4/Conv2D/strided_slice/stack_1:output:0.conv2d_4/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_4/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         
conv2d_4/Conv2D/ReshapeReshapeinput&conv2d_4/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%conv2d_4/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_4_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ó
conv2d_4/Conv2D/Conv2DConv2D conv2d_4/Conv2D/Reshape:output:0-conv2d_4/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
t
conv2d_4/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         f
conv2d_4/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÈ
conv2d_4/Conv2D/concatConcatV2&conv2d_4/Conv2D/strided_slice:output:0(conv2d_4/Conv2D/concat/values_1:output:0$conv2d_4/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:¤
conv2d_4/Conv2D/Reshape_1Reshapeconv2d_4/Conv2D/Conv2D:output:0conv2d_4/Conv2D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿs
!conv2d_4/squeeze_batch_dims/ShapeShape"conv2d_4/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_4/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1conv2d_4/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ{
1conv2d_4/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
)conv2d_4/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_4/squeeze_batch_dims/Shape:output:08conv2d_4/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_4/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_4/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
)conv2d_4/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         À
#conv2d_4/squeeze_batch_dims/ReshapeReshape"conv2d_4/Conv2D/Reshape_1:output:02conv2d_4/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_4_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ò
#conv2d_4/squeeze_batch_dims/BiasAddBiasAdd,conv2d_4/squeeze_batch_dims/Reshape:output:0:conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+conv2d_4/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         r
'conv2d_4/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿø
"conv2d_4/squeeze_batch_dims/concatConcatV22conv2d_4/squeeze_batch_dims/strided_slice:output:04conv2d_4/squeeze_batch_dims/concat/values_1:output:00conv2d_4/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:É
%conv2d_4/squeeze_batch_dims/Reshape_1Reshape,conv2d_4/squeeze_batch_dims/BiasAdd:output:0+conv2d_4/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿs
conv2d_5/Conv2D/ShapeShape.conv2d_4/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:m
#conv2d_5/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_5/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿo
%conv2d_5/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv2d_5/Conv2D/strided_sliceStridedSliceconv2d_5/Conv2D/Shape:output:0,conv2d_5/Conv2D/strided_slice/stack:output:0.conv2d_5/Conv2D/strided_slice/stack_1:output:0.conv2d_5/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_5/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         ´
conv2d_5/Conv2D/ReshapeReshape.conv2d_4/squeeze_batch_dims/Reshape_1:output:0&conv2d_5/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%conv2d_5/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_5_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ó
conv2d_5/Conv2D/Conv2DConv2D conv2d_5/Conv2D/Reshape:output:0-conv2d_5/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
t
conv2d_5/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         f
conv2d_5/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÈ
conv2d_5/Conv2D/concatConcatV2&conv2d_5/Conv2D/strided_slice:output:0(conv2d_5/Conv2D/concat/values_1:output:0$conv2d_5/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:¤
conv2d_5/Conv2D/Reshape_1Reshapeconv2d_5/Conv2D/Conv2D:output:0conv2d_5/Conv2D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿs
!conv2d_5/squeeze_batch_dims/ShapeShape"conv2d_5/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_5/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1conv2d_5/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ{
1conv2d_5/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
)conv2d_5/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_5/squeeze_batch_dims/Shape:output:08conv2d_5/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_5/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_5/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
)conv2d_5/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         À
#conv2d_5/squeeze_batch_dims/ReshapeReshape"conv2d_5/Conv2D/Reshape_1:output:02conv2d_5/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_5_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ò
#conv2d_5/squeeze_batch_dims/BiasAddBiasAdd,conv2d_5/squeeze_batch_dims/Reshape:output:0:conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+conv2d_5/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         r
'conv2d_5/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿø
"conv2d_5/squeeze_batch_dims/concatConcatV22conv2d_5/squeeze_batch_dims/strided_slice:output:04conv2d_5/squeeze_batch_dims/concat/values_1:output:00conv2d_5/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:É
%conv2d_5/squeeze_batch_dims/Reshape_1Reshape,conv2d_5/squeeze_batch_dims/BiasAdd:output:0+conv2d_5/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity.conv2d_5/squeeze_batch_dims/Reshape_1:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp&^conv2d_4/Conv2D/Conv2D/ReadVariableOp3^conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_5/Conv2D/Conv2D/ReadVariableOp3^conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : 2N
%conv2d_4/Conv2D/Conv2D/ReadVariableOp%conv2d_4/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_5/Conv2D/Conv2D/ReadVariableOp%conv2d_5/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp:Z V
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
×
ó
__inference_call_50856
x.
my_cnn_block_2_50642:"
my_cnn_block_2_50644:.
my_cnn_block_2_50646:"
my_cnn_block_2_50648:N
<rnn_2_my_lstm_cell_2_dense_10_matmul_readvariableop_resource:'K
=rnn_2_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource:N
<rnn_2_my_lstm_cell_2_dense_11_matmul_readvariableop_resource:'K
=rnn_2_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource:N
<rnn_2_my_lstm_cell_2_dense_12_matmul_readvariableop_resource:'K
=rnn_2_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource:N
<rnn_2_my_lstm_cell_2_dense_13_matmul_readvariableop_resource:'K
=rnn_2_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource:<
*dense_14_tensordot_readvariableop_resource:6
(dense_14_biasadd_readvariableop_resource:
identity¢dense_14/BiasAdd/ReadVariableOp¢!dense_14/Tensordot/ReadVariableOp¢&my_cnn_block_2/StatefulPartitionedCall¢4rnn_2/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp¢3rnn_2/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp¢4rnn_2/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp¢3rnn_2/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp¢4rnn_2/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp¢3rnn_2/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp¢4rnn_2/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp¢3rnn_2/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp¢rnn_2/while
&my_cnn_block_2/StatefulPartitionedCallStatefulPartitionedCallxmy_cnn_block_2_50642my_cnn_block_2_50644my_cnn_block_2_50646my_cnn_block_2_50648*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *
fR
__inference_call_50641y
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         »
time_distributed_2/ReshapeReshape/my_cnn_block_2/StatefulPartitionedCall:output:0)time_distributed_2/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Dtime_distributed_2/global_average_pooling2d_2/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      à
2time_distributed_2/global_average_pooling2d_2/MeanMean#time_distributed_2/Reshape:output:0Mtime_distributed_2/global_average_pooling2d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
"time_distributed_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      Ç
time_distributed_2/Reshape_1Reshape;time_distributed_2/global_average_pooling2d_2/Mean:output:0+time_distributed_2/Reshape_1/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
"time_distributed_2/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         ¿
time_distributed_2/Reshape_2Reshape/my_cnn_block_2/StatefulPartitionedCall:output:0+time_distributed_2/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
rnn_2/ShapeShape%time_distributed_2/Reshape_1:output:0*
T0*
_output_shapes
:c
rnn_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
rnn_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
rnn_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
rnn_2/strided_sliceStridedSlicernn_2/Shape:output:0"rnn_2/strided_slice/stack:output:0$rnn_2/strided_slice/stack_1:output:0$rnn_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
rnn_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
rnn_2/zeros/packedPackrnn_2/strided_slice:output:0rnn_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
rnn_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
rnn_2/zerosFillrnn_2/zeros/packed:output:0rnn_2/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
rnn_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
rnn_2/zeros_1/packedPackrnn_2/strided_slice:output:0rnn_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:X
rnn_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
rnn_2/zeros_1Fillrnn_2/zeros_1/packed:output:0rnn_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
rnn_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
rnn_2/transpose	Transpose%time_distributed_2/Reshape_1:output:0rnn_2/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
rnn_2/Shape_1Shapernn_2/transpose:y:0*
T0*
_output_shapes
:e
rnn_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: g
rnn_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
rnn_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
rnn_2/strided_slice_1StridedSlicernn_2/Shape_1:output:0$rnn_2/strided_slice_1/stack:output:0&rnn_2/strided_slice_1/stack_1:output:0&rnn_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!rnn_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
rnn_2/TensorArrayV2TensorListReserve*rnn_2/TensorArrayV2/element_shape:output:0rnn_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
;rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ò
-rnn_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn_2/transpose:y:0Drnn_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒe
rnn_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: g
rnn_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
rnn_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
rnn_2/strided_slice_2StridedSlicernn_2/transpose:y:0$rnn_2/strided_slice_2/stack:output:0&rnn_2/strided_slice_2/stack_1:output:0&rnn_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskp
.rnn_2/my_lstm_cell_2/concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ß
)rnn_2/my_lstm_cell_2/concatenate_5/concatConcatV2rnn_2/strided_slice_2:output:0rnn_2/zeros:output:07rnn_2/my_lstm_cell_2/concatenate_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'°
3rnn_2/my_lstm_cell_2/dense_10/MatMul/ReadVariableOpReadVariableOp<rnn_2_my_lstm_cell_2_dense_10_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0Ñ
$rnn_2/my_lstm_cell_2/dense_10/MatMulMatMul2rnn_2/my_lstm_cell_2/concatenate_5/concat:output:0;rnn_2/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
4rnn_2/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOpReadVariableOp=rnn_2_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ð
%rnn_2/my_lstm_cell_2/dense_10/BiasAddBiasAdd.rnn_2/my_lstm_cell_2/dense_10/MatMul:product:0<rnn_2/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%rnn_2/my_lstm_cell_2/dense_10/SigmoidSigmoid.rnn_2/my_lstm_cell_2/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#rnn_2/my_lstm_cell_2/multiply_2/mulMulrnn_2/zeros_1:output:0)rnn_2/my_lstm_cell_2/dense_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
3rnn_2/my_lstm_cell_2/dense_11/MatMul/ReadVariableOpReadVariableOp<rnn_2_my_lstm_cell_2_dense_11_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0Ñ
$rnn_2/my_lstm_cell_2/dense_11/MatMulMatMul2rnn_2/my_lstm_cell_2/concatenate_5/concat:output:0;rnn_2/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
4rnn_2/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOpReadVariableOp=rnn_2_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ð
%rnn_2/my_lstm_cell_2/dense_11/BiasAddBiasAdd.rnn_2/my_lstm_cell_2/dense_11/MatMul:product:0<rnn_2/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%rnn_2/my_lstm_cell_2/dense_11/SigmoidSigmoid.rnn_2/my_lstm_cell_2/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
3rnn_2/my_lstm_cell_2/dense_12/MatMul/ReadVariableOpReadVariableOp<rnn_2_my_lstm_cell_2_dense_12_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0Ñ
$rnn_2/my_lstm_cell_2/dense_12/MatMulMatMul2rnn_2/my_lstm_cell_2/concatenate_5/concat:output:0;rnn_2/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
4rnn_2/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOpReadVariableOp=rnn_2_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ð
%rnn_2/my_lstm_cell_2/dense_12/BiasAddBiasAdd.rnn_2/my_lstm_cell_2/dense_12/MatMul:product:0<rnn_2/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"rnn_2/my_lstm_cell_2/dense_12/TanhTanh.rnn_2/my_lstm_cell_2/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
%rnn_2/my_lstm_cell_2/multiply_2/mul_1Mul)rnn_2/my_lstm_cell_2/dense_11/Sigmoid:y:0&rnn_2/my_lstm_cell_2/dense_12/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
rnn_2/my_lstm_cell_2/add_5/addAddV2'rnn_2/my_lstm_cell_2/multiply_2/mul:z:0)rnn_2/my_lstm_cell_2/multiply_2/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
3rnn_2/my_lstm_cell_2/dense_13/MatMul/ReadVariableOpReadVariableOp<rnn_2_my_lstm_cell_2_dense_13_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0Ñ
$rnn_2/my_lstm_cell_2/dense_13/MatMulMatMul2rnn_2/my_lstm_cell_2/concatenate_5/concat:output:0;rnn_2/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
4rnn_2/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOpReadVariableOp=rnn_2_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ð
%rnn_2/my_lstm_cell_2/dense_13/BiasAddBiasAdd.rnn_2/my_lstm_cell_2/dense_13/MatMul:product:0<rnn_2/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%rnn_2/my_lstm_cell_2/dense_13/SigmoidSigmoid.rnn_2/my_lstm_cell_2/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&rnn_2/my_lstm_cell_2/activation_2/TanhTanh"rnn_2/my_lstm_cell_2/add_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
%rnn_2/my_lstm_cell_2/multiply_2/mul_2Mul*rnn_2/my_lstm_cell_2/activation_2/Tanh:y:0)rnn_2/my_lstm_cell_2/dense_13/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
#rnn_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ê
rnn_2/TensorArrayV2_1TensorListReserve,rnn_2/TensorArrayV2_1/element_shape:output:0rnn_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒL

rnn_2/timeConst*
_output_shapes
: *
dtype0*
value	B : i
rnn_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿZ
rnn_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ê	
rnn_2/whileWhile!rnn_2/while/loop_counter:output:0'rnn_2/while/maximum_iterations:output:0rnn_2/time:output:0rnn_2/TensorArrayV2_1:handle:0rnn_2/zeros:output:0rnn_2/zeros_1:output:0rnn_2/strided_slice_1:output:0=rnn_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0<rnn_2_my_lstm_cell_2_dense_10_matmul_readvariableop_resource=rnn_2_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource<rnn_2_my_lstm_cell_2_dense_11_matmul_readvariableop_resource=rnn_2_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource<rnn_2_my_lstm_cell_2_dense_12_matmul_readvariableop_resource=rnn_2_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource<rnn_2_my_lstm_cell_2_dense_13_matmul_readvariableop_resource=rnn_2_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( *"
bodyR
rnn_2_while_body_50726*"
condR
rnn_2_while_cond_50725*U
output_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : *
parallel_iterations 
6rnn_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ô
(rnn_2/TensorArrayV2Stack/TensorListStackTensorListStackrnn_2/while:output:3?rnn_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0n
rnn_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿg
rnn_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
rnn_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¥
rnn_2/strided_slice_3StridedSlice1rnn_2/TensorArrayV2Stack/TensorListStack:tensor:0$rnn_2/strided_slice_3/stack:output:0&rnn_2/strided_slice_3/stack_1:output:0&rnn_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskk
rnn_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¨
rnn_2/transpose_1	Transpose1rnn_2/TensorArrayV2Stack/TensorListStack:tensor:0rnn_2/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_14/Tensordot/ReadVariableOpReadVariableOp*dense_14_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0a
dense_14/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_14/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ]
dense_14/Tensordot/ShapeShapernn_2/transpose_1:y:0*
T0*
_output_shapes
:b
 dense_14/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_14/Tensordot/GatherV2GatherV2!dense_14/Tensordot/Shape:output:0 dense_14/Tensordot/free:output:0)dense_14/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_14/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
dense_14/Tensordot/GatherV2_1GatherV2!dense_14/Tensordot/Shape:output:0 dense_14/Tensordot/axes:output:0+dense_14/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_14/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_14/Tensordot/ProdProd$dense_14/Tensordot/GatherV2:output:0!dense_14/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_14/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_14/Tensordot/Prod_1Prod&dense_14/Tensordot/GatherV2_1:output:0#dense_14/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_14/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
dense_14/Tensordot/concatConcatV2 dense_14/Tensordot/free:output:0 dense_14/Tensordot/axes:output:0'dense_14/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_14/Tensordot/stackPack dense_14/Tensordot/Prod:output:0"dense_14/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_14/Tensordot/transpose	Transposernn_2/transpose_1:y:0"dense_14/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
dense_14/Tensordot/ReshapeReshape dense_14/Tensordot/transpose:y:0!dense_14/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
dense_14/Tensordot/MatMulMatMul#dense_14/Tensordot/Reshape:output:0)dense_14/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_14/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_14/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
dense_14/Tensordot/concat_1ConcatV2$dense_14/Tensordot/GatherV2:output:0#dense_14/Tensordot/Const_2:output:0)dense_14/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_14/TensordotReshape#dense_14/Tensordot/MatMul:product:0$dense_14/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_14/BiasAddBiasAdddense_14/Tensordot:output:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
IdentityIdentitydense_14/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ÷
NoOpNoOp ^dense_14/BiasAdd/ReadVariableOp"^dense_14/Tensordot/ReadVariableOp'^my_cnn_block_2/StatefulPartitionedCall5^rnn_2/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp4^rnn_2/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp5^rnn_2/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp4^rnn_2/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp5^rnn_2/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp4^rnn_2/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp5^rnn_2/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp4^rnn_2/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp^rnn_2/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2F
!dense_14/Tensordot/ReadVariableOp!dense_14/Tensordot/ReadVariableOp2P
&my_cnn_block_2/StatefulPartitionedCall&my_cnn_block_2/StatefulPartitionedCall2l
4rnn_2/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp4rnn_2/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp2j
3rnn_2/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp3rnn_2/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp2l
4rnn_2/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp4rnn_2/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp2j
3rnn_2/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp3rnn_2/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp2l
4rnn_2/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp4rnn_2/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp2j
3rnn_2/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp3rnn_2/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp2l
4rnn_2/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp4rnn_2/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp2j
3rnn_2/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp3rnn_2/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp2
rnn_2/whilernn_2/while:V R
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
±
û
rnn_2_while_cond_50725(
$rnn_2_while_rnn_2_while_loop_counter.
*rnn_2_while_rnn_2_while_maximum_iterations
rnn_2_while_placeholder
rnn_2_while_placeholder_1
rnn_2_while_placeholder_2
rnn_2_while_placeholder_3*
&rnn_2_while_less_rnn_2_strided_slice_1?
;rnn_2_while_rnn_2_while_cond_50725___redundant_placeholder0?
;rnn_2_while_rnn_2_while_cond_50725___redundant_placeholder1?
;rnn_2_while_rnn_2_while_cond_50725___redundant_placeholder2?
;rnn_2_while_rnn_2_while_cond_50725___redundant_placeholder3?
;rnn_2_while_rnn_2_while_cond_50725___redundant_placeholder4?
;rnn_2_while_rnn_2_while_cond_50725___redundant_placeholder5?
;rnn_2_while_rnn_2_while_cond_50725___redundant_placeholder6?
;rnn_2_while_rnn_2_while_cond_50725___redundant_placeholder7?
;rnn_2_while_rnn_2_while_cond_50725___redundant_placeholder8
rnn_2_while_identity
z
rnn_2/while/LessLessrnn_2_while_placeholder&rnn_2_while_less_rnn_2_strided_slice_1*
T0*
_output_shapes
: W
rnn_2/while/IdentityIdentityrnn_2/while/Less:z:0*
T0
*
_output_shapes
: "5
rnn_2_while_identityrnn_2/while/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
/
ú
I__inference_my_lstm_cell_2_layer_call_and_return_conditional_losses_51022

inputs

states
states_19
'dense_10_matmul_readvariableop_resource:'6
(dense_10_biasadd_readvariableop_resource:9
'dense_11_matmul_readvariableop_resource:'6
(dense_11_biasadd_readvariableop_resource:9
'dense_12_matmul_readvariableop_resource:'6
(dense_12_biasadd_readvariableop_resource:9
'dense_13_matmul_readvariableop_resource:'6
(dense_13_biasadd_readvariableop_resource:
identity

identity_1

identity_2¢dense_10/BiasAdd/ReadVariableOp¢dense_10/MatMul/ReadVariableOp¢dense_11/BiasAdd/ReadVariableOp¢dense_11/MatMul/ReadVariableOp¢dense_12/BiasAdd/ReadVariableOp¢dense_12/MatMul/ReadVariableOp¢dense_13/BiasAdd/ReadVariableOp¢dense_13/MatMul/ReadVariableOp[
concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatenate_5/concatConcatV2inputsstates"concatenate_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0
dense_10/MatMulMatMulconcatenate_5/concat:output:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_10/SigmoidSigmoiddense_10/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
multiply_2/mulMulstates_1dense_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0
dense_11/MatMulMatMulconcatenate_5/concat:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_11/SigmoidSigmoiddense_11/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0
dense_12/MatMulMatMulconcatenate_5/concat:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dense_12/TanhTanhdense_12/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
multiply_2/mul_1Muldense_11/Sigmoid:y:0dense_12/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
	add_5/addAddV2multiply_2/mul:z:0multiply_2/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0
dense_13/MatMulMatMulconcatenate_5/concat:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_13/SigmoidSigmoiddense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
activation_2/TanhTanhadd_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
multiply_2/mul_2Mulactivation_2/Tanh:y:0dense_13/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitymultiply_2/mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe

Identity_1Identitymultiply_2/mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^

Identity_2Identityadd_5/add:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
¥_
Ï
while_body_53899
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0P
>while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource_0:'M
?while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource_0:P
>while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource_0:'M
?while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource_0:P
>while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource_0:'M
?while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource_0:P
>while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource_0:'M
?while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorN
<while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource:'K
=while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource:N
<while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource:'K
=while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource:N
<while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource:'K
=while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource:N
<while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource:'K
=while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource:¢4while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp¢3while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp¢4while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp¢3while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp¢4while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp¢3while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp¢4while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp¢3while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0p
.while/my_lstm_cell_2/concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ð
)while/my_lstm_cell_2/concatenate_5/concatConcatV20while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_27while/my_lstm_cell_2/concatenate_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'²
3while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOpReadVariableOp>while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0Ñ
$while/my_lstm_cell_2/dense_10/MatMulMatMul2while/my_lstm_cell_2/concatenate_5/concat:output:0;while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
4while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOpReadVariableOp?while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Ð
%while/my_lstm_cell_2/dense_10/BiasAddBiasAdd.while/my_lstm_cell_2/dense_10/MatMul:product:0<while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%while/my_lstm_cell_2/dense_10/SigmoidSigmoid.while/my_lstm_cell_2/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#while/my_lstm_cell_2/multiply_2/mulMulwhile_placeholder_3)while/my_lstm_cell_2/dense_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
3while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOpReadVariableOp>while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0Ñ
$while/my_lstm_cell_2/dense_11/MatMulMatMul2while/my_lstm_cell_2/concatenate_5/concat:output:0;while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
4while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOpReadVariableOp?while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Ð
%while/my_lstm_cell_2/dense_11/BiasAddBiasAdd.while/my_lstm_cell_2/dense_11/MatMul:product:0<while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%while/my_lstm_cell_2/dense_11/SigmoidSigmoid.while/my_lstm_cell_2/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
3while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOpReadVariableOp>while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0Ñ
$while/my_lstm_cell_2/dense_12/MatMulMatMul2while/my_lstm_cell_2/concatenate_5/concat:output:0;while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
4while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOpReadVariableOp?while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Ð
%while/my_lstm_cell_2/dense_12/BiasAddBiasAdd.while/my_lstm_cell_2/dense_12/MatMul:product:0<while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"while/my_lstm_cell_2/dense_12/TanhTanh.while/my_lstm_cell_2/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
%while/my_lstm_cell_2/multiply_2/mul_1Mul)while/my_lstm_cell_2/dense_11/Sigmoid:y:0&while/my_lstm_cell_2/dense_12/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
while/my_lstm_cell_2/add_5/addAddV2'while/my_lstm_cell_2/multiply_2/mul:z:0)while/my_lstm_cell_2/multiply_2/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
3while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOpReadVariableOp>while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0Ñ
$while/my_lstm_cell_2/dense_13/MatMulMatMul2while/my_lstm_cell_2/concatenate_5/concat:output:0;while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
4while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOpReadVariableOp?while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Ð
%while/my_lstm_cell_2/dense_13/BiasAddBiasAdd.while/my_lstm_cell_2/dense_13/MatMul:product:0<while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%while/my_lstm_cell_2/dense_13/SigmoidSigmoid.while/my_lstm_cell_2/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&while/my_lstm_cell_2/activation_2/TanhTanh"while/my_lstm_cell_2/add_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
%while/my_lstm_cell_2/multiply_2/mul_2Mul*while/my_lstm_cell_2/activation_2/Tanh:y:0)while/my_lstm_cell_2/dense_13/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder)while/my_lstm_cell_2/multiply_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity)while/my_lstm_cell_2/multiply_2/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity"while/my_lstm_cell_2/add_5/add:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/NoOpNoOp5^while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp4^while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp5^while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp4^while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp5^while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp4^while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp5^while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp4^while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"
=while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource?while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource_0"~
<while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource>while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource_0"
=while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource?while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource_0"~
<while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource>while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource_0"
=while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource?while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource_0"~
<while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource>while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource_0"
=while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource?while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource_0"~
<while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource>while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2l
4while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp4while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp2j
3while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp3while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp2l
4while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp4while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp2j
3while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp3while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp2l
4while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp4while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp2j
3while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp3while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp2l
4while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp4while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp2j
3while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp3while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
²h
±
@__inference_rnn_2_layer_call_and_return_conditional_losses_53829
inputs_0H
6my_lstm_cell_2_dense_10_matmul_readvariableop_resource:'E
7my_lstm_cell_2_dense_10_biasadd_readvariableop_resource:H
6my_lstm_cell_2_dense_11_matmul_readvariableop_resource:'E
7my_lstm_cell_2_dense_11_biasadd_readvariableop_resource:H
6my_lstm_cell_2_dense_12_matmul_readvariableop_resource:'E
7my_lstm_cell_2_dense_12_biasadd_readvariableop_resource:H
6my_lstm_cell_2_dense_13_matmul_readvariableop_resource:'E
7my_lstm_cell_2_dense_13_biasadd_readvariableop_resource:
identity¢.my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp¢-my_lstm_cell_2/dense_10/MatMul/ReadVariableOp¢.my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp¢-my_lstm_cell_2/dense_11/MatMul/ReadVariableOp¢.my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp¢-my_lstm_cell_2/dense_12/MatMul/ReadVariableOp¢.my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp¢-my_lstm_cell_2/dense_13/MatMul/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
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
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskj
(my_lstm_cell_2/concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ç
#my_lstm_cell_2/concatenate_5/concatConcatV2strided_slice_2:output:0zeros:output:01my_lstm_cell_2/concatenate_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'¤
-my_lstm_cell_2/dense_10/MatMul/ReadVariableOpReadVariableOp6my_lstm_cell_2_dense_10_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0¿
my_lstm_cell_2/dense_10/MatMulMatMul,my_lstm_cell_2/concatenate_5/concat:output:05my_lstm_cell_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOpReadVariableOp7my_lstm_cell_2_dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
my_lstm_cell_2/dense_10/BiasAddBiasAdd(my_lstm_cell_2/dense_10/MatMul:product:06my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
my_lstm_cell_2/dense_10/SigmoidSigmoid(my_lstm_cell_2/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
my_lstm_cell_2/multiply_2/mulMulzeros_1:output:0#my_lstm_cell_2/dense_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-my_lstm_cell_2/dense_11/MatMul/ReadVariableOpReadVariableOp6my_lstm_cell_2_dense_11_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0¿
my_lstm_cell_2/dense_11/MatMulMatMul,my_lstm_cell_2/concatenate_5/concat:output:05my_lstm_cell_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOpReadVariableOp7my_lstm_cell_2_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
my_lstm_cell_2/dense_11/BiasAddBiasAdd(my_lstm_cell_2/dense_11/MatMul:product:06my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
my_lstm_cell_2/dense_11/SigmoidSigmoid(my_lstm_cell_2/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-my_lstm_cell_2/dense_12/MatMul/ReadVariableOpReadVariableOp6my_lstm_cell_2_dense_12_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0¿
my_lstm_cell_2/dense_12/MatMulMatMul,my_lstm_cell_2/concatenate_5/concat:output:05my_lstm_cell_2/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOpReadVariableOp7my_lstm_cell_2_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
my_lstm_cell_2/dense_12/BiasAddBiasAdd(my_lstm_cell_2/dense_12/MatMul:product:06my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
my_lstm_cell_2/dense_12/TanhTanh(my_lstm_cell_2/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
my_lstm_cell_2/multiply_2/mul_1Mul#my_lstm_cell_2/dense_11/Sigmoid:y:0 my_lstm_cell_2/dense_12/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
my_lstm_cell_2/add_5/addAddV2!my_lstm_cell_2/multiply_2/mul:z:0#my_lstm_cell_2/multiply_2/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-my_lstm_cell_2/dense_13/MatMul/ReadVariableOpReadVariableOp6my_lstm_cell_2_dense_13_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0¿
my_lstm_cell_2/dense_13/MatMulMatMul,my_lstm_cell_2/concatenate_5/concat:output:05my_lstm_cell_2/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOpReadVariableOp7my_lstm_cell_2_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
my_lstm_cell_2/dense_13/BiasAddBiasAdd(my_lstm_cell_2/dense_13/MatMul:product:06my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
my_lstm_cell_2/dense_13/SigmoidSigmoid(my_lstm_cell_2/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
 my_lstm_cell_2/activation_2/TanhTanhmy_lstm_cell_2/add_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
my_lstm_cell_2/multiply_2/mul_2Mul$my_lstm_cell_2/activation_2/Tanh:y:0#my_lstm_cell_2/dense_13/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ø
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:06my_lstm_cell_2_dense_10_matmul_readvariableop_resource7my_lstm_cell_2_dense_10_biasadd_readvariableop_resource6my_lstm_cell_2_dense_11_matmul_readvariableop_resource7my_lstm_cell_2_dense_11_biasadd_readvariableop_resource6my_lstm_cell_2_dense_12_matmul_readvariableop_resource7my_lstm_cell_2_dense_12_biasadd_readvariableop_resource6my_lstm_cell_2_dense_13_matmul_readvariableop_resource7my_lstm_cell_2_dense_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( *
bodyR
while_body_53725*
condR
while_cond_53724*U
output_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿk
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÒ
NoOpNoOp/^my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp.^my_lstm_cell_2/dense_10/MatMul/ReadVariableOp/^my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp.^my_lstm_cell_2/dense_11/MatMul/ReadVariableOp/^my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp.^my_lstm_cell_2/dense_12/MatMul/ReadVariableOp/^my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp.^my_lstm_cell_2/dense_13/MatMul/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : 2`
.my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp.my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp2^
-my_lstm_cell_2/dense_10/MatMul/ReadVariableOp-my_lstm_cell_2/dense_10/MatMul/ReadVariableOp2`
.my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp.my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp2^
-my_lstm_cell_2/dense_11/MatMul/ReadVariableOp-my_lstm_cell_2/dense_11/MatMul/ReadVariableOp2`
.my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp.my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp2^
-my_lstm_cell_2/dense_12/MatMul/ReadVariableOp-my_lstm_cell_2/dense_12/MatMul/ReadVariableOp2`
.my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp.my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp2^
-my_lstm_cell_2/dense_13/MatMul/ReadVariableOp-my_lstm_cell_2/dense_13/MatMul/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0

¦
J__inference_my_lstm_model_2_layer_call_and_return_conditional_losses_52145
input_1.
my_cnn_block_2_52110:"
my_cnn_block_2_52112:.
my_cnn_block_2_52114:"
my_cnn_block_2_52116:
rnn_2_52122:'
rnn_2_52124:
rnn_2_52126:'
rnn_2_52128:
rnn_2_52130:'
rnn_2_52132:
rnn_2_52134:'
rnn_2_52136: 
dense_14_52139:
dense_14_52141:
identity¢ dense_14/StatefulPartitionedCall¢&my_cnn_block_2/StatefulPartitionedCall¢rnn_2/StatefulPartitionedCallÂ
&my_cnn_block_2/StatefulPartitionedCallStatefulPartitionedCallinput_1my_cnn_block_2_52110my_cnn_block_2_52112my_cnn_block_2_52114my_cnn_block_2_52116*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_my_cnn_block_2_layer_call_and_return_conditional_losses_51921ø
"time_distributed_2/PartitionedCallPartitionedCall/my_cnn_block_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_time_distributed_2_layer_call_and_return_conditional_losses_50941y
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         »
time_distributed_2/ReshapeReshape/my_cnn_block_2/StatefulPartitionedCall:output:0)time_distributed_2/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿä
rnn_2/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_2/PartitionedCall:output:0rnn_2_52122rnn_2_52124rnn_2_52126rnn_2_52128rnn_2_52130rnn_2_52132rnn_2_52134rnn_2_52136*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_rnn_2_layer_call_and_return_conditional_losses_51826
 dense_14/StatefulPartitionedCallStatefulPartitionedCall&rnn_2/StatefulPartitionedCall:output:0dense_14_52139dense_14_52141*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_51613|
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
NoOpNoOp!^dense_14/StatefulPartitionedCall'^my_cnn_block_2/StatefulPartitionedCall^rnn_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2P
&my_cnn_block_2/StatefulPartitionedCall&my_cnn_block_2/StatefulPartitionedCall2>
rnn_2/StatefulPartitionedCallrnn_2/StatefulPartitionedCall:\ X
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
 
§
J__inference_my_lstm_model_2_layer_call_and_return_conditional_losses_53180
x.
my_cnn_block_2_52966:"
my_cnn_block_2_52968:.
my_cnn_block_2_52970:"
my_cnn_block_2_52972:N
<rnn_2_my_lstm_cell_2_dense_10_matmul_readvariableop_resource:'K
=rnn_2_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource:N
<rnn_2_my_lstm_cell_2_dense_11_matmul_readvariableop_resource:'K
=rnn_2_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource:N
<rnn_2_my_lstm_cell_2_dense_12_matmul_readvariableop_resource:'K
=rnn_2_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource:N
<rnn_2_my_lstm_cell_2_dense_13_matmul_readvariableop_resource:'K
=rnn_2_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource:<
*dense_14_tensordot_readvariableop_resource:6
(dense_14_biasadd_readvariableop_resource:
identity¢dense_14/BiasAdd/ReadVariableOp¢!dense_14/Tensordot/ReadVariableOp¢&my_cnn_block_2/StatefulPartitionedCall¢4rnn_2/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp¢3rnn_2/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp¢4rnn_2/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp¢3rnn_2/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp¢4rnn_2/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp¢3rnn_2/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp¢4rnn_2/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp¢3rnn_2/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp¢rnn_2/while
&my_cnn_block_2/StatefulPartitionedCallStatefulPartitionedCallxmy_cnn_block_2_52966my_cnn_block_2_52968my_cnn_block_2_52970my_cnn_block_2_52972*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *
fR
__inference_call_52213y
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         »
time_distributed_2/ReshapeReshape/my_cnn_block_2/StatefulPartitionedCall:output:0)time_distributed_2/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Dtime_distributed_2/global_average_pooling2d_2/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      à
2time_distributed_2/global_average_pooling2d_2/MeanMean#time_distributed_2/Reshape:output:0Mtime_distributed_2/global_average_pooling2d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
"time_distributed_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      Ç
time_distributed_2/Reshape_1Reshape;time_distributed_2/global_average_pooling2d_2/Mean:output:0+time_distributed_2/Reshape_1/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
"time_distributed_2/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         ¿
time_distributed_2/Reshape_2Reshape/my_cnn_block_2/StatefulPartitionedCall:output:0+time_distributed_2/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
rnn_2/ShapeShape%time_distributed_2/Reshape_1:output:0*
T0*
_output_shapes
:c
rnn_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
rnn_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
rnn_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
rnn_2/strided_sliceStridedSlicernn_2/Shape:output:0"rnn_2/strided_slice/stack:output:0$rnn_2/strided_slice/stack_1:output:0$rnn_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
rnn_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
rnn_2/zeros/packedPackrnn_2/strided_slice:output:0rnn_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
rnn_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
rnn_2/zerosFillrnn_2/zeros/packed:output:0rnn_2/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
rnn_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
rnn_2/zeros_1/packedPackrnn_2/strided_slice:output:0rnn_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:X
rnn_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
rnn_2/zeros_1Fillrnn_2/zeros_1/packed:output:0rnn_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
rnn_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
rnn_2/transpose	Transpose%time_distributed_2/Reshape_1:output:0rnn_2/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
rnn_2/Shape_1Shapernn_2/transpose:y:0*
T0*
_output_shapes
:e
rnn_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: g
rnn_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
rnn_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
rnn_2/strided_slice_1StridedSlicernn_2/Shape_1:output:0$rnn_2/strided_slice_1/stack:output:0&rnn_2/strided_slice_1/stack_1:output:0&rnn_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!rnn_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
rnn_2/TensorArrayV2TensorListReserve*rnn_2/TensorArrayV2/element_shape:output:0rnn_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
;rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ò
-rnn_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn_2/transpose:y:0Drnn_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒe
rnn_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: g
rnn_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
rnn_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
rnn_2/strided_slice_2StridedSlicernn_2/transpose:y:0$rnn_2/strided_slice_2/stack:output:0&rnn_2/strided_slice_2/stack_1:output:0&rnn_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskp
.rnn_2/my_lstm_cell_2/concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ß
)rnn_2/my_lstm_cell_2/concatenate_5/concatConcatV2rnn_2/strided_slice_2:output:0rnn_2/zeros:output:07rnn_2/my_lstm_cell_2/concatenate_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'°
3rnn_2/my_lstm_cell_2/dense_10/MatMul/ReadVariableOpReadVariableOp<rnn_2_my_lstm_cell_2_dense_10_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0Ñ
$rnn_2/my_lstm_cell_2/dense_10/MatMulMatMul2rnn_2/my_lstm_cell_2/concatenate_5/concat:output:0;rnn_2/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
4rnn_2/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOpReadVariableOp=rnn_2_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ð
%rnn_2/my_lstm_cell_2/dense_10/BiasAddBiasAdd.rnn_2/my_lstm_cell_2/dense_10/MatMul:product:0<rnn_2/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%rnn_2/my_lstm_cell_2/dense_10/SigmoidSigmoid.rnn_2/my_lstm_cell_2/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#rnn_2/my_lstm_cell_2/multiply_2/mulMulrnn_2/zeros_1:output:0)rnn_2/my_lstm_cell_2/dense_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
3rnn_2/my_lstm_cell_2/dense_11/MatMul/ReadVariableOpReadVariableOp<rnn_2_my_lstm_cell_2_dense_11_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0Ñ
$rnn_2/my_lstm_cell_2/dense_11/MatMulMatMul2rnn_2/my_lstm_cell_2/concatenate_5/concat:output:0;rnn_2/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
4rnn_2/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOpReadVariableOp=rnn_2_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ð
%rnn_2/my_lstm_cell_2/dense_11/BiasAddBiasAdd.rnn_2/my_lstm_cell_2/dense_11/MatMul:product:0<rnn_2/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%rnn_2/my_lstm_cell_2/dense_11/SigmoidSigmoid.rnn_2/my_lstm_cell_2/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
3rnn_2/my_lstm_cell_2/dense_12/MatMul/ReadVariableOpReadVariableOp<rnn_2_my_lstm_cell_2_dense_12_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0Ñ
$rnn_2/my_lstm_cell_2/dense_12/MatMulMatMul2rnn_2/my_lstm_cell_2/concatenate_5/concat:output:0;rnn_2/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
4rnn_2/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOpReadVariableOp=rnn_2_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ð
%rnn_2/my_lstm_cell_2/dense_12/BiasAddBiasAdd.rnn_2/my_lstm_cell_2/dense_12/MatMul:product:0<rnn_2/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"rnn_2/my_lstm_cell_2/dense_12/TanhTanh.rnn_2/my_lstm_cell_2/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
%rnn_2/my_lstm_cell_2/multiply_2/mul_1Mul)rnn_2/my_lstm_cell_2/dense_11/Sigmoid:y:0&rnn_2/my_lstm_cell_2/dense_12/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
rnn_2/my_lstm_cell_2/add_5/addAddV2'rnn_2/my_lstm_cell_2/multiply_2/mul:z:0)rnn_2/my_lstm_cell_2/multiply_2/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
3rnn_2/my_lstm_cell_2/dense_13/MatMul/ReadVariableOpReadVariableOp<rnn_2_my_lstm_cell_2_dense_13_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0Ñ
$rnn_2/my_lstm_cell_2/dense_13/MatMulMatMul2rnn_2/my_lstm_cell_2/concatenate_5/concat:output:0;rnn_2/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
4rnn_2/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOpReadVariableOp=rnn_2_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ð
%rnn_2/my_lstm_cell_2/dense_13/BiasAddBiasAdd.rnn_2/my_lstm_cell_2/dense_13/MatMul:product:0<rnn_2/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%rnn_2/my_lstm_cell_2/dense_13/SigmoidSigmoid.rnn_2/my_lstm_cell_2/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&rnn_2/my_lstm_cell_2/activation_2/TanhTanh"rnn_2/my_lstm_cell_2/add_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
%rnn_2/my_lstm_cell_2/multiply_2/mul_2Mul*rnn_2/my_lstm_cell_2/activation_2/Tanh:y:0)rnn_2/my_lstm_cell_2/dense_13/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
#rnn_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ê
rnn_2/TensorArrayV2_1TensorListReserve,rnn_2/TensorArrayV2_1/element_shape:output:0rnn_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒL

rnn_2/timeConst*
_output_shapes
: *
dtype0*
value	B : i
rnn_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿZ
rnn_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ê	
rnn_2/whileWhile!rnn_2/while/loop_counter:output:0'rnn_2/while/maximum_iterations:output:0rnn_2/time:output:0rnn_2/TensorArrayV2_1:handle:0rnn_2/zeros:output:0rnn_2/zeros_1:output:0rnn_2/strided_slice_1:output:0=rnn_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0<rnn_2_my_lstm_cell_2_dense_10_matmul_readvariableop_resource=rnn_2_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource<rnn_2_my_lstm_cell_2_dense_11_matmul_readvariableop_resource=rnn_2_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource<rnn_2_my_lstm_cell_2_dense_12_matmul_readvariableop_resource=rnn_2_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource<rnn_2_my_lstm_cell_2_dense_13_matmul_readvariableop_resource=rnn_2_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( *"
bodyR
rnn_2_while_body_53050*"
condR
rnn_2_while_cond_53049*U
output_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : *
parallel_iterations 
6rnn_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ô
(rnn_2/TensorArrayV2Stack/TensorListStackTensorListStackrnn_2/while:output:3?rnn_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0n
rnn_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿg
rnn_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
rnn_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¥
rnn_2/strided_slice_3StridedSlice1rnn_2/TensorArrayV2Stack/TensorListStack:tensor:0$rnn_2/strided_slice_3/stack:output:0&rnn_2/strided_slice_3/stack_1:output:0&rnn_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskk
rnn_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¨
rnn_2/transpose_1	Transpose1rnn_2/TensorArrayV2Stack/TensorListStack:tensor:0rnn_2/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_14/Tensordot/ReadVariableOpReadVariableOp*dense_14_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0a
dense_14/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_14/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ]
dense_14/Tensordot/ShapeShapernn_2/transpose_1:y:0*
T0*
_output_shapes
:b
 dense_14/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_14/Tensordot/GatherV2GatherV2!dense_14/Tensordot/Shape:output:0 dense_14/Tensordot/free:output:0)dense_14/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_14/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
dense_14/Tensordot/GatherV2_1GatherV2!dense_14/Tensordot/Shape:output:0 dense_14/Tensordot/axes:output:0+dense_14/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_14/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_14/Tensordot/ProdProd$dense_14/Tensordot/GatherV2:output:0!dense_14/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_14/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_14/Tensordot/Prod_1Prod&dense_14/Tensordot/GatherV2_1:output:0#dense_14/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_14/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
dense_14/Tensordot/concatConcatV2 dense_14/Tensordot/free:output:0 dense_14/Tensordot/axes:output:0'dense_14/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_14/Tensordot/stackPack dense_14/Tensordot/Prod:output:0"dense_14/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_14/Tensordot/transpose	Transposernn_2/transpose_1:y:0"dense_14/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
dense_14/Tensordot/ReshapeReshape dense_14/Tensordot/transpose:y:0!dense_14/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
dense_14/Tensordot/MatMulMatMul#dense_14/Tensordot/Reshape:output:0)dense_14/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_14/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_14/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
dense_14/Tensordot/concat_1ConcatV2$dense_14/Tensordot/GatherV2:output:0#dense_14/Tensordot/Const_2:output:0)dense_14/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_14/TensordotReshape#dense_14/Tensordot/MatMul:product:0$dense_14/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_14/BiasAddBiasAdddense_14/Tensordot:output:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
IdentityIdentitydense_14/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ÷
NoOpNoOp ^dense_14/BiasAdd/ReadVariableOp"^dense_14/Tensordot/ReadVariableOp'^my_cnn_block_2/StatefulPartitionedCall5^rnn_2/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp4^rnn_2/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp5^rnn_2/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp4^rnn_2/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp5^rnn_2/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp4^rnn_2/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp5^rnn_2/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp4^rnn_2/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp^rnn_2/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2F
!dense_14/Tensordot/ReadVariableOp!dense_14/Tensordot/ReadVariableOp2P
&my_cnn_block_2/StatefulPartitionedCall&my_cnn_block_2/StatefulPartitionedCall2l
4rnn_2/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp4rnn_2/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp2j
3rnn_2/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp3rnn_2/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp2l
4rnn_2/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp4rnn_2/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp2j
3rnn_2/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp3rnn_2/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp2l
4rnn_2/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp4rnn_2/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp2j
3rnn_2/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp3rnn_2/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp2l
4rnn_2/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp4rnn_2/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp2j
3rnn_2/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp3rnn_2/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp2
rnn_2/whilernn_2/while:V R
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
Í

Ç
while_cond_51236
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_51236___redundant_placeholder03
/while_while_cond_51236___redundant_placeholder13
/while_while_cond_51236___redundant_placeholder23
/while_while_cond_51236___redundant_placeholder33
/while_while_cond_51236___redundant_placeholder43
/while_while_cond_51236___redundant_placeholder53
/while_while_cond_51236___redundant_placeholder63
/while_while_cond_51236___redundant_placeholder73
/while_while_cond_51236___redundant_placeholder8
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Í

Ç
while_cond_54246
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_54246___redundant_placeholder03
/while_while_cond_54246___redundant_placeholder13
/while_while_cond_54246___redundant_placeholder23
/while_while_cond_54246___redundant_placeholder33
/while_while_cond_54246___redundant_placeholder43
/while_while_cond_54246___redundant_placeholder53
/while_while_cond_54246___redundant_placeholder63
/while_while_cond_54246___redundant_placeholder73
/while_while_cond_54246___redundant_placeholder8
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ê
ú
C__inference_dense_14_layer_call_and_return_conditional_losses_51613

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±
û
rnn_2_while_cond_52297(
$rnn_2_while_rnn_2_while_loop_counter.
*rnn_2_while_rnn_2_while_maximum_iterations
rnn_2_while_placeholder
rnn_2_while_placeholder_1
rnn_2_while_placeholder_2
rnn_2_while_placeholder_3*
&rnn_2_while_less_rnn_2_strided_slice_1?
;rnn_2_while_rnn_2_while_cond_52297___redundant_placeholder0?
;rnn_2_while_rnn_2_while_cond_52297___redundant_placeholder1?
;rnn_2_while_rnn_2_while_cond_52297___redundant_placeholder2?
;rnn_2_while_rnn_2_while_cond_52297___redundant_placeholder3?
;rnn_2_while_rnn_2_while_cond_52297___redundant_placeholder4?
;rnn_2_while_rnn_2_while_cond_52297___redundant_placeholder5?
;rnn_2_while_rnn_2_while_cond_52297___redundant_placeholder6?
;rnn_2_while_rnn_2_while_cond_52297___redundant_placeholder7?
;rnn_2_while_rnn_2_while_cond_52297___redundant_placeholder8
rnn_2_while_identity
z
rnn_2/while/LessLessrnn_2_while_placeholder&rnn_2_while_less_rnn_2_strided_slice_1*
T0*
_output_shapes
: W
rnn_2/while/IdentityIdentityrnn_2/while/Less:z:0*
T0
*
_output_shapes
: "5
rnn_2_while_identityrnn_2/while/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:

V
:__inference_global_average_pooling2d_2_layer_call_fn_53451

inputs
identityÉ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_50897i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð

(__inference_dense_14_layer_call_fn_54360

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_51613s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó
N
2__inference_time_distributed_2_layer_call_fn_53467

inputs
identityÅ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_time_distributed_2_layer_call_and_return_conditional_losses_50941m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±
û
rnn_2_while_cond_52832(
$rnn_2_while_rnn_2_while_loop_counter.
*rnn_2_while_rnn_2_while_maximum_iterations
rnn_2_while_placeholder
rnn_2_while_placeholder_1
rnn_2_while_placeholder_2
rnn_2_while_placeholder_3*
&rnn_2_while_less_rnn_2_strided_slice_1?
;rnn_2_while_rnn_2_while_cond_52832___redundant_placeholder0?
;rnn_2_while_rnn_2_while_cond_52832___redundant_placeholder1?
;rnn_2_while_rnn_2_while_cond_52832___redundant_placeholder2?
;rnn_2_while_rnn_2_while_cond_52832___redundant_placeholder3?
;rnn_2_while_rnn_2_while_cond_52832___redundant_placeholder4?
;rnn_2_while_rnn_2_while_cond_52832___redundant_placeholder5?
;rnn_2_while_rnn_2_while_cond_52832___redundant_placeholder6?
;rnn_2_while_rnn_2_while_cond_52832___redundant_placeholder7?
;rnn_2_while_rnn_2_while_cond_52832___redundant_placeholder8
rnn_2_while_identity
z
rnn_2/while/LessLessrnn_2_while_placeholder&rnn_2_while_less_rnn_2_strided_slice_1*
T0*
_output_shapes
: W
rnn_2/while/IdentityIdentityrnn_2/while/Less:z:0*
T0
*
_output_shapes
: "5
rnn_2_while_identityrnn_2/while/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
©i
é
rnn_2_while_body_50726(
$rnn_2_while_rnn_2_while_loop_counter.
*rnn_2_while_rnn_2_while_maximum_iterations
rnn_2_while_placeholder
rnn_2_while_placeholder_1
rnn_2_while_placeholder_2
rnn_2_while_placeholder_3'
#rnn_2_while_rnn_2_strided_slice_1_0c
_rnn_2_while_tensorarrayv2read_tensorlistgetitem_rnn_2_tensorarrayunstack_tensorlistfromtensor_0V
Drnn_2_while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource_0:'S
Ernn_2_while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource_0:V
Drnn_2_while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource_0:'S
Ernn_2_while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource_0:V
Drnn_2_while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource_0:'S
Ernn_2_while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource_0:V
Drnn_2_while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource_0:'S
Ernn_2_while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource_0:
rnn_2_while_identity
rnn_2_while_identity_1
rnn_2_while_identity_2
rnn_2_while_identity_3
rnn_2_while_identity_4
rnn_2_while_identity_5%
!rnn_2_while_rnn_2_strided_slice_1a
]rnn_2_while_tensorarrayv2read_tensorlistgetitem_rnn_2_tensorarrayunstack_tensorlistfromtensorT
Brnn_2_while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource:'Q
Crnn_2_while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource:T
Brnn_2_while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource:'Q
Crnn_2_while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource:T
Brnn_2_while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource:'Q
Crnn_2_while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource:T
Brnn_2_while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource:'Q
Crnn_2_while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource:¢:rnn_2/while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp¢9rnn_2/while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp¢:rnn_2/while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp¢9rnn_2/while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp¢:rnn_2/while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp¢9rnn_2/while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp¢:rnn_2/while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp¢9rnn_2/while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp
=rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ä
/rnn_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_rnn_2_while_tensorarrayv2read_tensorlistgetitem_rnn_2_tensorarrayunstack_tensorlistfromtensor_0rnn_2_while_placeholderFrnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0v
4rnn_2/while/my_lstm_cell_2/concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
/rnn_2/while/my_lstm_cell_2/concatenate_5/concatConcatV26rnn_2/while/TensorArrayV2Read/TensorListGetItem:item:0rnn_2_while_placeholder_2=rnn_2/while/my_lstm_cell_2/concatenate_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'¾
9rnn_2/while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOpReadVariableOpDrnn_2_while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ã
*rnn_2/while/my_lstm_cell_2/dense_10/MatMulMatMul8rnn_2/while/my_lstm_cell_2/concatenate_5/concat:output:0Arnn_2/while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
:rnn_2/while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOpReadVariableOpErnn_2_while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0â
+rnn_2/while/my_lstm_cell_2/dense_10/BiasAddBiasAdd4rnn_2/while/my_lstm_cell_2/dense_10/MatMul:product:0Brnn_2/while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+rnn_2/while/my_lstm_cell_2/dense_10/SigmoidSigmoid4rnn_2/while/my_lstm_cell_2/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
)rnn_2/while/my_lstm_cell_2/multiply_2/mulMulrnn_2_while_placeholder_3/rnn_2/while/my_lstm_cell_2/dense_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
9rnn_2/while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOpReadVariableOpDrnn_2_while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ã
*rnn_2/while/my_lstm_cell_2/dense_11/MatMulMatMul8rnn_2/while/my_lstm_cell_2/concatenate_5/concat:output:0Arnn_2/while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
:rnn_2/while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOpReadVariableOpErnn_2_while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0â
+rnn_2/while/my_lstm_cell_2/dense_11/BiasAddBiasAdd4rnn_2/while/my_lstm_cell_2/dense_11/MatMul:product:0Brnn_2/while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+rnn_2/while/my_lstm_cell_2/dense_11/SigmoidSigmoid4rnn_2/while/my_lstm_cell_2/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
9rnn_2/while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOpReadVariableOpDrnn_2_while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ã
*rnn_2/while/my_lstm_cell_2/dense_12/MatMulMatMul8rnn_2/while/my_lstm_cell_2/concatenate_5/concat:output:0Arnn_2/while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
:rnn_2/while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOpReadVariableOpErnn_2_while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0â
+rnn_2/while/my_lstm_cell_2/dense_12/BiasAddBiasAdd4rnn_2/while/my_lstm_cell_2/dense_12/MatMul:product:0Brnn_2/while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(rnn_2/while/my_lstm_cell_2/dense_12/TanhTanh4rnn_2/while/my_lstm_cell_2/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
+rnn_2/while/my_lstm_cell_2/multiply_2/mul_1Mul/rnn_2/while/my_lstm_cell_2/dense_11/Sigmoid:y:0,rnn_2/while/my_lstm_cell_2/dense_12/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
$rnn_2/while/my_lstm_cell_2/add_5/addAddV2-rnn_2/while/my_lstm_cell_2/multiply_2/mul:z:0/rnn_2/while/my_lstm_cell_2/multiply_2/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
9rnn_2/while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOpReadVariableOpDrnn_2_while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ã
*rnn_2/while/my_lstm_cell_2/dense_13/MatMulMatMul8rnn_2/while/my_lstm_cell_2/concatenate_5/concat:output:0Arnn_2/while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
:rnn_2/while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOpReadVariableOpErnn_2_while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0â
+rnn_2/while/my_lstm_cell_2/dense_13/BiasAddBiasAdd4rnn_2/while/my_lstm_cell_2/dense_13/MatMul:product:0Brnn_2/while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+rnn_2/while/my_lstm_cell_2/dense_13/SigmoidSigmoid4rnn_2/while/my_lstm_cell_2/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,rnn_2/while/my_lstm_cell_2/activation_2/TanhTanh(rnn_2/while/my_lstm_cell_2/add_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
+rnn_2/while/my_lstm_cell_2/multiply_2/mul_2Mul0rnn_2/while/my_lstm_cell_2/activation_2/Tanh:y:0/rnn_2/while/my_lstm_cell_2/dense_13/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
0rnn_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrnn_2_while_placeholder_1rnn_2_while_placeholder/rnn_2/while/my_lstm_cell_2/multiply_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒS
rnn_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :n
rnn_2/while/addAddV2rnn_2_while_placeholderrnn_2/while/add/y:output:0*
T0*
_output_shapes
: U
rnn_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
rnn_2/while/add_1AddV2$rnn_2_while_rnn_2_while_loop_counterrnn_2/while/add_1/y:output:0*
T0*
_output_shapes
: k
rnn_2/while/IdentityIdentityrnn_2/while/add_1:z:0^rnn_2/while/NoOp*
T0*
_output_shapes
: 
rnn_2/while/Identity_1Identity*rnn_2_while_rnn_2_while_maximum_iterations^rnn_2/while/NoOp*
T0*
_output_shapes
: k
rnn_2/while/Identity_2Identityrnn_2/while/add:z:0^rnn_2/while/NoOp*
T0*
_output_shapes
: 
rnn_2/while/Identity_3Identity@rnn_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^rnn_2/while/NoOp*
T0*
_output_shapes
: 
rnn_2/while/Identity_4Identity/rnn_2/while/my_lstm_cell_2/multiply_2/mul_2:z:0^rnn_2/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn_2/while/Identity_5Identity(rnn_2/while/my_lstm_cell_2/add_5/add:z:0^rnn_2/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
rnn_2/while/NoOpNoOp;^rnn_2/while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp:^rnn_2/while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp;^rnn_2/while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp:^rnn_2/while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp;^rnn_2/while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp:^rnn_2/while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp;^rnn_2/while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp:^rnn_2/while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "5
rnn_2_while_identityrnn_2/while/Identity:output:0"9
rnn_2_while_identity_1rnn_2/while/Identity_1:output:0"9
rnn_2_while_identity_2rnn_2/while/Identity_2:output:0"9
rnn_2_while_identity_3rnn_2/while/Identity_3:output:0"9
rnn_2_while_identity_4rnn_2/while/Identity_4:output:0"9
rnn_2_while_identity_5rnn_2/while/Identity_5:output:0"
Crnn_2_while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resourceErnn_2_while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource_0"
Brnn_2_while_my_lstm_cell_2_dense_10_matmul_readvariableop_resourceDrnn_2_while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource_0"
Crnn_2_while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resourceErnn_2_while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource_0"
Brnn_2_while_my_lstm_cell_2_dense_11_matmul_readvariableop_resourceDrnn_2_while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource_0"
Crnn_2_while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resourceErnn_2_while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource_0"
Brnn_2_while_my_lstm_cell_2_dense_12_matmul_readvariableop_resourceDrnn_2_while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource_0"
Crnn_2_while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resourceErnn_2_while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource_0"
Brnn_2_while_my_lstm_cell_2_dense_13_matmul_readvariableop_resourceDrnn_2_while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource_0"H
!rnn_2_while_rnn_2_strided_slice_1#rnn_2_while_rnn_2_strided_slice_1_0"À
]rnn_2_while_tensorarrayv2read_tensorlistgetitem_rnn_2_tensorarrayunstack_tensorlistfromtensor_rnn_2_while_tensorarrayv2read_tensorlistgetitem_rnn_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2x
:rnn_2/while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp:rnn_2/while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp2v
9rnn_2/while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp9rnn_2/while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp2x
:rnn_2/while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp:rnn_2/while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp2v
9rnn_2/while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp9rnn_2/while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp2x
:rnn_2/while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp:rnn_2/while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp2v
9rnn_2/while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp9rnn_2/while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp2x
:rnn_2/while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp:rnn_2/while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp2v
9rnn_2/while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp9rnn_2/while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ÿ
ô
/__inference_my_lstm_model_2_layer_call_fn_52069
input_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:'
	unknown_4:
	unknown_5:'
	unknown_6:
	unknown_7:'
	unknown_8:
	unknown_9:'

unknown_10:

unknown_11:

unknown_12:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_my_lstm_model_2_layer_call_and_return_conditional_losses_52005s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
«N

__inference_call_50641	
inputH
.conv2d_4_conv2d_conv2d_readvariableop_resource:I
;conv2d_4_squeeze_batch_dims_biasadd_readvariableop_resource:H
.conv2d_5_conv2d_conv2d_readvariableop_resource:I
;conv2d_5_squeeze_batch_dims_biasadd_readvariableop_resource:
identity¢%conv2d_4/Conv2D/Conv2D/ReadVariableOp¢2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp¢%conv2d_5/Conv2D/Conv2D/ReadVariableOp¢2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOpJ
conv2d_4/Conv2D/ShapeShapeinput*
T0*
_output_shapes
:m
#conv2d_4/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_4/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿo
%conv2d_4/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv2d_4/Conv2D/strided_sliceStridedSliceconv2d_4/Conv2D/Shape:output:0,conv2d_4/Conv2D/strided_slice/stack:output:0.conv2d_4/Conv2D/strided_slice/stack_1:output:0.conv2d_4/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_4/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         
conv2d_4/Conv2D/ReshapeReshapeinput&conv2d_4/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%conv2d_4/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_4_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ó
conv2d_4/Conv2D/Conv2DConv2D conv2d_4/Conv2D/Reshape:output:0-conv2d_4/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
t
conv2d_4/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         f
conv2d_4/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÈ
conv2d_4/Conv2D/concatConcatV2&conv2d_4/Conv2D/strided_slice:output:0(conv2d_4/Conv2D/concat/values_1:output:0$conv2d_4/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:¤
conv2d_4/Conv2D/Reshape_1Reshapeconv2d_4/Conv2D/Conv2D:output:0conv2d_4/Conv2D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿs
!conv2d_4/squeeze_batch_dims/ShapeShape"conv2d_4/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_4/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1conv2d_4/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ{
1conv2d_4/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
)conv2d_4/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_4/squeeze_batch_dims/Shape:output:08conv2d_4/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_4/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_4/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
)conv2d_4/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         À
#conv2d_4/squeeze_batch_dims/ReshapeReshape"conv2d_4/Conv2D/Reshape_1:output:02conv2d_4/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_4_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ò
#conv2d_4/squeeze_batch_dims/BiasAddBiasAdd,conv2d_4/squeeze_batch_dims/Reshape:output:0:conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+conv2d_4/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         r
'conv2d_4/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿø
"conv2d_4/squeeze_batch_dims/concatConcatV22conv2d_4/squeeze_batch_dims/strided_slice:output:04conv2d_4/squeeze_batch_dims/concat/values_1:output:00conv2d_4/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:É
%conv2d_4/squeeze_batch_dims/Reshape_1Reshape,conv2d_4/squeeze_batch_dims/BiasAdd:output:0+conv2d_4/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿs
conv2d_5/Conv2D/ShapeShape.conv2d_4/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:m
#conv2d_5/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_5/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿo
%conv2d_5/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv2d_5/Conv2D/strided_sliceStridedSliceconv2d_5/Conv2D/Shape:output:0,conv2d_5/Conv2D/strided_slice/stack:output:0.conv2d_5/Conv2D/strided_slice/stack_1:output:0.conv2d_5/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_5/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         ´
conv2d_5/Conv2D/ReshapeReshape.conv2d_4/squeeze_batch_dims/Reshape_1:output:0&conv2d_5/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%conv2d_5/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_5_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ó
conv2d_5/Conv2D/Conv2DConv2D conv2d_5/Conv2D/Reshape:output:0-conv2d_5/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
t
conv2d_5/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         f
conv2d_5/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÈ
conv2d_5/Conv2D/concatConcatV2&conv2d_5/Conv2D/strided_slice:output:0(conv2d_5/Conv2D/concat/values_1:output:0$conv2d_5/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:¤
conv2d_5/Conv2D/Reshape_1Reshapeconv2d_5/Conv2D/Conv2D:output:0conv2d_5/Conv2D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿs
!conv2d_5/squeeze_batch_dims/ShapeShape"conv2d_5/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_5/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1conv2d_5/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ{
1conv2d_5/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
)conv2d_5/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_5/squeeze_batch_dims/Shape:output:08conv2d_5/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_5/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_5/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
)conv2d_5/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         À
#conv2d_5/squeeze_batch_dims/ReshapeReshape"conv2d_5/Conv2D/Reshape_1:output:02conv2d_5/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_5_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ò
#conv2d_5/squeeze_batch_dims/BiasAddBiasAdd,conv2d_5/squeeze_batch_dims/Reshape:output:0:conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+conv2d_5/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         r
'conv2d_5/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿø
"conv2d_5/squeeze_batch_dims/concatConcatV22conv2d_5/squeeze_batch_dims/strided_slice:output:04conv2d_5/squeeze_batch_dims/concat/values_1:output:00conv2d_5/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:É
%conv2d_5/squeeze_batch_dims/Reshape_1Reshape,conv2d_5/squeeze_batch_dims/BiasAdd:output:0+conv2d_5/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity.conv2d_5/squeeze_batch_dims/Reshape_1:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp&^conv2d_4/Conv2D/Conv2D/ReadVariableOp3^conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_5/Conv2D/Conv2D/ReadVariableOp3^conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : 2N
%conv2d_4/Conv2D/Conv2D/ReadVariableOp%conv2d_4/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_5/Conv2D/Conv2D/ReadVariableOp%conv2d_5/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp:Z V
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
¥_
Ï
while_body_54073
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0P
>while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource_0:'M
?while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource_0:P
>while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource_0:'M
?while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource_0:P
>while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource_0:'M
?while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource_0:P
>while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource_0:'M
?while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorN
<while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource:'K
=while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource:N
<while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource:'K
=while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource:N
<while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource:'K
=while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource:N
<while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource:'K
=while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource:¢4while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp¢3while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp¢4while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp¢3while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp¢4while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp¢3while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp¢4while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp¢3while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0p
.while/my_lstm_cell_2/concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ð
)while/my_lstm_cell_2/concatenate_5/concatConcatV20while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_27while/my_lstm_cell_2/concatenate_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'²
3while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOpReadVariableOp>while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0Ñ
$while/my_lstm_cell_2/dense_10/MatMulMatMul2while/my_lstm_cell_2/concatenate_5/concat:output:0;while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
4while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOpReadVariableOp?while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Ð
%while/my_lstm_cell_2/dense_10/BiasAddBiasAdd.while/my_lstm_cell_2/dense_10/MatMul:product:0<while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%while/my_lstm_cell_2/dense_10/SigmoidSigmoid.while/my_lstm_cell_2/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#while/my_lstm_cell_2/multiply_2/mulMulwhile_placeholder_3)while/my_lstm_cell_2/dense_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
3while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOpReadVariableOp>while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0Ñ
$while/my_lstm_cell_2/dense_11/MatMulMatMul2while/my_lstm_cell_2/concatenate_5/concat:output:0;while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
4while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOpReadVariableOp?while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Ð
%while/my_lstm_cell_2/dense_11/BiasAddBiasAdd.while/my_lstm_cell_2/dense_11/MatMul:product:0<while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%while/my_lstm_cell_2/dense_11/SigmoidSigmoid.while/my_lstm_cell_2/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
3while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOpReadVariableOp>while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0Ñ
$while/my_lstm_cell_2/dense_12/MatMulMatMul2while/my_lstm_cell_2/concatenate_5/concat:output:0;while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
4while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOpReadVariableOp?while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Ð
%while/my_lstm_cell_2/dense_12/BiasAddBiasAdd.while/my_lstm_cell_2/dense_12/MatMul:product:0<while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"while/my_lstm_cell_2/dense_12/TanhTanh.while/my_lstm_cell_2/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
%while/my_lstm_cell_2/multiply_2/mul_1Mul)while/my_lstm_cell_2/dense_11/Sigmoid:y:0&while/my_lstm_cell_2/dense_12/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
while/my_lstm_cell_2/add_5/addAddV2'while/my_lstm_cell_2/multiply_2/mul:z:0)while/my_lstm_cell_2/multiply_2/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
3while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOpReadVariableOp>while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0Ñ
$while/my_lstm_cell_2/dense_13/MatMulMatMul2while/my_lstm_cell_2/concatenate_5/concat:output:0;while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
4while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOpReadVariableOp?while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Ð
%while/my_lstm_cell_2/dense_13/BiasAddBiasAdd.while/my_lstm_cell_2/dense_13/MatMul:product:0<while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%while/my_lstm_cell_2/dense_13/SigmoidSigmoid.while/my_lstm_cell_2/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&while/my_lstm_cell_2/activation_2/TanhTanh"while/my_lstm_cell_2/add_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
%while/my_lstm_cell_2/multiply_2/mul_2Mul*while/my_lstm_cell_2/activation_2/Tanh:y:0)while/my_lstm_cell_2/dense_13/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder)while/my_lstm_cell_2/multiply_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity)while/my_lstm_cell_2/multiply_2/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity"while/my_lstm_cell_2/add_5/add:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/NoOpNoOp5^while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp4^while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp5^while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp4^while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp5^while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp4^while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp5^while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp4^while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"
=while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource?while_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource_0"~
<while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource>while_my_lstm_cell_2_dense_10_matmul_readvariableop_resource_0"
=while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource?while_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource_0"~
<while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource>while_my_lstm_cell_2_dense_11_matmul_readvariableop_resource_0"
=while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource?while_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource_0"~
<while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource>while_my_lstm_cell_2_dense_12_matmul_readvariableop_resource_0"
=while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource?while_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource_0"~
<while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource>while_my_lstm_cell_2_dense_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2l
4while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp4while/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp2j
3while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp3while/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp2l
4while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp4while/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp2j
3while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp3while/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp2l
4while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp4while/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp2j
3while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp3while/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp2l
4while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp4while/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp2j
3while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp3while/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
«N

__inference_call_52213	
inputH
.conv2d_4_conv2d_conv2d_readvariableop_resource:I
;conv2d_4_squeeze_batch_dims_biasadd_readvariableop_resource:H
.conv2d_5_conv2d_conv2d_readvariableop_resource:I
;conv2d_5_squeeze_batch_dims_biasadd_readvariableop_resource:
identity¢%conv2d_4/Conv2D/Conv2D/ReadVariableOp¢2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp¢%conv2d_5/Conv2D/Conv2D/ReadVariableOp¢2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOpJ
conv2d_4/Conv2D/ShapeShapeinput*
T0*
_output_shapes
:m
#conv2d_4/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_4/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿo
%conv2d_4/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv2d_4/Conv2D/strided_sliceStridedSliceconv2d_4/Conv2D/Shape:output:0,conv2d_4/Conv2D/strided_slice/stack:output:0.conv2d_4/Conv2D/strided_slice/stack_1:output:0.conv2d_4/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_4/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         
conv2d_4/Conv2D/ReshapeReshapeinput&conv2d_4/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%conv2d_4/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_4_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ó
conv2d_4/Conv2D/Conv2DConv2D conv2d_4/Conv2D/Reshape:output:0-conv2d_4/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
t
conv2d_4/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         f
conv2d_4/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÈ
conv2d_4/Conv2D/concatConcatV2&conv2d_4/Conv2D/strided_slice:output:0(conv2d_4/Conv2D/concat/values_1:output:0$conv2d_4/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:¤
conv2d_4/Conv2D/Reshape_1Reshapeconv2d_4/Conv2D/Conv2D:output:0conv2d_4/Conv2D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿs
!conv2d_4/squeeze_batch_dims/ShapeShape"conv2d_4/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_4/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1conv2d_4/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ{
1conv2d_4/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
)conv2d_4/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_4/squeeze_batch_dims/Shape:output:08conv2d_4/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_4/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_4/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
)conv2d_4/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         À
#conv2d_4/squeeze_batch_dims/ReshapeReshape"conv2d_4/Conv2D/Reshape_1:output:02conv2d_4/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_4_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ò
#conv2d_4/squeeze_batch_dims/BiasAddBiasAdd,conv2d_4/squeeze_batch_dims/Reshape:output:0:conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+conv2d_4/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         r
'conv2d_4/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿø
"conv2d_4/squeeze_batch_dims/concatConcatV22conv2d_4/squeeze_batch_dims/strided_slice:output:04conv2d_4/squeeze_batch_dims/concat/values_1:output:00conv2d_4/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:É
%conv2d_4/squeeze_batch_dims/Reshape_1Reshape,conv2d_4/squeeze_batch_dims/BiasAdd:output:0+conv2d_4/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿs
conv2d_5/Conv2D/ShapeShape.conv2d_4/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:m
#conv2d_5/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_5/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿo
%conv2d_5/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv2d_5/Conv2D/strided_sliceStridedSliceconv2d_5/Conv2D/Shape:output:0,conv2d_5/Conv2D/strided_slice/stack:output:0.conv2d_5/Conv2D/strided_slice/stack_1:output:0.conv2d_5/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_5/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         ´
conv2d_5/Conv2D/ReshapeReshape.conv2d_4/squeeze_batch_dims/Reshape_1:output:0&conv2d_5/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%conv2d_5/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_5_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ó
conv2d_5/Conv2D/Conv2DConv2D conv2d_5/Conv2D/Reshape:output:0-conv2d_5/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
t
conv2d_5/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         f
conv2d_5/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÈ
conv2d_5/Conv2D/concatConcatV2&conv2d_5/Conv2D/strided_slice:output:0(conv2d_5/Conv2D/concat/values_1:output:0$conv2d_5/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:¤
conv2d_5/Conv2D/Reshape_1Reshapeconv2d_5/Conv2D/Conv2D:output:0conv2d_5/Conv2D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿs
!conv2d_5/squeeze_batch_dims/ShapeShape"conv2d_5/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_5/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1conv2d_5/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ{
1conv2d_5/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
)conv2d_5/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_5/squeeze_batch_dims/Shape:output:08conv2d_5/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_5/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_5/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
)conv2d_5/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         À
#conv2d_5/squeeze_batch_dims/ReshapeReshape"conv2d_5/Conv2D/Reshape_1:output:02conv2d_5/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_5_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ò
#conv2d_5/squeeze_batch_dims/BiasAddBiasAdd,conv2d_5/squeeze_batch_dims/Reshape:output:0:conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+conv2d_5/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         r
'conv2d_5/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿø
"conv2d_5/squeeze_batch_dims/concatConcatV22conv2d_5/squeeze_batch_dims/strided_slice:output:04conv2d_5/squeeze_batch_dims/concat/values_1:output:00conv2d_5/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:É
%conv2d_5/squeeze_batch_dims/Reshape_1Reshape,conv2d_5/squeeze_batch_dims/BiasAdd:output:0+conv2d_5/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity.conv2d_5/squeeze_batch_dims/Reshape_1:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp&^conv2d_4/Conv2D/Conv2D/ReadVariableOp3^conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_5/Conv2D/Conv2D/ReadVariableOp3^conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : 2N
%conv2d_4/Conv2D/Conv2D/ReadVariableOp%conv2d_4/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_5/Conv2D/Conv2D/ReadVariableOp%conv2d_5/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp:Z V
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
 
§
J__inference_my_lstm_model_2_layer_call_and_return_conditional_losses_52963
x.
my_cnn_block_2_52749:"
my_cnn_block_2_52751:.
my_cnn_block_2_52753:"
my_cnn_block_2_52755:N
<rnn_2_my_lstm_cell_2_dense_10_matmul_readvariableop_resource:'K
=rnn_2_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource:N
<rnn_2_my_lstm_cell_2_dense_11_matmul_readvariableop_resource:'K
=rnn_2_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource:N
<rnn_2_my_lstm_cell_2_dense_12_matmul_readvariableop_resource:'K
=rnn_2_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource:N
<rnn_2_my_lstm_cell_2_dense_13_matmul_readvariableop_resource:'K
=rnn_2_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource:<
*dense_14_tensordot_readvariableop_resource:6
(dense_14_biasadd_readvariableop_resource:
identity¢dense_14/BiasAdd/ReadVariableOp¢!dense_14/Tensordot/ReadVariableOp¢&my_cnn_block_2/StatefulPartitionedCall¢4rnn_2/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp¢3rnn_2/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp¢4rnn_2/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp¢3rnn_2/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp¢4rnn_2/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp¢3rnn_2/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp¢4rnn_2/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp¢3rnn_2/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp¢rnn_2/while
&my_cnn_block_2/StatefulPartitionedCallStatefulPartitionedCallxmy_cnn_block_2_52749my_cnn_block_2_52751my_cnn_block_2_52753my_cnn_block_2_52755*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *
fR
__inference_call_50641y
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         »
time_distributed_2/ReshapeReshape/my_cnn_block_2/StatefulPartitionedCall:output:0)time_distributed_2/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Dtime_distributed_2/global_average_pooling2d_2/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      à
2time_distributed_2/global_average_pooling2d_2/MeanMean#time_distributed_2/Reshape:output:0Mtime_distributed_2/global_average_pooling2d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
"time_distributed_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      Ç
time_distributed_2/Reshape_1Reshape;time_distributed_2/global_average_pooling2d_2/Mean:output:0+time_distributed_2/Reshape_1/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
"time_distributed_2/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         ¿
time_distributed_2/Reshape_2Reshape/my_cnn_block_2/StatefulPartitionedCall:output:0+time_distributed_2/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
rnn_2/ShapeShape%time_distributed_2/Reshape_1:output:0*
T0*
_output_shapes
:c
rnn_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
rnn_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
rnn_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
rnn_2/strided_sliceStridedSlicernn_2/Shape:output:0"rnn_2/strided_slice/stack:output:0$rnn_2/strided_slice/stack_1:output:0$rnn_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
rnn_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
rnn_2/zeros/packedPackrnn_2/strided_slice:output:0rnn_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
rnn_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
rnn_2/zerosFillrnn_2/zeros/packed:output:0rnn_2/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
rnn_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
rnn_2/zeros_1/packedPackrnn_2/strided_slice:output:0rnn_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:X
rnn_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
rnn_2/zeros_1Fillrnn_2/zeros_1/packed:output:0rnn_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
rnn_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
rnn_2/transpose	Transpose%time_distributed_2/Reshape_1:output:0rnn_2/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
rnn_2/Shape_1Shapernn_2/transpose:y:0*
T0*
_output_shapes
:e
rnn_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: g
rnn_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
rnn_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
rnn_2/strided_slice_1StridedSlicernn_2/Shape_1:output:0$rnn_2/strided_slice_1/stack:output:0&rnn_2/strided_slice_1/stack_1:output:0&rnn_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!rnn_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
rnn_2/TensorArrayV2TensorListReserve*rnn_2/TensorArrayV2/element_shape:output:0rnn_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
;rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ò
-rnn_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn_2/transpose:y:0Drnn_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒe
rnn_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: g
rnn_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
rnn_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
rnn_2/strided_slice_2StridedSlicernn_2/transpose:y:0$rnn_2/strided_slice_2/stack:output:0&rnn_2/strided_slice_2/stack_1:output:0&rnn_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskp
.rnn_2/my_lstm_cell_2/concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ß
)rnn_2/my_lstm_cell_2/concatenate_5/concatConcatV2rnn_2/strided_slice_2:output:0rnn_2/zeros:output:07rnn_2/my_lstm_cell_2/concatenate_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'°
3rnn_2/my_lstm_cell_2/dense_10/MatMul/ReadVariableOpReadVariableOp<rnn_2_my_lstm_cell_2_dense_10_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0Ñ
$rnn_2/my_lstm_cell_2/dense_10/MatMulMatMul2rnn_2/my_lstm_cell_2/concatenate_5/concat:output:0;rnn_2/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
4rnn_2/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOpReadVariableOp=rnn_2_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ð
%rnn_2/my_lstm_cell_2/dense_10/BiasAddBiasAdd.rnn_2/my_lstm_cell_2/dense_10/MatMul:product:0<rnn_2/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%rnn_2/my_lstm_cell_2/dense_10/SigmoidSigmoid.rnn_2/my_lstm_cell_2/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#rnn_2/my_lstm_cell_2/multiply_2/mulMulrnn_2/zeros_1:output:0)rnn_2/my_lstm_cell_2/dense_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
3rnn_2/my_lstm_cell_2/dense_11/MatMul/ReadVariableOpReadVariableOp<rnn_2_my_lstm_cell_2_dense_11_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0Ñ
$rnn_2/my_lstm_cell_2/dense_11/MatMulMatMul2rnn_2/my_lstm_cell_2/concatenate_5/concat:output:0;rnn_2/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
4rnn_2/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOpReadVariableOp=rnn_2_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ð
%rnn_2/my_lstm_cell_2/dense_11/BiasAddBiasAdd.rnn_2/my_lstm_cell_2/dense_11/MatMul:product:0<rnn_2/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%rnn_2/my_lstm_cell_2/dense_11/SigmoidSigmoid.rnn_2/my_lstm_cell_2/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
3rnn_2/my_lstm_cell_2/dense_12/MatMul/ReadVariableOpReadVariableOp<rnn_2_my_lstm_cell_2_dense_12_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0Ñ
$rnn_2/my_lstm_cell_2/dense_12/MatMulMatMul2rnn_2/my_lstm_cell_2/concatenate_5/concat:output:0;rnn_2/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
4rnn_2/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOpReadVariableOp=rnn_2_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ð
%rnn_2/my_lstm_cell_2/dense_12/BiasAddBiasAdd.rnn_2/my_lstm_cell_2/dense_12/MatMul:product:0<rnn_2/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"rnn_2/my_lstm_cell_2/dense_12/TanhTanh.rnn_2/my_lstm_cell_2/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
%rnn_2/my_lstm_cell_2/multiply_2/mul_1Mul)rnn_2/my_lstm_cell_2/dense_11/Sigmoid:y:0&rnn_2/my_lstm_cell_2/dense_12/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
rnn_2/my_lstm_cell_2/add_5/addAddV2'rnn_2/my_lstm_cell_2/multiply_2/mul:z:0)rnn_2/my_lstm_cell_2/multiply_2/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
3rnn_2/my_lstm_cell_2/dense_13/MatMul/ReadVariableOpReadVariableOp<rnn_2_my_lstm_cell_2_dense_13_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0Ñ
$rnn_2/my_lstm_cell_2/dense_13/MatMulMatMul2rnn_2/my_lstm_cell_2/concatenate_5/concat:output:0;rnn_2/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
4rnn_2/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOpReadVariableOp=rnn_2_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ð
%rnn_2/my_lstm_cell_2/dense_13/BiasAddBiasAdd.rnn_2/my_lstm_cell_2/dense_13/MatMul:product:0<rnn_2/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%rnn_2/my_lstm_cell_2/dense_13/SigmoidSigmoid.rnn_2/my_lstm_cell_2/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&rnn_2/my_lstm_cell_2/activation_2/TanhTanh"rnn_2/my_lstm_cell_2/add_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
%rnn_2/my_lstm_cell_2/multiply_2/mul_2Mul*rnn_2/my_lstm_cell_2/activation_2/Tanh:y:0)rnn_2/my_lstm_cell_2/dense_13/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
#rnn_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ê
rnn_2/TensorArrayV2_1TensorListReserve,rnn_2/TensorArrayV2_1/element_shape:output:0rnn_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒL

rnn_2/timeConst*
_output_shapes
: *
dtype0*
value	B : i
rnn_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿZ
rnn_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ê	
rnn_2/whileWhile!rnn_2/while/loop_counter:output:0'rnn_2/while/maximum_iterations:output:0rnn_2/time:output:0rnn_2/TensorArrayV2_1:handle:0rnn_2/zeros:output:0rnn_2/zeros_1:output:0rnn_2/strided_slice_1:output:0=rnn_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0<rnn_2_my_lstm_cell_2_dense_10_matmul_readvariableop_resource=rnn_2_my_lstm_cell_2_dense_10_biasadd_readvariableop_resource<rnn_2_my_lstm_cell_2_dense_11_matmul_readvariableop_resource=rnn_2_my_lstm_cell_2_dense_11_biasadd_readvariableop_resource<rnn_2_my_lstm_cell_2_dense_12_matmul_readvariableop_resource=rnn_2_my_lstm_cell_2_dense_12_biasadd_readvariableop_resource<rnn_2_my_lstm_cell_2_dense_13_matmul_readvariableop_resource=rnn_2_my_lstm_cell_2_dense_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( *"
bodyR
rnn_2_while_body_52833*"
condR
rnn_2_while_cond_52832*U
output_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : *
parallel_iterations 
6rnn_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ô
(rnn_2/TensorArrayV2Stack/TensorListStackTensorListStackrnn_2/while:output:3?rnn_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0n
rnn_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿg
rnn_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
rnn_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¥
rnn_2/strided_slice_3StridedSlice1rnn_2/TensorArrayV2Stack/TensorListStack:tensor:0$rnn_2/strided_slice_3/stack:output:0&rnn_2/strided_slice_3/stack_1:output:0&rnn_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskk
rnn_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¨
rnn_2/transpose_1	Transpose1rnn_2/TensorArrayV2Stack/TensorListStack:tensor:0rnn_2/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_14/Tensordot/ReadVariableOpReadVariableOp*dense_14_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0a
dense_14/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_14/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ]
dense_14/Tensordot/ShapeShapernn_2/transpose_1:y:0*
T0*
_output_shapes
:b
 dense_14/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_14/Tensordot/GatherV2GatherV2!dense_14/Tensordot/Shape:output:0 dense_14/Tensordot/free:output:0)dense_14/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_14/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
dense_14/Tensordot/GatherV2_1GatherV2!dense_14/Tensordot/Shape:output:0 dense_14/Tensordot/axes:output:0+dense_14/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_14/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_14/Tensordot/ProdProd$dense_14/Tensordot/GatherV2:output:0!dense_14/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_14/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_14/Tensordot/Prod_1Prod&dense_14/Tensordot/GatherV2_1:output:0#dense_14/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_14/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
dense_14/Tensordot/concatConcatV2 dense_14/Tensordot/free:output:0 dense_14/Tensordot/axes:output:0'dense_14/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_14/Tensordot/stackPack dense_14/Tensordot/Prod:output:0"dense_14/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_14/Tensordot/transpose	Transposernn_2/transpose_1:y:0"dense_14/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
dense_14/Tensordot/ReshapeReshape dense_14/Tensordot/transpose:y:0!dense_14/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
dense_14/Tensordot/MatMulMatMul#dense_14/Tensordot/Reshape:output:0)dense_14/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_14/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_14/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
dense_14/Tensordot/concat_1ConcatV2$dense_14/Tensordot/GatherV2:output:0#dense_14/Tensordot/Const_2:output:0)dense_14/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_14/TensordotReshape#dense_14/Tensordot/MatMul:product:0$dense_14/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_14/BiasAddBiasAdddense_14/Tensordot:output:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
IdentityIdentitydense_14/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ÷
NoOpNoOp ^dense_14/BiasAdd/ReadVariableOp"^dense_14/Tensordot/ReadVariableOp'^my_cnn_block_2/StatefulPartitionedCall5^rnn_2/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp4^rnn_2/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp5^rnn_2/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp4^rnn_2/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp5^rnn_2/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp4^rnn_2/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp5^rnn_2/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp4^rnn_2/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp^rnn_2/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2F
!dense_14/Tensordot/ReadVariableOp!dense_14/Tensordot/ReadVariableOp2P
&my_cnn_block_2/StatefulPartitionedCall&my_cnn_block_2/StatefulPartitionedCall2l
4rnn_2/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp4rnn_2/my_lstm_cell_2/dense_10/BiasAdd/ReadVariableOp2j
3rnn_2/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp3rnn_2/my_lstm_cell_2/dense_10/MatMul/ReadVariableOp2l
4rnn_2/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp4rnn_2/my_lstm_cell_2/dense_11/BiasAdd/ReadVariableOp2j
3rnn_2/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp3rnn_2/my_lstm_cell_2/dense_11/MatMul/ReadVariableOp2l
4rnn_2/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp4rnn_2/my_lstm_cell_2/dense_12/BiasAdd/ReadVariableOp2j
3rnn_2/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp3rnn_2/my_lstm_cell_2/dense_12/MatMul/ReadVariableOp2l
4rnn_2/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp4rnn_2/my_lstm_cell_2/dense_13/BiasAdd/ReadVariableOp2j
3rnn_2/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp3rnn_2/my_lstm_cell_2/dense_13/MatMul/ReadVariableOp2
rnn_2/whilernn_2/while:V R
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
Ý
à
.__inference_my_cnn_block_2_layer_call_fn_53326	
input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_my_cnn_block_2_layer_call_and_return_conditional_losses_51921{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
Í

Ç
while_cond_53724
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_53724___redundant_placeholder03
/while_while_cond_53724___redundant_placeholder13
/while_while_cond_53724___redundant_placeholder23
/while_while_cond_53724___redundant_placeholder33
/while_while_cond_53724___redundant_placeholder43
/while_while_cond_53724___redundant_placeholder53
/while_while_cond_53724___redundant_placeholder63
/while_while_cond_53724___redundant_placeholder73
/while_while_cond_53724___redundant_placeholder8
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
«h
Ù
__inference__traced_save_54566
file_prefix.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableopC
?savev2_rnn_2_my_lstm_cell_2_dense_10_kernel_read_readvariableopA
=savev2_rnn_2_my_lstm_cell_2_dense_10_bias_read_readvariableopC
?savev2_rnn_2_my_lstm_cell_2_dense_11_kernel_read_readvariableopA
=savev2_rnn_2_my_lstm_cell_2_dense_11_bias_read_readvariableopC
?savev2_rnn_2_my_lstm_cell_2_dense_12_kernel_read_readvariableopA
=savev2_rnn_2_my_lstm_cell_2_dense_12_bias_read_readvariableopC
?savev2_rnn_2_my_lstm_cell_2_dense_13_kernel_read_readvariableopA
=savev2_rnn_2_my_lstm_cell_2_dense_13_bias_read_readvariableop.
*savev2_dense_14_kernel_read_readvariableop,
(savev2_dense_14_bias_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop5
1savev2_adam_conv2d_4_kernel_m_read_readvariableop3
/savev2_adam_conv2d_4_bias_m_read_readvariableop5
1savev2_adam_conv2d_5_kernel_m_read_readvariableop3
/savev2_adam_conv2d_5_bias_m_read_readvariableopJ
Fsavev2_adam_rnn_2_my_lstm_cell_2_dense_10_kernel_m_read_readvariableopH
Dsavev2_adam_rnn_2_my_lstm_cell_2_dense_10_bias_m_read_readvariableopJ
Fsavev2_adam_rnn_2_my_lstm_cell_2_dense_11_kernel_m_read_readvariableopH
Dsavev2_adam_rnn_2_my_lstm_cell_2_dense_11_bias_m_read_readvariableopJ
Fsavev2_adam_rnn_2_my_lstm_cell_2_dense_12_kernel_m_read_readvariableopH
Dsavev2_adam_rnn_2_my_lstm_cell_2_dense_12_bias_m_read_readvariableopJ
Fsavev2_adam_rnn_2_my_lstm_cell_2_dense_13_kernel_m_read_readvariableopH
Dsavev2_adam_rnn_2_my_lstm_cell_2_dense_13_bias_m_read_readvariableop5
1savev2_adam_dense_14_kernel_m_read_readvariableop3
/savev2_adam_dense_14_bias_m_read_readvariableop5
1savev2_adam_conv2d_4_kernel_v_read_readvariableop3
/savev2_adam_conv2d_4_bias_v_read_readvariableop5
1savev2_adam_conv2d_5_kernel_v_read_readvariableop3
/savev2_adam_conv2d_5_bias_v_read_readvariableopJ
Fsavev2_adam_rnn_2_my_lstm_cell_2_dense_10_kernel_v_read_readvariableopH
Dsavev2_adam_rnn_2_my_lstm_cell_2_dense_10_bias_v_read_readvariableopJ
Fsavev2_adam_rnn_2_my_lstm_cell_2_dense_11_kernel_v_read_readvariableopH
Dsavev2_adam_rnn_2_my_lstm_cell_2_dense_11_bias_v_read_readvariableopJ
Fsavev2_adam_rnn_2_my_lstm_cell_2_dense_12_kernel_v_read_readvariableopH
Dsavev2_adam_rnn_2_my_lstm_cell_2_dense_12_bias_v_read_readvariableopJ
Fsavev2_adam_rnn_2_my_lstm_cell_2_dense_13_kernel_v_read_readvariableopH
Dsavev2_adam_rnn_2_my_lstm_cell_2_dense_13_bias_v_read_readvariableop5
1savev2_adam_dense_14_kernel_v_read_readvariableop3
/savev2_adam_dense_14_bias_v_read_readvariableop
savev2_const

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
: ¿
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*è
valueÞBÛ4B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÕ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop?savev2_rnn_2_my_lstm_cell_2_dense_10_kernel_read_readvariableop=savev2_rnn_2_my_lstm_cell_2_dense_10_bias_read_readvariableop?savev2_rnn_2_my_lstm_cell_2_dense_11_kernel_read_readvariableop=savev2_rnn_2_my_lstm_cell_2_dense_11_bias_read_readvariableop?savev2_rnn_2_my_lstm_cell_2_dense_12_kernel_read_readvariableop=savev2_rnn_2_my_lstm_cell_2_dense_12_bias_read_readvariableop?savev2_rnn_2_my_lstm_cell_2_dense_13_kernel_read_readvariableop=savev2_rnn_2_my_lstm_cell_2_dense_13_bias_read_readvariableop*savev2_dense_14_kernel_read_readvariableop(savev2_dense_14_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop1savev2_adam_conv2d_4_kernel_m_read_readvariableop/savev2_adam_conv2d_4_bias_m_read_readvariableop1savev2_adam_conv2d_5_kernel_m_read_readvariableop/savev2_adam_conv2d_5_bias_m_read_readvariableopFsavev2_adam_rnn_2_my_lstm_cell_2_dense_10_kernel_m_read_readvariableopDsavev2_adam_rnn_2_my_lstm_cell_2_dense_10_bias_m_read_readvariableopFsavev2_adam_rnn_2_my_lstm_cell_2_dense_11_kernel_m_read_readvariableopDsavev2_adam_rnn_2_my_lstm_cell_2_dense_11_bias_m_read_readvariableopFsavev2_adam_rnn_2_my_lstm_cell_2_dense_12_kernel_m_read_readvariableopDsavev2_adam_rnn_2_my_lstm_cell_2_dense_12_bias_m_read_readvariableopFsavev2_adam_rnn_2_my_lstm_cell_2_dense_13_kernel_m_read_readvariableopDsavev2_adam_rnn_2_my_lstm_cell_2_dense_13_bias_m_read_readvariableop1savev2_adam_dense_14_kernel_m_read_readvariableop/savev2_adam_dense_14_bias_m_read_readvariableop1savev2_adam_conv2d_4_kernel_v_read_readvariableop/savev2_adam_conv2d_4_bias_v_read_readvariableop1savev2_adam_conv2d_5_kernel_v_read_readvariableop/savev2_adam_conv2d_5_bias_v_read_readvariableopFsavev2_adam_rnn_2_my_lstm_cell_2_dense_10_kernel_v_read_readvariableopDsavev2_adam_rnn_2_my_lstm_cell_2_dense_10_bias_v_read_readvariableopFsavev2_adam_rnn_2_my_lstm_cell_2_dense_11_kernel_v_read_readvariableopDsavev2_adam_rnn_2_my_lstm_cell_2_dense_11_bias_v_read_readvariableopFsavev2_adam_rnn_2_my_lstm_cell_2_dense_12_kernel_v_read_readvariableopDsavev2_adam_rnn_2_my_lstm_cell_2_dense_12_bias_v_read_readvariableopFsavev2_adam_rnn_2_my_lstm_cell_2_dense_13_kernel_v_read_readvariableopDsavev2_adam_rnn_2_my_lstm_cell_2_dense_13_bias_v_read_readvariableop1savev2_adam_dense_14_kernel_v_read_readvariableop/savev2_adam_dense_14_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *B
dtypes8
624	
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

identity_1Identity_1:output:0*«
_input_shapes
: :::::'::'::'::':::: : : : : : : : : :::::'::'::'::'::::::::'::'::'::':::: 2(
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
::$ 

_output_shapes

:': 

_output_shapes
::$ 

_output_shapes

:': 

_output_shapes
::$	 

_output_shapes

:': 


_output_shapes
::$ 

_output_shapes

:': 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :
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
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:': 

_output_shapes
::$ 

_output_shapes

:': 

_output_shapes
::$  

_output_shapes

:': !

_output_shapes
::$" 

_output_shapes

:': #

_output_shapes
::$$ 

_output_shapes

:: %

_output_shapes
::,&(
&
_output_shapes
:: '

_output_shapes
::,((
&
_output_shapes
:: )

_output_shapes
::$* 

_output_shapes

:': +

_output_shapes
::$, 

_output_shapes

:': -

_output_shapes
::$. 

_output_shapes

:': /

_output_shapes
::$0 

_output_shapes

:': 1

_output_shapes
::$2 

_output_shapes

:: 3

_output_shapes
::4

_output_shapes
: "µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*»
serving_default§
G
input_1<
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ@
output_14
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:øÒ
è
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
conv_block1
	global_pooling

timedistributed
	lstm_cell

rnn_buffer
output_layer
metrics_list
	optimizer
call

signatures"
_tf_keras_model
¦
0
1
2
3
4
5
6
7
8
9
10
11
12
13
 14
!15
"16
#17"
trackable_list_wrapper

0
1
2
3
4
5
6
7
8
9
10
11
12
13"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
$non_trainable_variables

%layers
metrics
&layer_regularization_losses
'layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
à
(trace_0
)trace_1
*trace_2
+trace_32õ
/__inference_my_lstm_model_2_layer_call_fn_51651
/__inference_my_lstm_model_2_layer_call_fn_52713
/__inference_my_lstm_model_2_layer_call_fn_52746
/__inference_my_lstm_model_2_layer_call_fn_52069®
¥²¡
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z(trace_0z)trace_1z*trace_2z+trace_3
Ì
,trace_0
-trace_1
.trace_2
/trace_32á
J__inference_my_lstm_model_2_layer_call_and_return_conditional_losses_52963
J__inference_my_lstm_model_2_layer_call_and_return_conditional_losses_53180
J__inference_my_lstm_model_2_layer_call_and_return_conditional_losses_52107
J__inference_my_lstm_model_2_layer_call_and_return_conditional_losses_52145®
¥²¡
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z,trace_0z-trace_1z.trace_2z/trace_3
ËBÈ
 __inference__wrapped_model_50887input_1"
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
À
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6conv_layers
7call"
_tf_keras_layer
¥
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses"
_tf_keras_layer
°
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses
		layer"
_tf_keras_layer
¹
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses
JinputConcat
Ksigmoid1_layer
L
multiplier
Msigmoid2_layer
N
tanh_layer
	Oadder
Psigmoid3_layer
Qtanh
Rlayer_norm_2"
_tf_keras_layer
Ã
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses
cell
Y
state_spec"
_tf_keras_rnn_layer
»
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
.
`0
a1"
trackable_list_wrapper
ë
biter

cbeta_1

dbeta_2
	edecay
flearning_ratemmmmmmmm m¡m¢m£m¤m¥m¦v§v¨v©vªv«v¬v­v®v¯v°v±v²v³v´"
	optimizer

gtrace_0
htrace_12á
__inference_call_52428
__inference_call_52645®
¥²¡
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zgtrace_0zhtrace_1
,
iserving_default"
signature_map
):'2conv2d_4/kernel
:2conv2d_4/bias
):'2conv2d_5/kernel
:2conv2d_5/bias
6:4'2$rnn_2/my_lstm_cell_2/dense_10/kernel
0:.2"rnn_2/my_lstm_cell_2/dense_10/bias
6:4'2$rnn_2/my_lstm_cell_2/dense_11/kernel
0:.2"rnn_2/my_lstm_cell_2/dense_11/bias
6:4'2$rnn_2/my_lstm_cell_2/dense_12/kernel
0:.2"rnn_2/my_lstm_cell_2/dense_12/bias
6:4'2$rnn_2/my_lstm_cell_2/dense_13/kernel
0:.2"rnn_2/my_lstm_cell_2/dense_13/bias
!:2dense_14/kernel
:2dense_14/bias
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
J
0
	1

2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
8
`loss
aaccuracy"
trackable_dict_wrapper
ðBí
/__inference_my_lstm_model_2_layer_call_fn_51651input_1"®
¥²¡
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
êBç
/__inference_my_lstm_model_2_layer_call_fn_52713x"®
¥²¡
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
êBç
/__inference_my_lstm_model_2_layer_call_fn_52746x"®
¥²¡
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ðBí
/__inference_my_lstm_model_2_layer_call_fn_52069input_1"®
¥²¡
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
J__inference_my_lstm_model_2_layer_call_and_return_conditional_losses_52963x"®
¥²¡
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
J__inference_my_lstm_model_2_layer_call_and_return_conditional_losses_53180x"®
¥²¡
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
J__inference_my_lstm_model_2_layer_call_and_return_conditional_losses_52107input_1"®
¥²¡
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
J__inference_my_lstm_model_2_layer_call_and_return_conditional_losses_52145input_1"®
¥²¡
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
­
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
Ì
otrace_0
ptrace_12
.__inference_my_cnn_block_2_layer_call_fn_53313
.__inference_my_cnn_block_2_layer_call_fn_53326²
©²¥
FullArgSpec(
args 
jself
jinput

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zotrace_0zptrace_1

qtrace_0
rtrace_12Ë
I__inference_my_cnn_block_2_layer_call_and_return_conditional_losses_53386
I__inference_my_cnn_block_2_layer_call_and_return_conditional_losses_53446²
©²¥
FullArgSpec(
args 
jself
jinput

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zqtrace_0zrtrace_1
.
s0
t1"
trackable_list_wrapper

utrace_0
vtrace_12å
__inference_call_53240
__inference_call_53300²
©²¥
FullArgSpec(
args 
jself
jinput

jtraining
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zutrace_0zvtrace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
þ
|trace_02á
:__inference_global_average_pooling2d_2_layer_call_fn_53451¢
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
 z|trace_0

}trace_02ü
U__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_53457¢
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
 z}trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
å
trace_0
trace_12ª
2__inference_time_distributed_2_layer_call_fn_53462
2__inference_time_distributed_2_layer_call_fn_53467¿
¶²²
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

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1

trace_0
trace_12à
M__inference_time_distributed_2_layer_call_and_return_conditional_losses_53484
M__inference_time_distributed_2_layer_call_and_return_conditional_losses_53501¿
¶²²
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

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
þ
trace_02ß
.__inference_my_lstm_cell_2_layer_call_fn_53528¬
£²
FullArgSpec'
args
jself
jinputs
jstates
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
 ztrace_0

trace_02ú
I__inference_my_lstm_cell_2_layer_call_and_return_conditional_losses_53571¬
£²
FullArgSpec'
args
jself
jinputs
jstates
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
 ztrace_0
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Á
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Á
 	variables
¡trainable_variables
¢regularization_losses
£	keras_api
¤__call__
+¥&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
Á
¦	variables
§trainable_variables
¨regularization_losses
©	keras_api
ª__call__
+«&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
«
¬	variables
­trainable_variables
®regularization_losses
¯	keras_api
°__call__
+±&call_and_return_all_conditional_losses"
_tf_keras_layer
Á
²	variables
³trainable_variables
´regularization_losses
µ	keras_api
¶__call__
+·&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
«
¸	variables
¹trainable_variables
ºregularization_losses
»	keras_api
¼__call__
+½&call_and_return_all_conditional_losses"
_tf_keras_layer
)
¾	keras_api"
_tf_keras_layer
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
¿
¿states
Ànon_trainable_variables
Álayers
Âmetrics
 Ãlayer_regularization_losses
Älayer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
÷
Åtrace_0
Ætrace_1
Çtrace_2
Ètrace_32
%__inference_rnn_2_layer_call_fn_53592
%__inference_rnn_2_layer_call_fn_53613
%__inference_rnn_2_layer_call_fn_53634
%__inference_rnn_2_layer_call_fn_53655å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÅtrace_0zÆtrace_1zÇtrace_2zÈtrace_3
ã
Étrace_0
Êtrace_1
Ëtrace_2
Ìtrace_32ð
@__inference_rnn_2_layer_call_and_return_conditional_losses_53829
@__inference_rnn_2_layer_call_and_return_conditional_losses_54003
@__inference_rnn_2_layer_call_and_return_conditional_losses_54177
@__inference_rnn_2_layer_call_and_return_conditional_losses_54351å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÉtrace_0zÊtrace_1zËtrace_2zÌtrace_3
 "
trackable_list_wrapper
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
²
Ínon_trainable_variables
Îlayers
Ïmetrics
 Ðlayer_regularization_losses
Ñlayer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
î
Òtrace_02Ï
(__inference_dense_14_layer_call_fn_54360¢
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
 zÒtrace_0

Ótrace_02ê
C__inference_dense_14_layer_call_and_return_conditional_losses_54390¢
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
 zÓtrace_0
P
Ô	variables
Õ	keras_api
	 total
	!count"
_tf_keras_metric
a
Ö	variables
×	keras_api
	"total
	#count
Ø
_fn_kwargs"
_tf_keras_metric
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ÑBÎ
__inference_call_52428x"®
¥²¡
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÑBÎ
__inference_call_52645x"®
¥²¡
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÊBÇ
#__inference_signature_wrapper_52680input_1"
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
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ñBî
.__inference_my_cnn_block_2_layer_call_fn_53313input"²
©²¥
FullArgSpec(
args 
jself
jinput

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñBî
.__inference_my_cnn_block_2_layer_call_fn_53326input"²
©²¥
FullArgSpec(
args 
jself
jinput

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
I__inference_my_cnn_block_2_layer_call_and_return_conditional_losses_53386input"²
©²¥
FullArgSpec(
args 
jself
jinput

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
I__inference_my_cnn_block_2_layer_call_and_return_conditional_losses_53446input"²
©²¥
FullArgSpec(
args 
jself
jinput

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ä
Ù	variables
Útrainable_variables
Ûregularization_losses
Ü	keras_api
Ý__call__
+Þ&call_and_return_all_conditional_losses

kernel
bias
!ß_jit_compiled_convolution_op"
_tf_keras_layer
ä
à	variables
átrainable_variables
âregularization_losses
ã	keras_api
ä__call__
+å&call_and_return_all_conditional_losses

kernel
bias
!æ_jit_compiled_convolution_op"
_tf_keras_layer
ÙBÖ
__inference_call_53240input"²
©²¥
FullArgSpec(
args 
jself
jinput

jtraining
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÙBÖ
__inference_call_53300input"²
©²¥
FullArgSpec(
args 
jself
jinput

jtraining
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
îBë
:__inference_global_average_pooling2d_2_layer_call_fn_53451inputs"¢
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
B
U__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_53457inputs"¢
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
trackable_list_wrapper
'
	0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
2__inference_time_distributed_2_layer_call_fn_53462inputs"¿
¶²²
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

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
2__inference_time_distributed_2_layer_call_fn_53467inputs"¿
¶²²
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

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
M__inference_time_distributed_2_layer_call_and_return_conditional_losses_53484inputs"¿
¶²²
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

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
M__inference_time_distributed_2_layer_call_and_return_conditional_losses_53501inputs"¿
¶²²
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

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
_
J0
K1
L2
M3
N4
O5
P6
Q7
R8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Bý
.__inference_my_lstm_cell_2_layer_call_fn_53528inputsstates/0states/1"¬
£²
FullArgSpec'
args
jself
jinputs
jstates
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
B
I__inference_my_lstm_cell_2_layer_call_and_return_conditional_losses_53571inputsstates/0states/1"¬
£²
FullArgSpec'
args
jself
jinputs
jstates
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
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
çnon_trainable_variables
èlayers
émetrics
 êlayer_regularization_losses
ëlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
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
¨2¥¢
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
¸
ìnon_trainable_variables
ílayers
îmetrics
 ïlayer_regularization_losses
ðlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
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
¨2¥¢
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
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ñnon_trainable_variables
òlayers
ómetrics
 ôlayer_regularization_losses
õlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
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
¨2¥¢
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
¸
önon_trainable_variables
÷layers
ømetrics
 ùlayer_regularization_losses
úlayer_metrics
 	variables
¡trainable_variables
¢regularization_losses
¤__call__
+¥&call_and_return_all_conditional_losses
'¥"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
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
¨2¥¢
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
¸
ûnon_trainable_variables
ülayers
ýmetrics
 þlayer_regularization_losses
ÿlayer_metrics
¦	variables
§trainable_variables
¨regularization_losses
ª__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
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
¨2¥¢
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
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¬	variables
­trainable_variables
®regularization_losses
°__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
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
¨2¥¢
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
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
²	variables
³trainable_variables
´regularization_losses
¶__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
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
¨2¥¢
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
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¸	variables
¹trainable_variables
ºregularization_losses
¼__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
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
¨2¥¢
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
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
%__inference_rnn_2_layer_call_fn_53592inputs/0"å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
%__inference_rnn_2_layer_call_fn_53613inputs/0"å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
%__inference_rnn_2_layer_call_fn_53634inputs"å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
%__inference_rnn_2_layer_call_fn_53655inputs"å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¹B¶
@__inference_rnn_2_layer_call_and_return_conditional_losses_53829inputs/0"å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¹B¶
@__inference_rnn_2_layer_call_and_return_conditional_losses_54003inputs/0"å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
·B´
@__inference_rnn_2_layer_call_and_return_conditional_losses_54177inputs"å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
·B´
@__inference_rnn_2_layer_call_and_return_conditional_losses_54351inputs"å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ÜBÙ
(__inference_dense_14_layer_call_fn_54360inputs"¢
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
÷Bô
C__inference_dense_14_layer_call_and_return_conditional_losses_54390inputs"¢
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
.
 0
!1"
trackable_list_wrapper
.
Ô	variables"
_generic_user_object
.
"0
#1"
trackable_list_wrapper
.
Ö	variables"
_generic_user_object
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
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ù	variables
Útrainable_variables
Ûregularization_losses
Ý__call__
+Þ&call_and_return_all_conditional_losses
'Þ"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
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
¨2¥¢
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
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
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
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
à	variables
átrainable_variables
âregularization_losses
ä__call__
+å&call_and_return_all_conditional_losses
'å"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
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
¨2¥¢
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
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
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
 0
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
.:,2Adam/conv2d_4/kernel/m
 :2Adam/conv2d_4/bias/m
.:,2Adam/conv2d_5/kernel/m
 :2Adam/conv2d_5/bias/m
;:9'2+Adam/rnn_2/my_lstm_cell_2/dense_10/kernel/m
5:32)Adam/rnn_2/my_lstm_cell_2/dense_10/bias/m
;:9'2+Adam/rnn_2/my_lstm_cell_2/dense_11/kernel/m
5:32)Adam/rnn_2/my_lstm_cell_2/dense_11/bias/m
;:9'2+Adam/rnn_2/my_lstm_cell_2/dense_12/kernel/m
5:32)Adam/rnn_2/my_lstm_cell_2/dense_12/bias/m
;:9'2+Adam/rnn_2/my_lstm_cell_2/dense_13/kernel/m
5:32)Adam/rnn_2/my_lstm_cell_2/dense_13/bias/m
&:$2Adam/dense_14/kernel/m
 :2Adam/dense_14/bias/m
.:,2Adam/conv2d_4/kernel/v
 :2Adam/conv2d_4/bias/v
.:,2Adam/conv2d_5/kernel/v
 :2Adam/conv2d_5/bias/v
;:9'2+Adam/rnn_2/my_lstm_cell_2/dense_10/kernel/v
5:32)Adam/rnn_2/my_lstm_cell_2/dense_10/bias/v
;:9'2+Adam/rnn_2/my_lstm_cell_2/dense_11/kernel/v
5:32)Adam/rnn_2/my_lstm_cell_2/dense_11/bias/v
;:9'2+Adam/rnn_2/my_lstm_cell_2/dense_12/kernel/v
5:32)Adam/rnn_2/my_lstm_cell_2/dense_12/bias/v
;:9'2+Adam/rnn_2/my_lstm_cell_2/dense_13/kernel/v
5:32)Adam/rnn_2/my_lstm_cell_2/dense_13/bias/v
&:$2Adam/dense_14/kernel/v
 :2Adam/dense_14/bias/v¬
 __inference__wrapped_model_50887<¢9
2¢/
-*
input_1ÿÿÿÿÿÿÿÿÿ
ª "7ª4
2
output_1&#
output_1ÿÿÿÿÿÿÿÿÿ
__inference_call_52428j:¢7
0¢-
'$
xÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
__inference_call_52645j:¢7
0¢-
'$
xÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
__inference_call_53240l>¢;
4¢1
+(
inputÿÿÿÿÿÿÿÿÿ
p
ª "$!ÿÿÿÿÿÿÿÿÿ
__inference_call_53300l>¢;
4¢1
+(
inputÿÿÿÿÿÿÿÿÿ
p 
ª "$!ÿÿÿÿÿÿÿÿÿ«
C__inference_dense_14_layer_call_and_return_conditional_losses_54390d3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
(__inference_dense_14_layer_call_fn_54360W3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÞ
U__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_53457R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 µ
:__inference_global_average_pooling2d_2_layer_call_fn_53451wR¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÆ
I__inference_my_cnn_block_2_layer_call_and_return_conditional_losses_53386y>¢;
4¢1
+(
inputÿÿÿÿÿÿÿÿÿ
p 
ª "1¢.
'$
0ÿÿÿÿÿÿÿÿÿ
 Æ
I__inference_my_cnn_block_2_layer_call_and_return_conditional_losses_53446y>¢;
4¢1
+(
inputÿÿÿÿÿÿÿÿÿ
p
ª "1¢.
'$
0ÿÿÿÿÿÿÿÿÿ
 
.__inference_my_cnn_block_2_layer_call_fn_53313l>¢;
4¢1
+(
inputÿÿÿÿÿÿÿÿÿ
p 
ª "$!ÿÿÿÿÿÿÿÿÿ
.__inference_my_cnn_block_2_layer_call_fn_53326l>¢;
4¢1
+(
inputÿÿÿÿÿÿÿÿÿ
p
ª "$!ÿÿÿÿÿÿÿÿÿË
I__inference_my_lstm_cell_2_layer_call_and_return_conditional_losses_53571ý|¢y
r¢o
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ
"
states/1ÿÿÿÿÿÿÿÿÿ
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ
EB

0/1/0ÿÿÿÿÿÿÿÿÿ

0/1/1ÿÿÿÿÿÿÿÿÿ
  
.__inference_my_lstm_cell_2_layer_call_fn_53528í|¢y
r¢o
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ
"
states/1ÿÿÿÿÿÿÿÿÿ
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ
A>

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿË
J__inference_my_lstm_model_2_layer_call_and_return_conditional_losses_52107}@¢=
6¢3
-*
input_1ÿÿÿÿÿÿÿÿÿ
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 Ë
J__inference_my_lstm_model_2_layer_call_and_return_conditional_losses_52145}@¢=
6¢3
-*
input_1ÿÿÿÿÿÿÿÿÿ
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 Å
J__inference_my_lstm_model_2_layer_call_and_return_conditional_losses_52963w:¢7
0¢-
'$
xÿÿÿÿÿÿÿÿÿ
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 Å
J__inference_my_lstm_model_2_layer_call_and_return_conditional_losses_53180w:¢7
0¢-
'$
xÿÿÿÿÿÿÿÿÿ
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 £
/__inference_my_lstm_model_2_layer_call_fn_51651p@¢=
6¢3
-*
input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ£
/__inference_my_lstm_model_2_layer_call_fn_52069p@¢=
6¢3
-*
input_1ÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_my_lstm_model_2_layer_call_fn_52713j:¢7
0¢-
'$
xÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_my_lstm_model_2_layer_call_fn_52746j:¢7
0¢-
'$
xÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿØ
@__inference_rnn_2_layer_call_and_return_conditional_losses_53829S¢P
I¢F
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ø
@__inference_rnn_2_layer_call_and_return_conditional_losses_54003S¢P
I¢F
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¾
@__inference_rnn_2_layer_call_and_return_conditional_losses_54177zC¢@
9¢6
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 ¾
@__inference_rnn_2_layer_call_and_return_conditional_losses_54351zC¢@
9¢6
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 °
%__inference_rnn_2_layer_call_fn_53592S¢P
I¢F
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
%__inference_rnn_2_layer_call_fn_53613S¢P
I¢F
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
%__inference_rnn_2_layer_call_fn_53634mC¢@
9¢6
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 

 
ª "ÿÿÿÿÿÿÿÿÿ
%__inference_rnn_2_layer_call_fn_53655mC¢@
9¢6
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 

 
ª "ÿÿÿÿÿÿÿÿÿº
#__inference_signature_wrapper_52680G¢D
¢ 
=ª:
8
input_1-*
input_1ÿÿÿÿÿÿÿÿÿ"7ª4
2
output_1&#
output_1ÿÿÿÿÿÿÿÿÿÔ
M__inference_time_distributed_2_layer_call_and_return_conditional_losses_53484L¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ô
M__inference_time_distributed_2_layer_call_and_return_conditional_losses_53501L¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 «
2__inference_time_distributed_2_layer_call_fn_53462uL¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ«
2__inference_time_distributed_2_layer_call_fn_53467uL¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ