เค$
ฝ
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
ฎ
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
ม
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
executor_typestring จ
@
StaticRegexFullMatch	
input

output
"
patternstring
๗
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
ฐ
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handle้่element_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handle้่element_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
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
"serve*2.10.12v2.10.0-76-gfdfc646704c8๗ฑ!
~
Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/v
w
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes
:*
dtype0

Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_4/kernel/v

)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v*
_output_shapes

:*
dtype0
?
$Adam/rnn/my_lstm_cell/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/rnn/my_lstm_cell/dense_3/bias/v

8Adam/rnn/my_lstm_cell/dense_3/bias/v/Read/ReadVariableOpReadVariableOp$Adam/rnn/my_lstm_cell/dense_3/bias/v*
_output_shapes
:*
dtype0
จ
&Adam/rnn/my_lstm_cell/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:'*7
shared_name(&Adam/rnn/my_lstm_cell/dense_3/kernel/v
ก
:Adam/rnn/my_lstm_cell/dense_3/kernel/v/Read/ReadVariableOpReadVariableOp&Adam/rnn/my_lstm_cell/dense_3/kernel/v*
_output_shapes

:'*
dtype0
?
$Adam/rnn/my_lstm_cell/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/rnn/my_lstm_cell/dense_2/bias/v

8Adam/rnn/my_lstm_cell/dense_2/bias/v/Read/ReadVariableOpReadVariableOp$Adam/rnn/my_lstm_cell/dense_2/bias/v*
_output_shapes
:*
dtype0
จ
&Adam/rnn/my_lstm_cell/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:'*7
shared_name(&Adam/rnn/my_lstm_cell/dense_2/kernel/v
ก
:Adam/rnn/my_lstm_cell/dense_2/kernel/v/Read/ReadVariableOpReadVariableOp&Adam/rnn/my_lstm_cell/dense_2/kernel/v*
_output_shapes

:'*
dtype0
?
$Adam/rnn/my_lstm_cell/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/rnn/my_lstm_cell/dense_1/bias/v

8Adam/rnn/my_lstm_cell/dense_1/bias/v/Read/ReadVariableOpReadVariableOp$Adam/rnn/my_lstm_cell/dense_1/bias/v*
_output_shapes
:*
dtype0
จ
&Adam/rnn/my_lstm_cell/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:'*7
shared_name(&Adam/rnn/my_lstm_cell/dense_1/kernel/v
ก
:Adam/rnn/my_lstm_cell/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp&Adam/rnn/my_lstm_cell/dense_1/kernel/v*
_output_shapes

:'*
dtype0

"Adam/rnn/my_lstm_cell/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/rnn/my_lstm_cell/dense/bias/v

6Adam/rnn/my_lstm_cell/dense/bias/v/Read/ReadVariableOpReadVariableOp"Adam/rnn/my_lstm_cell/dense/bias/v*
_output_shapes
:*
dtype0
ค
$Adam/rnn/my_lstm_cell/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:'*5
shared_name&$Adam/rnn/my_lstm_cell/dense/kernel/v

8Adam/rnn/my_lstm_cell/dense/kernel/v/Read/ReadVariableOpReadVariableOp$Adam/rnn/my_lstm_cell/dense/kernel/v*
_output_shapes

:'*
dtype0

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

Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_1/kernel/v

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

Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/v

(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:*
dtype0
~
Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/m
w
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes
:*
dtype0

Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_4/kernel/m

)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m*
_output_shapes

:*
dtype0
?
$Adam/rnn/my_lstm_cell/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/rnn/my_lstm_cell/dense_3/bias/m

8Adam/rnn/my_lstm_cell/dense_3/bias/m/Read/ReadVariableOpReadVariableOp$Adam/rnn/my_lstm_cell/dense_3/bias/m*
_output_shapes
:*
dtype0
จ
&Adam/rnn/my_lstm_cell/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:'*7
shared_name(&Adam/rnn/my_lstm_cell/dense_3/kernel/m
ก
:Adam/rnn/my_lstm_cell/dense_3/kernel/m/Read/ReadVariableOpReadVariableOp&Adam/rnn/my_lstm_cell/dense_3/kernel/m*
_output_shapes

:'*
dtype0
?
$Adam/rnn/my_lstm_cell/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/rnn/my_lstm_cell/dense_2/bias/m

8Adam/rnn/my_lstm_cell/dense_2/bias/m/Read/ReadVariableOpReadVariableOp$Adam/rnn/my_lstm_cell/dense_2/bias/m*
_output_shapes
:*
dtype0
จ
&Adam/rnn/my_lstm_cell/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:'*7
shared_name(&Adam/rnn/my_lstm_cell/dense_2/kernel/m
ก
:Adam/rnn/my_lstm_cell/dense_2/kernel/m/Read/ReadVariableOpReadVariableOp&Adam/rnn/my_lstm_cell/dense_2/kernel/m*
_output_shapes

:'*
dtype0
?
$Adam/rnn/my_lstm_cell/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/rnn/my_lstm_cell/dense_1/bias/m

8Adam/rnn/my_lstm_cell/dense_1/bias/m/Read/ReadVariableOpReadVariableOp$Adam/rnn/my_lstm_cell/dense_1/bias/m*
_output_shapes
:*
dtype0
จ
&Adam/rnn/my_lstm_cell/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:'*7
shared_name(&Adam/rnn/my_lstm_cell/dense_1/kernel/m
ก
:Adam/rnn/my_lstm_cell/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp&Adam/rnn/my_lstm_cell/dense_1/kernel/m*
_output_shapes

:'*
dtype0

"Adam/rnn/my_lstm_cell/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/rnn/my_lstm_cell/dense/bias/m

6Adam/rnn/my_lstm_cell/dense/bias/m/Read/ReadVariableOpReadVariableOp"Adam/rnn/my_lstm_cell/dense/bias/m*
_output_shapes
:*
dtype0
ค
$Adam/rnn/my_lstm_cell/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:'*5
shared_name&$Adam/rnn/my_lstm_cell/dense/kernel/m

8Adam/rnn/my_lstm_cell/dense/kernel/m/Read/ReadVariableOpReadVariableOp$Adam/rnn/my_lstm_cell/dense/kernel/m*
_output_shapes

:'*
dtype0

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

Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_1/kernel/m

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

Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/m

(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
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
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:*
dtype0
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:*
dtype0

rnn/my_lstm_cell/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namernn/my_lstm_cell/dense_3/bias

1rnn/my_lstm_cell/dense_3/bias/Read/ReadVariableOpReadVariableOprnn/my_lstm_cell/dense_3/bias*
_output_shapes
:*
dtype0

rnn/my_lstm_cell/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:'*0
shared_name!rnn/my_lstm_cell/dense_3/kernel

3rnn/my_lstm_cell/dense_3/kernel/Read/ReadVariableOpReadVariableOprnn/my_lstm_cell/dense_3/kernel*
_output_shapes

:'*
dtype0

rnn/my_lstm_cell/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namernn/my_lstm_cell/dense_2/bias

1rnn/my_lstm_cell/dense_2/bias/Read/ReadVariableOpReadVariableOprnn/my_lstm_cell/dense_2/bias*
_output_shapes
:*
dtype0

rnn/my_lstm_cell/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:'*0
shared_name!rnn/my_lstm_cell/dense_2/kernel

3rnn/my_lstm_cell/dense_2/kernel/Read/ReadVariableOpReadVariableOprnn/my_lstm_cell/dense_2/kernel*
_output_shapes

:'*
dtype0

rnn/my_lstm_cell/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namernn/my_lstm_cell/dense_1/bias

1rnn/my_lstm_cell/dense_1/bias/Read/ReadVariableOpReadVariableOprnn/my_lstm_cell/dense_1/bias*
_output_shapes
:*
dtype0

rnn/my_lstm_cell/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:'*0
shared_name!rnn/my_lstm_cell/dense_1/kernel

3rnn/my_lstm_cell/dense_1/kernel/Read/ReadVariableOpReadVariableOprnn/my_lstm_cell/dense_1/kernel*
_output_shapes

:'*
dtype0

rnn/my_lstm_cell/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namernn/my_lstm_cell/dense/bias

/rnn/my_lstm_cell/dense/bias/Read/ReadVariableOpReadVariableOprnn/my_lstm_cell/dense/bias*
_output_shapes
:*
dtype0

rnn/my_lstm_cell/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:'*.
shared_namernn/my_lstm_cell/dense/kernel

1rnn/my_lstm_cell/dense/kernel/Read/ReadVariableOpReadVariableOprnn/my_lstm_cell/dense/kernel*
_output_shapes

:'*
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

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

serving_default_input_1Placeholder*3
_output_shapes!
:?????????*
dtype0*(
shape:?????????
ง
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasrnn/my_lstm_cell/dense/kernelrnn/my_lstm_cell/dense/biasrnn/my_lstm_cell/dense_1/kernelrnn/my_lstm_cell/dense_1/biasrnn/my_lstm_cell/dense_2/kernelrnn/my_lstm_cell/dense_2/biasrnn/my_lstm_cell/dense_3/kernelrnn/my_lstm_cell/dense_3/biasdense_4/kerneldense_4/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_17504

NoOpNoOp
๚
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ด
valueฉBฅ B
ำ
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
ฐ
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
ซ
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
ค
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
ช
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses
cell
Y
state_spec*
ฆ
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
?
biter

cbeta_1

dbeta_2
	edecay
flearning_ratemmmmmmmm?mกmขmฃmคmฅmฆvงvจvฉvชvซvฌvญvฎvฏvฐvฑvฒvณvด*

gtrace_0
htrace_1* 

iserving_default* 
MG
VARIABLE_VALUEconv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEconv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUErnn/my_lstm_cell/dense/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUErnn/my_lstm_cell/dense/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUErnn/my_lstm_cell/dense_1/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUErnn/my_lstm_cell/dense_1/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUErnn/my_lstm_cell/dense_2/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUErnn/my_lstm_cell/dense_2/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUErnn/my_lstm_cell/dense_3/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUErnn/my_lstm_cell/dense_3/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_4/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_4/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
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
ฌ
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
ฌ
?	variables
กtrainable_variables
ขregularization_losses
ฃ	keras_api
ค__call__
+ฅ&call_and_return_all_conditional_losses

kernel
bias*
ฌ
ฆ	variables
งtrainable_variables
จregularization_losses
ฉ	keras_api
ช__call__
+ซ&call_and_return_all_conditional_losses

kernel
bias*

ฌ	variables
ญtrainable_variables
ฎregularization_losses
ฏ	keras_api
ฐ__call__
+ฑ&call_and_return_all_conditional_losses* 
ฌ
ฒ	variables
ณtrainable_variables
ดregularization_losses
ต	keras_api
ถ__call__
+ท&call_and_return_all_conditional_losses

kernel
bias*

ธ	variables
นtrainable_variables
บregularization_losses
ป	keras_api
ผ__call__
+ฝ&call_and_return_all_conditional_losses* 

พ	keras_api* 
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
ฅ
ฟstates
ภnon_trainable_variables
มlayers
ยmetrics
 รlayer_regularization_losses
ฤlayer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*
:
ลtrace_0
ฦtrace_1
วtrace_2
ศtrace_3* 
:
ษtrace_0
สtrace_1
หtrace_2
ฬtrace_3* 
* 

0
1*

0
1*
* 

อnon_trainable_variables
ฮlayers
ฯmetrics
 ะlayer_regularization_losses
ัlayer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses*

าtrace_0* 

ำtrace_0* 
:
ิ	variables
ี	keras_api
	 total
	!count*
K
ึ	variables
ื	keras_api
	"total
	#count
ุ
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
ฯ
ู	variables
ฺtrainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias
!฿_jit_compiled_convolution_op*
ฯ
เ	variables
แtrainable_variables
โregularization_losses
ใ	keras_api
ไ__call__
+ๅ&call_and_return_all_conditional_losses

kernel
bias
!ๆ_jit_compiled_convolution_op*
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
็non_trainable_variables
่layers
้metrics
 ๊layer_regularization_losses
๋layer_metrics
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
์non_trainable_variables
ํlayers
๎metrics
 ๏layer_regularization_losses
๐layer_metrics
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
๑non_trainable_variables
๒layers
๓metrics
 ๔layer_regularization_losses
๕layer_metrics
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
๖non_trainable_variables
๗layers
๘metrics
 ๙layer_regularization_losses
๚layer_metrics
?	variables
กtrainable_variables
ขregularization_losses
ค__call__
+ฅ&call_and_return_all_conditional_losses
'ฅ"call_and_return_conditional_losses*
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
๛non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
ฆ	variables
งtrainable_variables
จregularization_losses
ช__call__
+ซ&call_and_return_all_conditional_losses
'ซ"call_and_return_conditional_losses*
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
ฌ	variables
ญtrainable_variables
ฎregularization_losses
ฐ__call__
+ฑ&call_and_return_all_conditional_losses
'ฑ"call_and_return_conditional_losses* 
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
ฒ	variables
ณtrainable_variables
ดregularization_losses
ถ__call__
+ท&call_and_return_all_conditional_losses
'ท"call_and_return_conditional_losses*
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
ธ	variables
นtrainable_variables
บregularization_losses
ผ__call__
+ฝ&call_and_return_all_conditional_losses
'ฝ"call_and_return_conditional_losses* 
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
ิ	variables*

"0
#1*

ึ	variables*
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
ู	variables
ฺtrainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
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
เ	variables
แtrainable_variables
โregularization_losses
ไ__call__
+ๅ&call_and_return_all_conditional_losses
'ๅ"call_and_return_conditional_losses*
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
pj
VARIABLE_VALUEAdam/conv2d/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/conv2d/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE$Adam/rnn/my_lstm_cell/dense/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/rnn/my_lstm_cell/dense/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE&Adam/rnn/my_lstm_cell/dense_1/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE$Adam/rnn/my_lstm_cell/dense_1/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE&Adam/rnn/my_lstm_cell/dense_2/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE$Adam/rnn/my_lstm_cell/dense_2/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUE&Adam/rnn/my_lstm_cell/dense_3/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE$Adam/rnn/my_lstm_cell/dense_3/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_4/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_4/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/conv2d/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE$Adam/rnn/my_lstm_cell/dense/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/rnn/my_lstm_cell/dense/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE&Adam/rnn/my_lstm_cell/dense_1/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE$Adam/rnn/my_lstm_cell/dense_1/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE&Adam/rnn/my_lstm_cell/dense_2/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE$Adam/rnn/my_lstm_cell/dense_2/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUE&Adam/rnn/my_lstm_cell/dense_3/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE$Adam/rnn/my_lstm_cell/dense_3/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_4/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_4/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
๔
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp1rnn/my_lstm_cell/dense/kernel/Read/ReadVariableOp/rnn/my_lstm_cell/dense/bias/Read/ReadVariableOp3rnn/my_lstm_cell/dense_1/kernel/Read/ReadVariableOp1rnn/my_lstm_cell/dense_1/bias/Read/ReadVariableOp3rnn/my_lstm_cell/dense_2/kernel/Read/ReadVariableOp1rnn/my_lstm_cell/dense_2/bias/Read/ReadVariableOp3rnn/my_lstm_cell/dense_3/kernel/Read/ReadVariableOp1rnn/my_lstm_cell/dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp8Adam/rnn/my_lstm_cell/dense/kernel/m/Read/ReadVariableOp6Adam/rnn/my_lstm_cell/dense/bias/m/Read/ReadVariableOp:Adam/rnn/my_lstm_cell/dense_1/kernel/m/Read/ReadVariableOp8Adam/rnn/my_lstm_cell/dense_1/bias/m/Read/ReadVariableOp:Adam/rnn/my_lstm_cell/dense_2/kernel/m/Read/ReadVariableOp8Adam/rnn/my_lstm_cell/dense_2/bias/m/Read/ReadVariableOp:Adam/rnn/my_lstm_cell/dense_3/kernel/m/Read/ReadVariableOp8Adam/rnn/my_lstm_cell/dense_3/bias/m/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp8Adam/rnn/my_lstm_cell/dense/kernel/v/Read/ReadVariableOp6Adam/rnn/my_lstm_cell/dense/bias/v/Read/ReadVariableOp:Adam/rnn/my_lstm_cell/dense_1/kernel/v/Read/ReadVariableOp8Adam/rnn/my_lstm_cell/dense_1/bias/v/Read/ReadVariableOp:Adam/rnn/my_lstm_cell/dense_2/kernel/v/Read/ReadVariableOp8Adam/rnn/my_lstm_cell/dense_2/bias/v/Read/ReadVariableOp:Adam/rnn/my_lstm_cell/dense_3/kernel/v/Read/ReadVariableOp8Adam/rnn/my_lstm_cell/dense_3/bias/v/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOpConst*@
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
__inference__traced_save_19390
๓
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasrnn/my_lstm_cell/dense/kernelrnn/my_lstm_cell/dense/biasrnn/my_lstm_cell/dense_1/kernelrnn/my_lstm_cell/dense_1/biasrnn/my_lstm_cell/dense_2/kernelrnn/my_lstm_cell/dense_2/biasrnn/my_lstm_cell/dense_3/kernelrnn/my_lstm_cell/dense_3/biasdense_4/kerneldense_4/biastotal_1count_1totalcount	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateAdam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/m$Adam/rnn/my_lstm_cell/dense/kernel/m"Adam/rnn/my_lstm_cell/dense/bias/m&Adam/rnn/my_lstm_cell/dense_1/kernel/m$Adam/rnn/my_lstm_cell/dense_1/bias/m&Adam/rnn/my_lstm_cell/dense_2/kernel/m$Adam/rnn/my_lstm_cell/dense_2/bias/m&Adam/rnn/my_lstm_cell/dense_3/kernel/m$Adam/rnn/my_lstm_cell/dense_3/bias/mAdam/dense_4/kernel/mAdam/dense_4/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/v$Adam/rnn/my_lstm_cell/dense/kernel/v"Adam/rnn/my_lstm_cell/dense/bias/v&Adam/rnn/my_lstm_cell/dense_1/kernel/v$Adam/rnn/my_lstm_cell/dense_1/bias/v&Adam/rnn/my_lstm_cell/dense_2/kernel/v$Adam/rnn/my_lstm_cell/dense_2/bias/v&Adam/rnn/my_lstm_cell/dense_3/kernel/v$Adam/rnn/my_lstm_cell/dense_3/bias/vAdam/dense_4/kernel/vAdam/dense_4/bias/v*?
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
!__inference__traced_restore_19553?
ฑ)


while_body_16061
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_my_lstm_cell_16085_0:'(
while_my_lstm_cell_16087_0:,
while_my_lstm_cell_16089_0:'(
while_my_lstm_cell_16091_0:,
while_my_lstm_cell_16093_0:'(
while_my_lstm_cell_16095_0:,
while_my_lstm_cell_16097_0:'(
while_my_lstm_cell_16099_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_my_lstm_cell_16085:'&
while_my_lstm_cell_16087:*
while_my_lstm_cell_16089:'&
while_my_lstm_cell_16091:*
while_my_lstm_cell_16093:'&
while_my_lstm_cell_16095:*
while_my_lstm_cell_16097:'&
while_my_lstm_cell_16099:ข*while/my_lstm_cell/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ฆ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0ล
*while/my_lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_my_lstm_cell_16085_0while_my_lstm_cell_16087_0while_my_lstm_cell_16089_0while_my_lstm_cell_16091_0while_my_lstm_cell_16093_0while_my_lstm_cell_16095_0while_my_lstm_cell_16097_0while_my_lstm_cell_16099_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_my_lstm_cell_layer_call_and_return_conditional_losses_15846?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/my_lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:้่าM
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
: 
while/Identity_4Identity3while/my_lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????
while/Identity_5Identity3while/my_lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:?????????y

while/NoOpNoOp+^while/my_lstm_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_my_lstm_cell_16085while_my_lstm_cell_16085_0"6
while_my_lstm_cell_16087while_my_lstm_cell_16087_0"6
while_my_lstm_cell_16089while_my_lstm_cell_16089_0"6
while_my_lstm_cell_16091while_my_lstm_cell_16091_0"6
while_my_lstm_cell_16093while_my_lstm_cell_16093_0"6
while_my_lstm_cell_16095while_my_lstm_cell_16095_0"6
while_my_lstm_cell_16097while_my_lstm_cell_16097_0"6
while_my_lstm_cell_16099while_my_lstm_cell_16099_0"0
while_strided_slice_1while_strided_slice_1_0"จ
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :?????????:?????????: : : : : : : : : : 2X
*while/my_lstm_cell/StatefulPartitionedCall*while/my_lstm_cell/StatefulPartitionedCall: 
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
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
ด
o
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_18281

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
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
:
ฌ
>__inference_rnn_layer_call_and_return_conditional_losses_16389

inputs$
my_lstm_cell_16278:' 
my_lstm_cell_16280:$
my_lstm_cell_16282:' 
my_lstm_cell_16284:$
my_lstm_cell_16286:' 
my_lstm_cell_16288:$
my_lstm_cell_16290:' 
my_lstm_cell_16292:
identityข$my_lstm_cell/StatefulPartitionedCallขwhile;
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
valueB:ั
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
:?????????R
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
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????D
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
valueB:?
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
?????????ด
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่า
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   เ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่า_
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
valueB:้
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask฿
$my_lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0my_lstm_cell_16278my_lstm_cell_16280my_lstm_cell_16282my_lstm_cell_16284my_lstm_cell_16286my_lstm_cell_16288my_lstm_cell_16290my_lstm_cell_16292*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_my_lstm_cell_layer_call_and_return_conditional_losses_15846n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ธ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่าF
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
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ด
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0my_lstm_cell_16278my_lstm_cell_16280my_lstm_cell_16282my_lstm_cell_16284my_lstm_cell_16286my_lstm_cell_16288my_lstm_cell_16290my_lstm_cell_16292*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( *
bodyR
while_body_16301*
condR
while_cond_16300*U
output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ย
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
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
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:?????????u
NoOpNoOp%^my_lstm_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2L
$my_lstm_cell/StatefulPartitionedCall$my_lstm_cell/StatefulPartitionedCall2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
๒d
๕
>__inference_rnn_layer_call_and_return_conditional_losses_19175

inputsC
1my_lstm_cell_dense_matmul_readvariableop_resource:'@
2my_lstm_cell_dense_biasadd_readvariableop_resource:E
3my_lstm_cell_dense_1_matmul_readvariableop_resource:'B
4my_lstm_cell_dense_1_biasadd_readvariableop_resource:E
3my_lstm_cell_dense_2_matmul_readvariableop_resource:'B
4my_lstm_cell_dense_2_biasadd_readvariableop_resource:E
3my_lstm_cell_dense_3_matmul_readvariableop_resource:'B
4my_lstm_cell_dense_3_biasadd_readvariableop_resource:
identityข)my_lstm_cell/dense/BiasAdd/ReadVariableOpข(my_lstm_cell/dense/MatMul/ReadVariableOpข+my_lstm_cell/dense_1/BiasAdd/ReadVariableOpข*my_lstm_cell/dense_1/MatMul/ReadVariableOpข+my_lstm_cell/dense_2/BiasAdd/ReadVariableOpข*my_lstm_cell/dense_2/MatMul/ReadVariableOpข+my_lstm_cell/dense_3/BiasAdd/ReadVariableOpข*my_lstm_cell/dense_3/MatMul/ReadVariableOpขwhile;
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
valueB:ั
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
:?????????R
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
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????D
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
valueB:?
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
?????????ด
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่า
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   เ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่า_
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
valueB:้
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskh
&my_lstm_cell/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ร
!my_lstm_cell/concatenate_1/concatConcatV2strided_slice_2:output:0zeros:output:0/my_lstm_cell/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????'
(my_lstm_cell/dense/MatMul/ReadVariableOpReadVariableOp1my_lstm_cell_dense_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0ณ
my_lstm_cell/dense/MatMulMatMul*my_lstm_cell/concatenate_1/concat:output:00my_lstm_cell/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
)my_lstm_cell/dense/BiasAdd/ReadVariableOpReadVariableOp2my_lstm_cell_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ฏ
my_lstm_cell/dense/BiasAddBiasAdd#my_lstm_cell/dense/MatMul:product:01my_lstm_cell/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????|
my_lstm_cell/dense/SigmoidSigmoid#my_lstm_cell/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
my_lstm_cell/multiply/mulMulzeros_1:output:0my_lstm_cell/dense/Sigmoid:y:0*
T0*'
_output_shapes
:?????????
*my_lstm_cell/dense_1/MatMul/ReadVariableOpReadVariableOp3my_lstm_cell_dense_1_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0ท
my_lstm_cell/dense_1/MatMulMatMul*my_lstm_cell/concatenate_1/concat:output:02my_lstm_cell/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
+my_lstm_cell/dense_1/BiasAdd/ReadVariableOpReadVariableOp4my_lstm_cell_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ต
my_lstm_cell/dense_1/BiasAddBiasAdd%my_lstm_cell/dense_1/MatMul:product:03my_lstm_cell/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
my_lstm_cell/dense_1/SigmoidSigmoid%my_lstm_cell/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
*my_lstm_cell/dense_2/MatMul/ReadVariableOpReadVariableOp3my_lstm_cell_dense_2_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0ท
my_lstm_cell/dense_2/MatMulMatMul*my_lstm_cell/concatenate_1/concat:output:02my_lstm_cell/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
+my_lstm_cell/dense_2/BiasAdd/ReadVariableOpReadVariableOp4my_lstm_cell_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ต
my_lstm_cell/dense_2/BiasAddBiasAdd%my_lstm_cell/dense_2/MatMul:product:03my_lstm_cell/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????z
my_lstm_cell/dense_2/TanhTanh%my_lstm_cell/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
my_lstm_cell/multiply/mul_1Mul my_lstm_cell/dense_1/Sigmoid:y:0my_lstm_cell/dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????
my_lstm_cell/add_1/addAddV2my_lstm_cell/multiply/mul:z:0my_lstm_cell/multiply/mul_1:z:0*
T0*'
_output_shapes
:?????????
*my_lstm_cell/dense_3/MatMul/ReadVariableOpReadVariableOp3my_lstm_cell_dense_3_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0ท
my_lstm_cell/dense_3/MatMulMatMul*my_lstm_cell/concatenate_1/concat:output:02my_lstm_cell/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
+my_lstm_cell/dense_3/BiasAdd/ReadVariableOpReadVariableOp4my_lstm_cell_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ต
my_lstm_cell/dense_3/BiasAddBiasAdd%my_lstm_cell/dense_3/MatMul:product:03my_lstm_cell/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
my_lstm_cell/dense_3/SigmoidSigmoid%my_lstm_cell/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????r
my_lstm_cell/activation/TanhTanhmy_lstm_cell/add_1/add:z:0*
T0*'
_output_shapes
:?????????
my_lstm_cell/multiply/mul_2Mul my_lstm_cell/activation/Tanh:y:0 my_lstm_cell/dense_3/Sigmoid:y:0*
T0*'
_output_shapes
:?????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ธ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่าF
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
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ผ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01my_lstm_cell_dense_matmul_readvariableop_resource2my_lstm_cell_dense_biasadd_readvariableop_resource3my_lstm_cell_dense_1_matmul_readvariableop_resource4my_lstm_cell_dense_1_biasadd_readvariableop_resource3my_lstm_cell_dense_2_matmul_readvariableop_resource4my_lstm_cell_dense_2_biasadd_readvariableop_resource3my_lstm_cell_dense_3_matmul_readvariableop_resource4my_lstm_cell_dense_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( *
bodyR
while_body_19071*
condR
while_cond_19070*U
output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ย
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
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
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:?????????ถ
NoOpNoOp*^my_lstm_cell/dense/BiasAdd/ReadVariableOp)^my_lstm_cell/dense/MatMul/ReadVariableOp,^my_lstm_cell/dense_1/BiasAdd/ReadVariableOp+^my_lstm_cell/dense_1/MatMul/ReadVariableOp,^my_lstm_cell/dense_2/BiasAdd/ReadVariableOp+^my_lstm_cell/dense_2/MatMul/ReadVariableOp,^my_lstm_cell/dense_3/BiasAdd/ReadVariableOp+^my_lstm_cell/dense_3/MatMul/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2V
)my_lstm_cell/dense/BiasAdd/ReadVariableOp)my_lstm_cell/dense/BiasAdd/ReadVariableOp2T
(my_lstm_cell/dense/MatMul/ReadVariableOp(my_lstm_cell/dense/MatMul/ReadVariableOp2Z
+my_lstm_cell/dense_1/BiasAdd/ReadVariableOp+my_lstm_cell/dense_1/BiasAdd/ReadVariableOp2X
*my_lstm_cell/dense_1/MatMul/ReadVariableOp*my_lstm_cell/dense_1/MatMul/ReadVariableOp2Z
+my_lstm_cell/dense_2/BiasAdd/ReadVariableOp+my_lstm_cell/dense_2/BiasAdd/ReadVariableOp2X
*my_lstm_cell/dense_2/MatMul/ReadVariableOp*my_lstm_cell/dense_2/MatMul/ReadVariableOp2Z
+my_lstm_cell/dense_3/BiasAdd/ReadVariableOp+my_lstm_cell/dense_3/BiasAdd/ReadVariableOp2X
*my_lstm_cell/dense_3/MatMul/ReadVariableOp*my_lstm_cell/dense_3/MatMul/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
ษ
๙
B__inference_dense_4_layer_call_and_return_conditional_losses_16437

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identityขBiasAdd/ReadVariableOpขTensordot/ReadVariableOpz
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
value	B : ป
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
value	B : ฟ
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
:?????????
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ง
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
์	
ด
#__inference_rnn_layer_call_fn_18416
inputs_0
unknown:'
	unknown_0:
	unknown_1:'
	unknown_2:
	unknown_3:'
	unknown_4:
	unknown_5:'
	unknown_6:
identityขStatefulPartitionedCallฐ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_15958|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
ืM
ม
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_16745	
inputF
,conv2d_conv2d_conv2d_readvariableop_resource:G
9conv2d_squeeze_batch_dims_biasadd_readvariableop_resource:H
.conv2d_1_conv2d_conv2d_readvariableop_resource:I
;conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource:
identityข#conv2d/Conv2D/Conv2D/ReadVariableOpข0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOpข%conv2d_1/Conv2D/Conv2D/ReadVariableOpข2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOpH
conv2d/Conv2D/ShapeShapeinput*
T0*
_output_shapes
:k
!conv2d/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
#conv2d/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????m
#conv2d/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv2d/Conv2D/strided_sliceStridedSliceconv2d/Conv2D/Shape:output:0*conv2d/Conv2D/strided_slice/stack:output:0,conv2d/Conv2D/strided_slice/stack_1:output:0,conv2d/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskt
conv2d/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         
conv2d/Conv2D/ReshapeReshapeinput$conv2d/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????
#conv2d/Conv2D/Conv2D/ReadVariableOpReadVariableOp,conv2d_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0อ
conv2d/Conv2D/Conv2DConv2Dconv2d/Conv2D/Reshape:output:0+conv2d/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
r
conv2d/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         d
conv2d/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????ภ
conv2d/Conv2D/concatConcatV2$conv2d/Conv2D/strided_slice:output:0&conv2d/Conv2D/concat/values_1:output:0"conv2d/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:
conv2d/Conv2D/Reshape_1Reshapeconv2d/Conv2D/Conv2D:output:0conv2d/Conv2D/concat:output:0*
T0*3
_output_shapes!
:?????????o
conv2d/squeeze_batch_dims/ShapeShape conv2d/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:w
-conv2d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
/conv2d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????y
/conv2d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ั
'conv2d/squeeze_batch_dims/strided_sliceStridedSlice(conv2d/squeeze_batch_dims/Shape:output:06conv2d/squeeze_batch_dims/strided_slice/stack:output:08conv2d/squeeze_batch_dims/strided_slice/stack_1:output:08conv2d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
'conv2d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         บ
!conv2d/squeeze_batch_dims/ReshapeReshape conv2d/Conv2D/Reshape_1:output:00conv2d/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????ฆ
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp9conv2d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ฬ
!conv2d/squeeze_batch_dims/BiasAddBiasAdd*conv2d/squeeze_batch_dims/Reshape:output:08conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????~
)conv2d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         p
%conv2d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????๐
 conv2d/squeeze_batch_dims/concatConcatV20conv2d/squeeze_batch_dims/strided_slice:output:02conv2d/squeeze_batch_dims/concat/values_1:output:0.conv2d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:ร
#conv2d/squeeze_batch_dims/Reshape_1Reshape*conv2d/squeeze_batch_dims/BiasAdd:output:0)conv2d/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:?????????q
conv2d_1/Conv2D/ShapeShape,conv2d/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:m
#conv2d_1/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_1/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????o
%conv2d_1/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv2d_1/Conv2D/strided_sliceStridedSliceconv2d_1/Conv2D/Shape:output:0,conv2d_1/Conv2D/strided_slice/stack:output:0.conv2d_1/Conv2D/strided_slice/stack_1:output:0.conv2d_1/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_1/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ฒ
conv2d_1/Conv2D/ReshapeReshape,conv2d/squeeze_batch_dims/Reshape_1:output:0&conv2d_1/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????
%conv2d_1/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ำ
conv2d_1/Conv2D/Conv2DConv2D conv2d_1/Conv2D/Reshape:output:0-conv2d_1/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
t
conv2d_1/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         f
conv2d_1/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????ศ
conv2d_1/Conv2D/concatConcatV2&conv2d_1/Conv2D/strided_slice:output:0(conv2d_1/Conv2D/concat/values_1:output:0$conv2d_1/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:ค
conv2d_1/Conv2D/Reshape_1Reshapeconv2d_1/Conv2D/Conv2D:output:0conv2d_1/Conv2D/concat:output:0*
T0*3
_output_shapes!
:?????????s
!conv2d_1/squeeze_batch_dims/ShapeShape"conv2d_1/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_1/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1conv2d_1/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????{
1conv2d_1/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)conv2d_1/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_1/squeeze_batch_dims/Shape:output:08conv2d_1/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_1/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_1/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
)conv2d_1/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ภ
#conv2d_1/squeeze_batch_dims/ReshapeReshape"conv2d_1/Conv2D/Reshape_1:output:02conv2d_1/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????ช
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0า
#conv2d_1/squeeze_batch_dims/BiasAddBiasAdd,conv2d_1/squeeze_batch_dims/Reshape:output:0:conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
+conv2d_1/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         r
'conv2d_1/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????๘
"conv2d_1/squeeze_batch_dims/concatConcatV22conv2d_1/squeeze_batch_dims/strided_slice:output:04conv2d_1/squeeze_batch_dims/concat/values_1:output:00conv2d_1/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:ษ
%conv2d_1/squeeze_batch_dims/Reshape_1Reshape,conv2d_1/squeeze_batch_dims/BiasAdd:output:0+conv2d_1/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:?????????
IdentityIdentity.conv2d_1/squeeze_batch_dims/Reshape_1:output:0^NoOp*
T0*3
_output_shapes!
:??????????
NoOpNoOp$^conv2d/Conv2D/Conv2D/ReadVariableOp1^conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_1/Conv2D/Conv2D/ReadVariableOp3^conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : 2J
#conv2d/Conv2D/Conv2D/ReadVariableOp#conv2d/Conv2D/Conv2D/ReadVariableOp2d
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_1/Conv2D/Conv2D/ReadVariableOp%conv2d_1/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp:Z V
3
_output_shapes!
:?????????

_user_specified_nameinput
ผb
ท
rnn_while_body_17657$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3#
rnn_while_rnn_strided_slice_1_0_
[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0O
=rnn_while_my_lstm_cell_dense_matmul_readvariableop_resource_0:'L
>rnn_while_my_lstm_cell_dense_biasadd_readvariableop_resource_0:Q
?rnn_while_my_lstm_cell_dense_1_matmul_readvariableop_resource_0:'N
@rnn_while_my_lstm_cell_dense_1_biasadd_readvariableop_resource_0:Q
?rnn_while_my_lstm_cell_dense_2_matmul_readvariableop_resource_0:'N
@rnn_while_my_lstm_cell_dense_2_biasadd_readvariableop_resource_0:Q
?rnn_while_my_lstm_cell_dense_3_matmul_readvariableop_resource_0:'N
@rnn_while_my_lstm_cell_dense_3_biasadd_readvariableop_resource_0:
rnn_while_identity
rnn_while_identity_1
rnn_while_identity_2
rnn_while_identity_3
rnn_while_identity_4
rnn_while_identity_5!
rnn_while_rnn_strided_slice_1]
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensorM
;rnn_while_my_lstm_cell_dense_matmul_readvariableop_resource:'J
<rnn_while_my_lstm_cell_dense_biasadd_readvariableop_resource:O
=rnn_while_my_lstm_cell_dense_1_matmul_readvariableop_resource:'L
>rnn_while_my_lstm_cell_dense_1_biasadd_readvariableop_resource:O
=rnn_while_my_lstm_cell_dense_2_matmul_readvariableop_resource:'L
>rnn_while_my_lstm_cell_dense_2_biasadd_readvariableop_resource:O
=rnn_while_my_lstm_cell_dense_3_matmul_readvariableop_resource:'L
>rnn_while_my_lstm_cell_dense_3_biasadd_readvariableop_resource:ข3rnn/while/my_lstm_cell/dense/BiasAdd/ReadVariableOpข2rnn/while/my_lstm_cell/dense/MatMul/ReadVariableOpข5rnn/while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOpข4rnn/while/my_lstm_cell/dense_1/MatMul/ReadVariableOpข5rnn/while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOpข4rnn/while/my_lstm_cell/dense_2/MatMul/ReadVariableOpข5rnn/while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOpข4rnn/while/my_lstm_cell/dense_3/MatMul/ReadVariableOp
;rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   บ
-rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0rnn_while_placeholderDrnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0r
0rnn/while/my_lstm_cell/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
+rnn/while/my_lstm_cell/concatenate_1/concatConcatV24rnn/while/TensorArrayV2Read/TensorListGetItem:item:0rnn_while_placeholder_29rnn/while/my_lstm_cell/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????'ฐ
2rnn/while/my_lstm_cell/dense/MatMul/ReadVariableOpReadVariableOp=rnn_while_my_lstm_cell_dense_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ั
#rnn/while/my_lstm_cell/dense/MatMulMatMul4rnn/while/my_lstm_cell/concatenate_1/concat:output:0:rnn/while/my_lstm_cell/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ฎ
3rnn/while/my_lstm_cell/dense/BiasAdd/ReadVariableOpReadVariableOp>rnn_while_my_lstm_cell_dense_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0อ
$rnn/while/my_lstm_cell/dense/BiasAddBiasAdd-rnn/while/my_lstm_cell/dense/MatMul:product:0;rnn/while/my_lstm_cell/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
$rnn/while/my_lstm_cell/dense/SigmoidSigmoid-rnn/while/my_lstm_cell/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
#rnn/while/my_lstm_cell/multiply/mulMulrnn_while_placeholder_3(rnn/while/my_lstm_cell/dense/Sigmoid:y:0*
T0*'
_output_shapes
:?????????ด
4rnn/while/my_lstm_cell/dense_1/MatMul/ReadVariableOpReadVariableOp?rnn_while_my_lstm_cell_dense_1_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ี
%rnn/while/my_lstm_cell/dense_1/MatMulMatMul4rnn/while/my_lstm_cell/concatenate_1/concat:output:0<rnn/while/my_lstm_cell/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ฒ
5rnn/while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOpReadVariableOp@rnn_while_my_lstm_cell_dense_1_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0ำ
&rnn/while/my_lstm_cell/dense_1/BiasAddBiasAdd/rnn/while/my_lstm_cell/dense_1/MatMul:product:0=rnn/while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
&rnn/while/my_lstm_cell/dense_1/SigmoidSigmoid/rnn/while/my_lstm_cell/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????ด
4rnn/while/my_lstm_cell/dense_2/MatMul/ReadVariableOpReadVariableOp?rnn_while_my_lstm_cell_dense_2_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ี
%rnn/while/my_lstm_cell/dense_2/MatMulMatMul4rnn/while/my_lstm_cell/concatenate_1/concat:output:0<rnn/while/my_lstm_cell/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ฒ
5rnn/while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOpReadVariableOp@rnn_while_my_lstm_cell_dense_2_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0ำ
&rnn/while/my_lstm_cell/dense_2/BiasAddBiasAdd/rnn/while/my_lstm_cell/dense_2/MatMul:product:0=rnn/while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
#rnn/while/my_lstm_cell/dense_2/TanhTanh/rnn/while/my_lstm_cell/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????ณ
%rnn/while/my_lstm_cell/multiply/mul_1Mul*rnn/while/my_lstm_cell/dense_1/Sigmoid:y:0'rnn/while/my_lstm_cell/dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????ฏ
 rnn/while/my_lstm_cell/add_1/addAddV2'rnn/while/my_lstm_cell/multiply/mul:z:0)rnn/while/my_lstm_cell/multiply/mul_1:z:0*
T0*'
_output_shapes
:?????????ด
4rnn/while/my_lstm_cell/dense_3/MatMul/ReadVariableOpReadVariableOp?rnn_while_my_lstm_cell_dense_3_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ี
%rnn/while/my_lstm_cell/dense_3/MatMulMatMul4rnn/while/my_lstm_cell/concatenate_1/concat:output:0<rnn/while/my_lstm_cell/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ฒ
5rnn/while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOpReadVariableOp@rnn_while_my_lstm_cell_dense_3_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0ำ
&rnn/while/my_lstm_cell/dense_3/BiasAddBiasAdd/rnn/while/my_lstm_cell/dense_3/MatMul:product:0=rnn/while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
&rnn/while/my_lstm_cell/dense_3/SigmoidSigmoid/rnn/while/my_lstm_cell/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
&rnn/while/my_lstm_cell/activation/TanhTanh$rnn/while/my_lstm_cell/add_1/add:z:0*
T0*'
_output_shapes
:?????????ถ
%rnn/while/my_lstm_cell/multiply/mul_2Mul*rnn/while/my_lstm_cell/activation/Tanh:y:0*rnn/while/my_lstm_cell/dense_3/Sigmoid:y:0*
T0*'
_output_shapes
:??????????
.rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrnn_while_placeholder_1rnn_while_placeholder)rnn/while/my_lstm_cell/multiply/mul_2:z:0*
_output_shapes
: *
element_dtype0:้่าQ
rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :h
rnn/while/addAddV2rnn_while_placeholderrnn/while/add/y:output:0*
T0*
_output_shapes
: S
rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :w
rnn/while/add_1AddV2 rnn_while_rnn_while_loop_counterrnn/while/add_1/y:output:0*
T0*
_output_shapes
: e
rnn/while/IdentityIdentityrnn/while/add_1:z:0^rnn/while/NoOp*
T0*
_output_shapes
: z
rnn/while/Identity_1Identity&rnn_while_rnn_while_maximum_iterations^rnn/while/NoOp*
T0*
_output_shapes
: e
rnn/while/Identity_2Identityrnn/while/add:z:0^rnn/while/NoOp*
T0*
_output_shapes
: 
rnn/while/Identity_3Identity>rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^rnn/while/NoOp*
T0*
_output_shapes
: 
rnn/while/Identity_4Identity)rnn/while/my_lstm_cell/multiply/mul_2:z:0^rnn/while/NoOp*
T0*'
_output_shapes
:?????????
rnn/while/Identity_5Identity$rnn/while/my_lstm_cell/add_1/add:z:0^rnn/while/NoOp*
T0*'
_output_shapes
:?????????
rnn/while/NoOpNoOp4^rnn/while/my_lstm_cell/dense/BiasAdd/ReadVariableOp3^rnn/while/my_lstm_cell/dense/MatMul/ReadVariableOp6^rnn/while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp5^rnn/while/my_lstm_cell/dense_1/MatMul/ReadVariableOp6^rnn/while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp5^rnn/while/my_lstm_cell/dense_2/MatMul/ReadVariableOp6^rnn/while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp5^rnn/while/my_lstm_cell/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "1
rnn_while_identityrnn/while/Identity:output:0"5
rnn_while_identity_1rnn/while/Identity_1:output:0"5
rnn_while_identity_2rnn/while/Identity_2:output:0"5
rnn_while_identity_3rnn/while/Identity_3:output:0"5
rnn_while_identity_4rnn/while/Identity_4:output:0"5
rnn_while_identity_5rnn/while/Identity_5:output:0"
>rnn_while_my_lstm_cell_dense_1_biasadd_readvariableop_resource@rnn_while_my_lstm_cell_dense_1_biasadd_readvariableop_resource_0"
=rnn_while_my_lstm_cell_dense_1_matmul_readvariableop_resource?rnn_while_my_lstm_cell_dense_1_matmul_readvariableop_resource_0"
>rnn_while_my_lstm_cell_dense_2_biasadd_readvariableop_resource@rnn_while_my_lstm_cell_dense_2_biasadd_readvariableop_resource_0"
=rnn_while_my_lstm_cell_dense_2_matmul_readvariableop_resource?rnn_while_my_lstm_cell_dense_2_matmul_readvariableop_resource_0"
>rnn_while_my_lstm_cell_dense_3_biasadd_readvariableop_resource@rnn_while_my_lstm_cell_dense_3_biasadd_readvariableop_resource_0"
=rnn_while_my_lstm_cell_dense_3_matmul_readvariableop_resource?rnn_while_my_lstm_cell_dense_3_matmul_readvariableop_resource_0"~
<rnn_while_my_lstm_cell_dense_biasadd_readvariableop_resource>rnn_while_my_lstm_cell_dense_biasadd_readvariableop_resource_0"|
;rnn_while_my_lstm_cell_dense_matmul_readvariableop_resource=rnn_while_my_lstm_cell_dense_matmul_readvariableop_resource_0"@
rnn_while_rnn_strided_slice_1rnn_while_rnn_strided_slice_1_0"ธ
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :?????????:?????????: : : : : : : : : : 2j
3rnn/while/my_lstm_cell/dense/BiasAdd/ReadVariableOp3rnn/while/my_lstm_cell/dense/BiasAdd/ReadVariableOp2h
2rnn/while/my_lstm_cell/dense/MatMul/ReadVariableOp2rnn/while/my_lstm_cell/dense/MatMul/ReadVariableOp2n
5rnn/while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp5rnn/while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp2l
4rnn/while/my_lstm_cell/dense_1/MatMul/ReadVariableOp4rnn/while/my_lstm_cell/dense_1/MatMul/ReadVariableOp2n
5rnn/while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp5rnn/while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp2l
4rnn/while/my_lstm_cell/dense_2/MatMul/ReadVariableOp4rnn/while/my_lstm_cell/dense_2/MatMul/ReadVariableOp2n
5rnn/while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp5rnn/while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp2l
4rnn/while/my_lstm_cell/dense_3/MatMul/ReadVariableOp4rnn/while/my_lstm_cell/dense_3/MatMul/ReadVariableOp: 
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
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
า
g
K__inference_time_distributed_layer_call_and_return_conditional_losses_18325

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
valueB:ั
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
valueB"????         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ฃ
global_average_pooling2d/MeanMeanReshape:output:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape&global_average_pooling2d/Mean:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????g
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&??????????????????:d `
<
_output_shapes*
(:&??????????????????
 
_user_specified_nameinputs
ด
o
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_15721

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
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
้
์
-__inference_my_lstm_model_layer_call_fn_17537
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
identityขStatefulPartitionedCall๛
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
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_my_lstm_model_layer_call_and_return_conditional_losses_16444s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
3
_output_shapes!
:?????????

_user_specified_namex
ษ
่
#__inference_signature_wrapper_17504
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
identityขStatefulPartitionedCallู
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
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_15711s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:?????????
!
_user_specified_name	input_1
ฎ

__inference_call_17469
x,
my_cnn_block_17255: 
my_cnn_block_17257:,
my_cnn_block_17259: 
my_cnn_block_17261:G
5rnn_my_lstm_cell_dense_matmul_readvariableop_resource:'D
6rnn_my_lstm_cell_dense_biasadd_readvariableop_resource:I
7rnn_my_lstm_cell_dense_1_matmul_readvariableop_resource:'F
8rnn_my_lstm_cell_dense_1_biasadd_readvariableop_resource:I
7rnn_my_lstm_cell_dense_2_matmul_readvariableop_resource:'F
8rnn_my_lstm_cell_dense_2_biasadd_readvariableop_resource:I
7rnn_my_lstm_cell_dense_3_matmul_readvariableop_resource:'F
8rnn_my_lstm_cell_dense_3_biasadd_readvariableop_resource:;
)dense_4_tensordot_readvariableop_resource:5
'dense_4_biasadd_readvariableop_resource:
identityขdense_4/BiasAdd/ReadVariableOpข dense_4/Tensordot/ReadVariableOpข$my_cnn_block/StatefulPartitionedCallข-rnn/my_lstm_cell/dense/BiasAdd/ReadVariableOpข,rnn/my_lstm_cell/dense/MatMul/ReadVariableOpข/rnn/my_lstm_cell/dense_1/BiasAdd/ReadVariableOpข.rnn/my_lstm_cell/dense_1/MatMul/ReadVariableOpข/rnn/my_lstm_cell/dense_2/BiasAdd/ReadVariableOpข.rnn/my_lstm_cell/dense_2/MatMul/ReadVariableOpข/rnn/my_lstm_cell/dense_3/BiasAdd/ReadVariableOpข.rnn/my_lstm_cell/dense_3/MatMul/ReadVariableOpข	rnn/while?
$my_cnn_block/StatefulPartitionedCallStatefulPartitionedCallxmy_cnn_block_17255my_cnn_block_17257my_cnn_block_17259my_cnn_block_17261*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *
fR
__inference_call_15465w
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ต
time_distributed/ReshapeReshape-my_cnn_block/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????
@time_distributed/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ึ
.time_distributed/global_average_pooling2d/MeanMean!time_distributed/Reshape:output:0Itime_distributed/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????u
 time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      ฟ
time_distributed/Reshape_1Reshape7time_distributed/global_average_pooling2d/Mean:output:0)time_distributed/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????y
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         น
time_distributed/Reshape_2Reshape-my_cnn_block/StatefulPartitionedCall:output:0)time_distributed/Reshape_2/shape:output:0*
T0*/
_output_shapes
:?????????\
	rnn/ShapeShape#time_distributed/Reshape_1:output:0*
T0*
_output_shapes
:a
rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ๅ
rnn/strided_sliceStridedSlicernn/Shape:output:0 rnn/strided_slice/stack:output:0"rnn/strided_slice/stack_1:output:0"rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
rnn/zeros/packedPackrnn/strided_slice:output:0rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:T
rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x
	rnn/zerosFillrnn/zeros/packed:output:0rnn/zeros/Const:output:0*
T0*'
_output_shapes
:?????????V
rnn/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
rnn/zeros_1/packedPackrnn/strided_slice:output:0rnn/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:V
rnn/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
rnn/zeros_1Fillrnn/zeros_1/packed:output:0rnn/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????g
rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
rnn/transpose	Transpose#time_distributed/Reshape_1:output:0rnn/transpose/perm:output:0*
T0*+
_output_shapes
:?????????L
rnn/Shape_1Shapernn/transpose:y:0*
T0*
_output_shapes
:c
rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:๏
rnn/strided_slice_1StridedSlicernn/Shape_1:output:0"rnn/strided_slice_1/stack:output:0$rnn/strided_slice_1/stack_1:output:0$rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????ภ
rnn/TensorArrayV2TensorListReserve(rnn/TensorArrayV2/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่า
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ์
+rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn/transpose:y:0Brnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่าc
rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
rnn/strided_slice_2StridedSlicernn/transpose:y:0"rnn/strided_slice_2/stack:output:0$rnn/strided_slice_2/stack_1:output:0$rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskl
*rnn/my_lstm_cell/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ำ
%rnn/my_lstm_cell/concatenate_1/concatConcatV2rnn/strided_slice_2:output:0rnn/zeros:output:03rnn/my_lstm_cell/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????'ข
,rnn/my_lstm_cell/dense/MatMul/ReadVariableOpReadVariableOp5rnn_my_lstm_cell_dense_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0ฟ
rnn/my_lstm_cell/dense/MatMulMatMul.rnn/my_lstm_cell/concatenate_1/concat:output:04rnn/my_lstm_cell/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
-rnn/my_lstm_cell/dense/BiasAdd/ReadVariableOpReadVariableOp6rnn_my_lstm_cell_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ป
rnn/my_lstm_cell/dense/BiasAddBiasAdd'rnn/my_lstm_cell/dense/MatMul:product:05rnn/my_lstm_cell/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
rnn/my_lstm_cell/dense/SigmoidSigmoid'rnn/my_lstm_cell/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
rnn/my_lstm_cell/multiply/mulMulrnn/zeros_1:output:0"rnn/my_lstm_cell/dense/Sigmoid:y:0*
T0*'
_output_shapes
:?????????ฆ
.rnn/my_lstm_cell/dense_1/MatMul/ReadVariableOpReadVariableOp7rnn_my_lstm_cell_dense_1_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0ร
rnn/my_lstm_cell/dense_1/MatMulMatMul.rnn/my_lstm_cell/concatenate_1/concat:output:06rnn/my_lstm_cell/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ค
/rnn/my_lstm_cell/dense_1/BiasAdd/ReadVariableOpReadVariableOp8rnn_my_lstm_cell_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ม
 rnn/my_lstm_cell/dense_1/BiasAddBiasAdd)rnn/my_lstm_cell/dense_1/MatMul:product:07rnn/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
 rnn/my_lstm_cell/dense_1/SigmoidSigmoid)rnn/my_lstm_cell/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????ฆ
.rnn/my_lstm_cell/dense_2/MatMul/ReadVariableOpReadVariableOp7rnn_my_lstm_cell_dense_2_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0ร
rnn/my_lstm_cell/dense_2/MatMulMatMul.rnn/my_lstm_cell/concatenate_1/concat:output:06rnn/my_lstm_cell/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ค
/rnn/my_lstm_cell/dense_2/BiasAdd/ReadVariableOpReadVariableOp8rnn_my_lstm_cell_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ม
 rnn/my_lstm_cell/dense_2/BiasAddBiasAdd)rnn/my_lstm_cell/dense_2/MatMul:product:07rnn/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
rnn/my_lstm_cell/dense_2/TanhTanh)rnn/my_lstm_cell/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????ก
rnn/my_lstm_cell/multiply/mul_1Mul$rnn/my_lstm_cell/dense_1/Sigmoid:y:0!rnn/my_lstm_cell/dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????
rnn/my_lstm_cell/add_1/addAddV2!rnn/my_lstm_cell/multiply/mul:z:0#rnn/my_lstm_cell/multiply/mul_1:z:0*
T0*'
_output_shapes
:?????????ฆ
.rnn/my_lstm_cell/dense_3/MatMul/ReadVariableOpReadVariableOp7rnn_my_lstm_cell_dense_3_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0ร
rnn/my_lstm_cell/dense_3/MatMulMatMul.rnn/my_lstm_cell/concatenate_1/concat:output:06rnn/my_lstm_cell/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ค
/rnn/my_lstm_cell/dense_3/BiasAdd/ReadVariableOpReadVariableOp8rnn_my_lstm_cell_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ม
 rnn/my_lstm_cell/dense_3/BiasAddBiasAdd)rnn/my_lstm_cell/dense_3/MatMul:product:07rnn/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
 rnn/my_lstm_cell/dense_3/SigmoidSigmoid)rnn/my_lstm_cell/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????z
 rnn/my_lstm_cell/activation/TanhTanhrnn/my_lstm_cell/add_1/add:z:0*
T0*'
_output_shapes
:?????????ค
rnn/my_lstm_cell/multiply/mul_2Mul$rnn/my_lstm_cell/activation/Tanh:y:0$rnn/my_lstm_cell/dense_3/Sigmoid:y:0*
T0*'
_output_shapes
:?????????r
!rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ฤ
rnn/TensorArrayV2_1TensorListReserve*rnn/TensorArrayV2_1/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่าJ
rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : g
rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????X
rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 	
	rnn/whileWhilernn/while/loop_counter:output:0%rnn/while/maximum_iterations:output:0rnn/time:output:0rnn/TensorArrayV2_1:handle:0rnn/zeros:output:0rnn/zeros_1:output:0rnn/strided_slice_1:output:0;rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:05rnn_my_lstm_cell_dense_matmul_readvariableop_resource6rnn_my_lstm_cell_dense_biasadd_readvariableop_resource7rnn_my_lstm_cell_dense_1_matmul_readvariableop_resource8rnn_my_lstm_cell_dense_1_biasadd_readvariableop_resource7rnn_my_lstm_cell_dense_2_matmul_readvariableop_resource8rnn_my_lstm_cell_dense_2_biasadd_readvariableop_resource7rnn_my_lstm_cell_dense_3_matmul_readvariableop_resource8rnn_my_lstm_cell_dense_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( * 
bodyR
rnn_while_body_17339* 
condR
rnn_while_cond_17338*U
output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : *
parallel_iterations 
4rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ฮ
&rnn/TensorArrayV2Stack/TensorListStackTensorListStackrnn/while:output:3=rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype0l
rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????e
rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
rnn/strided_slice_3StridedSlice/rnn/TensorArrayV2Stack/TensorListStack:tensor:0"rnn/strided_slice_3/stack:output:0$rnn/strided_slice_3/stack_1:output:0$rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maski
rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ข
rnn/transpose_1	Transpose/rnn/TensorArrayV2Stack/TensorListStack:tensor:0rnn/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Z
dense_4/Tensordot/ShapeShapernn/transpose_1:y:0*
T0*
_output_shapes
:a
dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ฿
dense_4/Tensordot/GatherV2_1GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/axes:output:0*dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ผ
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_4/Tensordot/transpose	Transposernn/transpose_1:y:0!dense_4/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????ข
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????ข
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????c
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ว
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????k
IdentityIdentitydense_4/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????ล
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp%^my_cnn_block/StatefulPartitionedCall.^rnn/my_lstm_cell/dense/BiasAdd/ReadVariableOp-^rnn/my_lstm_cell/dense/MatMul/ReadVariableOp0^rnn/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp/^rnn/my_lstm_cell/dense_1/MatMul/ReadVariableOp0^rnn/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp/^rnn/my_lstm_cell/dense_2/MatMul/ReadVariableOp0^rnn/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp/^rnn/my_lstm_cell/dense_3/MatMul/ReadVariableOp
^rnn/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp2L
$my_cnn_block/StatefulPartitionedCall$my_cnn_block/StatefulPartitionedCall2^
-rnn/my_lstm_cell/dense/BiasAdd/ReadVariableOp-rnn/my_lstm_cell/dense/BiasAdd/ReadVariableOp2\
,rnn/my_lstm_cell/dense/MatMul/ReadVariableOp,rnn/my_lstm_cell/dense/MatMul/ReadVariableOp2b
/rnn/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp/rnn/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp2`
.rnn/my_lstm_cell/dense_1/MatMul/ReadVariableOp.rnn/my_lstm_cell/dense_1/MatMul/ReadVariableOp2b
/rnn/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp/rnn/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp2`
.rnn/my_lstm_cell/dense_2/MatMul/ReadVariableOp.rnn/my_lstm_cell/dense_2/MatMul/ReadVariableOp2b
/rnn/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp/rnn/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp2`
.rnn/my_lstm_cell/dense_3/MatMul/ReadVariableOp.rnn/my_lstm_cell/dense_3/MatMul/ReadVariableOp2
	rnn/while	rnn/while:V R
3
_output_shapes!
:?????????

_user_specified_namex
อ

ว
while_cond_16060
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_16060___redundant_placeholder03
/while_while_cond_16060___redundant_placeholder13
/while_while_cond_16060___redundant_placeholder23
/while_while_cond_16060___redundant_placeholder33
/while_while_cond_16060___redundant_placeholder43
/while_while_cond_16060___redundant_placeholder53
/while_while_cond_16060___redundant_placeholder63
/while_while_cond_16060___redundant_placeholder73
/while_while_cond_16060___redundant_placeholder8
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
T: : : : :?????????:?????????: :::::::::: 
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
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
ๅ
ฟ
rnn_while_cond_17656$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3&
"rnn_while_less_rnn_strided_slice_1;
7rnn_while_rnn_while_cond_17656___redundant_placeholder0;
7rnn_while_rnn_while_cond_17656___redundant_placeholder1;
7rnn_while_rnn_while_cond_17656___redundant_placeholder2;
7rnn_while_rnn_while_cond_17656___redundant_placeholder3;
7rnn_while_rnn_while_cond_17656___redundant_placeholder4;
7rnn_while_rnn_while_cond_17656___redundant_placeholder5;
7rnn_while_rnn_while_cond_17656___redundant_placeholder6;
7rnn_while_rnn_while_cond_17656___redundant_placeholder7;
7rnn_while_rnn_while_cond_17656___redundant_placeholder8
rnn_while_identity
r
rnn/while/LessLessrnn_while_placeholder"rnn_while_less_rnn_strided_slice_1*
T0*
_output_shapes
: S
rnn/while/IdentityIdentityrnn/while/Less:z:0*
T0
*
_output_shapes
: "1
rnn_while_identityrnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :?????????:?????????: :::::::::: 
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
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
ืM
ม
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_16235	
inputF
,conv2d_conv2d_conv2d_readvariableop_resource:G
9conv2d_squeeze_batch_dims_biasadd_readvariableop_resource:H
.conv2d_1_conv2d_conv2d_readvariableop_resource:I
;conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource:
identityข#conv2d/Conv2D/Conv2D/ReadVariableOpข0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOpข%conv2d_1/Conv2D/Conv2D/ReadVariableOpข2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOpH
conv2d/Conv2D/ShapeShapeinput*
T0*
_output_shapes
:k
!conv2d/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
#conv2d/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????m
#conv2d/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv2d/Conv2D/strided_sliceStridedSliceconv2d/Conv2D/Shape:output:0*conv2d/Conv2D/strided_slice/stack:output:0,conv2d/Conv2D/strided_slice/stack_1:output:0,conv2d/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskt
conv2d/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         
conv2d/Conv2D/ReshapeReshapeinput$conv2d/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????
#conv2d/Conv2D/Conv2D/ReadVariableOpReadVariableOp,conv2d_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0อ
conv2d/Conv2D/Conv2DConv2Dconv2d/Conv2D/Reshape:output:0+conv2d/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
r
conv2d/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         d
conv2d/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????ภ
conv2d/Conv2D/concatConcatV2$conv2d/Conv2D/strided_slice:output:0&conv2d/Conv2D/concat/values_1:output:0"conv2d/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:
conv2d/Conv2D/Reshape_1Reshapeconv2d/Conv2D/Conv2D:output:0conv2d/Conv2D/concat:output:0*
T0*3
_output_shapes!
:?????????o
conv2d/squeeze_batch_dims/ShapeShape conv2d/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:w
-conv2d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
/conv2d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????y
/conv2d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ั
'conv2d/squeeze_batch_dims/strided_sliceStridedSlice(conv2d/squeeze_batch_dims/Shape:output:06conv2d/squeeze_batch_dims/strided_slice/stack:output:08conv2d/squeeze_batch_dims/strided_slice/stack_1:output:08conv2d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
'conv2d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         บ
!conv2d/squeeze_batch_dims/ReshapeReshape conv2d/Conv2D/Reshape_1:output:00conv2d/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????ฆ
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp9conv2d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ฬ
!conv2d/squeeze_batch_dims/BiasAddBiasAdd*conv2d/squeeze_batch_dims/Reshape:output:08conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????~
)conv2d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         p
%conv2d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????๐
 conv2d/squeeze_batch_dims/concatConcatV20conv2d/squeeze_batch_dims/strided_slice:output:02conv2d/squeeze_batch_dims/concat/values_1:output:0.conv2d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:ร
#conv2d/squeeze_batch_dims/Reshape_1Reshape*conv2d/squeeze_batch_dims/BiasAdd:output:0)conv2d/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:?????????q
conv2d_1/Conv2D/ShapeShape,conv2d/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:m
#conv2d_1/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_1/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????o
%conv2d_1/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv2d_1/Conv2D/strided_sliceStridedSliceconv2d_1/Conv2D/Shape:output:0,conv2d_1/Conv2D/strided_slice/stack:output:0.conv2d_1/Conv2D/strided_slice/stack_1:output:0.conv2d_1/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_1/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ฒ
conv2d_1/Conv2D/ReshapeReshape,conv2d/squeeze_batch_dims/Reshape_1:output:0&conv2d_1/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????
%conv2d_1/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ำ
conv2d_1/Conv2D/Conv2DConv2D conv2d_1/Conv2D/Reshape:output:0-conv2d_1/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
t
conv2d_1/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         f
conv2d_1/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????ศ
conv2d_1/Conv2D/concatConcatV2&conv2d_1/Conv2D/strided_slice:output:0(conv2d_1/Conv2D/concat/values_1:output:0$conv2d_1/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:ค
conv2d_1/Conv2D/Reshape_1Reshapeconv2d_1/Conv2D/Conv2D:output:0conv2d_1/Conv2D/concat:output:0*
T0*3
_output_shapes!
:?????????s
!conv2d_1/squeeze_batch_dims/ShapeShape"conv2d_1/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_1/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1conv2d_1/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????{
1conv2d_1/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)conv2d_1/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_1/squeeze_batch_dims/Shape:output:08conv2d_1/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_1/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_1/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
)conv2d_1/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ภ
#conv2d_1/squeeze_batch_dims/ReshapeReshape"conv2d_1/Conv2D/Reshape_1:output:02conv2d_1/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????ช
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0า
#conv2d_1/squeeze_batch_dims/BiasAddBiasAdd,conv2d_1/squeeze_batch_dims/Reshape:output:0:conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
+conv2d_1/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         r
'conv2d_1/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????๘
"conv2d_1/squeeze_batch_dims/concatConcatV22conv2d_1/squeeze_batch_dims/strided_slice:output:04conv2d_1/squeeze_batch_dims/concat/values_1:output:00conv2d_1/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:ษ
%conv2d_1/squeeze_batch_dims/Reshape_1Reshape,conv2d_1/squeeze_batch_dims/BiasAdd:output:0+conv2d_1/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:?????????
IdentityIdentity.conv2d_1/squeeze_batch_dims/Reshape_1:output:0^NoOp*
T0*3
_output_shapes!
:??????????
NoOpNoOp$^conv2d/Conv2D/Conv2D/ReadVariableOp1^conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_1/Conv2D/Conv2D/ReadVariableOp3^conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : 2J
#conv2d/Conv2D/Conv2D/ReadVariableOp#conv2d/Conv2D/Conv2D/ReadVariableOp2d
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_1/Conv2D/Conv2D/ReadVariableOp%conv2d_1/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp:Z V
3
_output_shapes!
:?????????

_user_specified_nameinput
ผb
ท
rnn_while_body_17339$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3#
rnn_while_rnn_strided_slice_1_0_
[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0O
=rnn_while_my_lstm_cell_dense_matmul_readvariableop_resource_0:'L
>rnn_while_my_lstm_cell_dense_biasadd_readvariableop_resource_0:Q
?rnn_while_my_lstm_cell_dense_1_matmul_readvariableop_resource_0:'N
@rnn_while_my_lstm_cell_dense_1_biasadd_readvariableop_resource_0:Q
?rnn_while_my_lstm_cell_dense_2_matmul_readvariableop_resource_0:'N
@rnn_while_my_lstm_cell_dense_2_biasadd_readvariableop_resource_0:Q
?rnn_while_my_lstm_cell_dense_3_matmul_readvariableop_resource_0:'N
@rnn_while_my_lstm_cell_dense_3_biasadd_readvariableop_resource_0:
rnn_while_identity
rnn_while_identity_1
rnn_while_identity_2
rnn_while_identity_3
rnn_while_identity_4
rnn_while_identity_5!
rnn_while_rnn_strided_slice_1]
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensorM
;rnn_while_my_lstm_cell_dense_matmul_readvariableop_resource:'J
<rnn_while_my_lstm_cell_dense_biasadd_readvariableop_resource:O
=rnn_while_my_lstm_cell_dense_1_matmul_readvariableop_resource:'L
>rnn_while_my_lstm_cell_dense_1_biasadd_readvariableop_resource:O
=rnn_while_my_lstm_cell_dense_2_matmul_readvariableop_resource:'L
>rnn_while_my_lstm_cell_dense_2_biasadd_readvariableop_resource:O
=rnn_while_my_lstm_cell_dense_3_matmul_readvariableop_resource:'L
>rnn_while_my_lstm_cell_dense_3_biasadd_readvariableop_resource:ข3rnn/while/my_lstm_cell/dense/BiasAdd/ReadVariableOpข2rnn/while/my_lstm_cell/dense/MatMul/ReadVariableOpข5rnn/while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOpข4rnn/while/my_lstm_cell/dense_1/MatMul/ReadVariableOpข5rnn/while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOpข4rnn/while/my_lstm_cell/dense_2/MatMul/ReadVariableOpข5rnn/while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOpข4rnn/while/my_lstm_cell/dense_3/MatMul/ReadVariableOp
;rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   บ
-rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0rnn_while_placeholderDrnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0r
0rnn/while/my_lstm_cell/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
+rnn/while/my_lstm_cell/concatenate_1/concatConcatV24rnn/while/TensorArrayV2Read/TensorListGetItem:item:0rnn_while_placeholder_29rnn/while/my_lstm_cell/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????'ฐ
2rnn/while/my_lstm_cell/dense/MatMul/ReadVariableOpReadVariableOp=rnn_while_my_lstm_cell_dense_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ั
#rnn/while/my_lstm_cell/dense/MatMulMatMul4rnn/while/my_lstm_cell/concatenate_1/concat:output:0:rnn/while/my_lstm_cell/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ฎ
3rnn/while/my_lstm_cell/dense/BiasAdd/ReadVariableOpReadVariableOp>rnn_while_my_lstm_cell_dense_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0อ
$rnn/while/my_lstm_cell/dense/BiasAddBiasAdd-rnn/while/my_lstm_cell/dense/MatMul:product:0;rnn/while/my_lstm_cell/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
$rnn/while/my_lstm_cell/dense/SigmoidSigmoid-rnn/while/my_lstm_cell/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
#rnn/while/my_lstm_cell/multiply/mulMulrnn_while_placeholder_3(rnn/while/my_lstm_cell/dense/Sigmoid:y:0*
T0*'
_output_shapes
:?????????ด
4rnn/while/my_lstm_cell/dense_1/MatMul/ReadVariableOpReadVariableOp?rnn_while_my_lstm_cell_dense_1_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ี
%rnn/while/my_lstm_cell/dense_1/MatMulMatMul4rnn/while/my_lstm_cell/concatenate_1/concat:output:0<rnn/while/my_lstm_cell/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ฒ
5rnn/while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOpReadVariableOp@rnn_while_my_lstm_cell_dense_1_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0ำ
&rnn/while/my_lstm_cell/dense_1/BiasAddBiasAdd/rnn/while/my_lstm_cell/dense_1/MatMul:product:0=rnn/while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
&rnn/while/my_lstm_cell/dense_1/SigmoidSigmoid/rnn/while/my_lstm_cell/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????ด
4rnn/while/my_lstm_cell/dense_2/MatMul/ReadVariableOpReadVariableOp?rnn_while_my_lstm_cell_dense_2_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ี
%rnn/while/my_lstm_cell/dense_2/MatMulMatMul4rnn/while/my_lstm_cell/concatenate_1/concat:output:0<rnn/while/my_lstm_cell/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ฒ
5rnn/while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOpReadVariableOp@rnn_while_my_lstm_cell_dense_2_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0ำ
&rnn/while/my_lstm_cell/dense_2/BiasAddBiasAdd/rnn/while/my_lstm_cell/dense_2/MatMul:product:0=rnn/while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
#rnn/while/my_lstm_cell/dense_2/TanhTanh/rnn/while/my_lstm_cell/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????ณ
%rnn/while/my_lstm_cell/multiply/mul_1Mul*rnn/while/my_lstm_cell/dense_1/Sigmoid:y:0'rnn/while/my_lstm_cell/dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????ฏ
 rnn/while/my_lstm_cell/add_1/addAddV2'rnn/while/my_lstm_cell/multiply/mul:z:0)rnn/while/my_lstm_cell/multiply/mul_1:z:0*
T0*'
_output_shapes
:?????????ด
4rnn/while/my_lstm_cell/dense_3/MatMul/ReadVariableOpReadVariableOp?rnn_while_my_lstm_cell_dense_3_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ี
%rnn/while/my_lstm_cell/dense_3/MatMulMatMul4rnn/while/my_lstm_cell/concatenate_1/concat:output:0<rnn/while/my_lstm_cell/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ฒ
5rnn/while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOpReadVariableOp@rnn_while_my_lstm_cell_dense_3_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0ำ
&rnn/while/my_lstm_cell/dense_3/BiasAddBiasAdd/rnn/while/my_lstm_cell/dense_3/MatMul:product:0=rnn/while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
&rnn/while/my_lstm_cell/dense_3/SigmoidSigmoid/rnn/while/my_lstm_cell/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
&rnn/while/my_lstm_cell/activation/TanhTanh$rnn/while/my_lstm_cell/add_1/add:z:0*
T0*'
_output_shapes
:?????????ถ
%rnn/while/my_lstm_cell/multiply/mul_2Mul*rnn/while/my_lstm_cell/activation/Tanh:y:0*rnn/while/my_lstm_cell/dense_3/Sigmoid:y:0*
T0*'
_output_shapes
:??????????
.rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrnn_while_placeholder_1rnn_while_placeholder)rnn/while/my_lstm_cell/multiply/mul_2:z:0*
_output_shapes
: *
element_dtype0:้่าQ
rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :h
rnn/while/addAddV2rnn_while_placeholderrnn/while/add/y:output:0*
T0*
_output_shapes
: S
rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :w
rnn/while/add_1AddV2 rnn_while_rnn_while_loop_counterrnn/while/add_1/y:output:0*
T0*
_output_shapes
: e
rnn/while/IdentityIdentityrnn/while/add_1:z:0^rnn/while/NoOp*
T0*
_output_shapes
: z
rnn/while/Identity_1Identity&rnn_while_rnn_while_maximum_iterations^rnn/while/NoOp*
T0*
_output_shapes
: e
rnn/while/Identity_2Identityrnn/while/add:z:0^rnn/while/NoOp*
T0*
_output_shapes
: 
rnn/while/Identity_3Identity>rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^rnn/while/NoOp*
T0*
_output_shapes
: 
rnn/while/Identity_4Identity)rnn/while/my_lstm_cell/multiply/mul_2:z:0^rnn/while/NoOp*
T0*'
_output_shapes
:?????????
rnn/while/Identity_5Identity$rnn/while/my_lstm_cell/add_1/add:z:0^rnn/while/NoOp*
T0*'
_output_shapes
:?????????
rnn/while/NoOpNoOp4^rnn/while/my_lstm_cell/dense/BiasAdd/ReadVariableOp3^rnn/while/my_lstm_cell/dense/MatMul/ReadVariableOp6^rnn/while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp5^rnn/while/my_lstm_cell/dense_1/MatMul/ReadVariableOp6^rnn/while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp5^rnn/while/my_lstm_cell/dense_2/MatMul/ReadVariableOp6^rnn/while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp5^rnn/while/my_lstm_cell/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "1
rnn_while_identityrnn/while/Identity:output:0"5
rnn_while_identity_1rnn/while/Identity_1:output:0"5
rnn_while_identity_2rnn/while/Identity_2:output:0"5
rnn_while_identity_3rnn/while/Identity_3:output:0"5
rnn_while_identity_4rnn/while/Identity_4:output:0"5
rnn_while_identity_5rnn/while/Identity_5:output:0"
>rnn_while_my_lstm_cell_dense_1_biasadd_readvariableop_resource@rnn_while_my_lstm_cell_dense_1_biasadd_readvariableop_resource_0"
=rnn_while_my_lstm_cell_dense_1_matmul_readvariableop_resource?rnn_while_my_lstm_cell_dense_1_matmul_readvariableop_resource_0"
>rnn_while_my_lstm_cell_dense_2_biasadd_readvariableop_resource@rnn_while_my_lstm_cell_dense_2_biasadd_readvariableop_resource_0"
=rnn_while_my_lstm_cell_dense_2_matmul_readvariableop_resource?rnn_while_my_lstm_cell_dense_2_matmul_readvariableop_resource_0"
>rnn_while_my_lstm_cell_dense_3_biasadd_readvariableop_resource@rnn_while_my_lstm_cell_dense_3_biasadd_readvariableop_resource_0"
=rnn_while_my_lstm_cell_dense_3_matmul_readvariableop_resource?rnn_while_my_lstm_cell_dense_3_matmul_readvariableop_resource_0"~
<rnn_while_my_lstm_cell_dense_biasadd_readvariableop_resource>rnn_while_my_lstm_cell_dense_biasadd_readvariableop_resource_0"|
;rnn_while_my_lstm_cell_dense_matmul_readvariableop_resource=rnn_while_my_lstm_cell_dense_matmul_readvariableop_resource_0"@
rnn_while_rnn_strided_slice_1rnn_while_rnn_strided_slice_1_0"ธ
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :?????????:?????????: : : : : : : : : : 2j
3rnn/while/my_lstm_cell/dense/BiasAdd/ReadVariableOp3rnn/while/my_lstm_cell/dense/BiasAdd/ReadVariableOp2h
2rnn/while/my_lstm_cell/dense/MatMul/ReadVariableOp2rnn/while/my_lstm_cell/dense/MatMul/ReadVariableOp2n
5rnn/while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp5rnn/while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp2l
4rnn/while/my_lstm_cell/dense_1/MatMul/ReadVariableOp4rnn/while/my_lstm_cell/dense_1/MatMul/ReadVariableOp2n
5rnn/while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp5rnn/while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp2l
4rnn/while/my_lstm_cell/dense_2/MatMul/ReadVariableOp4rnn/while/my_lstm_cell/dense_2/MatMul/ReadVariableOp2n
5rnn/while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp5rnn/while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp2l
4rnn/while/my_lstm_cell/dense_3/MatMul/ReadVariableOp4rnn/while/my_lstm_cell/dense_3/MatMul/ReadVariableOp: 
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
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
ษ
๙
B__inference_dense_4_layer_call_and_return_conditional_losses_19214

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identityขBiasAdd/ReadVariableOpขTensordot/ReadVariableOpz
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
value	B : ป
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
value	B : ฟ
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
:?????????
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ง
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
๏
L
0__inference_time_distributed_layer_call_fn_18291

inputs
identityร
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_15765m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&??????????????????:d `
<
_output_shapes*
(:&??????????????????
 
_user_specified_nameinputs

g
K__inference_time_distributed_layer_call_and_return_conditional_losses_15744

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
valueB:ั
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
valueB"????         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????แ
(global_average_pooling2d/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_15721\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshape1global_average_pooling2d/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????g
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&??????????????????:d `
<
_output_shapes*
(:&??????????????????
 
_user_specified_nameinputs
ๅ
ฟ
rnn_while_cond_17121$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3&
"rnn_while_less_rnn_strided_slice_1;
7rnn_while_rnn_while_cond_17121___redundant_placeholder0;
7rnn_while_rnn_while_cond_17121___redundant_placeholder1;
7rnn_while_rnn_while_cond_17121___redundant_placeholder2;
7rnn_while_rnn_while_cond_17121___redundant_placeholder3;
7rnn_while_rnn_while_cond_17121___redundant_placeholder4;
7rnn_while_rnn_while_cond_17121___redundant_placeholder5;
7rnn_while_rnn_while_cond_17121___redundant_placeholder6;
7rnn_while_rnn_while_cond_17121___redundant_placeholder7;
7rnn_while_rnn_while_cond_17121___redundant_placeholder8
rnn_while_identity
r
rnn/while/LessLessrnn_while_placeholder"rnn_while_less_rnn_strided_slice_1*
T0*
_output_shapes
: S
rnn/while/IdentityIdentityrnn/while/Less:z:0*
T0
*
_output_shapes
: "1
rnn_while_identityrnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :?????????:?????????: :::::::::: 
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
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
ฑ)


while_body_15870
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_my_lstm_cell_15894_0:'(
while_my_lstm_cell_15896_0:,
while_my_lstm_cell_15898_0:'(
while_my_lstm_cell_15900_0:,
while_my_lstm_cell_15902_0:'(
while_my_lstm_cell_15904_0:,
while_my_lstm_cell_15906_0:'(
while_my_lstm_cell_15908_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_my_lstm_cell_15894:'&
while_my_lstm_cell_15896:*
while_my_lstm_cell_15898:'&
while_my_lstm_cell_15900:*
while_my_lstm_cell_15902:'&
while_my_lstm_cell_15904:*
while_my_lstm_cell_15906:'&
while_my_lstm_cell_15908:ข*while/my_lstm_cell/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ฆ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0ล
*while/my_lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_my_lstm_cell_15894_0while_my_lstm_cell_15896_0while_my_lstm_cell_15898_0while_my_lstm_cell_15900_0while_my_lstm_cell_15902_0while_my_lstm_cell_15904_0while_my_lstm_cell_15906_0while_my_lstm_cell_15908_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_my_lstm_cell_layer_call_and_return_conditional_losses_15846?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/my_lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:้่าM
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
: 
while/Identity_4Identity3while/my_lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????
while/Identity_5Identity3while/my_lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:?????????y

while/NoOpNoOp+^while/my_lstm_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_my_lstm_cell_15894while_my_lstm_cell_15894_0"6
while_my_lstm_cell_15896while_my_lstm_cell_15896_0"6
while_my_lstm_cell_15898while_my_lstm_cell_15898_0"6
while_my_lstm_cell_15900while_my_lstm_cell_15900_0"6
while_my_lstm_cell_15902while_my_lstm_cell_15902_0"6
while_my_lstm_cell_15904while_my_lstm_cell_15904_0"6
while_my_lstm_cell_15906while_my_lstm_cell_15906_0"6
while_my_lstm_cell_15908while_my_lstm_cell_15908_0"0
while_strided_slice_1while_strided_slice_1_0"จ
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :?????????:?????????: : : : : : : : : : 2X
*while/my_lstm_cell/StatefulPartitionedCall*while/my_lstm_cell/StatefulPartitionedCall: 
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
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
ๅ
ฟ
rnn_while_cond_17873$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3&
"rnn_while_less_rnn_strided_slice_1;
7rnn_while_rnn_while_cond_17873___redundant_placeholder0;
7rnn_while_rnn_while_cond_17873___redundant_placeholder1;
7rnn_while_rnn_while_cond_17873___redundant_placeholder2;
7rnn_while_rnn_while_cond_17873___redundant_placeholder3;
7rnn_while_rnn_while_cond_17873___redundant_placeholder4;
7rnn_while_rnn_while_cond_17873___redundant_placeholder5;
7rnn_while_rnn_while_cond_17873___redundant_placeholder6;
7rnn_while_rnn_while_cond_17873___redundant_placeholder7;
7rnn_while_rnn_while_cond_17873___redundant_placeholder8
rnn_while_identity
r
rnn/while/LessLessrnn_while_placeholder"rnn_while_less_rnn_strided_slice_1*
T0*
_output_shapes
: S
rnn/while/IdentityIdentityrnn/while/Less:z:0*
T0
*
_output_shapes
: "1
rnn_while_identityrnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :?????????:?????????: :::::::::: 
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
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
อ

ว
while_cond_19070
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_19070___redundant_placeholder03
/while_while_cond_19070___redundant_placeholder13
/while_while_cond_19070___redundant_placeholder23
/while_while_cond_19070___redundant_placeholder33
/while_while_cond_19070___redundant_placeholder43
/while_while_cond_19070___redundant_placeholder53
/while_while_cond_19070___redundant_placeholder63
/while_while_cond_19070___redundant_placeholder73
/while_while_cond_19070___redundant_placeholder8
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
T: : : : :?????????:?????????: :::::::::: 
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
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
เ[
๛
while_body_18549
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_my_lstm_cell_dense_matmul_readvariableop_resource_0:'H
:while_my_lstm_cell_dense_biasadd_readvariableop_resource_0:M
;while_my_lstm_cell_dense_1_matmul_readvariableop_resource_0:'J
<while_my_lstm_cell_dense_1_biasadd_readvariableop_resource_0:M
;while_my_lstm_cell_dense_2_matmul_readvariableop_resource_0:'J
<while_my_lstm_cell_dense_2_biasadd_readvariableop_resource_0:M
;while_my_lstm_cell_dense_3_matmul_readvariableop_resource_0:'J
<while_my_lstm_cell_dense_3_biasadd_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_my_lstm_cell_dense_matmul_readvariableop_resource:'F
8while_my_lstm_cell_dense_biasadd_readvariableop_resource:K
9while_my_lstm_cell_dense_1_matmul_readvariableop_resource:'H
:while_my_lstm_cell_dense_1_biasadd_readvariableop_resource:K
9while_my_lstm_cell_dense_2_matmul_readvariableop_resource:'H
:while_my_lstm_cell_dense_2_biasadd_readvariableop_resource:K
9while_my_lstm_cell_dense_3_matmul_readvariableop_resource:'H
:while_my_lstm_cell_dense_3_biasadd_readvariableop_resource:ข/while/my_lstm_cell/dense/BiasAdd/ReadVariableOpข.while/my_lstm_cell/dense/MatMul/ReadVariableOpข1while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOpข0while/my_lstm_cell/dense_1/MatMul/ReadVariableOpข1while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOpข0while/my_lstm_cell/dense_2/MatMul/ReadVariableOpข1while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOpข0while/my_lstm_cell/dense_3/MatMul/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ฆ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0n
,while/my_lstm_cell/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :์
'while/my_lstm_cell/concatenate_1/concatConcatV20while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_25while/my_lstm_cell/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????'จ
.while/my_lstm_cell/dense/MatMul/ReadVariableOpReadVariableOp9while_my_lstm_cell_dense_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ล
while/my_lstm_cell/dense/MatMulMatMul0while/my_lstm_cell/concatenate_1/concat:output:06while/my_lstm_cell/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ฆ
/while/my_lstm_cell/dense/BiasAdd/ReadVariableOpReadVariableOp:while_my_lstm_cell_dense_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0ม
 while/my_lstm_cell/dense/BiasAddBiasAdd)while/my_lstm_cell/dense/MatMul:product:07while/my_lstm_cell/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
 while/my_lstm_cell/dense/SigmoidSigmoid)while/my_lstm_cell/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
while/my_lstm_cell/multiply/mulMulwhile_placeholder_3$while/my_lstm_cell/dense/Sigmoid:y:0*
T0*'
_output_shapes
:?????????ฌ
0while/my_lstm_cell/dense_1/MatMul/ReadVariableOpReadVariableOp;while_my_lstm_cell_dense_1_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ษ
!while/my_lstm_cell/dense_1/MatMulMatMul0while/my_lstm_cell/concatenate_1/concat:output:08while/my_lstm_cell/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ช
1while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOpReadVariableOp<while_my_lstm_cell_dense_1_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0ว
"while/my_lstm_cell/dense_1/BiasAddBiasAdd+while/my_lstm_cell/dense_1/MatMul:product:09while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
"while/my_lstm_cell/dense_1/SigmoidSigmoid+while/my_lstm_cell/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????ฌ
0while/my_lstm_cell/dense_2/MatMul/ReadVariableOpReadVariableOp;while_my_lstm_cell_dense_2_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ษ
!while/my_lstm_cell/dense_2/MatMulMatMul0while/my_lstm_cell/concatenate_1/concat:output:08while/my_lstm_cell/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ช
1while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOpReadVariableOp<while_my_lstm_cell_dense_2_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0ว
"while/my_lstm_cell/dense_2/BiasAddBiasAdd+while/my_lstm_cell/dense_2/MatMul:product:09while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
while/my_lstm_cell/dense_2/TanhTanh+while/my_lstm_cell/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????ง
!while/my_lstm_cell/multiply/mul_1Mul&while/my_lstm_cell/dense_1/Sigmoid:y:0#while/my_lstm_cell/dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????ฃ
while/my_lstm_cell/add_1/addAddV2#while/my_lstm_cell/multiply/mul:z:0%while/my_lstm_cell/multiply/mul_1:z:0*
T0*'
_output_shapes
:?????????ฌ
0while/my_lstm_cell/dense_3/MatMul/ReadVariableOpReadVariableOp;while_my_lstm_cell_dense_3_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ษ
!while/my_lstm_cell/dense_3/MatMulMatMul0while/my_lstm_cell/concatenate_1/concat:output:08while/my_lstm_cell/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ช
1while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOpReadVariableOp<while_my_lstm_cell_dense_3_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0ว
"while/my_lstm_cell/dense_3/BiasAddBiasAdd+while/my_lstm_cell/dense_3/MatMul:product:09while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
"while/my_lstm_cell/dense_3/SigmoidSigmoid+while/my_lstm_cell/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????~
"while/my_lstm_cell/activation/TanhTanh while/my_lstm_cell/add_1/add:z:0*
T0*'
_output_shapes
:?????????ช
!while/my_lstm_cell/multiply/mul_2Mul&while/my_lstm_cell/activation/Tanh:y:0&while/my_lstm_cell/dense_3/Sigmoid:y:0*
T0*'
_output_shapes
:?????????ฮ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder%while/my_lstm_cell/multiply/mul_2:z:0*
_output_shapes
: *
element_dtype0:้่าM
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
: 
while/Identity_4Identity%while/my_lstm_cell/multiply/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:?????????}
while/Identity_5Identity while/my_lstm_cell/add_1/add:z:0^while/NoOp*
T0*'
_output_shapes
:?????????ไ

while/NoOpNoOp0^while/my_lstm_cell/dense/BiasAdd/ReadVariableOp/^while/my_lstm_cell/dense/MatMul/ReadVariableOp2^while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp1^while/my_lstm_cell/dense_1/MatMul/ReadVariableOp2^while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp1^while/my_lstm_cell/dense_2/MatMul/ReadVariableOp2^while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp1^while/my_lstm_cell/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"z
:while_my_lstm_cell_dense_1_biasadd_readvariableop_resource<while_my_lstm_cell_dense_1_biasadd_readvariableop_resource_0"x
9while_my_lstm_cell_dense_1_matmul_readvariableop_resource;while_my_lstm_cell_dense_1_matmul_readvariableop_resource_0"z
:while_my_lstm_cell_dense_2_biasadd_readvariableop_resource<while_my_lstm_cell_dense_2_biasadd_readvariableop_resource_0"x
9while_my_lstm_cell_dense_2_matmul_readvariableop_resource;while_my_lstm_cell_dense_2_matmul_readvariableop_resource_0"z
:while_my_lstm_cell_dense_3_biasadd_readvariableop_resource<while_my_lstm_cell_dense_3_biasadd_readvariableop_resource_0"x
9while_my_lstm_cell_dense_3_matmul_readvariableop_resource;while_my_lstm_cell_dense_3_matmul_readvariableop_resource_0"v
8while_my_lstm_cell_dense_biasadd_readvariableop_resource:while_my_lstm_cell_dense_biasadd_readvariableop_resource_0"t
7while_my_lstm_cell_dense_matmul_readvariableop_resource9while_my_lstm_cell_dense_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"จ
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :?????????:?????????: : : : : : : : : : 2b
/while/my_lstm_cell/dense/BiasAdd/ReadVariableOp/while/my_lstm_cell/dense/BiasAdd/ReadVariableOp2`
.while/my_lstm_cell/dense/MatMul/ReadVariableOp.while/my_lstm_cell/dense/MatMul/ReadVariableOp2f
1while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp1while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp2d
0while/my_lstm_cell/dense_1/MatMul/ReadVariableOp0while/my_lstm_cell/dense_1/MatMul/ReadVariableOp2f
1while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp1while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp2d
0while/my_lstm_cell/dense_2/MatMul/ReadVariableOp0while/my_lstm_cell/dense_2/MatMul/ReadVariableOp2f
1while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp1while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp2d
0while/my_lstm_cell/dense_3/MatMul/ReadVariableOp0while/my_lstm_cell/dense_3/MatMul/ReadVariableOp: 
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
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
ฑ)


while_body_16301
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_my_lstm_cell_16325_0:'(
while_my_lstm_cell_16327_0:,
while_my_lstm_cell_16329_0:'(
while_my_lstm_cell_16331_0:,
while_my_lstm_cell_16333_0:'(
while_my_lstm_cell_16335_0:,
while_my_lstm_cell_16337_0:'(
while_my_lstm_cell_16339_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_my_lstm_cell_16325:'&
while_my_lstm_cell_16327:*
while_my_lstm_cell_16329:'&
while_my_lstm_cell_16331:*
while_my_lstm_cell_16333:'&
while_my_lstm_cell_16335:*
while_my_lstm_cell_16337:'&
while_my_lstm_cell_16339:ข*while/my_lstm_cell/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ฆ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0ล
*while/my_lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_my_lstm_cell_16325_0while_my_lstm_cell_16327_0while_my_lstm_cell_16329_0while_my_lstm_cell_16331_0while_my_lstm_cell_16333_0while_my_lstm_cell_16335_0while_my_lstm_cell_16337_0while_my_lstm_cell_16339_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_my_lstm_cell_layer_call_and_return_conditional_losses_15846?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/my_lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:้่าM
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
: 
while/Identity_4Identity3while/my_lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????
while/Identity_5Identity3while/my_lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:?????????y

while/NoOpNoOp+^while/my_lstm_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_my_lstm_cell_16325while_my_lstm_cell_16325_0"6
while_my_lstm_cell_16327while_my_lstm_cell_16327_0"6
while_my_lstm_cell_16329while_my_lstm_cell_16329_0"6
while_my_lstm_cell_16331while_my_lstm_cell_16331_0"6
while_my_lstm_cell_16333while_my_lstm_cell_16333_0"6
while_my_lstm_cell_16335while_my_lstm_cell_16335_0"6
while_my_lstm_cell_16337while_my_lstm_cell_16337_0"6
while_my_lstm_cell_16339while_my_lstm_cell_16339_0"0
while_strided_slice_1while_strided_slice_1_0"จ
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :?????????:?????????: : : : : : : : : : 2X
*while/my_lstm_cell/StatefulPartitionedCall*while/my_lstm_cell/StatefulPartitionedCall: 
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
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
อ

ว
while_cond_18896
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_18896___redundant_placeholder03
/while_while_cond_18896___redundant_placeholder13
/while_while_cond_18896___redundant_placeholder23
/while_while_cond_18896___redundant_placeholder33
/while_while_cond_18896___redundant_placeholder43
/while_while_cond_18896___redundant_placeholder53
/while_while_cond_18896___redundant_placeholder63
/while_while_cond_18896___redundant_placeholder73
/while_while_cond_18896___redundant_placeholder8
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
T: : : : :?????????:?????????: :::::::::: 
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
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
:
ฌ
>__inference_rnn_layer_call_and_return_conditional_losses_16650

inputs$
my_lstm_cell_16539:' 
my_lstm_cell_16541:$
my_lstm_cell_16543:' 
my_lstm_cell_16545:$
my_lstm_cell_16547:' 
my_lstm_cell_16549:$
my_lstm_cell_16551:' 
my_lstm_cell_16553:
identityข$my_lstm_cell/StatefulPartitionedCallขwhile;
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
valueB:ั
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
:?????????R
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
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????D
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
valueB:?
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
?????????ด
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่า
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   เ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่า_
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
valueB:้
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask฿
$my_lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0my_lstm_cell_16539my_lstm_cell_16541my_lstm_cell_16543my_lstm_cell_16545my_lstm_cell_16547my_lstm_cell_16549my_lstm_cell_16551my_lstm_cell_16553*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_my_lstm_cell_layer_call_and_return_conditional_losses_15846n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ธ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่าF
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
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ด
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0my_lstm_cell_16539my_lstm_cell_16541my_lstm_cell_16543my_lstm_cell_16545my_lstm_cell_16547my_lstm_cell_16549my_lstm_cell_16551my_lstm_cell_16553*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( *
bodyR
while_body_16562*
condR
while_cond_16561*U
output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ย
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
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
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:?????????u
NoOpNoOp%^my_lstm_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2L
$my_lstm_cell/StatefulPartitionedCall$my_lstm_cell/StatefulPartitionedCall2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
ืM
ม
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_18270	
inputF
,conv2d_conv2d_conv2d_readvariableop_resource:G
9conv2d_squeeze_batch_dims_biasadd_readvariableop_resource:H
.conv2d_1_conv2d_conv2d_readvariableop_resource:I
;conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource:
identityข#conv2d/Conv2D/Conv2D/ReadVariableOpข0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOpข%conv2d_1/Conv2D/Conv2D/ReadVariableOpข2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOpH
conv2d/Conv2D/ShapeShapeinput*
T0*
_output_shapes
:k
!conv2d/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
#conv2d/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????m
#conv2d/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv2d/Conv2D/strided_sliceStridedSliceconv2d/Conv2D/Shape:output:0*conv2d/Conv2D/strided_slice/stack:output:0,conv2d/Conv2D/strided_slice/stack_1:output:0,conv2d/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskt
conv2d/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         
conv2d/Conv2D/ReshapeReshapeinput$conv2d/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????
#conv2d/Conv2D/Conv2D/ReadVariableOpReadVariableOp,conv2d_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0อ
conv2d/Conv2D/Conv2DConv2Dconv2d/Conv2D/Reshape:output:0+conv2d/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
r
conv2d/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         d
conv2d/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????ภ
conv2d/Conv2D/concatConcatV2$conv2d/Conv2D/strided_slice:output:0&conv2d/Conv2D/concat/values_1:output:0"conv2d/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:
conv2d/Conv2D/Reshape_1Reshapeconv2d/Conv2D/Conv2D:output:0conv2d/Conv2D/concat:output:0*
T0*3
_output_shapes!
:?????????o
conv2d/squeeze_batch_dims/ShapeShape conv2d/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:w
-conv2d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
/conv2d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????y
/conv2d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ั
'conv2d/squeeze_batch_dims/strided_sliceStridedSlice(conv2d/squeeze_batch_dims/Shape:output:06conv2d/squeeze_batch_dims/strided_slice/stack:output:08conv2d/squeeze_batch_dims/strided_slice/stack_1:output:08conv2d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
'conv2d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         บ
!conv2d/squeeze_batch_dims/ReshapeReshape conv2d/Conv2D/Reshape_1:output:00conv2d/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????ฆ
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp9conv2d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ฬ
!conv2d/squeeze_batch_dims/BiasAddBiasAdd*conv2d/squeeze_batch_dims/Reshape:output:08conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????~
)conv2d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         p
%conv2d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????๐
 conv2d/squeeze_batch_dims/concatConcatV20conv2d/squeeze_batch_dims/strided_slice:output:02conv2d/squeeze_batch_dims/concat/values_1:output:0.conv2d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:ร
#conv2d/squeeze_batch_dims/Reshape_1Reshape*conv2d/squeeze_batch_dims/BiasAdd:output:0)conv2d/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:?????????q
conv2d_1/Conv2D/ShapeShape,conv2d/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:m
#conv2d_1/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_1/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????o
%conv2d_1/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv2d_1/Conv2D/strided_sliceStridedSliceconv2d_1/Conv2D/Shape:output:0,conv2d_1/Conv2D/strided_slice/stack:output:0.conv2d_1/Conv2D/strided_slice/stack_1:output:0.conv2d_1/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_1/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ฒ
conv2d_1/Conv2D/ReshapeReshape,conv2d/squeeze_batch_dims/Reshape_1:output:0&conv2d_1/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????
%conv2d_1/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ำ
conv2d_1/Conv2D/Conv2DConv2D conv2d_1/Conv2D/Reshape:output:0-conv2d_1/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
t
conv2d_1/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         f
conv2d_1/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????ศ
conv2d_1/Conv2D/concatConcatV2&conv2d_1/Conv2D/strided_slice:output:0(conv2d_1/Conv2D/concat/values_1:output:0$conv2d_1/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:ค
conv2d_1/Conv2D/Reshape_1Reshapeconv2d_1/Conv2D/Conv2D:output:0conv2d_1/Conv2D/concat:output:0*
T0*3
_output_shapes!
:?????????s
!conv2d_1/squeeze_batch_dims/ShapeShape"conv2d_1/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_1/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1conv2d_1/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????{
1conv2d_1/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)conv2d_1/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_1/squeeze_batch_dims/Shape:output:08conv2d_1/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_1/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_1/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
)conv2d_1/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ภ
#conv2d_1/squeeze_batch_dims/ReshapeReshape"conv2d_1/Conv2D/Reshape_1:output:02conv2d_1/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????ช
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0า
#conv2d_1/squeeze_batch_dims/BiasAddBiasAdd,conv2d_1/squeeze_batch_dims/Reshape:output:0:conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
+conv2d_1/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         r
'conv2d_1/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????๘
"conv2d_1/squeeze_batch_dims/concatConcatV22conv2d_1/squeeze_batch_dims/strided_slice:output:04conv2d_1/squeeze_batch_dims/concat/values_1:output:00conv2d_1/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:ษ
%conv2d_1/squeeze_batch_dims/Reshape_1Reshape,conv2d_1/squeeze_batch_dims/BiasAdd:output:0+conv2d_1/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:?????????
IdentityIdentity.conv2d_1/squeeze_batch_dims/Reshape_1:output:0^NoOp*
T0*3
_output_shapes!
:??????????
NoOpNoOp$^conv2d/Conv2D/Conv2D/ReadVariableOp1^conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_1/Conv2D/Conv2D/ReadVariableOp3^conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : 2J
#conv2d/Conv2D/Conv2D/ReadVariableOp#conv2d/Conv2D/Conv2D/ReadVariableOp2d
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_1/Conv2D/Conv2D/ReadVariableOp%conv2d_1/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp:Z V
3
_output_shapes!
:?????????

_user_specified_nameinput
ย	
ฒ
#__inference_rnn_layer_call_fn_18458

inputs
unknown:'
	unknown_0:
	unknown_1:'
	unknown_2:
	unknown_3:'
	unknown_4:
	unknown_5:'
	unknown_6:
identityขStatefulPartitionedCallฅ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_16389s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
า
g
K__inference_time_distributed_layer_call_and_return_conditional_losses_18308

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
valueB:ั
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
valueB"????         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ฃ
global_average_pooling2d/MeanMeanReshape:output:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape&global_average_pooling2d/Mean:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????g
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&??????????????????:d `
<
_output_shapes*
(:&??????????????????
 
_user_specified_nameinputs
ๅ
ฟ
rnn_while_cond_17338$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3&
"rnn_while_less_rnn_strided_slice_1;
7rnn_while_rnn_while_cond_17338___redundant_placeholder0;
7rnn_while_rnn_while_cond_17338___redundant_placeholder1;
7rnn_while_rnn_while_cond_17338___redundant_placeholder2;
7rnn_while_rnn_while_cond_17338___redundant_placeholder3;
7rnn_while_rnn_while_cond_17338___redundant_placeholder4;
7rnn_while_rnn_while_cond_17338___redundant_placeholder5;
7rnn_while_rnn_while_cond_17338___redundant_placeholder6;
7rnn_while_rnn_while_cond_17338___redundant_placeholder7;
7rnn_while_rnn_while_cond_17338___redundant_placeholder8
rnn_while_identity
r
rnn/while/LessLessrnn_while_placeholder"rnn_while_less_rnn_strided_slice_1*
T0*
_output_shapes
: S
rnn/while/IdentityIdentityrnn/while/Less:z:0*
T0
*
_output_shapes
: "1
rnn_while_identityrnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :?????????:?????????: :::::::::: 
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
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
ืM
ม
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_18210	
inputF
,conv2d_conv2d_conv2d_readvariableop_resource:G
9conv2d_squeeze_batch_dims_biasadd_readvariableop_resource:H
.conv2d_1_conv2d_conv2d_readvariableop_resource:I
;conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource:
identityข#conv2d/Conv2D/Conv2D/ReadVariableOpข0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOpข%conv2d_1/Conv2D/Conv2D/ReadVariableOpข2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOpH
conv2d/Conv2D/ShapeShapeinput*
T0*
_output_shapes
:k
!conv2d/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
#conv2d/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????m
#conv2d/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv2d/Conv2D/strided_sliceStridedSliceconv2d/Conv2D/Shape:output:0*conv2d/Conv2D/strided_slice/stack:output:0,conv2d/Conv2D/strided_slice/stack_1:output:0,conv2d/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskt
conv2d/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         
conv2d/Conv2D/ReshapeReshapeinput$conv2d/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????
#conv2d/Conv2D/Conv2D/ReadVariableOpReadVariableOp,conv2d_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0อ
conv2d/Conv2D/Conv2DConv2Dconv2d/Conv2D/Reshape:output:0+conv2d/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
r
conv2d/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         d
conv2d/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????ภ
conv2d/Conv2D/concatConcatV2$conv2d/Conv2D/strided_slice:output:0&conv2d/Conv2D/concat/values_1:output:0"conv2d/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:
conv2d/Conv2D/Reshape_1Reshapeconv2d/Conv2D/Conv2D:output:0conv2d/Conv2D/concat:output:0*
T0*3
_output_shapes!
:?????????o
conv2d/squeeze_batch_dims/ShapeShape conv2d/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:w
-conv2d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
/conv2d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????y
/conv2d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ั
'conv2d/squeeze_batch_dims/strided_sliceStridedSlice(conv2d/squeeze_batch_dims/Shape:output:06conv2d/squeeze_batch_dims/strided_slice/stack:output:08conv2d/squeeze_batch_dims/strided_slice/stack_1:output:08conv2d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
'conv2d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         บ
!conv2d/squeeze_batch_dims/ReshapeReshape conv2d/Conv2D/Reshape_1:output:00conv2d/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????ฆ
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp9conv2d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ฬ
!conv2d/squeeze_batch_dims/BiasAddBiasAdd*conv2d/squeeze_batch_dims/Reshape:output:08conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????~
)conv2d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         p
%conv2d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????๐
 conv2d/squeeze_batch_dims/concatConcatV20conv2d/squeeze_batch_dims/strided_slice:output:02conv2d/squeeze_batch_dims/concat/values_1:output:0.conv2d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:ร
#conv2d/squeeze_batch_dims/Reshape_1Reshape*conv2d/squeeze_batch_dims/BiasAdd:output:0)conv2d/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:?????????q
conv2d_1/Conv2D/ShapeShape,conv2d/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:m
#conv2d_1/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_1/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????o
%conv2d_1/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv2d_1/Conv2D/strided_sliceStridedSliceconv2d_1/Conv2D/Shape:output:0,conv2d_1/Conv2D/strided_slice/stack:output:0.conv2d_1/Conv2D/strided_slice/stack_1:output:0.conv2d_1/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_1/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ฒ
conv2d_1/Conv2D/ReshapeReshape,conv2d/squeeze_batch_dims/Reshape_1:output:0&conv2d_1/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????
%conv2d_1/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ำ
conv2d_1/Conv2D/Conv2DConv2D conv2d_1/Conv2D/Reshape:output:0-conv2d_1/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
t
conv2d_1/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         f
conv2d_1/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????ศ
conv2d_1/Conv2D/concatConcatV2&conv2d_1/Conv2D/strided_slice:output:0(conv2d_1/Conv2D/concat/values_1:output:0$conv2d_1/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:ค
conv2d_1/Conv2D/Reshape_1Reshapeconv2d_1/Conv2D/Conv2D:output:0conv2d_1/Conv2D/concat:output:0*
T0*3
_output_shapes!
:?????????s
!conv2d_1/squeeze_batch_dims/ShapeShape"conv2d_1/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_1/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1conv2d_1/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????{
1conv2d_1/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)conv2d_1/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_1/squeeze_batch_dims/Shape:output:08conv2d_1/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_1/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_1/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
)conv2d_1/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ภ
#conv2d_1/squeeze_batch_dims/ReshapeReshape"conv2d_1/Conv2D/Reshape_1:output:02conv2d_1/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????ช
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0า
#conv2d_1/squeeze_batch_dims/BiasAddBiasAdd,conv2d_1/squeeze_batch_dims/Reshape:output:0:conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
+conv2d_1/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         r
'conv2d_1/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????๘
"conv2d_1/squeeze_batch_dims/concatConcatV22conv2d_1/squeeze_batch_dims/strided_slice:output:04conv2d_1/squeeze_batch_dims/concat/values_1:output:00conv2d_1/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:ษ
%conv2d_1/squeeze_batch_dims/Reshape_1Reshape,conv2d_1/squeeze_batch_dims/BiasAdd:output:0+conv2d_1/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:?????????
IdentityIdentity.conv2d_1/squeeze_batch_dims/Reshape_1:output:0^NoOp*
T0*3
_output_shapes!
:??????????
NoOpNoOp$^conv2d/Conv2D/Conv2D/ReadVariableOp1^conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_1/Conv2D/Conv2D/ReadVariableOp3^conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : 2J
#conv2d/Conv2D/Conv2D/ReadVariableOp#conv2d/Conv2D/Conv2D/ReadVariableOp2d
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_1/Conv2D/Conv2D/ReadVariableOp%conv2d_1/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp:Z V
3
_output_shapes!
:?????????

_user_specified_nameinput
ฆ
?
H__inference_my_lstm_model_layer_call_and_return_conditional_losses_16829
x,
my_cnn_block_16794: 
my_cnn_block_16796:,
my_cnn_block_16798: 
my_cnn_block_16800:
	rnn_16806:'
	rnn_16808:
	rnn_16810:'
	rnn_16812:
	rnn_16814:'
	rnn_16816:
	rnn_16818:'
	rnn_16820:
dense_4_16823:
dense_4_16825:
identityขdense_4/StatefulPartitionedCallข$my_cnn_block/StatefulPartitionedCallขrnn/StatefulPartitionedCallฐ
$my_cnn_block/StatefulPartitionedCallStatefulPartitionedCallxmy_cnn_block_16794my_cnn_block_16796my_cnn_block_16798my_cnn_block_16800*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_16745๒
 time_distributed/PartitionedCallPartitionedCall-my_cnn_block/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_15765w
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ต
time_distributed/ReshapeReshape-my_cnn_block/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????ฮ
rnn/StatefulPartitionedCallStatefulPartitionedCall)time_distributed/PartitionedCall:output:0	rnn_16806	rnn_16808	rnn_16810	rnn_16812	rnn_16814	rnn_16816	rnn_16818	rnn_16820*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_16650
dense_4/StatefulPartitionedCallStatefulPartitionedCall$rnn/StatefulPartitionedCall:output:0dense_4_16823dense_4_16825*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_16437{
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????ญ
NoOpNoOp ^dense_4/StatefulPartitionedCall%^my_cnn_block/StatefulPartitionedCall^rnn/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2L
$my_cnn_block/StatefulPartitionedCall$my_cnn_block/StatefulPartitionedCall2:
rnn/StatefulPartitionedCallrnn/StatefulPartitionedCall:V R
3
_output_shapes!
:?????????

_user_specified_namex
อ

ว
while_cond_16300
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_16300___redundant_placeholder03
/while_while_cond_16300___redundant_placeholder13
/while_while_cond_16300___redundant_placeholder23
/while_while_cond_16300___redundant_placeholder33
/while_while_cond_16300___redundant_placeholder43
/while_while_cond_16300___redundant_placeholder53
/while_while_cond_16300___redundant_placeholder63
/while_while_cond_16300___redundant_placeholder73
/while_while_cond_16300___redundant_placeholder8
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
T: : : : :?????????:?????????: :::::::::: 
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
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
์	
ด
#__inference_rnn_layer_call_fn_18437
inputs_0
unknown:'
	unknown_0:
	unknown_1:'
	unknown_2:
	unknown_3:'
	unknown_4:
	unknown_5:'
	unknown_6:
identityขStatefulPartitionedCallฐ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_16149|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
ง
?
 __inference__wrapped_model_15711
input_1-
my_lstm_model_15681:!
my_lstm_model_15683:-
my_lstm_model_15685:!
my_lstm_model_15687:%
my_lstm_model_15689:'!
my_lstm_model_15691:%
my_lstm_model_15693:'!
my_lstm_model_15695:%
my_lstm_model_15697:'!
my_lstm_model_15699:%
my_lstm_model_15701:'!
my_lstm_model_15703:%
my_lstm_model_15705:!
my_lstm_model_15707:
identityข%my_lstm_model/StatefulPartitionedCall่
%my_lstm_model/StatefulPartitionedCallStatefulPartitionedCallinput_1my_lstm_model_15681my_lstm_model_15683my_lstm_model_15685my_lstm_model_15687my_lstm_model_15689my_lstm_model_15691my_lstm_model_15693my_lstm_model_15695my_lstm_model_15697my_lstm_model_15699my_lstm_model_15701my_lstm_model_15703my_lstm_model_15705my_lstm_model_15707*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *
fR
__inference_call_15680
IdentityIdentity.my_lstm_model/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????n
NoOpNoOp&^my_lstm_model/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : 2N
%my_lstm_model/StatefulPartitionedCall%my_lstm_model/StatefulPartitionedCall:\ X
3
_output_shapes!
:?????????
!
_user_specified_name	input_1
ห
ใ!
!__inference__traced_restore_19553
file_prefix8
assignvariableop_conv2d_kernel:,
assignvariableop_1_conv2d_bias:<
"assignvariableop_2_conv2d_1_kernel:.
 assignvariableop_3_conv2d_1_bias:B
0assignvariableop_4_rnn_my_lstm_cell_dense_kernel:'<
.assignvariableop_5_rnn_my_lstm_cell_dense_bias:D
2assignvariableop_6_rnn_my_lstm_cell_dense_1_kernel:'>
0assignvariableop_7_rnn_my_lstm_cell_dense_1_bias:D
2assignvariableop_8_rnn_my_lstm_cell_dense_2_kernel:'>
0assignvariableop_9_rnn_my_lstm_cell_dense_2_bias:E
3assignvariableop_10_rnn_my_lstm_cell_dense_3_kernel:'?
1assignvariableop_11_rnn_my_lstm_cell_dense_3_bias:4
"assignvariableop_12_dense_4_kernel:.
 assignvariableop_13_dense_4_bias:%
assignvariableop_14_total_1: %
assignvariableop_15_count_1: #
assignvariableop_16_total: #
assignvariableop_17_count: '
assignvariableop_18_adam_iter:	 )
assignvariableop_19_adam_beta_1: )
assignvariableop_20_adam_beta_2: (
assignvariableop_21_adam_decay: 0
&assignvariableop_22_adam_learning_rate: B
(assignvariableop_23_adam_conv2d_kernel_m:4
&assignvariableop_24_adam_conv2d_bias_m:D
*assignvariableop_25_adam_conv2d_1_kernel_m:6
(assignvariableop_26_adam_conv2d_1_bias_m:J
8assignvariableop_27_adam_rnn_my_lstm_cell_dense_kernel_m:'D
6assignvariableop_28_adam_rnn_my_lstm_cell_dense_bias_m:L
:assignvariableop_29_adam_rnn_my_lstm_cell_dense_1_kernel_m:'F
8assignvariableop_30_adam_rnn_my_lstm_cell_dense_1_bias_m:L
:assignvariableop_31_adam_rnn_my_lstm_cell_dense_2_kernel_m:'F
8assignvariableop_32_adam_rnn_my_lstm_cell_dense_2_bias_m:L
:assignvariableop_33_adam_rnn_my_lstm_cell_dense_3_kernel_m:'F
8assignvariableop_34_adam_rnn_my_lstm_cell_dense_3_bias_m:;
)assignvariableop_35_adam_dense_4_kernel_m:5
'assignvariableop_36_adam_dense_4_bias_m:B
(assignvariableop_37_adam_conv2d_kernel_v:4
&assignvariableop_38_adam_conv2d_bias_v:D
*assignvariableop_39_adam_conv2d_1_kernel_v:6
(assignvariableop_40_adam_conv2d_1_bias_v:J
8assignvariableop_41_adam_rnn_my_lstm_cell_dense_kernel_v:'D
6assignvariableop_42_adam_rnn_my_lstm_cell_dense_bias_v:L
:assignvariableop_43_adam_rnn_my_lstm_cell_dense_1_kernel_v:'F
8assignvariableop_44_adam_rnn_my_lstm_cell_dense_1_bias_v:L
:assignvariableop_45_adam_rnn_my_lstm_cell_dense_2_kernel_v:'F
8assignvariableop_46_adam_rnn_my_lstm_cell_dense_2_bias_v:L
:assignvariableop_47_adam_rnn_my_lstm_cell_dense_3_kernel_v:'F
8assignvariableop_48_adam_rnn_my_lstm_cell_dense_3_bias_v:;
)assignvariableop_49_adam_dense_4_kernel_v:5
'assignvariableop_50_adam_dense_4_bias_v:
identity_52ขAssignVariableOpขAssignVariableOp_1ขAssignVariableOp_10ขAssignVariableOp_11ขAssignVariableOp_12ขAssignVariableOp_13ขAssignVariableOp_14ขAssignVariableOp_15ขAssignVariableOp_16ขAssignVariableOp_17ขAssignVariableOp_18ขAssignVariableOp_19ขAssignVariableOp_2ขAssignVariableOp_20ขAssignVariableOp_21ขAssignVariableOp_22ขAssignVariableOp_23ขAssignVariableOp_24ขAssignVariableOp_25ขAssignVariableOp_26ขAssignVariableOp_27ขAssignVariableOp_28ขAssignVariableOp_29ขAssignVariableOp_3ขAssignVariableOp_30ขAssignVariableOp_31ขAssignVariableOp_32ขAssignVariableOp_33ขAssignVariableOp_34ขAssignVariableOp_35ขAssignVariableOp_36ขAssignVariableOp_37ขAssignVariableOp_38ขAssignVariableOp_39ขAssignVariableOp_4ขAssignVariableOp_40ขAssignVariableOp_41ขAssignVariableOp_42ขAssignVariableOp_43ขAssignVariableOp_44ขAssignVariableOp_45ขAssignVariableOp_46ขAssignVariableOp_47ขAssignVariableOp_48ขAssignVariableOp_49ขAssignVariableOp_5ขAssignVariableOp_50ขAssignVariableOp_6ขAssignVariableOp_7ขAssignVariableOp_8ขAssignVariableOp_9ย
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*่
value?B?4B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHุ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ฅ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ๆ
_output_shapesำ
ะ::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp0assignvariableop_4_rnn_my_lstm_cell_dense_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp.assignvariableop_5_rnn_my_lstm_cell_dense_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:ก
AssignVariableOp_6AssignVariableOp2assignvariableop_6_rnn_my_lstm_cell_dense_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp0assignvariableop_7_rnn_my_lstm_cell_dense_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:ก
AssignVariableOp_8AssignVariableOp2assignvariableop_8_rnn_my_lstm_cell_dense_2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp0assignvariableop_9_rnn_my_lstm_cell_dense_2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:ค
AssignVariableOp_10AssignVariableOp3assignvariableop_10_rnn_my_lstm_cell_dense_3_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:ข
AssignVariableOp_11AssignVariableOp1assignvariableop_11_rnn_my_lstm_cell_dense_3_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_4_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_4_biasIdentity_13:output:0"/device:CPU:0*
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
:
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_conv2d_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_conv2d_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv2d_1_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv2d_1_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:ฉ
AssignVariableOp_27AssignVariableOp8assignvariableop_27_adam_rnn_my_lstm_cell_dense_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:ง
AssignVariableOp_28AssignVariableOp6assignvariableop_28_adam_rnn_my_lstm_cell_dense_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:ซ
AssignVariableOp_29AssignVariableOp:assignvariableop_29_adam_rnn_my_lstm_cell_dense_1_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:ฉ
AssignVariableOp_30AssignVariableOp8assignvariableop_30_adam_rnn_my_lstm_cell_dense_1_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:ซ
AssignVariableOp_31AssignVariableOp:assignvariableop_31_adam_rnn_my_lstm_cell_dense_2_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:ฉ
AssignVariableOp_32AssignVariableOp8assignvariableop_32_adam_rnn_my_lstm_cell_dense_2_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:ซ
AssignVariableOp_33AssignVariableOp:assignvariableop_33_adam_rnn_my_lstm_cell_dense_3_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:ฉ
AssignVariableOp_34AssignVariableOp8assignvariableop_34_adam_rnn_my_lstm_cell_dense_3_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_dense_4_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_dense_4_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_conv2d_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp&assignvariableop_38_adam_conv2d_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_conv2d_1_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_conv2d_1_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:ฉ
AssignVariableOp_41AssignVariableOp8assignvariableop_41_adam_rnn_my_lstm_cell_dense_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:ง
AssignVariableOp_42AssignVariableOp6assignvariableop_42_adam_rnn_my_lstm_cell_dense_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:ซ
AssignVariableOp_43AssignVariableOp:assignvariableop_43_adam_rnn_my_lstm_cell_dense_1_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:ฉ
AssignVariableOp_44AssignVariableOp8assignvariableop_44_adam_rnn_my_lstm_cell_dense_1_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:ซ
AssignVariableOp_45AssignVariableOp:assignvariableop_45_adam_rnn_my_lstm_cell_dense_2_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:ฉ
AssignVariableOp_46AssignVariableOp8assignvariableop_46_adam_rnn_my_lstm_cell_dense_2_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:ซ
AssignVariableOp_47AssignVariableOp:assignvariableop_47_adam_rnn_my_lstm_cell_dense_3_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:ฉ
AssignVariableOp_48AssignVariableOp8assignvariableop_48_adam_rnn_my_lstm_cell_dense_3_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_49AssignVariableOp)assignvariableop_49_adam_dense_4_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_50AssignVariableOp'assignvariableop_50_adam_dense_4_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ฑ	
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
๓-
เ
G__inference_my_lstm_cell_layer_call_and_return_conditional_losses_15846

inputs

states
states_16
$dense_matmul_readvariableop_resource:'3
%dense_biasadd_readvariableop_resource:8
&dense_1_matmul_readvariableop_resource:'5
'dense_1_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:'5
'dense_2_biasadd_readvariableop_resource:8
&dense_3_matmul_readvariableop_resource:'5
'dense_3_biasadd_readvariableop_resource:
identity

identity_1

identity_2ขdense/BiasAdd/ReadVariableOpขdense/MatMul/ReadVariableOpขdense_1/BiasAdd/ReadVariableOpขdense_1/MatMul/ReadVariableOpขdense_2/BiasAdd/ReadVariableOpขdense_2/MatMul/ReadVariableOpขdense_3/BiasAdd/ReadVariableOpขdense_3/MatMul/ReadVariableOp[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatenate_1/concatConcatV2inputsstates"concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????'
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0
dense/MatMulMatMulconcatenate_1/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????b
multiply/mulMulstates_1dense/Sigmoid:y:0*
T0*'
_output_shapes
:?????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0
dense_1/MatMulMatMulconcatenate_1/concat:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0
dense_2/MatMulMatMulconcatenate_1/concat:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`
dense_2/TanhTanhdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????n
multiply/mul_1Muldense_1/Sigmoid:y:0dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????j
	add_1/addAddV2multiply/mul:z:0multiply/mul_1:z:0*
T0*'
_output_shapes
:?????????
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0
dense_3/MatMulMatMulconcatenate_1/concat:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????X
activation/TanhTanhadd_1/add:z:0*
T0*'
_output_shapes
:?????????q
multiply/mul_2Mulactivation/Tanh:y:0dense_3/Sigmoid:y:0*
T0*'
_output_shapes
:?????????a
IdentityIdentitymultiply/mul_2:z:0^NoOp*
T0*'
_output_shapes
:?????????c

Identity_1Identitymultiply/mul_2:z:0^NoOp*
T0*'
_output_shapes
:?????????^

Identity_2Identityadd_1/add:z:0^NoOp*
T0*'
_output_shapes
:?????????ฦ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????:?????????:?????????: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_namestates:OK
'
_output_shapes
:?????????
 
_user_specified_namestates
ฑ)


while_body_16562
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_my_lstm_cell_16586_0:'(
while_my_lstm_cell_16588_0:,
while_my_lstm_cell_16590_0:'(
while_my_lstm_cell_16592_0:,
while_my_lstm_cell_16594_0:'(
while_my_lstm_cell_16596_0:,
while_my_lstm_cell_16598_0:'(
while_my_lstm_cell_16600_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_my_lstm_cell_16586:'&
while_my_lstm_cell_16588:*
while_my_lstm_cell_16590:'&
while_my_lstm_cell_16592:*
while_my_lstm_cell_16594:'&
while_my_lstm_cell_16596:*
while_my_lstm_cell_16598:'&
while_my_lstm_cell_16600:ข*while/my_lstm_cell/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ฆ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0ล
*while/my_lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_my_lstm_cell_16586_0while_my_lstm_cell_16588_0while_my_lstm_cell_16590_0while_my_lstm_cell_16592_0while_my_lstm_cell_16594_0while_my_lstm_cell_16596_0while_my_lstm_cell_16598_0while_my_lstm_cell_16600_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_my_lstm_cell_layer_call_and_return_conditional_losses_15846?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/my_lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:้่าM
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
: 
while/Identity_4Identity3while/my_lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????
while/Identity_5Identity3while/my_lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:?????????y

while/NoOpNoOp+^while/my_lstm_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_my_lstm_cell_16586while_my_lstm_cell_16586_0"6
while_my_lstm_cell_16588while_my_lstm_cell_16588_0"6
while_my_lstm_cell_16590while_my_lstm_cell_16590_0"6
while_my_lstm_cell_16592while_my_lstm_cell_16592_0"6
while_my_lstm_cell_16594while_my_lstm_cell_16594_0"6
while_my_lstm_cell_16596while_my_lstm_cell_16596_0"6
while_my_lstm_cell_16598while_my_lstm_cell_16598_0"6
while_my_lstm_cell_16600while_my_lstm_cell_16600_0"0
while_strided_slice_1while_strided_slice_1_0"จ
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :?????????:?????????: : : : : : : : : : 2X
*while/my_lstm_cell/StatefulPartitionedCall*while/my_lstm_cell/StatefulPartitionedCall: 
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
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
๒d
๕
>__inference_rnn_layer_call_and_return_conditional_losses_19001

inputsC
1my_lstm_cell_dense_matmul_readvariableop_resource:'@
2my_lstm_cell_dense_biasadd_readvariableop_resource:E
3my_lstm_cell_dense_1_matmul_readvariableop_resource:'B
4my_lstm_cell_dense_1_biasadd_readvariableop_resource:E
3my_lstm_cell_dense_2_matmul_readvariableop_resource:'B
4my_lstm_cell_dense_2_biasadd_readvariableop_resource:E
3my_lstm_cell_dense_3_matmul_readvariableop_resource:'B
4my_lstm_cell_dense_3_biasadd_readvariableop_resource:
identityข)my_lstm_cell/dense/BiasAdd/ReadVariableOpข(my_lstm_cell/dense/MatMul/ReadVariableOpข+my_lstm_cell/dense_1/BiasAdd/ReadVariableOpข*my_lstm_cell/dense_1/MatMul/ReadVariableOpข+my_lstm_cell/dense_2/BiasAdd/ReadVariableOpข*my_lstm_cell/dense_2/MatMul/ReadVariableOpข+my_lstm_cell/dense_3/BiasAdd/ReadVariableOpข*my_lstm_cell/dense_3/MatMul/ReadVariableOpขwhile;
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
valueB:ั
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
:?????????R
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
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????D
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
valueB:?
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
?????????ด
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่า
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   เ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่า_
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
valueB:้
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskh
&my_lstm_cell/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ร
!my_lstm_cell/concatenate_1/concatConcatV2strided_slice_2:output:0zeros:output:0/my_lstm_cell/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????'
(my_lstm_cell/dense/MatMul/ReadVariableOpReadVariableOp1my_lstm_cell_dense_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0ณ
my_lstm_cell/dense/MatMulMatMul*my_lstm_cell/concatenate_1/concat:output:00my_lstm_cell/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
)my_lstm_cell/dense/BiasAdd/ReadVariableOpReadVariableOp2my_lstm_cell_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ฏ
my_lstm_cell/dense/BiasAddBiasAdd#my_lstm_cell/dense/MatMul:product:01my_lstm_cell/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????|
my_lstm_cell/dense/SigmoidSigmoid#my_lstm_cell/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
my_lstm_cell/multiply/mulMulzeros_1:output:0my_lstm_cell/dense/Sigmoid:y:0*
T0*'
_output_shapes
:?????????
*my_lstm_cell/dense_1/MatMul/ReadVariableOpReadVariableOp3my_lstm_cell_dense_1_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0ท
my_lstm_cell/dense_1/MatMulMatMul*my_lstm_cell/concatenate_1/concat:output:02my_lstm_cell/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
+my_lstm_cell/dense_1/BiasAdd/ReadVariableOpReadVariableOp4my_lstm_cell_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ต
my_lstm_cell/dense_1/BiasAddBiasAdd%my_lstm_cell/dense_1/MatMul:product:03my_lstm_cell/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
my_lstm_cell/dense_1/SigmoidSigmoid%my_lstm_cell/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
*my_lstm_cell/dense_2/MatMul/ReadVariableOpReadVariableOp3my_lstm_cell_dense_2_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0ท
my_lstm_cell/dense_2/MatMulMatMul*my_lstm_cell/concatenate_1/concat:output:02my_lstm_cell/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
+my_lstm_cell/dense_2/BiasAdd/ReadVariableOpReadVariableOp4my_lstm_cell_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ต
my_lstm_cell/dense_2/BiasAddBiasAdd%my_lstm_cell/dense_2/MatMul:product:03my_lstm_cell/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????z
my_lstm_cell/dense_2/TanhTanh%my_lstm_cell/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
my_lstm_cell/multiply/mul_1Mul my_lstm_cell/dense_1/Sigmoid:y:0my_lstm_cell/dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????
my_lstm_cell/add_1/addAddV2my_lstm_cell/multiply/mul:z:0my_lstm_cell/multiply/mul_1:z:0*
T0*'
_output_shapes
:?????????
*my_lstm_cell/dense_3/MatMul/ReadVariableOpReadVariableOp3my_lstm_cell_dense_3_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0ท
my_lstm_cell/dense_3/MatMulMatMul*my_lstm_cell/concatenate_1/concat:output:02my_lstm_cell/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
+my_lstm_cell/dense_3/BiasAdd/ReadVariableOpReadVariableOp4my_lstm_cell_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ต
my_lstm_cell/dense_3/BiasAddBiasAdd%my_lstm_cell/dense_3/MatMul:product:03my_lstm_cell/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
my_lstm_cell/dense_3/SigmoidSigmoid%my_lstm_cell/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????r
my_lstm_cell/activation/TanhTanhmy_lstm_cell/add_1/add:z:0*
T0*'
_output_shapes
:?????????
my_lstm_cell/multiply/mul_2Mul my_lstm_cell/activation/Tanh:y:0 my_lstm_cell/dense_3/Sigmoid:y:0*
T0*'
_output_shapes
:?????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ธ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่าF
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
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ผ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01my_lstm_cell_dense_matmul_readvariableop_resource2my_lstm_cell_dense_biasadd_readvariableop_resource3my_lstm_cell_dense_1_matmul_readvariableop_resource4my_lstm_cell_dense_1_biasadd_readvariableop_resource3my_lstm_cell_dense_2_matmul_readvariableop_resource4my_lstm_cell_dense_2_biasadd_readvariableop_resource3my_lstm_cell_dense_3_matmul_readvariableop_resource4my_lstm_cell_dense_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( *
bodyR
while_body_18897*
condR
while_cond_18896*U
output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ย
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
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
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:?????????ถ
NoOpNoOp*^my_lstm_cell/dense/BiasAdd/ReadVariableOp)^my_lstm_cell/dense/MatMul/ReadVariableOp,^my_lstm_cell/dense_1/BiasAdd/ReadVariableOp+^my_lstm_cell/dense_1/MatMul/ReadVariableOp,^my_lstm_cell/dense_2/BiasAdd/ReadVariableOp+^my_lstm_cell/dense_2/MatMul/ReadVariableOp,^my_lstm_cell/dense_3/BiasAdd/ReadVariableOp+^my_lstm_cell/dense_3/MatMul/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2V
)my_lstm_cell/dense/BiasAdd/ReadVariableOp)my_lstm_cell/dense/BiasAdd/ReadVariableOp2T
(my_lstm_cell/dense/MatMul/ReadVariableOp(my_lstm_cell/dense/MatMul/ReadVariableOp2Z
+my_lstm_cell/dense_1/BiasAdd/ReadVariableOp+my_lstm_cell/dense_1/BiasAdd/ReadVariableOp2X
*my_lstm_cell/dense_1/MatMul/ReadVariableOp*my_lstm_cell/dense_1/MatMul/ReadVariableOp2Z
+my_lstm_cell/dense_2/BiasAdd/ReadVariableOp+my_lstm_cell/dense_2/BiasAdd/ReadVariableOp2X
*my_lstm_cell/dense_2/MatMul/ReadVariableOp*my_lstm_cell/dense_2/MatMul/ReadVariableOp2Z
+my_lstm_cell/dense_3/BiasAdd/ReadVariableOp+my_lstm_cell/dense_3/BiasAdd/ReadVariableOp2X
*my_lstm_cell/dense_3/MatMul/ReadVariableOp*my_lstm_cell/dense_3/MatMul/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
ธ

H__inference_my_lstm_model_layer_call_and_return_conditional_losses_16931
input_1,
my_cnn_block_16896: 
my_cnn_block_16898:,
my_cnn_block_16900: 
my_cnn_block_16902:
	rnn_16908:'
	rnn_16910:
	rnn_16912:'
	rnn_16914:
	rnn_16916:'
	rnn_16918:
	rnn_16920:'
	rnn_16922:
dense_4_16925:
dense_4_16927:
identityขdense_4/StatefulPartitionedCallข$my_cnn_block/StatefulPartitionedCallขrnn/StatefulPartitionedCallถ
$my_cnn_block/StatefulPartitionedCallStatefulPartitionedCallinput_1my_cnn_block_16896my_cnn_block_16898my_cnn_block_16900my_cnn_block_16902*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_16235๒
 time_distributed/PartitionedCallPartitionedCall-my_cnn_block/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_15744w
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ต
time_distributed/ReshapeReshape-my_cnn_block/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????ฮ
rnn/StatefulPartitionedCallStatefulPartitionedCall)time_distributed/PartitionedCall:output:0	rnn_16908	rnn_16910	rnn_16912	rnn_16914	rnn_16916	rnn_16918	rnn_16920	rnn_16922*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_16389
dense_4/StatefulPartitionedCallStatefulPartitionedCall$rnn/StatefulPartitionedCall:output:0dense_4_16925dense_4_16927*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_16437{
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????ญ
NoOpNoOp ^dense_4/StatefulPartitionedCall%^my_cnn_block/StatefulPartitionedCall^rnn/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2L
$my_cnn_block/StatefulPartitionedCall$my_cnn_block/StatefulPartitionedCall2:
rnn/StatefulPartitionedCallrnn/StatefulPartitionedCall:\ X
3
_output_shapes!
:?????????
!
_user_specified_name	input_1
ฐe
๗
>__inference_rnn_layer_call_and_return_conditional_losses_18827
inputs_0C
1my_lstm_cell_dense_matmul_readvariableop_resource:'@
2my_lstm_cell_dense_biasadd_readvariableop_resource:E
3my_lstm_cell_dense_1_matmul_readvariableop_resource:'B
4my_lstm_cell_dense_1_biasadd_readvariableop_resource:E
3my_lstm_cell_dense_2_matmul_readvariableop_resource:'B
4my_lstm_cell_dense_2_biasadd_readvariableop_resource:E
3my_lstm_cell_dense_3_matmul_readvariableop_resource:'B
4my_lstm_cell_dense_3_biasadd_readvariableop_resource:
identityข)my_lstm_cell/dense/BiasAdd/ReadVariableOpข(my_lstm_cell/dense/MatMul/ReadVariableOpข+my_lstm_cell/dense_1/BiasAdd/ReadVariableOpข*my_lstm_cell/dense_1/MatMul/ReadVariableOpข+my_lstm_cell/dense_2/BiasAdd/ReadVariableOpข*my_lstm_cell/dense_2/MatMul/ReadVariableOpข+my_lstm_cell/dense_3/BiasAdd/ReadVariableOpข*my_lstm_cell/dense_3/MatMul/ReadVariableOpขwhile=
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
valueB:ั
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
:?????????R
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
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
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
valueB:?
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
?????????ด
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่า
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   เ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่า_
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
valueB:้
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskh
&my_lstm_cell/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ร
!my_lstm_cell/concatenate_1/concatConcatV2strided_slice_2:output:0zeros:output:0/my_lstm_cell/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????'
(my_lstm_cell/dense/MatMul/ReadVariableOpReadVariableOp1my_lstm_cell_dense_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0ณ
my_lstm_cell/dense/MatMulMatMul*my_lstm_cell/concatenate_1/concat:output:00my_lstm_cell/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
)my_lstm_cell/dense/BiasAdd/ReadVariableOpReadVariableOp2my_lstm_cell_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ฏ
my_lstm_cell/dense/BiasAddBiasAdd#my_lstm_cell/dense/MatMul:product:01my_lstm_cell/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????|
my_lstm_cell/dense/SigmoidSigmoid#my_lstm_cell/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
my_lstm_cell/multiply/mulMulzeros_1:output:0my_lstm_cell/dense/Sigmoid:y:0*
T0*'
_output_shapes
:?????????
*my_lstm_cell/dense_1/MatMul/ReadVariableOpReadVariableOp3my_lstm_cell_dense_1_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0ท
my_lstm_cell/dense_1/MatMulMatMul*my_lstm_cell/concatenate_1/concat:output:02my_lstm_cell/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
+my_lstm_cell/dense_1/BiasAdd/ReadVariableOpReadVariableOp4my_lstm_cell_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ต
my_lstm_cell/dense_1/BiasAddBiasAdd%my_lstm_cell/dense_1/MatMul:product:03my_lstm_cell/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
my_lstm_cell/dense_1/SigmoidSigmoid%my_lstm_cell/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
*my_lstm_cell/dense_2/MatMul/ReadVariableOpReadVariableOp3my_lstm_cell_dense_2_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0ท
my_lstm_cell/dense_2/MatMulMatMul*my_lstm_cell/concatenate_1/concat:output:02my_lstm_cell/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
+my_lstm_cell/dense_2/BiasAdd/ReadVariableOpReadVariableOp4my_lstm_cell_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ต
my_lstm_cell/dense_2/BiasAddBiasAdd%my_lstm_cell/dense_2/MatMul:product:03my_lstm_cell/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????z
my_lstm_cell/dense_2/TanhTanh%my_lstm_cell/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
my_lstm_cell/multiply/mul_1Mul my_lstm_cell/dense_1/Sigmoid:y:0my_lstm_cell/dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????
my_lstm_cell/add_1/addAddV2my_lstm_cell/multiply/mul:z:0my_lstm_cell/multiply/mul_1:z:0*
T0*'
_output_shapes
:?????????
*my_lstm_cell/dense_3/MatMul/ReadVariableOpReadVariableOp3my_lstm_cell_dense_3_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0ท
my_lstm_cell/dense_3/MatMulMatMul*my_lstm_cell/concatenate_1/concat:output:02my_lstm_cell/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
+my_lstm_cell/dense_3/BiasAdd/ReadVariableOpReadVariableOp4my_lstm_cell_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ต
my_lstm_cell/dense_3/BiasAddBiasAdd%my_lstm_cell/dense_3/MatMul:product:03my_lstm_cell/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
my_lstm_cell/dense_3/SigmoidSigmoid%my_lstm_cell/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????r
my_lstm_cell/activation/TanhTanhmy_lstm_cell/add_1/add:z:0*
T0*'
_output_shapes
:?????????
my_lstm_cell/multiply/mul_2Mul my_lstm_cell/activation/Tanh:y:0 my_lstm_cell/dense_3/Sigmoid:y:0*
T0*'
_output_shapes
:?????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ธ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่าF
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
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ผ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01my_lstm_cell_dense_matmul_readvariableop_resource2my_lstm_cell_dense_biasadd_readvariableop_resource3my_lstm_cell_dense_1_matmul_readvariableop_resource4my_lstm_cell_dense_1_biasadd_readvariableop_resource3my_lstm_cell_dense_2_matmul_readvariableop_resource4my_lstm_cell_dense_2_biasadd_readvariableop_resource3my_lstm_cell_dense_3_matmul_readvariableop_resource4my_lstm_cell_dense_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( *
bodyR
while_body_18723*
condR
while_cond_18722*U
output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ห
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
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
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :??????????????????ถ
NoOpNoOp*^my_lstm_cell/dense/BiasAdd/ReadVariableOp)^my_lstm_cell/dense/MatMul/ReadVariableOp,^my_lstm_cell/dense_1/BiasAdd/ReadVariableOp+^my_lstm_cell/dense_1/MatMul/ReadVariableOp,^my_lstm_cell/dense_2/BiasAdd/ReadVariableOp+^my_lstm_cell/dense_2/MatMul/ReadVariableOp,^my_lstm_cell/dense_3/BiasAdd/ReadVariableOp+^my_lstm_cell/dense_3/MatMul/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????????????: : : : : : : : 2V
)my_lstm_cell/dense/BiasAdd/ReadVariableOp)my_lstm_cell/dense/BiasAdd/ReadVariableOp2T
(my_lstm_cell/dense/MatMul/ReadVariableOp(my_lstm_cell/dense/MatMul/ReadVariableOp2Z
+my_lstm_cell/dense_1/BiasAdd/ReadVariableOp+my_lstm_cell/dense_1/BiasAdd/ReadVariableOp2X
*my_lstm_cell/dense_1/MatMul/ReadVariableOp*my_lstm_cell/dense_1/MatMul/ReadVariableOp2Z
+my_lstm_cell/dense_2/BiasAdd/ReadVariableOp+my_lstm_cell/dense_2/BiasAdd/ReadVariableOp2X
*my_lstm_cell/dense_2/MatMul/ReadVariableOp*my_lstm_cell/dense_2/MatMul/ReadVariableOp2Z
+my_lstm_cell/dense_3/BiasAdd/ReadVariableOp+my_lstm_cell/dense_3/BiasAdd/ReadVariableOp2X
*my_lstm_cell/dense_3/MatMul/ReadVariableOp*my_lstm_cell/dense_3/MatMul/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
ู
?
,__inference_my_cnn_block_layer_call_fn_18150	
input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_16745{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
3
_output_shapes!
:?????????

_user_specified_nameinput
้
์
-__inference_my_lstm_model_layer_call_fn_17570
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
identityขStatefulPartitionedCall๛
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
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_my_lstm_model_layer_call_and_return_conditional_losses_16829s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
3
_output_shapes!
:?????????

_user_specified_namex
ฎ

__inference_call_15680
x,
my_cnn_block_15466: 
my_cnn_block_15468:,
my_cnn_block_15470: 
my_cnn_block_15472:G
5rnn_my_lstm_cell_dense_matmul_readvariableop_resource:'D
6rnn_my_lstm_cell_dense_biasadd_readvariableop_resource:I
7rnn_my_lstm_cell_dense_1_matmul_readvariableop_resource:'F
8rnn_my_lstm_cell_dense_1_biasadd_readvariableop_resource:I
7rnn_my_lstm_cell_dense_2_matmul_readvariableop_resource:'F
8rnn_my_lstm_cell_dense_2_biasadd_readvariableop_resource:I
7rnn_my_lstm_cell_dense_3_matmul_readvariableop_resource:'F
8rnn_my_lstm_cell_dense_3_biasadd_readvariableop_resource:;
)dense_4_tensordot_readvariableop_resource:5
'dense_4_biasadd_readvariableop_resource:
identityขdense_4/BiasAdd/ReadVariableOpข dense_4/Tensordot/ReadVariableOpข$my_cnn_block/StatefulPartitionedCallข-rnn/my_lstm_cell/dense/BiasAdd/ReadVariableOpข,rnn/my_lstm_cell/dense/MatMul/ReadVariableOpข/rnn/my_lstm_cell/dense_1/BiasAdd/ReadVariableOpข.rnn/my_lstm_cell/dense_1/MatMul/ReadVariableOpข/rnn/my_lstm_cell/dense_2/BiasAdd/ReadVariableOpข.rnn/my_lstm_cell/dense_2/MatMul/ReadVariableOpข/rnn/my_lstm_cell/dense_3/BiasAdd/ReadVariableOpข.rnn/my_lstm_cell/dense_3/MatMul/ReadVariableOpข	rnn/while?
$my_cnn_block/StatefulPartitionedCallStatefulPartitionedCallxmy_cnn_block_15466my_cnn_block_15468my_cnn_block_15470my_cnn_block_15472*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *
fR
__inference_call_15465w
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ต
time_distributed/ReshapeReshape-my_cnn_block/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????
@time_distributed/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ึ
.time_distributed/global_average_pooling2d/MeanMean!time_distributed/Reshape:output:0Itime_distributed/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????u
 time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      ฟ
time_distributed/Reshape_1Reshape7time_distributed/global_average_pooling2d/Mean:output:0)time_distributed/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????y
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         น
time_distributed/Reshape_2Reshape-my_cnn_block/StatefulPartitionedCall:output:0)time_distributed/Reshape_2/shape:output:0*
T0*/
_output_shapes
:?????????\
	rnn/ShapeShape#time_distributed/Reshape_1:output:0*
T0*
_output_shapes
:a
rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ๅ
rnn/strided_sliceStridedSlicernn/Shape:output:0 rnn/strided_slice/stack:output:0"rnn/strided_slice/stack_1:output:0"rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
rnn/zeros/packedPackrnn/strided_slice:output:0rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:T
rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x
	rnn/zerosFillrnn/zeros/packed:output:0rnn/zeros/Const:output:0*
T0*'
_output_shapes
:?????????V
rnn/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
rnn/zeros_1/packedPackrnn/strided_slice:output:0rnn/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:V
rnn/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
rnn/zeros_1Fillrnn/zeros_1/packed:output:0rnn/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????g
rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
rnn/transpose	Transpose#time_distributed/Reshape_1:output:0rnn/transpose/perm:output:0*
T0*+
_output_shapes
:?????????L
rnn/Shape_1Shapernn/transpose:y:0*
T0*
_output_shapes
:c
rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:๏
rnn/strided_slice_1StridedSlicernn/Shape_1:output:0"rnn/strided_slice_1/stack:output:0$rnn/strided_slice_1/stack_1:output:0$rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????ภ
rnn/TensorArrayV2TensorListReserve(rnn/TensorArrayV2/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่า
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ์
+rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn/transpose:y:0Brnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่าc
rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
rnn/strided_slice_2StridedSlicernn/transpose:y:0"rnn/strided_slice_2/stack:output:0$rnn/strided_slice_2/stack_1:output:0$rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskl
*rnn/my_lstm_cell/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ำ
%rnn/my_lstm_cell/concatenate_1/concatConcatV2rnn/strided_slice_2:output:0rnn/zeros:output:03rnn/my_lstm_cell/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????'ข
,rnn/my_lstm_cell/dense/MatMul/ReadVariableOpReadVariableOp5rnn_my_lstm_cell_dense_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0ฟ
rnn/my_lstm_cell/dense/MatMulMatMul.rnn/my_lstm_cell/concatenate_1/concat:output:04rnn/my_lstm_cell/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
-rnn/my_lstm_cell/dense/BiasAdd/ReadVariableOpReadVariableOp6rnn_my_lstm_cell_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ป
rnn/my_lstm_cell/dense/BiasAddBiasAdd'rnn/my_lstm_cell/dense/MatMul:product:05rnn/my_lstm_cell/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
rnn/my_lstm_cell/dense/SigmoidSigmoid'rnn/my_lstm_cell/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
rnn/my_lstm_cell/multiply/mulMulrnn/zeros_1:output:0"rnn/my_lstm_cell/dense/Sigmoid:y:0*
T0*'
_output_shapes
:?????????ฆ
.rnn/my_lstm_cell/dense_1/MatMul/ReadVariableOpReadVariableOp7rnn_my_lstm_cell_dense_1_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0ร
rnn/my_lstm_cell/dense_1/MatMulMatMul.rnn/my_lstm_cell/concatenate_1/concat:output:06rnn/my_lstm_cell/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ค
/rnn/my_lstm_cell/dense_1/BiasAdd/ReadVariableOpReadVariableOp8rnn_my_lstm_cell_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ม
 rnn/my_lstm_cell/dense_1/BiasAddBiasAdd)rnn/my_lstm_cell/dense_1/MatMul:product:07rnn/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
 rnn/my_lstm_cell/dense_1/SigmoidSigmoid)rnn/my_lstm_cell/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????ฆ
.rnn/my_lstm_cell/dense_2/MatMul/ReadVariableOpReadVariableOp7rnn_my_lstm_cell_dense_2_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0ร
rnn/my_lstm_cell/dense_2/MatMulMatMul.rnn/my_lstm_cell/concatenate_1/concat:output:06rnn/my_lstm_cell/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ค
/rnn/my_lstm_cell/dense_2/BiasAdd/ReadVariableOpReadVariableOp8rnn_my_lstm_cell_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ม
 rnn/my_lstm_cell/dense_2/BiasAddBiasAdd)rnn/my_lstm_cell/dense_2/MatMul:product:07rnn/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
rnn/my_lstm_cell/dense_2/TanhTanh)rnn/my_lstm_cell/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????ก
rnn/my_lstm_cell/multiply/mul_1Mul$rnn/my_lstm_cell/dense_1/Sigmoid:y:0!rnn/my_lstm_cell/dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????
rnn/my_lstm_cell/add_1/addAddV2!rnn/my_lstm_cell/multiply/mul:z:0#rnn/my_lstm_cell/multiply/mul_1:z:0*
T0*'
_output_shapes
:?????????ฆ
.rnn/my_lstm_cell/dense_3/MatMul/ReadVariableOpReadVariableOp7rnn_my_lstm_cell_dense_3_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0ร
rnn/my_lstm_cell/dense_3/MatMulMatMul.rnn/my_lstm_cell/concatenate_1/concat:output:06rnn/my_lstm_cell/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ค
/rnn/my_lstm_cell/dense_3/BiasAdd/ReadVariableOpReadVariableOp8rnn_my_lstm_cell_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ม
 rnn/my_lstm_cell/dense_3/BiasAddBiasAdd)rnn/my_lstm_cell/dense_3/MatMul:product:07rnn/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
 rnn/my_lstm_cell/dense_3/SigmoidSigmoid)rnn/my_lstm_cell/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????z
 rnn/my_lstm_cell/activation/TanhTanhrnn/my_lstm_cell/add_1/add:z:0*
T0*'
_output_shapes
:?????????ค
rnn/my_lstm_cell/multiply/mul_2Mul$rnn/my_lstm_cell/activation/Tanh:y:0$rnn/my_lstm_cell/dense_3/Sigmoid:y:0*
T0*'
_output_shapes
:?????????r
!rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ฤ
rnn/TensorArrayV2_1TensorListReserve*rnn/TensorArrayV2_1/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่าJ
rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : g
rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????X
rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 	
	rnn/whileWhilernn/while/loop_counter:output:0%rnn/while/maximum_iterations:output:0rnn/time:output:0rnn/TensorArrayV2_1:handle:0rnn/zeros:output:0rnn/zeros_1:output:0rnn/strided_slice_1:output:0;rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:05rnn_my_lstm_cell_dense_matmul_readvariableop_resource6rnn_my_lstm_cell_dense_biasadd_readvariableop_resource7rnn_my_lstm_cell_dense_1_matmul_readvariableop_resource8rnn_my_lstm_cell_dense_1_biasadd_readvariableop_resource7rnn_my_lstm_cell_dense_2_matmul_readvariableop_resource8rnn_my_lstm_cell_dense_2_biasadd_readvariableop_resource7rnn_my_lstm_cell_dense_3_matmul_readvariableop_resource8rnn_my_lstm_cell_dense_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( * 
bodyR
rnn_while_body_15550* 
condR
rnn_while_cond_15549*U
output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : *
parallel_iterations 
4rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ฮ
&rnn/TensorArrayV2Stack/TensorListStackTensorListStackrnn/while:output:3=rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype0l
rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????e
rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
rnn/strided_slice_3StridedSlice/rnn/TensorArrayV2Stack/TensorListStack:tensor:0"rnn/strided_slice_3/stack:output:0$rnn/strided_slice_3/stack_1:output:0$rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maski
rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ข
rnn/transpose_1	Transpose/rnn/TensorArrayV2Stack/TensorListStack:tensor:0rnn/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Z
dense_4/Tensordot/ShapeShapernn/transpose_1:y:0*
T0*
_output_shapes
:a
dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ฿
dense_4/Tensordot/GatherV2_1GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/axes:output:0*dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ผ
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_4/Tensordot/transpose	Transposernn/transpose_1:y:0!dense_4/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????ข
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????ข
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????c
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ว
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????k
IdentityIdentitydense_4/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????ล
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp%^my_cnn_block/StatefulPartitionedCall.^rnn/my_lstm_cell/dense/BiasAdd/ReadVariableOp-^rnn/my_lstm_cell/dense/MatMul/ReadVariableOp0^rnn/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp/^rnn/my_lstm_cell/dense_1/MatMul/ReadVariableOp0^rnn/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp/^rnn/my_lstm_cell/dense_2/MatMul/ReadVariableOp0^rnn/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp/^rnn/my_lstm_cell/dense_3/MatMul/ReadVariableOp
^rnn/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp2L
$my_cnn_block/StatefulPartitionedCall$my_cnn_block/StatefulPartitionedCall2^
-rnn/my_lstm_cell/dense/BiasAdd/ReadVariableOp-rnn/my_lstm_cell/dense/BiasAdd/ReadVariableOp2\
,rnn/my_lstm_cell/dense/MatMul/ReadVariableOp,rnn/my_lstm_cell/dense/MatMul/ReadVariableOp2b
/rnn/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp/rnn/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp2`
.rnn/my_lstm_cell/dense_1/MatMul/ReadVariableOp.rnn/my_lstm_cell/dense_1/MatMul/ReadVariableOp2b
/rnn/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp/rnn/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp2`
.rnn/my_lstm_cell/dense_2/MatMul/ReadVariableOp.rnn/my_lstm_cell/dense_2/MatMul/ReadVariableOp2b
/rnn/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp/rnn/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp2`
.rnn/my_lstm_cell/dense_3/MatMul/ReadVariableOp.rnn/my_lstm_cell/dense_3/MatMul/ReadVariableOp2
	rnn/while	rnn/while:V R
3
_output_shapes!
:?????????

_user_specified_namex
ฆ
?
H__inference_my_lstm_model_layer_call_and_return_conditional_losses_16444
x,
my_cnn_block_16236: 
my_cnn_block_16238:,
my_cnn_block_16240: 
my_cnn_block_16242:
	rnn_16390:'
	rnn_16392:
	rnn_16394:'
	rnn_16396:
	rnn_16398:'
	rnn_16400:
	rnn_16402:'
	rnn_16404:
dense_4_16438:
dense_4_16440:
identityขdense_4/StatefulPartitionedCallข$my_cnn_block/StatefulPartitionedCallขrnn/StatefulPartitionedCallฐ
$my_cnn_block/StatefulPartitionedCallStatefulPartitionedCallxmy_cnn_block_16236my_cnn_block_16238my_cnn_block_16240my_cnn_block_16242*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_16235๒
 time_distributed/PartitionedCallPartitionedCall-my_cnn_block/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_15744w
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ต
time_distributed/ReshapeReshape-my_cnn_block/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????ฮ
rnn/StatefulPartitionedCallStatefulPartitionedCall)time_distributed/PartitionedCall:output:0	rnn_16390	rnn_16392	rnn_16394	rnn_16396	rnn_16398	rnn_16400	rnn_16402	rnn_16404*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_16389
dense_4/StatefulPartitionedCallStatefulPartitionedCall$rnn/StatefulPartitionedCall:output:0dense_4_16438dense_4_16440*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_16437{
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????ญ
NoOpNoOp ^dense_4/StatefulPartitionedCall%^my_cnn_block/StatefulPartitionedCall^rnn/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2L
$my_cnn_block/StatefulPartitionedCall$my_cnn_block/StatefulPartitionedCall2:
rnn/StatefulPartitionedCallrnn/StatefulPartitionedCall:V R
3
_output_shapes!
:?????????

_user_specified_namex
ธ

H__inference_my_lstm_model_layer_call_and_return_conditional_losses_16969
input_1,
my_cnn_block_16934: 
my_cnn_block_16936:,
my_cnn_block_16938: 
my_cnn_block_16940:
	rnn_16946:'
	rnn_16948:
	rnn_16950:'
	rnn_16952:
	rnn_16954:'
	rnn_16956:
	rnn_16958:'
	rnn_16960:
dense_4_16963:
dense_4_16965:
identityขdense_4/StatefulPartitionedCallข$my_cnn_block/StatefulPartitionedCallขrnn/StatefulPartitionedCallถ
$my_cnn_block/StatefulPartitionedCallStatefulPartitionedCallinput_1my_cnn_block_16934my_cnn_block_16936my_cnn_block_16938my_cnn_block_16940*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_16745๒
 time_distributed/PartitionedCallPartitionedCall-my_cnn_block/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_15765w
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ต
time_distributed/ReshapeReshape-my_cnn_block/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????ฮ
rnn/StatefulPartitionedCallStatefulPartitionedCall)time_distributed/PartitionedCall:output:0	rnn_16946	rnn_16948	rnn_16950	rnn_16952	rnn_16954	rnn_16956	rnn_16958	rnn_16960*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_16650
dense_4/StatefulPartitionedCallStatefulPartitionedCall$rnn/StatefulPartitionedCall:output:0dense_4_16963dense_4_16965*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_16437{
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????ญ
NoOpNoOp ^dense_4/StatefulPartitionedCall%^my_cnn_block/StatefulPartitionedCall^rnn/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2L
$my_cnn_block/StatefulPartitionedCall$my_cnn_block/StatefulPartitionedCall2:
rnn/StatefulPartitionedCallrnn/StatefulPartitionedCall:\ X
3
_output_shapes!
:?????????
!
_user_specified_name	input_1
๛
๒
-__inference_my_lstm_model_layer_call_fn_16893
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
identityขStatefulPartitionedCall
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
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_my_lstm_model_layer_call_and_return_conditional_losses_16829s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:?????????
!
_user_specified_name	input_1
เ[
๛
while_body_19071
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_my_lstm_cell_dense_matmul_readvariableop_resource_0:'H
:while_my_lstm_cell_dense_biasadd_readvariableop_resource_0:M
;while_my_lstm_cell_dense_1_matmul_readvariableop_resource_0:'J
<while_my_lstm_cell_dense_1_biasadd_readvariableop_resource_0:M
;while_my_lstm_cell_dense_2_matmul_readvariableop_resource_0:'J
<while_my_lstm_cell_dense_2_biasadd_readvariableop_resource_0:M
;while_my_lstm_cell_dense_3_matmul_readvariableop_resource_0:'J
<while_my_lstm_cell_dense_3_biasadd_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_my_lstm_cell_dense_matmul_readvariableop_resource:'F
8while_my_lstm_cell_dense_biasadd_readvariableop_resource:K
9while_my_lstm_cell_dense_1_matmul_readvariableop_resource:'H
:while_my_lstm_cell_dense_1_biasadd_readvariableop_resource:K
9while_my_lstm_cell_dense_2_matmul_readvariableop_resource:'H
:while_my_lstm_cell_dense_2_biasadd_readvariableop_resource:K
9while_my_lstm_cell_dense_3_matmul_readvariableop_resource:'H
:while_my_lstm_cell_dense_3_biasadd_readvariableop_resource:ข/while/my_lstm_cell/dense/BiasAdd/ReadVariableOpข.while/my_lstm_cell/dense/MatMul/ReadVariableOpข1while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOpข0while/my_lstm_cell/dense_1/MatMul/ReadVariableOpข1while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOpข0while/my_lstm_cell/dense_2/MatMul/ReadVariableOpข1while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOpข0while/my_lstm_cell/dense_3/MatMul/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ฆ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0n
,while/my_lstm_cell/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :์
'while/my_lstm_cell/concatenate_1/concatConcatV20while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_25while/my_lstm_cell/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????'จ
.while/my_lstm_cell/dense/MatMul/ReadVariableOpReadVariableOp9while_my_lstm_cell_dense_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ล
while/my_lstm_cell/dense/MatMulMatMul0while/my_lstm_cell/concatenate_1/concat:output:06while/my_lstm_cell/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ฆ
/while/my_lstm_cell/dense/BiasAdd/ReadVariableOpReadVariableOp:while_my_lstm_cell_dense_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0ม
 while/my_lstm_cell/dense/BiasAddBiasAdd)while/my_lstm_cell/dense/MatMul:product:07while/my_lstm_cell/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
 while/my_lstm_cell/dense/SigmoidSigmoid)while/my_lstm_cell/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
while/my_lstm_cell/multiply/mulMulwhile_placeholder_3$while/my_lstm_cell/dense/Sigmoid:y:0*
T0*'
_output_shapes
:?????????ฌ
0while/my_lstm_cell/dense_1/MatMul/ReadVariableOpReadVariableOp;while_my_lstm_cell_dense_1_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ษ
!while/my_lstm_cell/dense_1/MatMulMatMul0while/my_lstm_cell/concatenate_1/concat:output:08while/my_lstm_cell/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ช
1while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOpReadVariableOp<while_my_lstm_cell_dense_1_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0ว
"while/my_lstm_cell/dense_1/BiasAddBiasAdd+while/my_lstm_cell/dense_1/MatMul:product:09while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
"while/my_lstm_cell/dense_1/SigmoidSigmoid+while/my_lstm_cell/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????ฌ
0while/my_lstm_cell/dense_2/MatMul/ReadVariableOpReadVariableOp;while_my_lstm_cell_dense_2_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ษ
!while/my_lstm_cell/dense_2/MatMulMatMul0while/my_lstm_cell/concatenate_1/concat:output:08while/my_lstm_cell/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ช
1while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOpReadVariableOp<while_my_lstm_cell_dense_2_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0ว
"while/my_lstm_cell/dense_2/BiasAddBiasAdd+while/my_lstm_cell/dense_2/MatMul:product:09while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
while/my_lstm_cell/dense_2/TanhTanh+while/my_lstm_cell/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????ง
!while/my_lstm_cell/multiply/mul_1Mul&while/my_lstm_cell/dense_1/Sigmoid:y:0#while/my_lstm_cell/dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????ฃ
while/my_lstm_cell/add_1/addAddV2#while/my_lstm_cell/multiply/mul:z:0%while/my_lstm_cell/multiply/mul_1:z:0*
T0*'
_output_shapes
:?????????ฌ
0while/my_lstm_cell/dense_3/MatMul/ReadVariableOpReadVariableOp;while_my_lstm_cell_dense_3_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ษ
!while/my_lstm_cell/dense_3/MatMulMatMul0while/my_lstm_cell/concatenate_1/concat:output:08while/my_lstm_cell/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ช
1while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOpReadVariableOp<while_my_lstm_cell_dense_3_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0ว
"while/my_lstm_cell/dense_3/BiasAddBiasAdd+while/my_lstm_cell/dense_3/MatMul:product:09while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
"while/my_lstm_cell/dense_3/SigmoidSigmoid+while/my_lstm_cell/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????~
"while/my_lstm_cell/activation/TanhTanh while/my_lstm_cell/add_1/add:z:0*
T0*'
_output_shapes
:?????????ช
!while/my_lstm_cell/multiply/mul_2Mul&while/my_lstm_cell/activation/Tanh:y:0&while/my_lstm_cell/dense_3/Sigmoid:y:0*
T0*'
_output_shapes
:?????????ฮ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder%while/my_lstm_cell/multiply/mul_2:z:0*
_output_shapes
: *
element_dtype0:้่าM
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
: 
while/Identity_4Identity%while/my_lstm_cell/multiply/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:?????????}
while/Identity_5Identity while/my_lstm_cell/add_1/add:z:0^while/NoOp*
T0*'
_output_shapes
:?????????ไ

while/NoOpNoOp0^while/my_lstm_cell/dense/BiasAdd/ReadVariableOp/^while/my_lstm_cell/dense/MatMul/ReadVariableOp2^while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp1^while/my_lstm_cell/dense_1/MatMul/ReadVariableOp2^while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp1^while/my_lstm_cell/dense_2/MatMul/ReadVariableOp2^while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp1^while/my_lstm_cell/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"z
:while_my_lstm_cell_dense_1_biasadd_readvariableop_resource<while_my_lstm_cell_dense_1_biasadd_readvariableop_resource_0"x
9while_my_lstm_cell_dense_1_matmul_readvariableop_resource;while_my_lstm_cell_dense_1_matmul_readvariableop_resource_0"z
:while_my_lstm_cell_dense_2_biasadd_readvariableop_resource<while_my_lstm_cell_dense_2_biasadd_readvariableop_resource_0"x
9while_my_lstm_cell_dense_2_matmul_readvariableop_resource;while_my_lstm_cell_dense_2_matmul_readvariableop_resource_0"z
:while_my_lstm_cell_dense_3_biasadd_readvariableop_resource<while_my_lstm_cell_dense_3_biasadd_readvariableop_resource_0"x
9while_my_lstm_cell_dense_3_matmul_readvariableop_resource;while_my_lstm_cell_dense_3_matmul_readvariableop_resource_0"v
8while_my_lstm_cell_dense_biasadd_readvariableop_resource:while_my_lstm_cell_dense_biasadd_readvariableop_resource_0"t
7while_my_lstm_cell_dense_matmul_readvariableop_resource9while_my_lstm_cell_dense_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"จ
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :?????????:?????????: : : : : : : : : : 2b
/while/my_lstm_cell/dense/BiasAdd/ReadVariableOp/while/my_lstm_cell/dense/BiasAdd/ReadVariableOp2`
.while/my_lstm_cell/dense/MatMul/ReadVariableOp.while/my_lstm_cell/dense/MatMul/ReadVariableOp2f
1while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp1while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp2d
0while/my_lstm_cell/dense_1/MatMul/ReadVariableOp0while/my_lstm_cell/dense_1/MatMul/ReadVariableOp2f
1while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp1while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp2d
0while/my_lstm_cell/dense_2/MatMul/ReadVariableOp0while/my_lstm_cell/dense_2/MatMul/ReadVariableOp2f
1while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp1while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp2d
0while/my_lstm_cell/dense_3/MatMul/ReadVariableOp0while/my_lstm_cell/dense_3/MatMul/ReadVariableOp: 
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
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
เ[
๛
while_body_18897
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_my_lstm_cell_dense_matmul_readvariableop_resource_0:'H
:while_my_lstm_cell_dense_biasadd_readvariableop_resource_0:M
;while_my_lstm_cell_dense_1_matmul_readvariableop_resource_0:'J
<while_my_lstm_cell_dense_1_biasadd_readvariableop_resource_0:M
;while_my_lstm_cell_dense_2_matmul_readvariableop_resource_0:'J
<while_my_lstm_cell_dense_2_biasadd_readvariableop_resource_0:M
;while_my_lstm_cell_dense_3_matmul_readvariableop_resource_0:'J
<while_my_lstm_cell_dense_3_biasadd_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_my_lstm_cell_dense_matmul_readvariableop_resource:'F
8while_my_lstm_cell_dense_biasadd_readvariableop_resource:K
9while_my_lstm_cell_dense_1_matmul_readvariableop_resource:'H
:while_my_lstm_cell_dense_1_biasadd_readvariableop_resource:K
9while_my_lstm_cell_dense_2_matmul_readvariableop_resource:'H
:while_my_lstm_cell_dense_2_biasadd_readvariableop_resource:K
9while_my_lstm_cell_dense_3_matmul_readvariableop_resource:'H
:while_my_lstm_cell_dense_3_biasadd_readvariableop_resource:ข/while/my_lstm_cell/dense/BiasAdd/ReadVariableOpข.while/my_lstm_cell/dense/MatMul/ReadVariableOpข1while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOpข0while/my_lstm_cell/dense_1/MatMul/ReadVariableOpข1while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOpข0while/my_lstm_cell/dense_2/MatMul/ReadVariableOpข1while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOpข0while/my_lstm_cell/dense_3/MatMul/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ฆ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0n
,while/my_lstm_cell/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :์
'while/my_lstm_cell/concatenate_1/concatConcatV20while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_25while/my_lstm_cell/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????'จ
.while/my_lstm_cell/dense/MatMul/ReadVariableOpReadVariableOp9while_my_lstm_cell_dense_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ล
while/my_lstm_cell/dense/MatMulMatMul0while/my_lstm_cell/concatenate_1/concat:output:06while/my_lstm_cell/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ฆ
/while/my_lstm_cell/dense/BiasAdd/ReadVariableOpReadVariableOp:while_my_lstm_cell_dense_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0ม
 while/my_lstm_cell/dense/BiasAddBiasAdd)while/my_lstm_cell/dense/MatMul:product:07while/my_lstm_cell/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
 while/my_lstm_cell/dense/SigmoidSigmoid)while/my_lstm_cell/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
while/my_lstm_cell/multiply/mulMulwhile_placeholder_3$while/my_lstm_cell/dense/Sigmoid:y:0*
T0*'
_output_shapes
:?????????ฌ
0while/my_lstm_cell/dense_1/MatMul/ReadVariableOpReadVariableOp;while_my_lstm_cell_dense_1_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ษ
!while/my_lstm_cell/dense_1/MatMulMatMul0while/my_lstm_cell/concatenate_1/concat:output:08while/my_lstm_cell/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ช
1while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOpReadVariableOp<while_my_lstm_cell_dense_1_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0ว
"while/my_lstm_cell/dense_1/BiasAddBiasAdd+while/my_lstm_cell/dense_1/MatMul:product:09while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
"while/my_lstm_cell/dense_1/SigmoidSigmoid+while/my_lstm_cell/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????ฌ
0while/my_lstm_cell/dense_2/MatMul/ReadVariableOpReadVariableOp;while_my_lstm_cell_dense_2_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ษ
!while/my_lstm_cell/dense_2/MatMulMatMul0while/my_lstm_cell/concatenate_1/concat:output:08while/my_lstm_cell/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ช
1while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOpReadVariableOp<while_my_lstm_cell_dense_2_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0ว
"while/my_lstm_cell/dense_2/BiasAddBiasAdd+while/my_lstm_cell/dense_2/MatMul:product:09while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
while/my_lstm_cell/dense_2/TanhTanh+while/my_lstm_cell/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????ง
!while/my_lstm_cell/multiply/mul_1Mul&while/my_lstm_cell/dense_1/Sigmoid:y:0#while/my_lstm_cell/dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????ฃ
while/my_lstm_cell/add_1/addAddV2#while/my_lstm_cell/multiply/mul:z:0%while/my_lstm_cell/multiply/mul_1:z:0*
T0*'
_output_shapes
:?????????ฌ
0while/my_lstm_cell/dense_3/MatMul/ReadVariableOpReadVariableOp;while_my_lstm_cell_dense_3_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ษ
!while/my_lstm_cell/dense_3/MatMulMatMul0while/my_lstm_cell/concatenate_1/concat:output:08while/my_lstm_cell/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ช
1while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOpReadVariableOp<while_my_lstm_cell_dense_3_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0ว
"while/my_lstm_cell/dense_3/BiasAddBiasAdd+while/my_lstm_cell/dense_3/MatMul:product:09while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
"while/my_lstm_cell/dense_3/SigmoidSigmoid+while/my_lstm_cell/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????~
"while/my_lstm_cell/activation/TanhTanh while/my_lstm_cell/add_1/add:z:0*
T0*'
_output_shapes
:?????????ช
!while/my_lstm_cell/multiply/mul_2Mul&while/my_lstm_cell/activation/Tanh:y:0&while/my_lstm_cell/dense_3/Sigmoid:y:0*
T0*'
_output_shapes
:?????????ฮ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder%while/my_lstm_cell/multiply/mul_2:z:0*
_output_shapes
: *
element_dtype0:้่าM
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
: 
while/Identity_4Identity%while/my_lstm_cell/multiply/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:?????????}
while/Identity_5Identity while/my_lstm_cell/add_1/add:z:0^while/NoOp*
T0*'
_output_shapes
:?????????ไ

while/NoOpNoOp0^while/my_lstm_cell/dense/BiasAdd/ReadVariableOp/^while/my_lstm_cell/dense/MatMul/ReadVariableOp2^while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp1^while/my_lstm_cell/dense_1/MatMul/ReadVariableOp2^while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp1^while/my_lstm_cell/dense_2/MatMul/ReadVariableOp2^while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp1^while/my_lstm_cell/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"z
:while_my_lstm_cell_dense_1_biasadd_readvariableop_resource<while_my_lstm_cell_dense_1_biasadd_readvariableop_resource_0"x
9while_my_lstm_cell_dense_1_matmul_readvariableop_resource;while_my_lstm_cell_dense_1_matmul_readvariableop_resource_0"z
:while_my_lstm_cell_dense_2_biasadd_readvariableop_resource<while_my_lstm_cell_dense_2_biasadd_readvariableop_resource_0"x
9while_my_lstm_cell_dense_2_matmul_readvariableop_resource;while_my_lstm_cell_dense_2_matmul_readvariableop_resource_0"z
:while_my_lstm_cell_dense_3_biasadd_readvariableop_resource<while_my_lstm_cell_dense_3_biasadd_readvariableop_resource_0"x
9while_my_lstm_cell_dense_3_matmul_readvariableop_resource;while_my_lstm_cell_dense_3_matmul_readvariableop_resource_0"v
8while_my_lstm_cell_dense_biasadd_readvariableop_resource:while_my_lstm_cell_dense_biasadd_readvariableop_resource_0"t
7while_my_lstm_cell_dense_matmul_readvariableop_resource9while_my_lstm_cell_dense_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"จ
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :?????????:?????????: : : : : : : : : : 2b
/while/my_lstm_cell/dense/BiasAdd/ReadVariableOp/while/my_lstm_cell/dense/BiasAdd/ReadVariableOp2`
.while/my_lstm_cell/dense/MatMul/ReadVariableOp.while/my_lstm_cell/dense/MatMul/ReadVariableOp2f
1while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp1while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp2d
0while/my_lstm_cell/dense_1/MatMul/ReadVariableOp0while/my_lstm_cell/dense_1/MatMul/ReadVariableOp2f
1while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp1while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp2d
0while/my_lstm_cell/dense_2/MatMul/ReadVariableOp0while/my_lstm_cell/dense_2/MatMul/ReadVariableOp2f
1while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp1while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp2d
0while/my_lstm_cell/dense_3/MatMul/ReadVariableOp0while/my_lstm_cell/dense_3/MatMul/ReadVariableOp: 
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
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
ฆM

__inference_call_17037	
inputF
,conv2d_conv2d_conv2d_readvariableop_resource:G
9conv2d_squeeze_batch_dims_biasadd_readvariableop_resource:H
.conv2d_1_conv2d_conv2d_readvariableop_resource:I
;conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource:
identityข#conv2d/Conv2D/Conv2D/ReadVariableOpข0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOpข%conv2d_1/Conv2D/Conv2D/ReadVariableOpข2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOpH
conv2d/Conv2D/ShapeShapeinput*
T0*
_output_shapes
:k
!conv2d/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
#conv2d/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????m
#conv2d/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv2d/Conv2D/strided_sliceStridedSliceconv2d/Conv2D/Shape:output:0*conv2d/Conv2D/strided_slice/stack:output:0,conv2d/Conv2D/strided_slice/stack_1:output:0,conv2d/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskt
conv2d/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         
conv2d/Conv2D/ReshapeReshapeinput$conv2d/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????
#conv2d/Conv2D/Conv2D/ReadVariableOpReadVariableOp,conv2d_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0อ
conv2d/Conv2D/Conv2DConv2Dconv2d/Conv2D/Reshape:output:0+conv2d/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
r
conv2d/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         d
conv2d/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????ภ
conv2d/Conv2D/concatConcatV2$conv2d/Conv2D/strided_slice:output:0&conv2d/Conv2D/concat/values_1:output:0"conv2d/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:
conv2d/Conv2D/Reshape_1Reshapeconv2d/Conv2D/Conv2D:output:0conv2d/Conv2D/concat:output:0*
T0*3
_output_shapes!
:?????????o
conv2d/squeeze_batch_dims/ShapeShape conv2d/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:w
-conv2d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
/conv2d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????y
/conv2d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ั
'conv2d/squeeze_batch_dims/strided_sliceStridedSlice(conv2d/squeeze_batch_dims/Shape:output:06conv2d/squeeze_batch_dims/strided_slice/stack:output:08conv2d/squeeze_batch_dims/strided_slice/stack_1:output:08conv2d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
'conv2d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         บ
!conv2d/squeeze_batch_dims/ReshapeReshape conv2d/Conv2D/Reshape_1:output:00conv2d/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????ฆ
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp9conv2d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ฬ
!conv2d/squeeze_batch_dims/BiasAddBiasAdd*conv2d/squeeze_batch_dims/Reshape:output:08conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????~
)conv2d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         p
%conv2d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????๐
 conv2d/squeeze_batch_dims/concatConcatV20conv2d/squeeze_batch_dims/strided_slice:output:02conv2d/squeeze_batch_dims/concat/values_1:output:0.conv2d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:ร
#conv2d/squeeze_batch_dims/Reshape_1Reshape*conv2d/squeeze_batch_dims/BiasAdd:output:0)conv2d/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:?????????q
conv2d_1/Conv2D/ShapeShape,conv2d/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:m
#conv2d_1/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_1/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????o
%conv2d_1/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv2d_1/Conv2D/strided_sliceStridedSliceconv2d_1/Conv2D/Shape:output:0,conv2d_1/Conv2D/strided_slice/stack:output:0.conv2d_1/Conv2D/strided_slice/stack_1:output:0.conv2d_1/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_1/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ฒ
conv2d_1/Conv2D/ReshapeReshape,conv2d/squeeze_batch_dims/Reshape_1:output:0&conv2d_1/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????
%conv2d_1/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ำ
conv2d_1/Conv2D/Conv2DConv2D conv2d_1/Conv2D/Reshape:output:0-conv2d_1/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
t
conv2d_1/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         f
conv2d_1/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????ศ
conv2d_1/Conv2D/concatConcatV2&conv2d_1/Conv2D/strided_slice:output:0(conv2d_1/Conv2D/concat/values_1:output:0$conv2d_1/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:ค
conv2d_1/Conv2D/Reshape_1Reshapeconv2d_1/Conv2D/Conv2D:output:0conv2d_1/Conv2D/concat:output:0*
T0*3
_output_shapes!
:?????????s
!conv2d_1/squeeze_batch_dims/ShapeShape"conv2d_1/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_1/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1conv2d_1/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????{
1conv2d_1/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)conv2d_1/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_1/squeeze_batch_dims/Shape:output:08conv2d_1/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_1/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_1/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
)conv2d_1/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ภ
#conv2d_1/squeeze_batch_dims/ReshapeReshape"conv2d_1/Conv2D/Reshape_1:output:02conv2d_1/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????ช
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0า
#conv2d_1/squeeze_batch_dims/BiasAddBiasAdd,conv2d_1/squeeze_batch_dims/Reshape:output:0:conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
+conv2d_1/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         r
'conv2d_1/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????๘
"conv2d_1/squeeze_batch_dims/concatConcatV22conv2d_1/squeeze_batch_dims/strided_slice:output:04conv2d_1/squeeze_batch_dims/concat/values_1:output:00conv2d_1/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:ษ
%conv2d_1/squeeze_batch_dims/Reshape_1Reshape,conv2d_1/squeeze_batch_dims/BiasAdd:output:0+conv2d_1/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:?????????
IdentityIdentity.conv2d_1/squeeze_batch_dims/Reshape_1:output:0^NoOp*
T0*3
_output_shapes!
:??????????
NoOpNoOp$^conv2d/Conv2D/Conv2D/ReadVariableOp1^conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_1/Conv2D/Conv2D/ReadVariableOp3^conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : 2J
#conv2d/Conv2D/Conv2D/ReadVariableOp#conv2d/Conv2D/Conv2D/ReadVariableOp2d
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_1/Conv2D/Conv2D/ReadVariableOp%conv2d_1/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp:Z V
3
_output_shapes!
:?????????

_user_specified_nameinput

g
K__inference_time_distributed_layer_call_and_return_conditional_losses_15765

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
valueB:ั
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
valueB"????         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????แ
(global_average_pooling2d/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_15721\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
	Reshape_1Reshape1global_average_pooling2d/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????g
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&??????????????????:d `
<
_output_shapes*
(:&??????????????????
 
_user_specified_nameinputs
อ

ว
while_cond_16561
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_16561___redundant_placeholder03
/while_while_cond_16561___redundant_placeholder13
/while_while_cond_16561___redundant_placeholder23
/while_while_cond_16561___redundant_placeholder33
/while_while_cond_16561___redundant_placeholder43
/while_while_cond_16561___redundant_placeholder53
/while_while_cond_16561___redundant_placeholder63
/while_while_cond_16561___redundant_placeholder73
/while_while_cond_16561___redundant_placeholder8
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
T: : : : :?????????:?????????: :::::::::: 
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
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
๛-
โ
G__inference_my_lstm_cell_layer_call_and_return_conditional_losses_18395

inputs
states_0
states_16
$dense_matmul_readvariableop_resource:'3
%dense_biasadd_readvariableop_resource:8
&dense_1_matmul_readvariableop_resource:'5
'dense_1_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:'5
'dense_2_biasadd_readvariableop_resource:8
&dense_3_matmul_readvariableop_resource:'5
'dense_3_biasadd_readvariableop_resource:
identity

identity_1

identity_2ขdense/BiasAdd/ReadVariableOpขdense/MatMul/ReadVariableOpขdense_1/BiasAdd/ReadVariableOpขdense_1/MatMul/ReadVariableOpขdense_2/BiasAdd/ReadVariableOpขdense_2/MatMul/ReadVariableOpขdense_3/BiasAdd/ReadVariableOpขdense_3/MatMul/ReadVariableOp[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatenate_1/concatConcatV2inputsstates_0"concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????'
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0
dense/MatMulMatMulconcatenate_1/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????b
multiply/mulMulstates_1dense/Sigmoid:y:0*
T0*'
_output_shapes
:?????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0
dense_1/MatMulMatMulconcatenate_1/concat:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0
dense_2/MatMulMatMulconcatenate_1/concat:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`
dense_2/TanhTanhdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????n
multiply/mul_1Muldense_1/Sigmoid:y:0dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????j
	add_1/addAddV2multiply/mul:z:0multiply/mul_1:z:0*
T0*'
_output_shapes
:?????????
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0
dense_3/MatMulMatMulconcatenate_1/concat:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????X
activation/TanhTanhadd_1/add:z:0*
T0*'
_output_shapes
:?????????q
multiply/mul_2Mulactivation/Tanh:y:0dense_3/Sigmoid:y:0*
T0*'
_output_shapes
:?????????a
IdentityIdentitymultiply/mul_2:z:0^NoOp*
T0*'
_output_shapes
:?????????c

Identity_1Identitymultiply/mul_2:z:0^NoOp*
T0*'
_output_shapes
:?????????^

Identity_2Identityadd_1/add:z:0^NoOp*
T0*'
_output_shapes
:?????????ฦ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????:?????????:?????????: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/1
ถ
๗
,__inference_my_lstm_cell_layer_call_fn_18352

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

identity_2ขStatefulPartitionedCall่
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_my_lstm_cell_layer_call_and_return_conditional_losses_15846o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????:?????????:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/1
ผb
ท
rnn_while_body_17122$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3#
rnn_while_rnn_strided_slice_1_0_
[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0O
=rnn_while_my_lstm_cell_dense_matmul_readvariableop_resource_0:'L
>rnn_while_my_lstm_cell_dense_biasadd_readvariableop_resource_0:Q
?rnn_while_my_lstm_cell_dense_1_matmul_readvariableop_resource_0:'N
@rnn_while_my_lstm_cell_dense_1_biasadd_readvariableop_resource_0:Q
?rnn_while_my_lstm_cell_dense_2_matmul_readvariableop_resource_0:'N
@rnn_while_my_lstm_cell_dense_2_biasadd_readvariableop_resource_0:Q
?rnn_while_my_lstm_cell_dense_3_matmul_readvariableop_resource_0:'N
@rnn_while_my_lstm_cell_dense_3_biasadd_readvariableop_resource_0:
rnn_while_identity
rnn_while_identity_1
rnn_while_identity_2
rnn_while_identity_3
rnn_while_identity_4
rnn_while_identity_5!
rnn_while_rnn_strided_slice_1]
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensorM
;rnn_while_my_lstm_cell_dense_matmul_readvariableop_resource:'J
<rnn_while_my_lstm_cell_dense_biasadd_readvariableop_resource:O
=rnn_while_my_lstm_cell_dense_1_matmul_readvariableop_resource:'L
>rnn_while_my_lstm_cell_dense_1_biasadd_readvariableop_resource:O
=rnn_while_my_lstm_cell_dense_2_matmul_readvariableop_resource:'L
>rnn_while_my_lstm_cell_dense_2_biasadd_readvariableop_resource:O
=rnn_while_my_lstm_cell_dense_3_matmul_readvariableop_resource:'L
>rnn_while_my_lstm_cell_dense_3_biasadd_readvariableop_resource:ข3rnn/while/my_lstm_cell/dense/BiasAdd/ReadVariableOpข2rnn/while/my_lstm_cell/dense/MatMul/ReadVariableOpข5rnn/while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOpข4rnn/while/my_lstm_cell/dense_1/MatMul/ReadVariableOpข5rnn/while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOpข4rnn/while/my_lstm_cell/dense_2/MatMul/ReadVariableOpข5rnn/while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOpข4rnn/while/my_lstm_cell/dense_3/MatMul/ReadVariableOp
;rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   บ
-rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0rnn_while_placeholderDrnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0r
0rnn/while/my_lstm_cell/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
+rnn/while/my_lstm_cell/concatenate_1/concatConcatV24rnn/while/TensorArrayV2Read/TensorListGetItem:item:0rnn_while_placeholder_29rnn/while/my_lstm_cell/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????'ฐ
2rnn/while/my_lstm_cell/dense/MatMul/ReadVariableOpReadVariableOp=rnn_while_my_lstm_cell_dense_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ั
#rnn/while/my_lstm_cell/dense/MatMulMatMul4rnn/while/my_lstm_cell/concatenate_1/concat:output:0:rnn/while/my_lstm_cell/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ฎ
3rnn/while/my_lstm_cell/dense/BiasAdd/ReadVariableOpReadVariableOp>rnn_while_my_lstm_cell_dense_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0อ
$rnn/while/my_lstm_cell/dense/BiasAddBiasAdd-rnn/while/my_lstm_cell/dense/MatMul:product:0;rnn/while/my_lstm_cell/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
$rnn/while/my_lstm_cell/dense/SigmoidSigmoid-rnn/while/my_lstm_cell/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
#rnn/while/my_lstm_cell/multiply/mulMulrnn_while_placeholder_3(rnn/while/my_lstm_cell/dense/Sigmoid:y:0*
T0*'
_output_shapes
:?????????ด
4rnn/while/my_lstm_cell/dense_1/MatMul/ReadVariableOpReadVariableOp?rnn_while_my_lstm_cell_dense_1_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ี
%rnn/while/my_lstm_cell/dense_1/MatMulMatMul4rnn/while/my_lstm_cell/concatenate_1/concat:output:0<rnn/while/my_lstm_cell/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ฒ
5rnn/while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOpReadVariableOp@rnn_while_my_lstm_cell_dense_1_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0ำ
&rnn/while/my_lstm_cell/dense_1/BiasAddBiasAdd/rnn/while/my_lstm_cell/dense_1/MatMul:product:0=rnn/while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
&rnn/while/my_lstm_cell/dense_1/SigmoidSigmoid/rnn/while/my_lstm_cell/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????ด
4rnn/while/my_lstm_cell/dense_2/MatMul/ReadVariableOpReadVariableOp?rnn_while_my_lstm_cell_dense_2_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ี
%rnn/while/my_lstm_cell/dense_2/MatMulMatMul4rnn/while/my_lstm_cell/concatenate_1/concat:output:0<rnn/while/my_lstm_cell/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ฒ
5rnn/while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOpReadVariableOp@rnn_while_my_lstm_cell_dense_2_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0ำ
&rnn/while/my_lstm_cell/dense_2/BiasAddBiasAdd/rnn/while/my_lstm_cell/dense_2/MatMul:product:0=rnn/while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
#rnn/while/my_lstm_cell/dense_2/TanhTanh/rnn/while/my_lstm_cell/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????ณ
%rnn/while/my_lstm_cell/multiply/mul_1Mul*rnn/while/my_lstm_cell/dense_1/Sigmoid:y:0'rnn/while/my_lstm_cell/dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????ฏ
 rnn/while/my_lstm_cell/add_1/addAddV2'rnn/while/my_lstm_cell/multiply/mul:z:0)rnn/while/my_lstm_cell/multiply/mul_1:z:0*
T0*'
_output_shapes
:?????????ด
4rnn/while/my_lstm_cell/dense_3/MatMul/ReadVariableOpReadVariableOp?rnn_while_my_lstm_cell_dense_3_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ี
%rnn/while/my_lstm_cell/dense_3/MatMulMatMul4rnn/while/my_lstm_cell/concatenate_1/concat:output:0<rnn/while/my_lstm_cell/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ฒ
5rnn/while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOpReadVariableOp@rnn_while_my_lstm_cell_dense_3_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0ำ
&rnn/while/my_lstm_cell/dense_3/BiasAddBiasAdd/rnn/while/my_lstm_cell/dense_3/MatMul:product:0=rnn/while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
&rnn/while/my_lstm_cell/dense_3/SigmoidSigmoid/rnn/while/my_lstm_cell/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
&rnn/while/my_lstm_cell/activation/TanhTanh$rnn/while/my_lstm_cell/add_1/add:z:0*
T0*'
_output_shapes
:?????????ถ
%rnn/while/my_lstm_cell/multiply/mul_2Mul*rnn/while/my_lstm_cell/activation/Tanh:y:0*rnn/while/my_lstm_cell/dense_3/Sigmoid:y:0*
T0*'
_output_shapes
:??????????
.rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrnn_while_placeholder_1rnn_while_placeholder)rnn/while/my_lstm_cell/multiply/mul_2:z:0*
_output_shapes
: *
element_dtype0:้่าQ
rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :h
rnn/while/addAddV2rnn_while_placeholderrnn/while/add/y:output:0*
T0*
_output_shapes
: S
rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :w
rnn/while/add_1AddV2 rnn_while_rnn_while_loop_counterrnn/while/add_1/y:output:0*
T0*
_output_shapes
: e
rnn/while/IdentityIdentityrnn/while/add_1:z:0^rnn/while/NoOp*
T0*
_output_shapes
: z
rnn/while/Identity_1Identity&rnn_while_rnn_while_maximum_iterations^rnn/while/NoOp*
T0*
_output_shapes
: e
rnn/while/Identity_2Identityrnn/while/add:z:0^rnn/while/NoOp*
T0*
_output_shapes
: 
rnn/while/Identity_3Identity>rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^rnn/while/NoOp*
T0*
_output_shapes
: 
rnn/while/Identity_4Identity)rnn/while/my_lstm_cell/multiply/mul_2:z:0^rnn/while/NoOp*
T0*'
_output_shapes
:?????????
rnn/while/Identity_5Identity$rnn/while/my_lstm_cell/add_1/add:z:0^rnn/while/NoOp*
T0*'
_output_shapes
:?????????
rnn/while/NoOpNoOp4^rnn/while/my_lstm_cell/dense/BiasAdd/ReadVariableOp3^rnn/while/my_lstm_cell/dense/MatMul/ReadVariableOp6^rnn/while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp5^rnn/while/my_lstm_cell/dense_1/MatMul/ReadVariableOp6^rnn/while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp5^rnn/while/my_lstm_cell/dense_2/MatMul/ReadVariableOp6^rnn/while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp5^rnn/while/my_lstm_cell/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "1
rnn_while_identityrnn/while/Identity:output:0"5
rnn_while_identity_1rnn/while/Identity_1:output:0"5
rnn_while_identity_2rnn/while/Identity_2:output:0"5
rnn_while_identity_3rnn/while/Identity_3:output:0"5
rnn_while_identity_4rnn/while/Identity_4:output:0"5
rnn_while_identity_5rnn/while/Identity_5:output:0"
>rnn_while_my_lstm_cell_dense_1_biasadd_readvariableop_resource@rnn_while_my_lstm_cell_dense_1_biasadd_readvariableop_resource_0"
=rnn_while_my_lstm_cell_dense_1_matmul_readvariableop_resource?rnn_while_my_lstm_cell_dense_1_matmul_readvariableop_resource_0"
>rnn_while_my_lstm_cell_dense_2_biasadd_readvariableop_resource@rnn_while_my_lstm_cell_dense_2_biasadd_readvariableop_resource_0"
=rnn_while_my_lstm_cell_dense_2_matmul_readvariableop_resource?rnn_while_my_lstm_cell_dense_2_matmul_readvariableop_resource_0"
>rnn_while_my_lstm_cell_dense_3_biasadd_readvariableop_resource@rnn_while_my_lstm_cell_dense_3_biasadd_readvariableop_resource_0"
=rnn_while_my_lstm_cell_dense_3_matmul_readvariableop_resource?rnn_while_my_lstm_cell_dense_3_matmul_readvariableop_resource_0"~
<rnn_while_my_lstm_cell_dense_biasadd_readvariableop_resource>rnn_while_my_lstm_cell_dense_biasadd_readvariableop_resource_0"|
;rnn_while_my_lstm_cell_dense_matmul_readvariableop_resource=rnn_while_my_lstm_cell_dense_matmul_readvariableop_resource_0"@
rnn_while_rnn_strided_slice_1rnn_while_rnn_strided_slice_1_0"ธ
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :?????????:?????????: : : : : : : : : : 2j
3rnn/while/my_lstm_cell/dense/BiasAdd/ReadVariableOp3rnn/while/my_lstm_cell/dense/BiasAdd/ReadVariableOp2h
2rnn/while/my_lstm_cell/dense/MatMul/ReadVariableOp2rnn/while/my_lstm_cell/dense/MatMul/ReadVariableOp2n
5rnn/while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp5rnn/while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp2l
4rnn/while/my_lstm_cell/dense_1/MatMul/ReadVariableOp4rnn/while/my_lstm_cell/dense_1/MatMul/ReadVariableOp2n
5rnn/while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp5rnn/while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp2l
4rnn/while/my_lstm_cell/dense_2/MatMul/ReadVariableOp4rnn/while/my_lstm_cell/dense_2/MatMul/ReadVariableOp2n
5rnn/while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp5rnn/while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp2l
4rnn/while/my_lstm_cell/dense_3/MatMul/ReadVariableOp4rnn/while/my_lstm_cell/dense_3/MatMul/ReadVariableOp: 
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
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
เ
ฝ
H__inference_my_lstm_model_layer_call_and_return_conditional_losses_17787
x,
my_cnn_block_17573: 
my_cnn_block_17575:,
my_cnn_block_17577: 
my_cnn_block_17579:G
5rnn_my_lstm_cell_dense_matmul_readvariableop_resource:'D
6rnn_my_lstm_cell_dense_biasadd_readvariableop_resource:I
7rnn_my_lstm_cell_dense_1_matmul_readvariableop_resource:'F
8rnn_my_lstm_cell_dense_1_biasadd_readvariableop_resource:I
7rnn_my_lstm_cell_dense_2_matmul_readvariableop_resource:'F
8rnn_my_lstm_cell_dense_2_biasadd_readvariableop_resource:I
7rnn_my_lstm_cell_dense_3_matmul_readvariableop_resource:'F
8rnn_my_lstm_cell_dense_3_biasadd_readvariableop_resource:;
)dense_4_tensordot_readvariableop_resource:5
'dense_4_biasadd_readvariableop_resource:
identityขdense_4/BiasAdd/ReadVariableOpข dense_4/Tensordot/ReadVariableOpข$my_cnn_block/StatefulPartitionedCallข-rnn/my_lstm_cell/dense/BiasAdd/ReadVariableOpข,rnn/my_lstm_cell/dense/MatMul/ReadVariableOpข/rnn/my_lstm_cell/dense_1/BiasAdd/ReadVariableOpข.rnn/my_lstm_cell/dense_1/MatMul/ReadVariableOpข/rnn/my_lstm_cell/dense_2/BiasAdd/ReadVariableOpข.rnn/my_lstm_cell/dense_2/MatMul/ReadVariableOpข/rnn/my_lstm_cell/dense_3/BiasAdd/ReadVariableOpข.rnn/my_lstm_cell/dense_3/MatMul/ReadVariableOpข	rnn/while?
$my_cnn_block/StatefulPartitionedCallStatefulPartitionedCallxmy_cnn_block_17573my_cnn_block_17575my_cnn_block_17577my_cnn_block_17579*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *
fR
__inference_call_15465w
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ต
time_distributed/ReshapeReshape-my_cnn_block/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????
@time_distributed/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ึ
.time_distributed/global_average_pooling2d/MeanMean!time_distributed/Reshape:output:0Itime_distributed/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????u
 time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      ฟ
time_distributed/Reshape_1Reshape7time_distributed/global_average_pooling2d/Mean:output:0)time_distributed/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????y
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         น
time_distributed/Reshape_2Reshape-my_cnn_block/StatefulPartitionedCall:output:0)time_distributed/Reshape_2/shape:output:0*
T0*/
_output_shapes
:?????????\
	rnn/ShapeShape#time_distributed/Reshape_1:output:0*
T0*
_output_shapes
:a
rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ๅ
rnn/strided_sliceStridedSlicernn/Shape:output:0 rnn/strided_slice/stack:output:0"rnn/strided_slice/stack_1:output:0"rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
rnn/zeros/packedPackrnn/strided_slice:output:0rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:T
rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x
	rnn/zerosFillrnn/zeros/packed:output:0rnn/zeros/Const:output:0*
T0*'
_output_shapes
:?????????V
rnn/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
rnn/zeros_1/packedPackrnn/strided_slice:output:0rnn/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:V
rnn/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
rnn/zeros_1Fillrnn/zeros_1/packed:output:0rnn/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????g
rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
rnn/transpose	Transpose#time_distributed/Reshape_1:output:0rnn/transpose/perm:output:0*
T0*+
_output_shapes
:?????????L
rnn/Shape_1Shapernn/transpose:y:0*
T0*
_output_shapes
:c
rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:๏
rnn/strided_slice_1StridedSlicernn/Shape_1:output:0"rnn/strided_slice_1/stack:output:0$rnn/strided_slice_1/stack_1:output:0$rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????ภ
rnn/TensorArrayV2TensorListReserve(rnn/TensorArrayV2/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่า
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ์
+rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn/transpose:y:0Brnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่าc
rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
rnn/strided_slice_2StridedSlicernn/transpose:y:0"rnn/strided_slice_2/stack:output:0$rnn/strided_slice_2/stack_1:output:0$rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskl
*rnn/my_lstm_cell/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ำ
%rnn/my_lstm_cell/concatenate_1/concatConcatV2rnn/strided_slice_2:output:0rnn/zeros:output:03rnn/my_lstm_cell/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????'ข
,rnn/my_lstm_cell/dense/MatMul/ReadVariableOpReadVariableOp5rnn_my_lstm_cell_dense_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0ฟ
rnn/my_lstm_cell/dense/MatMulMatMul.rnn/my_lstm_cell/concatenate_1/concat:output:04rnn/my_lstm_cell/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
-rnn/my_lstm_cell/dense/BiasAdd/ReadVariableOpReadVariableOp6rnn_my_lstm_cell_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ป
rnn/my_lstm_cell/dense/BiasAddBiasAdd'rnn/my_lstm_cell/dense/MatMul:product:05rnn/my_lstm_cell/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
rnn/my_lstm_cell/dense/SigmoidSigmoid'rnn/my_lstm_cell/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
rnn/my_lstm_cell/multiply/mulMulrnn/zeros_1:output:0"rnn/my_lstm_cell/dense/Sigmoid:y:0*
T0*'
_output_shapes
:?????????ฆ
.rnn/my_lstm_cell/dense_1/MatMul/ReadVariableOpReadVariableOp7rnn_my_lstm_cell_dense_1_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0ร
rnn/my_lstm_cell/dense_1/MatMulMatMul.rnn/my_lstm_cell/concatenate_1/concat:output:06rnn/my_lstm_cell/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ค
/rnn/my_lstm_cell/dense_1/BiasAdd/ReadVariableOpReadVariableOp8rnn_my_lstm_cell_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ม
 rnn/my_lstm_cell/dense_1/BiasAddBiasAdd)rnn/my_lstm_cell/dense_1/MatMul:product:07rnn/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
 rnn/my_lstm_cell/dense_1/SigmoidSigmoid)rnn/my_lstm_cell/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????ฆ
.rnn/my_lstm_cell/dense_2/MatMul/ReadVariableOpReadVariableOp7rnn_my_lstm_cell_dense_2_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0ร
rnn/my_lstm_cell/dense_2/MatMulMatMul.rnn/my_lstm_cell/concatenate_1/concat:output:06rnn/my_lstm_cell/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ค
/rnn/my_lstm_cell/dense_2/BiasAdd/ReadVariableOpReadVariableOp8rnn_my_lstm_cell_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ม
 rnn/my_lstm_cell/dense_2/BiasAddBiasAdd)rnn/my_lstm_cell/dense_2/MatMul:product:07rnn/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
rnn/my_lstm_cell/dense_2/TanhTanh)rnn/my_lstm_cell/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????ก
rnn/my_lstm_cell/multiply/mul_1Mul$rnn/my_lstm_cell/dense_1/Sigmoid:y:0!rnn/my_lstm_cell/dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????
rnn/my_lstm_cell/add_1/addAddV2!rnn/my_lstm_cell/multiply/mul:z:0#rnn/my_lstm_cell/multiply/mul_1:z:0*
T0*'
_output_shapes
:?????????ฆ
.rnn/my_lstm_cell/dense_3/MatMul/ReadVariableOpReadVariableOp7rnn_my_lstm_cell_dense_3_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0ร
rnn/my_lstm_cell/dense_3/MatMulMatMul.rnn/my_lstm_cell/concatenate_1/concat:output:06rnn/my_lstm_cell/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ค
/rnn/my_lstm_cell/dense_3/BiasAdd/ReadVariableOpReadVariableOp8rnn_my_lstm_cell_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ม
 rnn/my_lstm_cell/dense_3/BiasAddBiasAdd)rnn/my_lstm_cell/dense_3/MatMul:product:07rnn/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
 rnn/my_lstm_cell/dense_3/SigmoidSigmoid)rnn/my_lstm_cell/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????z
 rnn/my_lstm_cell/activation/TanhTanhrnn/my_lstm_cell/add_1/add:z:0*
T0*'
_output_shapes
:?????????ค
rnn/my_lstm_cell/multiply/mul_2Mul$rnn/my_lstm_cell/activation/Tanh:y:0$rnn/my_lstm_cell/dense_3/Sigmoid:y:0*
T0*'
_output_shapes
:?????????r
!rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ฤ
rnn/TensorArrayV2_1TensorListReserve*rnn/TensorArrayV2_1/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่าJ
rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : g
rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????X
rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 	
	rnn/whileWhilernn/while/loop_counter:output:0%rnn/while/maximum_iterations:output:0rnn/time:output:0rnn/TensorArrayV2_1:handle:0rnn/zeros:output:0rnn/zeros_1:output:0rnn/strided_slice_1:output:0;rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:05rnn_my_lstm_cell_dense_matmul_readvariableop_resource6rnn_my_lstm_cell_dense_biasadd_readvariableop_resource7rnn_my_lstm_cell_dense_1_matmul_readvariableop_resource8rnn_my_lstm_cell_dense_1_biasadd_readvariableop_resource7rnn_my_lstm_cell_dense_2_matmul_readvariableop_resource8rnn_my_lstm_cell_dense_2_biasadd_readvariableop_resource7rnn_my_lstm_cell_dense_3_matmul_readvariableop_resource8rnn_my_lstm_cell_dense_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( * 
bodyR
rnn_while_body_17657* 
condR
rnn_while_cond_17656*U
output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : *
parallel_iterations 
4rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ฮ
&rnn/TensorArrayV2Stack/TensorListStackTensorListStackrnn/while:output:3=rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype0l
rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????e
rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
rnn/strided_slice_3StridedSlice/rnn/TensorArrayV2Stack/TensorListStack:tensor:0"rnn/strided_slice_3/stack:output:0$rnn/strided_slice_3/stack_1:output:0$rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maski
rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ข
rnn/transpose_1	Transpose/rnn/TensorArrayV2Stack/TensorListStack:tensor:0rnn/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Z
dense_4/Tensordot/ShapeShapernn/transpose_1:y:0*
T0*
_output_shapes
:a
dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ฿
dense_4/Tensordot/GatherV2_1GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/axes:output:0*dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ผ
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_4/Tensordot/transpose	Transposernn/transpose_1:y:0!dense_4/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????ข
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????ข
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????c
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ว
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????k
IdentityIdentitydense_4/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????ล
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp%^my_cnn_block/StatefulPartitionedCall.^rnn/my_lstm_cell/dense/BiasAdd/ReadVariableOp-^rnn/my_lstm_cell/dense/MatMul/ReadVariableOp0^rnn/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp/^rnn/my_lstm_cell/dense_1/MatMul/ReadVariableOp0^rnn/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp/^rnn/my_lstm_cell/dense_2/MatMul/ReadVariableOp0^rnn/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp/^rnn/my_lstm_cell/dense_3/MatMul/ReadVariableOp
^rnn/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp2L
$my_cnn_block/StatefulPartitionedCall$my_cnn_block/StatefulPartitionedCall2^
-rnn/my_lstm_cell/dense/BiasAdd/ReadVariableOp-rnn/my_lstm_cell/dense/BiasAdd/ReadVariableOp2\
,rnn/my_lstm_cell/dense/MatMul/ReadVariableOp,rnn/my_lstm_cell/dense/MatMul/ReadVariableOp2b
/rnn/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp/rnn/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp2`
.rnn/my_lstm_cell/dense_1/MatMul/ReadVariableOp.rnn/my_lstm_cell/dense_1/MatMul/ReadVariableOp2b
/rnn/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp/rnn/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp2`
.rnn/my_lstm_cell/dense_2/MatMul/ReadVariableOp.rnn/my_lstm_cell/dense_2/MatMul/ReadVariableOp2b
/rnn/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp/rnn/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp2`
.rnn/my_lstm_cell/dense_3/MatMul/ReadVariableOp.rnn/my_lstm_cell/dense_3/MatMul/ReadVariableOp2
	rnn/while	rnn/while:V R
3
_output_shapes!
:?????????

_user_specified_namex
อ

ว
while_cond_18548
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_18548___redundant_placeholder03
/while_while_cond_18548___redundant_placeholder13
/while_while_cond_18548___redundant_placeholder23
/while_while_cond_18548___redundant_placeholder33
/while_while_cond_18548___redundant_placeholder43
/while_while_cond_18548___redundant_placeholder53
/while_while_cond_18548___redundant_placeholder63
/while_while_cond_18548___redundant_placeholder73
/while_while_cond_18548___redundant_placeholder8
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
T: : : : :?????????:?????????: :::::::::: 
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
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
อ:
ฌ
>__inference_rnn_layer_call_and_return_conditional_losses_16149

inputs$
my_lstm_cell_16038:' 
my_lstm_cell_16040:$
my_lstm_cell_16042:' 
my_lstm_cell_16044:$
my_lstm_cell_16046:' 
my_lstm_cell_16048:$
my_lstm_cell_16050:' 
my_lstm_cell_16052:
identityข$my_lstm_cell/StatefulPartitionedCallขwhile;
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
valueB:ั
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
:?????????R
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
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
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
valueB:?
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
?????????ด
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่า
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   เ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่า_
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
valueB:้
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask฿
$my_lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0my_lstm_cell_16038my_lstm_cell_16040my_lstm_cell_16042my_lstm_cell_16044my_lstm_cell_16046my_lstm_cell_16048my_lstm_cell_16050my_lstm_cell_16052*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_my_lstm_cell_layer_call_and_return_conditional_losses_15846n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ธ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่าF
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
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ด
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0my_lstm_cell_16038my_lstm_cell_16040my_lstm_cell_16042my_lstm_cell_16044my_lstm_cell_16046my_lstm_cell_16048my_lstm_cell_16050my_lstm_cell_16052*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( *
bodyR
while_body_16061*
condR
while_cond_16060*U
output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ห
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
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
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :??????????????????u
NoOpNoOp%^my_lstm_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????????????: : : : : : : : 2L
$my_lstm_cell/StatefulPartitionedCall$my_lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
ๅ
ฟ
rnn_while_cond_15549$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3&
"rnn_while_less_rnn_strided_slice_1;
7rnn_while_rnn_while_cond_15549___redundant_placeholder0;
7rnn_while_rnn_while_cond_15549___redundant_placeholder1;
7rnn_while_rnn_while_cond_15549___redundant_placeholder2;
7rnn_while_rnn_while_cond_15549___redundant_placeholder3;
7rnn_while_rnn_while_cond_15549___redundant_placeholder4;
7rnn_while_rnn_while_cond_15549___redundant_placeholder5;
7rnn_while_rnn_while_cond_15549___redundant_placeholder6;
7rnn_while_rnn_while_cond_15549___redundant_placeholder7;
7rnn_while_rnn_while_cond_15549___redundant_placeholder8
rnn_while_identity
r
rnn/while/LessLessrnn_while_placeholder"rnn_while_less_rnn_strided_slice_1*
T0*
_output_shapes
: S
rnn/while/IdentityIdentityrnn/while/Less:z:0*
T0
*
_output_shapes
: "1
rnn_while_identityrnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : :?????????:?????????: :::::::::: 
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
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
ฆM

__inference_call_18124	
inputF
,conv2d_conv2d_conv2d_readvariableop_resource:G
9conv2d_squeeze_batch_dims_biasadd_readvariableop_resource:H
.conv2d_1_conv2d_conv2d_readvariableop_resource:I
;conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource:
identityข#conv2d/Conv2D/Conv2D/ReadVariableOpข0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOpข%conv2d_1/Conv2D/Conv2D/ReadVariableOpข2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOpH
conv2d/Conv2D/ShapeShapeinput*
T0*
_output_shapes
:k
!conv2d/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
#conv2d/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????m
#conv2d/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv2d/Conv2D/strided_sliceStridedSliceconv2d/Conv2D/Shape:output:0*conv2d/Conv2D/strided_slice/stack:output:0,conv2d/Conv2D/strided_slice/stack_1:output:0,conv2d/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskt
conv2d/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         
conv2d/Conv2D/ReshapeReshapeinput$conv2d/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????
#conv2d/Conv2D/Conv2D/ReadVariableOpReadVariableOp,conv2d_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0อ
conv2d/Conv2D/Conv2DConv2Dconv2d/Conv2D/Reshape:output:0+conv2d/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
r
conv2d/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         d
conv2d/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????ภ
conv2d/Conv2D/concatConcatV2$conv2d/Conv2D/strided_slice:output:0&conv2d/Conv2D/concat/values_1:output:0"conv2d/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:
conv2d/Conv2D/Reshape_1Reshapeconv2d/Conv2D/Conv2D:output:0conv2d/Conv2D/concat:output:0*
T0*3
_output_shapes!
:?????????o
conv2d/squeeze_batch_dims/ShapeShape conv2d/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:w
-conv2d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
/conv2d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????y
/conv2d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ั
'conv2d/squeeze_batch_dims/strided_sliceStridedSlice(conv2d/squeeze_batch_dims/Shape:output:06conv2d/squeeze_batch_dims/strided_slice/stack:output:08conv2d/squeeze_batch_dims/strided_slice/stack_1:output:08conv2d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
'conv2d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         บ
!conv2d/squeeze_batch_dims/ReshapeReshape conv2d/Conv2D/Reshape_1:output:00conv2d/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????ฆ
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp9conv2d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ฬ
!conv2d/squeeze_batch_dims/BiasAddBiasAdd*conv2d/squeeze_batch_dims/Reshape:output:08conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????~
)conv2d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         p
%conv2d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????๐
 conv2d/squeeze_batch_dims/concatConcatV20conv2d/squeeze_batch_dims/strided_slice:output:02conv2d/squeeze_batch_dims/concat/values_1:output:0.conv2d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:ร
#conv2d/squeeze_batch_dims/Reshape_1Reshape*conv2d/squeeze_batch_dims/BiasAdd:output:0)conv2d/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:?????????q
conv2d_1/Conv2D/ShapeShape,conv2d/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:m
#conv2d_1/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_1/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????o
%conv2d_1/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv2d_1/Conv2D/strided_sliceStridedSliceconv2d_1/Conv2D/Shape:output:0,conv2d_1/Conv2D/strided_slice/stack:output:0.conv2d_1/Conv2D/strided_slice/stack_1:output:0.conv2d_1/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_1/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ฒ
conv2d_1/Conv2D/ReshapeReshape,conv2d/squeeze_batch_dims/Reshape_1:output:0&conv2d_1/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????
%conv2d_1/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ำ
conv2d_1/Conv2D/Conv2DConv2D conv2d_1/Conv2D/Reshape:output:0-conv2d_1/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
t
conv2d_1/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         f
conv2d_1/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????ศ
conv2d_1/Conv2D/concatConcatV2&conv2d_1/Conv2D/strided_slice:output:0(conv2d_1/Conv2D/concat/values_1:output:0$conv2d_1/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:ค
conv2d_1/Conv2D/Reshape_1Reshapeconv2d_1/Conv2D/Conv2D:output:0conv2d_1/Conv2D/concat:output:0*
T0*3
_output_shapes!
:?????????s
!conv2d_1/squeeze_batch_dims/ShapeShape"conv2d_1/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_1/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1conv2d_1/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????{
1conv2d_1/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)conv2d_1/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_1/squeeze_batch_dims/Shape:output:08conv2d_1/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_1/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_1/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
)conv2d_1/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ภ
#conv2d_1/squeeze_batch_dims/ReshapeReshape"conv2d_1/Conv2D/Reshape_1:output:02conv2d_1/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????ช
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0า
#conv2d_1/squeeze_batch_dims/BiasAddBiasAdd,conv2d_1/squeeze_batch_dims/Reshape:output:0:conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
+conv2d_1/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         r
'conv2d_1/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????๘
"conv2d_1/squeeze_batch_dims/concatConcatV22conv2d_1/squeeze_batch_dims/strided_slice:output:04conv2d_1/squeeze_batch_dims/concat/values_1:output:00conv2d_1/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:ษ
%conv2d_1/squeeze_batch_dims/Reshape_1Reshape,conv2d_1/squeeze_batch_dims/BiasAdd:output:0+conv2d_1/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:?????????
IdentityIdentity.conv2d_1/squeeze_batch_dims/Reshape_1:output:0^NoOp*
T0*3
_output_shapes!
:??????????
NoOpNoOp$^conv2d/Conv2D/Conv2D/ReadVariableOp1^conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_1/Conv2D/Conv2D/ReadVariableOp3^conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : 2J
#conv2d/Conv2D/Conv2D/ReadVariableOp#conv2d/Conv2D/Conv2D/ReadVariableOp2d
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_1/Conv2D/Conv2D/ReadVariableOp%conv2d_1/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp:Z V
3
_output_shapes!
:?????????

_user_specified_nameinput
ฮ

'__inference_dense_4_layer_call_fn_19184

inputs
unknown:
	unknown_0:
identityขStatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_16437s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
๏
L
0__inference_time_distributed_layer_call_fn_18286

inputs
identityร
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_15744m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&??????????????????:d `
<
_output_shapes*
(:&??????????????????
 
_user_specified_nameinputs

T
8__inference_global_average_pooling2d_layer_call_fn_18275

inputs
identityว
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_15721i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
๛
๒
-__inference_my_lstm_model_layer_call_fn_16475
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
identityขStatefulPartitionedCall
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
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_my_lstm_model_layer_call_and_return_conditional_losses_16444s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:?????????
!
_user_specified_name	input_1
ย	
ฒ
#__inference_rnn_layer_call_fn_18479

inputs
unknown:'
	unknown_0:
	unknown_1:'
	unknown_2:
	unknown_3:'
	unknown_4:
	unknown_5:'
	unknown_6:
identityขStatefulPartitionedCallฅ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_16650s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
ผb
ท
rnn_while_body_15550$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3#
rnn_while_rnn_strided_slice_1_0_
[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0O
=rnn_while_my_lstm_cell_dense_matmul_readvariableop_resource_0:'L
>rnn_while_my_lstm_cell_dense_biasadd_readvariableop_resource_0:Q
?rnn_while_my_lstm_cell_dense_1_matmul_readvariableop_resource_0:'N
@rnn_while_my_lstm_cell_dense_1_biasadd_readvariableop_resource_0:Q
?rnn_while_my_lstm_cell_dense_2_matmul_readvariableop_resource_0:'N
@rnn_while_my_lstm_cell_dense_2_biasadd_readvariableop_resource_0:Q
?rnn_while_my_lstm_cell_dense_3_matmul_readvariableop_resource_0:'N
@rnn_while_my_lstm_cell_dense_3_biasadd_readvariableop_resource_0:
rnn_while_identity
rnn_while_identity_1
rnn_while_identity_2
rnn_while_identity_3
rnn_while_identity_4
rnn_while_identity_5!
rnn_while_rnn_strided_slice_1]
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensorM
;rnn_while_my_lstm_cell_dense_matmul_readvariableop_resource:'J
<rnn_while_my_lstm_cell_dense_biasadd_readvariableop_resource:O
=rnn_while_my_lstm_cell_dense_1_matmul_readvariableop_resource:'L
>rnn_while_my_lstm_cell_dense_1_biasadd_readvariableop_resource:O
=rnn_while_my_lstm_cell_dense_2_matmul_readvariableop_resource:'L
>rnn_while_my_lstm_cell_dense_2_biasadd_readvariableop_resource:O
=rnn_while_my_lstm_cell_dense_3_matmul_readvariableop_resource:'L
>rnn_while_my_lstm_cell_dense_3_biasadd_readvariableop_resource:ข3rnn/while/my_lstm_cell/dense/BiasAdd/ReadVariableOpข2rnn/while/my_lstm_cell/dense/MatMul/ReadVariableOpข5rnn/while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOpข4rnn/while/my_lstm_cell/dense_1/MatMul/ReadVariableOpข5rnn/while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOpข4rnn/while/my_lstm_cell/dense_2/MatMul/ReadVariableOpข5rnn/while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOpข4rnn/while/my_lstm_cell/dense_3/MatMul/ReadVariableOp
;rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   บ
-rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0rnn_while_placeholderDrnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0r
0rnn/while/my_lstm_cell/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
+rnn/while/my_lstm_cell/concatenate_1/concatConcatV24rnn/while/TensorArrayV2Read/TensorListGetItem:item:0rnn_while_placeholder_29rnn/while/my_lstm_cell/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????'ฐ
2rnn/while/my_lstm_cell/dense/MatMul/ReadVariableOpReadVariableOp=rnn_while_my_lstm_cell_dense_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ั
#rnn/while/my_lstm_cell/dense/MatMulMatMul4rnn/while/my_lstm_cell/concatenate_1/concat:output:0:rnn/while/my_lstm_cell/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ฎ
3rnn/while/my_lstm_cell/dense/BiasAdd/ReadVariableOpReadVariableOp>rnn_while_my_lstm_cell_dense_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0อ
$rnn/while/my_lstm_cell/dense/BiasAddBiasAdd-rnn/while/my_lstm_cell/dense/MatMul:product:0;rnn/while/my_lstm_cell/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
$rnn/while/my_lstm_cell/dense/SigmoidSigmoid-rnn/while/my_lstm_cell/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
#rnn/while/my_lstm_cell/multiply/mulMulrnn_while_placeholder_3(rnn/while/my_lstm_cell/dense/Sigmoid:y:0*
T0*'
_output_shapes
:?????????ด
4rnn/while/my_lstm_cell/dense_1/MatMul/ReadVariableOpReadVariableOp?rnn_while_my_lstm_cell_dense_1_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ี
%rnn/while/my_lstm_cell/dense_1/MatMulMatMul4rnn/while/my_lstm_cell/concatenate_1/concat:output:0<rnn/while/my_lstm_cell/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ฒ
5rnn/while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOpReadVariableOp@rnn_while_my_lstm_cell_dense_1_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0ำ
&rnn/while/my_lstm_cell/dense_1/BiasAddBiasAdd/rnn/while/my_lstm_cell/dense_1/MatMul:product:0=rnn/while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
&rnn/while/my_lstm_cell/dense_1/SigmoidSigmoid/rnn/while/my_lstm_cell/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????ด
4rnn/while/my_lstm_cell/dense_2/MatMul/ReadVariableOpReadVariableOp?rnn_while_my_lstm_cell_dense_2_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ี
%rnn/while/my_lstm_cell/dense_2/MatMulMatMul4rnn/while/my_lstm_cell/concatenate_1/concat:output:0<rnn/while/my_lstm_cell/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ฒ
5rnn/while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOpReadVariableOp@rnn_while_my_lstm_cell_dense_2_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0ำ
&rnn/while/my_lstm_cell/dense_2/BiasAddBiasAdd/rnn/while/my_lstm_cell/dense_2/MatMul:product:0=rnn/while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
#rnn/while/my_lstm_cell/dense_2/TanhTanh/rnn/while/my_lstm_cell/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????ณ
%rnn/while/my_lstm_cell/multiply/mul_1Mul*rnn/while/my_lstm_cell/dense_1/Sigmoid:y:0'rnn/while/my_lstm_cell/dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????ฏ
 rnn/while/my_lstm_cell/add_1/addAddV2'rnn/while/my_lstm_cell/multiply/mul:z:0)rnn/while/my_lstm_cell/multiply/mul_1:z:0*
T0*'
_output_shapes
:?????????ด
4rnn/while/my_lstm_cell/dense_3/MatMul/ReadVariableOpReadVariableOp?rnn_while_my_lstm_cell_dense_3_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ี
%rnn/while/my_lstm_cell/dense_3/MatMulMatMul4rnn/while/my_lstm_cell/concatenate_1/concat:output:0<rnn/while/my_lstm_cell/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ฒ
5rnn/while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOpReadVariableOp@rnn_while_my_lstm_cell_dense_3_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0ำ
&rnn/while/my_lstm_cell/dense_3/BiasAddBiasAdd/rnn/while/my_lstm_cell/dense_3/MatMul:product:0=rnn/while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
&rnn/while/my_lstm_cell/dense_3/SigmoidSigmoid/rnn/while/my_lstm_cell/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
&rnn/while/my_lstm_cell/activation/TanhTanh$rnn/while/my_lstm_cell/add_1/add:z:0*
T0*'
_output_shapes
:?????????ถ
%rnn/while/my_lstm_cell/multiply/mul_2Mul*rnn/while/my_lstm_cell/activation/Tanh:y:0*rnn/while/my_lstm_cell/dense_3/Sigmoid:y:0*
T0*'
_output_shapes
:??????????
.rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrnn_while_placeholder_1rnn_while_placeholder)rnn/while/my_lstm_cell/multiply/mul_2:z:0*
_output_shapes
: *
element_dtype0:้่าQ
rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :h
rnn/while/addAddV2rnn_while_placeholderrnn/while/add/y:output:0*
T0*
_output_shapes
: S
rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :w
rnn/while/add_1AddV2 rnn_while_rnn_while_loop_counterrnn/while/add_1/y:output:0*
T0*
_output_shapes
: e
rnn/while/IdentityIdentityrnn/while/add_1:z:0^rnn/while/NoOp*
T0*
_output_shapes
: z
rnn/while/Identity_1Identity&rnn_while_rnn_while_maximum_iterations^rnn/while/NoOp*
T0*
_output_shapes
: e
rnn/while/Identity_2Identityrnn/while/add:z:0^rnn/while/NoOp*
T0*
_output_shapes
: 
rnn/while/Identity_3Identity>rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^rnn/while/NoOp*
T0*
_output_shapes
: 
rnn/while/Identity_4Identity)rnn/while/my_lstm_cell/multiply/mul_2:z:0^rnn/while/NoOp*
T0*'
_output_shapes
:?????????
rnn/while/Identity_5Identity$rnn/while/my_lstm_cell/add_1/add:z:0^rnn/while/NoOp*
T0*'
_output_shapes
:?????????
rnn/while/NoOpNoOp4^rnn/while/my_lstm_cell/dense/BiasAdd/ReadVariableOp3^rnn/while/my_lstm_cell/dense/MatMul/ReadVariableOp6^rnn/while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp5^rnn/while/my_lstm_cell/dense_1/MatMul/ReadVariableOp6^rnn/while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp5^rnn/while/my_lstm_cell/dense_2/MatMul/ReadVariableOp6^rnn/while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp5^rnn/while/my_lstm_cell/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "1
rnn_while_identityrnn/while/Identity:output:0"5
rnn_while_identity_1rnn/while/Identity_1:output:0"5
rnn_while_identity_2rnn/while/Identity_2:output:0"5
rnn_while_identity_3rnn/while/Identity_3:output:0"5
rnn_while_identity_4rnn/while/Identity_4:output:0"5
rnn_while_identity_5rnn/while/Identity_5:output:0"
>rnn_while_my_lstm_cell_dense_1_biasadd_readvariableop_resource@rnn_while_my_lstm_cell_dense_1_biasadd_readvariableop_resource_0"
=rnn_while_my_lstm_cell_dense_1_matmul_readvariableop_resource?rnn_while_my_lstm_cell_dense_1_matmul_readvariableop_resource_0"
>rnn_while_my_lstm_cell_dense_2_biasadd_readvariableop_resource@rnn_while_my_lstm_cell_dense_2_biasadd_readvariableop_resource_0"
=rnn_while_my_lstm_cell_dense_2_matmul_readvariableop_resource?rnn_while_my_lstm_cell_dense_2_matmul_readvariableop_resource_0"
>rnn_while_my_lstm_cell_dense_3_biasadd_readvariableop_resource@rnn_while_my_lstm_cell_dense_3_biasadd_readvariableop_resource_0"
=rnn_while_my_lstm_cell_dense_3_matmul_readvariableop_resource?rnn_while_my_lstm_cell_dense_3_matmul_readvariableop_resource_0"~
<rnn_while_my_lstm_cell_dense_biasadd_readvariableop_resource>rnn_while_my_lstm_cell_dense_biasadd_readvariableop_resource_0"|
;rnn_while_my_lstm_cell_dense_matmul_readvariableop_resource=rnn_while_my_lstm_cell_dense_matmul_readvariableop_resource_0"@
rnn_while_rnn_strided_slice_1rnn_while_rnn_strided_slice_1_0"ธ
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :?????????:?????????: : : : : : : : : : 2j
3rnn/while/my_lstm_cell/dense/BiasAdd/ReadVariableOp3rnn/while/my_lstm_cell/dense/BiasAdd/ReadVariableOp2h
2rnn/while/my_lstm_cell/dense/MatMul/ReadVariableOp2rnn/while/my_lstm_cell/dense/MatMul/ReadVariableOp2n
5rnn/while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp5rnn/while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp2l
4rnn/while/my_lstm_cell/dense_1/MatMul/ReadVariableOp4rnn/while/my_lstm_cell/dense_1/MatMul/ReadVariableOp2n
5rnn/while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp5rnn/while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp2l
4rnn/while/my_lstm_cell/dense_2/MatMul/ReadVariableOp4rnn/while/my_lstm_cell/dense_2/MatMul/ReadVariableOp2n
5rnn/while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp5rnn/while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp2l
4rnn/while/my_lstm_cell/dense_3/MatMul/ReadVariableOp4rnn/while/my_lstm_cell/dense_3/MatMul/ReadVariableOp: 
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
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?e
ร
__inference__traced_save_19390
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop<
8savev2_rnn_my_lstm_cell_dense_kernel_read_readvariableop:
6savev2_rnn_my_lstm_cell_dense_bias_read_readvariableop>
:savev2_rnn_my_lstm_cell_dense_1_kernel_read_readvariableop<
8savev2_rnn_my_lstm_cell_dense_1_bias_read_readvariableop>
:savev2_rnn_my_lstm_cell_dense_2_kernel_read_readvariableop<
8savev2_rnn_my_lstm_cell_dense_2_bias_read_readvariableop>
:savev2_rnn_my_lstm_cell_dense_3_kernel_read_readvariableop<
8savev2_rnn_my_lstm_cell_dense_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableopC
?savev2_adam_rnn_my_lstm_cell_dense_kernel_m_read_readvariableopA
=savev2_adam_rnn_my_lstm_cell_dense_bias_m_read_readvariableopE
Asavev2_adam_rnn_my_lstm_cell_dense_1_kernel_m_read_readvariableopC
?savev2_adam_rnn_my_lstm_cell_dense_1_bias_m_read_readvariableopE
Asavev2_adam_rnn_my_lstm_cell_dense_2_kernel_m_read_readvariableopC
?savev2_adam_rnn_my_lstm_cell_dense_2_bias_m_read_readvariableopE
Asavev2_adam_rnn_my_lstm_cell_dense_3_kernel_m_read_readvariableopC
?savev2_adam_rnn_my_lstm_cell_dense_3_bias_m_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableopC
?savev2_adam_rnn_my_lstm_cell_dense_kernel_v_read_readvariableopA
=savev2_adam_rnn_my_lstm_cell_dense_bias_v_read_readvariableopE
Asavev2_adam_rnn_my_lstm_cell_dense_1_kernel_v_read_readvariableopC
?savev2_adam_rnn_my_lstm_cell_dense_1_bias_v_read_readvariableopE
Asavev2_adam_rnn_my_lstm_cell_dense_2_kernel_v_read_readvariableopC
?savev2_adam_rnn_my_lstm_cell_dense_2_bias_v_read_readvariableopE
Asavev2_adam_rnn_my_lstm_cell_dense_3_kernel_v_read_readvariableopC
?savev2_adam_rnn_my_lstm_cell_dense_3_bias_v_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableop
savev2_const

identity_1ขMergeV2Checkpointsw
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
: ฟ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*่
value?B?4B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHี
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ๏
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop8savev2_rnn_my_lstm_cell_dense_kernel_read_readvariableop6savev2_rnn_my_lstm_cell_dense_bias_read_readvariableop:savev2_rnn_my_lstm_cell_dense_1_kernel_read_readvariableop8savev2_rnn_my_lstm_cell_dense_1_bias_read_readvariableop:savev2_rnn_my_lstm_cell_dense_2_kernel_read_readvariableop8savev2_rnn_my_lstm_cell_dense_2_bias_read_readvariableop:savev2_rnn_my_lstm_cell_dense_3_kernel_read_readvariableop8savev2_rnn_my_lstm_cell_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop?savev2_adam_rnn_my_lstm_cell_dense_kernel_m_read_readvariableop=savev2_adam_rnn_my_lstm_cell_dense_bias_m_read_readvariableopAsavev2_adam_rnn_my_lstm_cell_dense_1_kernel_m_read_readvariableop?savev2_adam_rnn_my_lstm_cell_dense_1_bias_m_read_readvariableopAsavev2_adam_rnn_my_lstm_cell_dense_2_kernel_m_read_readvariableop?savev2_adam_rnn_my_lstm_cell_dense_2_bias_m_read_readvariableopAsavev2_adam_rnn_my_lstm_cell_dense_3_kernel_m_read_readvariableop?savev2_adam_rnn_my_lstm_cell_dense_3_bias_m_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop?savev2_adam_rnn_my_lstm_cell_dense_kernel_v_read_readvariableop=savev2_adam_rnn_my_lstm_cell_dense_bias_v_read_readvariableopAsavev2_adam_rnn_my_lstm_cell_dense_1_kernel_v_read_readvariableop?savev2_adam_rnn_my_lstm_cell_dense_1_bias_v_read_readvariableopAsavev2_adam_rnn_my_lstm_cell_dense_2_kernel_v_read_readvariableop?savev2_adam_rnn_my_lstm_cell_dense_2_bias_v_read_readvariableopAsavev2_adam_rnn_my_lstm_cell_dense_3_kernel_v_read_readvariableop?savev2_adam_rnn_my_lstm_cell_dense_3_bias_v_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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

identity_1Identity_1:output:0*ซ
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
: 
ผb
ท
rnn_while_body_17874$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3#
rnn_while_rnn_strided_slice_1_0_
[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0O
=rnn_while_my_lstm_cell_dense_matmul_readvariableop_resource_0:'L
>rnn_while_my_lstm_cell_dense_biasadd_readvariableop_resource_0:Q
?rnn_while_my_lstm_cell_dense_1_matmul_readvariableop_resource_0:'N
@rnn_while_my_lstm_cell_dense_1_biasadd_readvariableop_resource_0:Q
?rnn_while_my_lstm_cell_dense_2_matmul_readvariableop_resource_0:'N
@rnn_while_my_lstm_cell_dense_2_biasadd_readvariableop_resource_0:Q
?rnn_while_my_lstm_cell_dense_3_matmul_readvariableop_resource_0:'N
@rnn_while_my_lstm_cell_dense_3_biasadd_readvariableop_resource_0:
rnn_while_identity
rnn_while_identity_1
rnn_while_identity_2
rnn_while_identity_3
rnn_while_identity_4
rnn_while_identity_5!
rnn_while_rnn_strided_slice_1]
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensorM
;rnn_while_my_lstm_cell_dense_matmul_readvariableop_resource:'J
<rnn_while_my_lstm_cell_dense_biasadd_readvariableop_resource:O
=rnn_while_my_lstm_cell_dense_1_matmul_readvariableop_resource:'L
>rnn_while_my_lstm_cell_dense_1_biasadd_readvariableop_resource:O
=rnn_while_my_lstm_cell_dense_2_matmul_readvariableop_resource:'L
>rnn_while_my_lstm_cell_dense_2_biasadd_readvariableop_resource:O
=rnn_while_my_lstm_cell_dense_3_matmul_readvariableop_resource:'L
>rnn_while_my_lstm_cell_dense_3_biasadd_readvariableop_resource:ข3rnn/while/my_lstm_cell/dense/BiasAdd/ReadVariableOpข2rnn/while/my_lstm_cell/dense/MatMul/ReadVariableOpข5rnn/while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOpข4rnn/while/my_lstm_cell/dense_1/MatMul/ReadVariableOpข5rnn/while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOpข4rnn/while/my_lstm_cell/dense_2/MatMul/ReadVariableOpข5rnn/while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOpข4rnn/while/my_lstm_cell/dense_3/MatMul/ReadVariableOp
;rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   บ
-rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0rnn_while_placeholderDrnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0r
0rnn/while/my_lstm_cell/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
+rnn/while/my_lstm_cell/concatenate_1/concatConcatV24rnn/while/TensorArrayV2Read/TensorListGetItem:item:0rnn_while_placeholder_29rnn/while/my_lstm_cell/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????'ฐ
2rnn/while/my_lstm_cell/dense/MatMul/ReadVariableOpReadVariableOp=rnn_while_my_lstm_cell_dense_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ั
#rnn/while/my_lstm_cell/dense/MatMulMatMul4rnn/while/my_lstm_cell/concatenate_1/concat:output:0:rnn/while/my_lstm_cell/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ฎ
3rnn/while/my_lstm_cell/dense/BiasAdd/ReadVariableOpReadVariableOp>rnn_while_my_lstm_cell_dense_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0อ
$rnn/while/my_lstm_cell/dense/BiasAddBiasAdd-rnn/while/my_lstm_cell/dense/MatMul:product:0;rnn/while/my_lstm_cell/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
$rnn/while/my_lstm_cell/dense/SigmoidSigmoid-rnn/while/my_lstm_cell/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
#rnn/while/my_lstm_cell/multiply/mulMulrnn_while_placeholder_3(rnn/while/my_lstm_cell/dense/Sigmoid:y:0*
T0*'
_output_shapes
:?????????ด
4rnn/while/my_lstm_cell/dense_1/MatMul/ReadVariableOpReadVariableOp?rnn_while_my_lstm_cell_dense_1_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ี
%rnn/while/my_lstm_cell/dense_1/MatMulMatMul4rnn/while/my_lstm_cell/concatenate_1/concat:output:0<rnn/while/my_lstm_cell/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ฒ
5rnn/while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOpReadVariableOp@rnn_while_my_lstm_cell_dense_1_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0ำ
&rnn/while/my_lstm_cell/dense_1/BiasAddBiasAdd/rnn/while/my_lstm_cell/dense_1/MatMul:product:0=rnn/while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
&rnn/while/my_lstm_cell/dense_1/SigmoidSigmoid/rnn/while/my_lstm_cell/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????ด
4rnn/while/my_lstm_cell/dense_2/MatMul/ReadVariableOpReadVariableOp?rnn_while_my_lstm_cell_dense_2_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ี
%rnn/while/my_lstm_cell/dense_2/MatMulMatMul4rnn/while/my_lstm_cell/concatenate_1/concat:output:0<rnn/while/my_lstm_cell/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ฒ
5rnn/while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOpReadVariableOp@rnn_while_my_lstm_cell_dense_2_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0ำ
&rnn/while/my_lstm_cell/dense_2/BiasAddBiasAdd/rnn/while/my_lstm_cell/dense_2/MatMul:product:0=rnn/while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
#rnn/while/my_lstm_cell/dense_2/TanhTanh/rnn/while/my_lstm_cell/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????ณ
%rnn/while/my_lstm_cell/multiply/mul_1Mul*rnn/while/my_lstm_cell/dense_1/Sigmoid:y:0'rnn/while/my_lstm_cell/dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????ฏ
 rnn/while/my_lstm_cell/add_1/addAddV2'rnn/while/my_lstm_cell/multiply/mul:z:0)rnn/while/my_lstm_cell/multiply/mul_1:z:0*
T0*'
_output_shapes
:?????????ด
4rnn/while/my_lstm_cell/dense_3/MatMul/ReadVariableOpReadVariableOp?rnn_while_my_lstm_cell_dense_3_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ี
%rnn/while/my_lstm_cell/dense_3/MatMulMatMul4rnn/while/my_lstm_cell/concatenate_1/concat:output:0<rnn/while/my_lstm_cell/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ฒ
5rnn/while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOpReadVariableOp@rnn_while_my_lstm_cell_dense_3_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0ำ
&rnn/while/my_lstm_cell/dense_3/BiasAddBiasAdd/rnn/while/my_lstm_cell/dense_3/MatMul:product:0=rnn/while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
&rnn/while/my_lstm_cell/dense_3/SigmoidSigmoid/rnn/while/my_lstm_cell/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
&rnn/while/my_lstm_cell/activation/TanhTanh$rnn/while/my_lstm_cell/add_1/add:z:0*
T0*'
_output_shapes
:?????????ถ
%rnn/while/my_lstm_cell/multiply/mul_2Mul*rnn/while/my_lstm_cell/activation/Tanh:y:0*rnn/while/my_lstm_cell/dense_3/Sigmoid:y:0*
T0*'
_output_shapes
:??????????
.rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrnn_while_placeholder_1rnn_while_placeholder)rnn/while/my_lstm_cell/multiply/mul_2:z:0*
_output_shapes
: *
element_dtype0:้่าQ
rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :h
rnn/while/addAddV2rnn_while_placeholderrnn/while/add/y:output:0*
T0*
_output_shapes
: S
rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :w
rnn/while/add_1AddV2 rnn_while_rnn_while_loop_counterrnn/while/add_1/y:output:0*
T0*
_output_shapes
: e
rnn/while/IdentityIdentityrnn/while/add_1:z:0^rnn/while/NoOp*
T0*
_output_shapes
: z
rnn/while/Identity_1Identity&rnn_while_rnn_while_maximum_iterations^rnn/while/NoOp*
T0*
_output_shapes
: e
rnn/while/Identity_2Identityrnn/while/add:z:0^rnn/while/NoOp*
T0*
_output_shapes
: 
rnn/while/Identity_3Identity>rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^rnn/while/NoOp*
T0*
_output_shapes
: 
rnn/while/Identity_4Identity)rnn/while/my_lstm_cell/multiply/mul_2:z:0^rnn/while/NoOp*
T0*'
_output_shapes
:?????????
rnn/while/Identity_5Identity$rnn/while/my_lstm_cell/add_1/add:z:0^rnn/while/NoOp*
T0*'
_output_shapes
:?????????
rnn/while/NoOpNoOp4^rnn/while/my_lstm_cell/dense/BiasAdd/ReadVariableOp3^rnn/while/my_lstm_cell/dense/MatMul/ReadVariableOp6^rnn/while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp5^rnn/while/my_lstm_cell/dense_1/MatMul/ReadVariableOp6^rnn/while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp5^rnn/while/my_lstm_cell/dense_2/MatMul/ReadVariableOp6^rnn/while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp5^rnn/while/my_lstm_cell/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "1
rnn_while_identityrnn/while/Identity:output:0"5
rnn_while_identity_1rnn/while/Identity_1:output:0"5
rnn_while_identity_2rnn/while/Identity_2:output:0"5
rnn_while_identity_3rnn/while/Identity_3:output:0"5
rnn_while_identity_4rnn/while/Identity_4:output:0"5
rnn_while_identity_5rnn/while/Identity_5:output:0"
>rnn_while_my_lstm_cell_dense_1_biasadd_readvariableop_resource@rnn_while_my_lstm_cell_dense_1_biasadd_readvariableop_resource_0"
=rnn_while_my_lstm_cell_dense_1_matmul_readvariableop_resource?rnn_while_my_lstm_cell_dense_1_matmul_readvariableop_resource_0"
>rnn_while_my_lstm_cell_dense_2_biasadd_readvariableop_resource@rnn_while_my_lstm_cell_dense_2_biasadd_readvariableop_resource_0"
=rnn_while_my_lstm_cell_dense_2_matmul_readvariableop_resource?rnn_while_my_lstm_cell_dense_2_matmul_readvariableop_resource_0"
>rnn_while_my_lstm_cell_dense_3_biasadd_readvariableop_resource@rnn_while_my_lstm_cell_dense_3_biasadd_readvariableop_resource_0"
=rnn_while_my_lstm_cell_dense_3_matmul_readvariableop_resource?rnn_while_my_lstm_cell_dense_3_matmul_readvariableop_resource_0"~
<rnn_while_my_lstm_cell_dense_biasadd_readvariableop_resource>rnn_while_my_lstm_cell_dense_biasadd_readvariableop_resource_0"|
;rnn_while_my_lstm_cell_dense_matmul_readvariableop_resource=rnn_while_my_lstm_cell_dense_matmul_readvariableop_resource_0"@
rnn_while_rnn_strided_slice_1rnn_while_rnn_strided_slice_1_0"ธ
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :?????????:?????????: : : : : : : : : : 2j
3rnn/while/my_lstm_cell/dense/BiasAdd/ReadVariableOp3rnn/while/my_lstm_cell/dense/BiasAdd/ReadVariableOp2h
2rnn/while/my_lstm_cell/dense/MatMul/ReadVariableOp2rnn/while/my_lstm_cell/dense/MatMul/ReadVariableOp2n
5rnn/while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp5rnn/while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp2l
4rnn/while/my_lstm_cell/dense_1/MatMul/ReadVariableOp4rnn/while/my_lstm_cell/dense_1/MatMul/ReadVariableOp2n
5rnn/while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp5rnn/while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp2l
4rnn/while/my_lstm_cell/dense_2/MatMul/ReadVariableOp4rnn/while/my_lstm_cell/dense_2/MatMul/ReadVariableOp2n
5rnn/while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp5rnn/while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp2l
4rnn/while/my_lstm_cell/dense_3/MatMul/ReadVariableOp4rnn/while/my_lstm_cell/dense_3/MatMul/ReadVariableOp: 
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
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
ฎ

__inference_call_17252
x,
my_cnn_block_17038: 
my_cnn_block_17040:,
my_cnn_block_17042: 
my_cnn_block_17044:G
5rnn_my_lstm_cell_dense_matmul_readvariableop_resource:'D
6rnn_my_lstm_cell_dense_biasadd_readvariableop_resource:I
7rnn_my_lstm_cell_dense_1_matmul_readvariableop_resource:'F
8rnn_my_lstm_cell_dense_1_biasadd_readvariableop_resource:I
7rnn_my_lstm_cell_dense_2_matmul_readvariableop_resource:'F
8rnn_my_lstm_cell_dense_2_biasadd_readvariableop_resource:I
7rnn_my_lstm_cell_dense_3_matmul_readvariableop_resource:'F
8rnn_my_lstm_cell_dense_3_biasadd_readvariableop_resource:;
)dense_4_tensordot_readvariableop_resource:5
'dense_4_biasadd_readvariableop_resource:
identityขdense_4/BiasAdd/ReadVariableOpข dense_4/Tensordot/ReadVariableOpข$my_cnn_block/StatefulPartitionedCallข-rnn/my_lstm_cell/dense/BiasAdd/ReadVariableOpข,rnn/my_lstm_cell/dense/MatMul/ReadVariableOpข/rnn/my_lstm_cell/dense_1/BiasAdd/ReadVariableOpข.rnn/my_lstm_cell/dense_1/MatMul/ReadVariableOpข/rnn/my_lstm_cell/dense_2/BiasAdd/ReadVariableOpข.rnn/my_lstm_cell/dense_2/MatMul/ReadVariableOpข/rnn/my_lstm_cell/dense_3/BiasAdd/ReadVariableOpข.rnn/my_lstm_cell/dense_3/MatMul/ReadVariableOpข	rnn/while?
$my_cnn_block/StatefulPartitionedCallStatefulPartitionedCallxmy_cnn_block_17038my_cnn_block_17040my_cnn_block_17042my_cnn_block_17044*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *
fR
__inference_call_17037w
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ต
time_distributed/ReshapeReshape-my_cnn_block/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????
@time_distributed/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ึ
.time_distributed/global_average_pooling2d/MeanMean!time_distributed/Reshape:output:0Itime_distributed/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????u
 time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      ฟ
time_distributed/Reshape_1Reshape7time_distributed/global_average_pooling2d/Mean:output:0)time_distributed/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????y
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         น
time_distributed/Reshape_2Reshape-my_cnn_block/StatefulPartitionedCall:output:0)time_distributed/Reshape_2/shape:output:0*
T0*/
_output_shapes
:?????????\
	rnn/ShapeShape#time_distributed/Reshape_1:output:0*
T0*
_output_shapes
:a
rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ๅ
rnn/strided_sliceStridedSlicernn/Shape:output:0 rnn/strided_slice/stack:output:0"rnn/strided_slice/stack_1:output:0"rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
rnn/zeros/packedPackrnn/strided_slice:output:0rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:T
rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x
	rnn/zerosFillrnn/zeros/packed:output:0rnn/zeros/Const:output:0*
T0*'
_output_shapes
:?????????V
rnn/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
rnn/zeros_1/packedPackrnn/strided_slice:output:0rnn/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:V
rnn/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
rnn/zeros_1Fillrnn/zeros_1/packed:output:0rnn/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????g
rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
rnn/transpose	Transpose#time_distributed/Reshape_1:output:0rnn/transpose/perm:output:0*
T0*+
_output_shapes
:?????????L
rnn/Shape_1Shapernn/transpose:y:0*
T0*
_output_shapes
:c
rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:๏
rnn/strided_slice_1StridedSlicernn/Shape_1:output:0"rnn/strided_slice_1/stack:output:0$rnn/strided_slice_1/stack_1:output:0$rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????ภ
rnn/TensorArrayV2TensorListReserve(rnn/TensorArrayV2/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่า
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ์
+rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn/transpose:y:0Brnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่าc
rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
rnn/strided_slice_2StridedSlicernn/transpose:y:0"rnn/strided_slice_2/stack:output:0$rnn/strided_slice_2/stack_1:output:0$rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskl
*rnn/my_lstm_cell/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ำ
%rnn/my_lstm_cell/concatenate_1/concatConcatV2rnn/strided_slice_2:output:0rnn/zeros:output:03rnn/my_lstm_cell/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????'ข
,rnn/my_lstm_cell/dense/MatMul/ReadVariableOpReadVariableOp5rnn_my_lstm_cell_dense_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0ฟ
rnn/my_lstm_cell/dense/MatMulMatMul.rnn/my_lstm_cell/concatenate_1/concat:output:04rnn/my_lstm_cell/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
-rnn/my_lstm_cell/dense/BiasAdd/ReadVariableOpReadVariableOp6rnn_my_lstm_cell_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ป
rnn/my_lstm_cell/dense/BiasAddBiasAdd'rnn/my_lstm_cell/dense/MatMul:product:05rnn/my_lstm_cell/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
rnn/my_lstm_cell/dense/SigmoidSigmoid'rnn/my_lstm_cell/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
rnn/my_lstm_cell/multiply/mulMulrnn/zeros_1:output:0"rnn/my_lstm_cell/dense/Sigmoid:y:0*
T0*'
_output_shapes
:?????????ฆ
.rnn/my_lstm_cell/dense_1/MatMul/ReadVariableOpReadVariableOp7rnn_my_lstm_cell_dense_1_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0ร
rnn/my_lstm_cell/dense_1/MatMulMatMul.rnn/my_lstm_cell/concatenate_1/concat:output:06rnn/my_lstm_cell/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ค
/rnn/my_lstm_cell/dense_1/BiasAdd/ReadVariableOpReadVariableOp8rnn_my_lstm_cell_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ม
 rnn/my_lstm_cell/dense_1/BiasAddBiasAdd)rnn/my_lstm_cell/dense_1/MatMul:product:07rnn/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
 rnn/my_lstm_cell/dense_1/SigmoidSigmoid)rnn/my_lstm_cell/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????ฆ
.rnn/my_lstm_cell/dense_2/MatMul/ReadVariableOpReadVariableOp7rnn_my_lstm_cell_dense_2_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0ร
rnn/my_lstm_cell/dense_2/MatMulMatMul.rnn/my_lstm_cell/concatenate_1/concat:output:06rnn/my_lstm_cell/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ค
/rnn/my_lstm_cell/dense_2/BiasAdd/ReadVariableOpReadVariableOp8rnn_my_lstm_cell_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ม
 rnn/my_lstm_cell/dense_2/BiasAddBiasAdd)rnn/my_lstm_cell/dense_2/MatMul:product:07rnn/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
rnn/my_lstm_cell/dense_2/TanhTanh)rnn/my_lstm_cell/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????ก
rnn/my_lstm_cell/multiply/mul_1Mul$rnn/my_lstm_cell/dense_1/Sigmoid:y:0!rnn/my_lstm_cell/dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????
rnn/my_lstm_cell/add_1/addAddV2!rnn/my_lstm_cell/multiply/mul:z:0#rnn/my_lstm_cell/multiply/mul_1:z:0*
T0*'
_output_shapes
:?????????ฆ
.rnn/my_lstm_cell/dense_3/MatMul/ReadVariableOpReadVariableOp7rnn_my_lstm_cell_dense_3_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0ร
rnn/my_lstm_cell/dense_3/MatMulMatMul.rnn/my_lstm_cell/concatenate_1/concat:output:06rnn/my_lstm_cell/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ค
/rnn/my_lstm_cell/dense_3/BiasAdd/ReadVariableOpReadVariableOp8rnn_my_lstm_cell_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ม
 rnn/my_lstm_cell/dense_3/BiasAddBiasAdd)rnn/my_lstm_cell/dense_3/MatMul:product:07rnn/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
 rnn/my_lstm_cell/dense_3/SigmoidSigmoid)rnn/my_lstm_cell/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????z
 rnn/my_lstm_cell/activation/TanhTanhrnn/my_lstm_cell/add_1/add:z:0*
T0*'
_output_shapes
:?????????ค
rnn/my_lstm_cell/multiply/mul_2Mul$rnn/my_lstm_cell/activation/Tanh:y:0$rnn/my_lstm_cell/dense_3/Sigmoid:y:0*
T0*'
_output_shapes
:?????????r
!rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ฤ
rnn/TensorArrayV2_1TensorListReserve*rnn/TensorArrayV2_1/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่าJ
rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : g
rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????X
rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 	
	rnn/whileWhilernn/while/loop_counter:output:0%rnn/while/maximum_iterations:output:0rnn/time:output:0rnn/TensorArrayV2_1:handle:0rnn/zeros:output:0rnn/zeros_1:output:0rnn/strided_slice_1:output:0;rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:05rnn_my_lstm_cell_dense_matmul_readvariableop_resource6rnn_my_lstm_cell_dense_biasadd_readvariableop_resource7rnn_my_lstm_cell_dense_1_matmul_readvariableop_resource8rnn_my_lstm_cell_dense_1_biasadd_readvariableop_resource7rnn_my_lstm_cell_dense_2_matmul_readvariableop_resource8rnn_my_lstm_cell_dense_2_biasadd_readvariableop_resource7rnn_my_lstm_cell_dense_3_matmul_readvariableop_resource8rnn_my_lstm_cell_dense_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( * 
bodyR
rnn_while_body_17122* 
condR
rnn_while_cond_17121*U
output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : *
parallel_iterations 
4rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ฮ
&rnn/TensorArrayV2Stack/TensorListStackTensorListStackrnn/while:output:3=rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype0l
rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????e
rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
rnn/strided_slice_3StridedSlice/rnn/TensorArrayV2Stack/TensorListStack:tensor:0"rnn/strided_slice_3/stack:output:0$rnn/strided_slice_3/stack_1:output:0$rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maski
rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ข
rnn/transpose_1	Transpose/rnn/TensorArrayV2Stack/TensorListStack:tensor:0rnn/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Z
dense_4/Tensordot/ShapeShapernn/transpose_1:y:0*
T0*
_output_shapes
:a
dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ฿
dense_4/Tensordot/GatherV2_1GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/axes:output:0*dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ผ
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_4/Tensordot/transpose	Transposernn/transpose_1:y:0!dense_4/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????ข
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????ข
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????c
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ว
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????k
IdentityIdentitydense_4/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????ล
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp%^my_cnn_block/StatefulPartitionedCall.^rnn/my_lstm_cell/dense/BiasAdd/ReadVariableOp-^rnn/my_lstm_cell/dense/MatMul/ReadVariableOp0^rnn/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp/^rnn/my_lstm_cell/dense_1/MatMul/ReadVariableOp0^rnn/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp/^rnn/my_lstm_cell/dense_2/MatMul/ReadVariableOp0^rnn/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp/^rnn/my_lstm_cell/dense_3/MatMul/ReadVariableOp
^rnn/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp2L
$my_cnn_block/StatefulPartitionedCall$my_cnn_block/StatefulPartitionedCall2^
-rnn/my_lstm_cell/dense/BiasAdd/ReadVariableOp-rnn/my_lstm_cell/dense/BiasAdd/ReadVariableOp2\
,rnn/my_lstm_cell/dense/MatMul/ReadVariableOp,rnn/my_lstm_cell/dense/MatMul/ReadVariableOp2b
/rnn/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp/rnn/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp2`
.rnn/my_lstm_cell/dense_1/MatMul/ReadVariableOp.rnn/my_lstm_cell/dense_1/MatMul/ReadVariableOp2b
/rnn/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp/rnn/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp2`
.rnn/my_lstm_cell/dense_2/MatMul/ReadVariableOp.rnn/my_lstm_cell/dense_2/MatMul/ReadVariableOp2b
/rnn/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp/rnn/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp2`
.rnn/my_lstm_cell/dense_3/MatMul/ReadVariableOp.rnn/my_lstm_cell/dense_3/MatMul/ReadVariableOp2
	rnn/while	rnn/while:V R
3
_output_shapes!
:?????????

_user_specified_namex
ฐe
๗
>__inference_rnn_layer_call_and_return_conditional_losses_18653
inputs_0C
1my_lstm_cell_dense_matmul_readvariableop_resource:'@
2my_lstm_cell_dense_biasadd_readvariableop_resource:E
3my_lstm_cell_dense_1_matmul_readvariableop_resource:'B
4my_lstm_cell_dense_1_biasadd_readvariableop_resource:E
3my_lstm_cell_dense_2_matmul_readvariableop_resource:'B
4my_lstm_cell_dense_2_biasadd_readvariableop_resource:E
3my_lstm_cell_dense_3_matmul_readvariableop_resource:'B
4my_lstm_cell_dense_3_biasadd_readvariableop_resource:
identityข)my_lstm_cell/dense/BiasAdd/ReadVariableOpข(my_lstm_cell/dense/MatMul/ReadVariableOpข+my_lstm_cell/dense_1/BiasAdd/ReadVariableOpข*my_lstm_cell/dense_1/MatMul/ReadVariableOpข+my_lstm_cell/dense_2/BiasAdd/ReadVariableOpข*my_lstm_cell/dense_2/MatMul/ReadVariableOpข+my_lstm_cell/dense_3/BiasAdd/ReadVariableOpข*my_lstm_cell/dense_3/MatMul/ReadVariableOpขwhile=
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
valueB:ั
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
:?????????R
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
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
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
valueB:?
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
?????????ด
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่า
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   เ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่า_
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
valueB:้
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskh
&my_lstm_cell/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ร
!my_lstm_cell/concatenate_1/concatConcatV2strided_slice_2:output:0zeros:output:0/my_lstm_cell/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????'
(my_lstm_cell/dense/MatMul/ReadVariableOpReadVariableOp1my_lstm_cell_dense_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0ณ
my_lstm_cell/dense/MatMulMatMul*my_lstm_cell/concatenate_1/concat:output:00my_lstm_cell/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
)my_lstm_cell/dense/BiasAdd/ReadVariableOpReadVariableOp2my_lstm_cell_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ฏ
my_lstm_cell/dense/BiasAddBiasAdd#my_lstm_cell/dense/MatMul:product:01my_lstm_cell/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????|
my_lstm_cell/dense/SigmoidSigmoid#my_lstm_cell/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
my_lstm_cell/multiply/mulMulzeros_1:output:0my_lstm_cell/dense/Sigmoid:y:0*
T0*'
_output_shapes
:?????????
*my_lstm_cell/dense_1/MatMul/ReadVariableOpReadVariableOp3my_lstm_cell_dense_1_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0ท
my_lstm_cell/dense_1/MatMulMatMul*my_lstm_cell/concatenate_1/concat:output:02my_lstm_cell/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
+my_lstm_cell/dense_1/BiasAdd/ReadVariableOpReadVariableOp4my_lstm_cell_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ต
my_lstm_cell/dense_1/BiasAddBiasAdd%my_lstm_cell/dense_1/MatMul:product:03my_lstm_cell/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
my_lstm_cell/dense_1/SigmoidSigmoid%my_lstm_cell/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
*my_lstm_cell/dense_2/MatMul/ReadVariableOpReadVariableOp3my_lstm_cell_dense_2_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0ท
my_lstm_cell/dense_2/MatMulMatMul*my_lstm_cell/concatenate_1/concat:output:02my_lstm_cell/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
+my_lstm_cell/dense_2/BiasAdd/ReadVariableOpReadVariableOp4my_lstm_cell_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ต
my_lstm_cell/dense_2/BiasAddBiasAdd%my_lstm_cell/dense_2/MatMul:product:03my_lstm_cell/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????z
my_lstm_cell/dense_2/TanhTanh%my_lstm_cell/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
my_lstm_cell/multiply/mul_1Mul my_lstm_cell/dense_1/Sigmoid:y:0my_lstm_cell/dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????
my_lstm_cell/add_1/addAddV2my_lstm_cell/multiply/mul:z:0my_lstm_cell/multiply/mul_1:z:0*
T0*'
_output_shapes
:?????????
*my_lstm_cell/dense_3/MatMul/ReadVariableOpReadVariableOp3my_lstm_cell_dense_3_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0ท
my_lstm_cell/dense_3/MatMulMatMul*my_lstm_cell/concatenate_1/concat:output:02my_lstm_cell/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
+my_lstm_cell/dense_3/BiasAdd/ReadVariableOpReadVariableOp4my_lstm_cell_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ต
my_lstm_cell/dense_3/BiasAddBiasAdd%my_lstm_cell/dense_3/MatMul:product:03my_lstm_cell/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
my_lstm_cell/dense_3/SigmoidSigmoid%my_lstm_cell/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????r
my_lstm_cell/activation/TanhTanhmy_lstm_cell/add_1/add:z:0*
T0*'
_output_shapes
:?????????
my_lstm_cell/multiply/mul_2Mul my_lstm_cell/activation/Tanh:y:0 my_lstm_cell/dense_3/Sigmoid:y:0*
T0*'
_output_shapes
:?????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ธ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่าF
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
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ผ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01my_lstm_cell_dense_matmul_readvariableop_resource2my_lstm_cell_dense_biasadd_readvariableop_resource3my_lstm_cell_dense_1_matmul_readvariableop_resource4my_lstm_cell_dense_1_biasadd_readvariableop_resource3my_lstm_cell_dense_2_matmul_readvariableop_resource4my_lstm_cell_dense_2_biasadd_readvariableop_resource3my_lstm_cell_dense_3_matmul_readvariableop_resource4my_lstm_cell_dense_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( *
bodyR
while_body_18549*
condR
while_cond_18548*U
output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ห
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
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
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :??????????????????ถ
NoOpNoOp*^my_lstm_cell/dense/BiasAdd/ReadVariableOp)^my_lstm_cell/dense/MatMul/ReadVariableOp,^my_lstm_cell/dense_1/BiasAdd/ReadVariableOp+^my_lstm_cell/dense_1/MatMul/ReadVariableOp,^my_lstm_cell/dense_2/BiasAdd/ReadVariableOp+^my_lstm_cell/dense_2/MatMul/ReadVariableOp,^my_lstm_cell/dense_3/BiasAdd/ReadVariableOp+^my_lstm_cell/dense_3/MatMul/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????????????: : : : : : : : 2V
)my_lstm_cell/dense/BiasAdd/ReadVariableOp)my_lstm_cell/dense/BiasAdd/ReadVariableOp2T
(my_lstm_cell/dense/MatMul/ReadVariableOp(my_lstm_cell/dense/MatMul/ReadVariableOp2Z
+my_lstm_cell/dense_1/BiasAdd/ReadVariableOp+my_lstm_cell/dense_1/BiasAdd/ReadVariableOp2X
*my_lstm_cell/dense_1/MatMul/ReadVariableOp*my_lstm_cell/dense_1/MatMul/ReadVariableOp2Z
+my_lstm_cell/dense_2/BiasAdd/ReadVariableOp+my_lstm_cell/dense_2/BiasAdd/ReadVariableOp2X
*my_lstm_cell/dense_2/MatMul/ReadVariableOp*my_lstm_cell/dense_2/MatMul/ReadVariableOp2Z
+my_lstm_cell/dense_3/BiasAdd/ReadVariableOp+my_lstm_cell/dense_3/BiasAdd/ReadVariableOp2X
*my_lstm_cell/dense_3/MatMul/ReadVariableOp*my_lstm_cell/dense_3/MatMul/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
อ:
ฌ
>__inference_rnn_layer_call_and_return_conditional_losses_15958

inputs$
my_lstm_cell_15847:' 
my_lstm_cell_15849:$
my_lstm_cell_15851:' 
my_lstm_cell_15853:$
my_lstm_cell_15855:' 
my_lstm_cell_15857:$
my_lstm_cell_15859:' 
my_lstm_cell_15861:
identityข$my_lstm_cell/StatefulPartitionedCallขwhile;
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
valueB:ั
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
:?????????R
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
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
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
valueB:?
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
?????????ด
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่า
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   เ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่า_
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
valueB:้
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask฿
$my_lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0my_lstm_cell_15847my_lstm_cell_15849my_lstm_cell_15851my_lstm_cell_15853my_lstm_cell_15855my_lstm_cell_15857my_lstm_cell_15859my_lstm_cell_15861*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_my_lstm_cell_layer_call_and_return_conditional_losses_15846n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ธ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่าF
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
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ด
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0my_lstm_cell_15847my_lstm_cell_15849my_lstm_cell_15851my_lstm_cell_15853my_lstm_cell_15855my_lstm_cell_15857my_lstm_cell_15859my_lstm_cell_15861*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( *
bodyR
while_body_15870*
condR
while_cond_15869*U
output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ห
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
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
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :??????????????????u
NoOpNoOp%^my_lstm_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????????????: : : : : : : : 2L
$my_lstm_cell/StatefulPartitionedCall$my_lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
ู
?
,__inference_my_cnn_block_layer_call_fn_18137	
input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_16235{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
3
_output_shapes!
:?????????

_user_specified_nameinput
อ

ว
while_cond_18722
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_18722___redundant_placeholder03
/while_while_cond_18722___redundant_placeholder13
/while_while_cond_18722___redundant_placeholder23
/while_while_cond_18722___redundant_placeholder33
/while_while_cond_18722___redundant_placeholder43
/while_while_cond_18722___redundant_placeholder53
/while_while_cond_18722___redundant_placeholder63
/while_while_cond_18722___redundant_placeholder73
/while_while_cond_18722___redundant_placeholder8
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
T: : : : :?????????:?????????: :::::::::: 
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
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
อ

ว
while_cond_15869
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_15869___redundant_placeholder03
/while_while_cond_15869___redundant_placeholder13
/while_while_cond_15869___redundant_placeholder23
/while_while_cond_15869___redundant_placeholder33
/while_while_cond_15869___redundant_placeholder43
/while_while_cond_15869___redundant_placeholder53
/while_while_cond_15869___redundant_placeholder63
/while_while_cond_15869___redundant_placeholder73
/while_while_cond_15869___redundant_placeholder8
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
T: : : : :?????????:?????????: :::::::::: 
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
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
ฆM

__inference_call_15465	
inputF
,conv2d_conv2d_conv2d_readvariableop_resource:G
9conv2d_squeeze_batch_dims_biasadd_readvariableop_resource:H
.conv2d_1_conv2d_conv2d_readvariableop_resource:I
;conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource:
identityข#conv2d/Conv2D/Conv2D/ReadVariableOpข0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOpข%conv2d_1/Conv2D/Conv2D/ReadVariableOpข2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOpH
conv2d/Conv2D/ShapeShapeinput*
T0*
_output_shapes
:k
!conv2d/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
#conv2d/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????m
#conv2d/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv2d/Conv2D/strided_sliceStridedSliceconv2d/Conv2D/Shape:output:0*conv2d/Conv2D/strided_slice/stack:output:0,conv2d/Conv2D/strided_slice/stack_1:output:0,conv2d/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskt
conv2d/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         
conv2d/Conv2D/ReshapeReshapeinput$conv2d/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????
#conv2d/Conv2D/Conv2D/ReadVariableOpReadVariableOp,conv2d_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0อ
conv2d/Conv2D/Conv2DConv2Dconv2d/Conv2D/Reshape:output:0+conv2d/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
r
conv2d/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         d
conv2d/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????ภ
conv2d/Conv2D/concatConcatV2$conv2d/Conv2D/strided_slice:output:0&conv2d/Conv2D/concat/values_1:output:0"conv2d/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:
conv2d/Conv2D/Reshape_1Reshapeconv2d/Conv2D/Conv2D:output:0conv2d/Conv2D/concat:output:0*
T0*3
_output_shapes!
:?????????o
conv2d/squeeze_batch_dims/ShapeShape conv2d/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:w
-conv2d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
/conv2d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????y
/conv2d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ั
'conv2d/squeeze_batch_dims/strided_sliceStridedSlice(conv2d/squeeze_batch_dims/Shape:output:06conv2d/squeeze_batch_dims/strided_slice/stack:output:08conv2d/squeeze_batch_dims/strided_slice/stack_1:output:08conv2d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
'conv2d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         บ
!conv2d/squeeze_batch_dims/ReshapeReshape conv2d/Conv2D/Reshape_1:output:00conv2d/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????ฆ
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp9conv2d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ฬ
!conv2d/squeeze_batch_dims/BiasAddBiasAdd*conv2d/squeeze_batch_dims/Reshape:output:08conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????~
)conv2d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         p
%conv2d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????๐
 conv2d/squeeze_batch_dims/concatConcatV20conv2d/squeeze_batch_dims/strided_slice:output:02conv2d/squeeze_batch_dims/concat/values_1:output:0.conv2d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:ร
#conv2d/squeeze_batch_dims/Reshape_1Reshape*conv2d/squeeze_batch_dims/BiasAdd:output:0)conv2d/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:?????????q
conv2d_1/Conv2D/ShapeShape,conv2d/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:m
#conv2d_1/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_1/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????o
%conv2d_1/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv2d_1/Conv2D/strided_sliceStridedSliceconv2d_1/Conv2D/Shape:output:0,conv2d_1/Conv2D/strided_slice/stack:output:0.conv2d_1/Conv2D/strided_slice/stack_1:output:0.conv2d_1/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_1/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ฒ
conv2d_1/Conv2D/ReshapeReshape,conv2d/squeeze_batch_dims/Reshape_1:output:0&conv2d_1/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????
%conv2d_1/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ำ
conv2d_1/Conv2D/Conv2DConv2D conv2d_1/Conv2D/Reshape:output:0-conv2d_1/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
t
conv2d_1/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         f
conv2d_1/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????ศ
conv2d_1/Conv2D/concatConcatV2&conv2d_1/Conv2D/strided_slice:output:0(conv2d_1/Conv2D/concat/values_1:output:0$conv2d_1/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:ค
conv2d_1/Conv2D/Reshape_1Reshapeconv2d_1/Conv2D/Conv2D:output:0conv2d_1/Conv2D/concat:output:0*
T0*3
_output_shapes!
:?????????s
!conv2d_1/squeeze_batch_dims/ShapeShape"conv2d_1/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_1/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1conv2d_1/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????{
1conv2d_1/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)conv2d_1/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_1/squeeze_batch_dims/Shape:output:08conv2d_1/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_1/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_1/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
)conv2d_1/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ภ
#conv2d_1/squeeze_batch_dims/ReshapeReshape"conv2d_1/Conv2D/Reshape_1:output:02conv2d_1/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????ช
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0า
#conv2d_1/squeeze_batch_dims/BiasAddBiasAdd,conv2d_1/squeeze_batch_dims/Reshape:output:0:conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
+conv2d_1/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         r
'conv2d_1/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????๘
"conv2d_1/squeeze_batch_dims/concatConcatV22conv2d_1/squeeze_batch_dims/strided_slice:output:04conv2d_1/squeeze_batch_dims/concat/values_1:output:00conv2d_1/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:ษ
%conv2d_1/squeeze_batch_dims/Reshape_1Reshape,conv2d_1/squeeze_batch_dims/BiasAdd:output:0+conv2d_1/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:?????????
IdentityIdentity.conv2d_1/squeeze_batch_dims/Reshape_1:output:0^NoOp*
T0*3
_output_shapes!
:??????????
NoOpNoOp$^conv2d/Conv2D/Conv2D/ReadVariableOp1^conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_1/Conv2D/Conv2D/ReadVariableOp3^conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : 2J
#conv2d/Conv2D/Conv2D/ReadVariableOp#conv2d/Conv2D/Conv2D/ReadVariableOp2d
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_1/Conv2D/Conv2D/ReadVariableOp%conv2d_1/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp:Z V
3
_output_shapes!
:?????????

_user_specified_nameinput
ฆM

__inference_call_18064	
inputF
,conv2d_conv2d_conv2d_readvariableop_resource:G
9conv2d_squeeze_batch_dims_biasadd_readvariableop_resource:H
.conv2d_1_conv2d_conv2d_readvariableop_resource:I
;conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource:
identityข#conv2d/Conv2D/Conv2D/ReadVariableOpข0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOpข%conv2d_1/Conv2D/Conv2D/ReadVariableOpข2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOpH
conv2d/Conv2D/ShapeShapeinput*
T0*
_output_shapes
:k
!conv2d/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
#conv2d/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????m
#conv2d/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv2d/Conv2D/strided_sliceStridedSliceconv2d/Conv2D/Shape:output:0*conv2d/Conv2D/strided_slice/stack:output:0,conv2d/Conv2D/strided_slice/stack_1:output:0,conv2d/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskt
conv2d/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         
conv2d/Conv2D/ReshapeReshapeinput$conv2d/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????
#conv2d/Conv2D/Conv2D/ReadVariableOpReadVariableOp,conv2d_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0อ
conv2d/Conv2D/Conv2DConv2Dconv2d/Conv2D/Reshape:output:0+conv2d/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
r
conv2d/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         d
conv2d/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????ภ
conv2d/Conv2D/concatConcatV2$conv2d/Conv2D/strided_slice:output:0&conv2d/Conv2D/concat/values_1:output:0"conv2d/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:
conv2d/Conv2D/Reshape_1Reshapeconv2d/Conv2D/Conv2D:output:0conv2d/Conv2D/concat:output:0*
T0*3
_output_shapes!
:?????????o
conv2d/squeeze_batch_dims/ShapeShape conv2d/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:w
-conv2d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
/conv2d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????y
/conv2d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ั
'conv2d/squeeze_batch_dims/strided_sliceStridedSlice(conv2d/squeeze_batch_dims/Shape:output:06conv2d/squeeze_batch_dims/strided_slice/stack:output:08conv2d/squeeze_batch_dims/strided_slice/stack_1:output:08conv2d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
'conv2d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         บ
!conv2d/squeeze_batch_dims/ReshapeReshape conv2d/Conv2D/Reshape_1:output:00conv2d/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????ฆ
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp9conv2d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ฬ
!conv2d/squeeze_batch_dims/BiasAddBiasAdd*conv2d/squeeze_batch_dims/Reshape:output:08conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????~
)conv2d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         p
%conv2d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????๐
 conv2d/squeeze_batch_dims/concatConcatV20conv2d/squeeze_batch_dims/strided_slice:output:02conv2d/squeeze_batch_dims/concat/values_1:output:0.conv2d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:ร
#conv2d/squeeze_batch_dims/Reshape_1Reshape*conv2d/squeeze_batch_dims/BiasAdd:output:0)conv2d/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:?????????q
conv2d_1/Conv2D/ShapeShape,conv2d/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:m
#conv2d_1/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_1/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????o
%conv2d_1/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv2d_1/Conv2D/strided_sliceStridedSliceconv2d_1/Conv2D/Shape:output:0,conv2d_1/Conv2D/strided_slice/stack:output:0.conv2d_1/Conv2D/strided_slice/stack_1:output:0.conv2d_1/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_1/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ฒ
conv2d_1/Conv2D/ReshapeReshape,conv2d/squeeze_batch_dims/Reshape_1:output:0&conv2d_1/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????
%conv2d_1/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ำ
conv2d_1/Conv2D/Conv2DConv2D conv2d_1/Conv2D/Reshape:output:0-conv2d_1/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
t
conv2d_1/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         f
conv2d_1/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????ศ
conv2d_1/Conv2D/concatConcatV2&conv2d_1/Conv2D/strided_slice:output:0(conv2d_1/Conv2D/concat/values_1:output:0$conv2d_1/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:ค
conv2d_1/Conv2D/Reshape_1Reshapeconv2d_1/Conv2D/Conv2D:output:0conv2d_1/Conv2D/concat:output:0*
T0*3
_output_shapes!
:?????????s
!conv2d_1/squeeze_batch_dims/ShapeShape"conv2d_1/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_1/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1conv2d_1/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????{
1conv2d_1/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)conv2d_1/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_1/squeeze_batch_dims/Shape:output:08conv2d_1/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_1/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_1/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
)conv2d_1/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ภ
#conv2d_1/squeeze_batch_dims/ReshapeReshape"conv2d_1/Conv2D/Reshape_1:output:02conv2d_1/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????ช
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0า
#conv2d_1/squeeze_batch_dims/BiasAddBiasAdd,conv2d_1/squeeze_batch_dims/Reshape:output:0:conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
+conv2d_1/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         r
'conv2d_1/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????๘
"conv2d_1/squeeze_batch_dims/concatConcatV22conv2d_1/squeeze_batch_dims/strided_slice:output:04conv2d_1/squeeze_batch_dims/concat/values_1:output:00conv2d_1/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:ษ
%conv2d_1/squeeze_batch_dims/Reshape_1Reshape,conv2d_1/squeeze_batch_dims/BiasAdd:output:0+conv2d_1/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:?????????
IdentityIdentity.conv2d_1/squeeze_batch_dims/Reshape_1:output:0^NoOp*
T0*3
_output_shapes!
:??????????
NoOpNoOp$^conv2d/Conv2D/Conv2D/ReadVariableOp1^conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_1/Conv2D/Conv2D/ReadVariableOp3^conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : 2J
#conv2d/Conv2D/Conv2D/ReadVariableOp#conv2d/Conv2D/Conv2D/ReadVariableOp2d
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_1/Conv2D/Conv2D/ReadVariableOp%conv2d_1/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp:Z V
3
_output_shapes!
:?????????

_user_specified_nameinput
เ
ฝ
H__inference_my_lstm_model_layer_call_and_return_conditional_losses_18004
x,
my_cnn_block_17790: 
my_cnn_block_17792:,
my_cnn_block_17794: 
my_cnn_block_17796:G
5rnn_my_lstm_cell_dense_matmul_readvariableop_resource:'D
6rnn_my_lstm_cell_dense_biasadd_readvariableop_resource:I
7rnn_my_lstm_cell_dense_1_matmul_readvariableop_resource:'F
8rnn_my_lstm_cell_dense_1_biasadd_readvariableop_resource:I
7rnn_my_lstm_cell_dense_2_matmul_readvariableop_resource:'F
8rnn_my_lstm_cell_dense_2_biasadd_readvariableop_resource:I
7rnn_my_lstm_cell_dense_3_matmul_readvariableop_resource:'F
8rnn_my_lstm_cell_dense_3_biasadd_readvariableop_resource:;
)dense_4_tensordot_readvariableop_resource:5
'dense_4_biasadd_readvariableop_resource:
identityขdense_4/BiasAdd/ReadVariableOpข dense_4/Tensordot/ReadVariableOpข$my_cnn_block/StatefulPartitionedCallข-rnn/my_lstm_cell/dense/BiasAdd/ReadVariableOpข,rnn/my_lstm_cell/dense/MatMul/ReadVariableOpข/rnn/my_lstm_cell/dense_1/BiasAdd/ReadVariableOpข.rnn/my_lstm_cell/dense_1/MatMul/ReadVariableOpข/rnn/my_lstm_cell/dense_2/BiasAdd/ReadVariableOpข.rnn/my_lstm_cell/dense_2/MatMul/ReadVariableOpข/rnn/my_lstm_cell/dense_3/BiasAdd/ReadVariableOpข.rnn/my_lstm_cell/dense_3/MatMul/ReadVariableOpข	rnn/while?
$my_cnn_block/StatefulPartitionedCallStatefulPartitionedCallxmy_cnn_block_17790my_cnn_block_17792my_cnn_block_17794my_cnn_block_17796*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *
fR
__inference_call_17037w
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ต
time_distributed/ReshapeReshape-my_cnn_block/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????
@time_distributed/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ึ
.time_distributed/global_average_pooling2d/MeanMean!time_distributed/Reshape:output:0Itime_distributed/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????u
 time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      ฟ
time_distributed/Reshape_1Reshape7time_distributed/global_average_pooling2d/Mean:output:0)time_distributed/Reshape_1/shape:output:0*
T0*+
_output_shapes
:?????????y
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         น
time_distributed/Reshape_2Reshape-my_cnn_block/StatefulPartitionedCall:output:0)time_distributed/Reshape_2/shape:output:0*
T0*/
_output_shapes
:?????????\
	rnn/ShapeShape#time_distributed/Reshape_1:output:0*
T0*
_output_shapes
:a
rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ๅ
rnn/strided_sliceStridedSlicernn/Shape:output:0 rnn/strided_slice/stack:output:0"rnn/strided_slice/stack_1:output:0"rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
rnn/zeros/packedPackrnn/strided_slice:output:0rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:T
rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x
	rnn/zerosFillrnn/zeros/packed:output:0rnn/zeros/Const:output:0*
T0*'
_output_shapes
:?????????V
rnn/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
rnn/zeros_1/packedPackrnn/strided_slice:output:0rnn/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:V
rnn/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
rnn/zeros_1Fillrnn/zeros_1/packed:output:0rnn/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????g
rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
rnn/transpose	Transpose#time_distributed/Reshape_1:output:0rnn/transpose/perm:output:0*
T0*+
_output_shapes
:?????????L
rnn/Shape_1Shapernn/transpose:y:0*
T0*
_output_shapes
:c
rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:๏
rnn/strided_slice_1StridedSlicernn/Shape_1:output:0"rnn/strided_slice_1/stack:output:0$rnn/strided_slice_1/stack_1:output:0$rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????ภ
rnn/TensorArrayV2TensorListReserve(rnn/TensorArrayV2/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่า
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ์
+rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn/transpose:y:0Brnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่าc
rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
rnn/strided_slice_2StridedSlicernn/transpose:y:0"rnn/strided_slice_2/stack:output:0$rnn/strided_slice_2/stack_1:output:0$rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskl
*rnn/my_lstm_cell/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ำ
%rnn/my_lstm_cell/concatenate_1/concatConcatV2rnn/strided_slice_2:output:0rnn/zeros:output:03rnn/my_lstm_cell/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????'ข
,rnn/my_lstm_cell/dense/MatMul/ReadVariableOpReadVariableOp5rnn_my_lstm_cell_dense_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0ฟ
rnn/my_lstm_cell/dense/MatMulMatMul.rnn/my_lstm_cell/concatenate_1/concat:output:04rnn/my_lstm_cell/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
-rnn/my_lstm_cell/dense/BiasAdd/ReadVariableOpReadVariableOp6rnn_my_lstm_cell_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ป
rnn/my_lstm_cell/dense/BiasAddBiasAdd'rnn/my_lstm_cell/dense/MatMul:product:05rnn/my_lstm_cell/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
rnn/my_lstm_cell/dense/SigmoidSigmoid'rnn/my_lstm_cell/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
rnn/my_lstm_cell/multiply/mulMulrnn/zeros_1:output:0"rnn/my_lstm_cell/dense/Sigmoid:y:0*
T0*'
_output_shapes
:?????????ฆ
.rnn/my_lstm_cell/dense_1/MatMul/ReadVariableOpReadVariableOp7rnn_my_lstm_cell_dense_1_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0ร
rnn/my_lstm_cell/dense_1/MatMulMatMul.rnn/my_lstm_cell/concatenate_1/concat:output:06rnn/my_lstm_cell/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ค
/rnn/my_lstm_cell/dense_1/BiasAdd/ReadVariableOpReadVariableOp8rnn_my_lstm_cell_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ม
 rnn/my_lstm_cell/dense_1/BiasAddBiasAdd)rnn/my_lstm_cell/dense_1/MatMul:product:07rnn/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
 rnn/my_lstm_cell/dense_1/SigmoidSigmoid)rnn/my_lstm_cell/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????ฆ
.rnn/my_lstm_cell/dense_2/MatMul/ReadVariableOpReadVariableOp7rnn_my_lstm_cell_dense_2_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0ร
rnn/my_lstm_cell/dense_2/MatMulMatMul.rnn/my_lstm_cell/concatenate_1/concat:output:06rnn/my_lstm_cell/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ค
/rnn/my_lstm_cell/dense_2/BiasAdd/ReadVariableOpReadVariableOp8rnn_my_lstm_cell_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ม
 rnn/my_lstm_cell/dense_2/BiasAddBiasAdd)rnn/my_lstm_cell/dense_2/MatMul:product:07rnn/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
rnn/my_lstm_cell/dense_2/TanhTanh)rnn/my_lstm_cell/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????ก
rnn/my_lstm_cell/multiply/mul_1Mul$rnn/my_lstm_cell/dense_1/Sigmoid:y:0!rnn/my_lstm_cell/dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????
rnn/my_lstm_cell/add_1/addAddV2!rnn/my_lstm_cell/multiply/mul:z:0#rnn/my_lstm_cell/multiply/mul_1:z:0*
T0*'
_output_shapes
:?????????ฆ
.rnn/my_lstm_cell/dense_3/MatMul/ReadVariableOpReadVariableOp7rnn_my_lstm_cell_dense_3_matmul_readvariableop_resource*
_output_shapes

:'*
dtype0ร
rnn/my_lstm_cell/dense_3/MatMulMatMul.rnn/my_lstm_cell/concatenate_1/concat:output:06rnn/my_lstm_cell/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ค
/rnn/my_lstm_cell/dense_3/BiasAdd/ReadVariableOpReadVariableOp8rnn_my_lstm_cell_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ม
 rnn/my_lstm_cell/dense_3/BiasAddBiasAdd)rnn/my_lstm_cell/dense_3/MatMul:product:07rnn/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
 rnn/my_lstm_cell/dense_3/SigmoidSigmoid)rnn/my_lstm_cell/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????z
 rnn/my_lstm_cell/activation/TanhTanhrnn/my_lstm_cell/add_1/add:z:0*
T0*'
_output_shapes
:?????????ค
rnn/my_lstm_cell/multiply/mul_2Mul$rnn/my_lstm_cell/activation/Tanh:y:0$rnn/my_lstm_cell/dense_3/Sigmoid:y:0*
T0*'
_output_shapes
:?????????r
!rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ฤ
rnn/TensorArrayV2_1TensorListReserve*rnn/TensorArrayV2_1/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:้่าJ
rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : g
rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????X
rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 	
	rnn/whileWhilernn/while/loop_counter:output:0%rnn/while/maximum_iterations:output:0rnn/time:output:0rnn/TensorArrayV2_1:handle:0rnn/zeros:output:0rnn/zeros_1:output:0rnn/strided_slice_1:output:0;rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:05rnn_my_lstm_cell_dense_matmul_readvariableop_resource6rnn_my_lstm_cell_dense_biasadd_readvariableop_resource7rnn_my_lstm_cell_dense_1_matmul_readvariableop_resource8rnn_my_lstm_cell_dense_1_biasadd_readvariableop_resource7rnn_my_lstm_cell_dense_2_matmul_readvariableop_resource8rnn_my_lstm_cell_dense_2_biasadd_readvariableop_resource7rnn_my_lstm_cell_dense_3_matmul_readvariableop_resource8rnn_my_lstm_cell_dense_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*V
_output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : **
_read_only_resource_inputs

	
*
_stateful_parallelism( * 
bodyR
rnn_while_body_17874* 
condR
rnn_while_cond_17873*U
output_shapesD
B: : : : :?????????:?????????: : : : : : : : : : *
parallel_iterations 
4rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ฮ
&rnn/TensorArrayV2Stack/TensorListStackTensorListStackrnn/while:output:3=rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype0l
rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????e
rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
rnn/strided_slice_3StridedSlice/rnn/TensorArrayV2Stack/TensorListStack:tensor:0"rnn/strided_slice_3/stack:output:0$rnn/strided_slice_3/stack_1:output:0$rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maski
rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ข
rnn/transpose_1	Transpose/rnn/TensorArrayV2Stack/TensorListStack:tensor:0rnn/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Z
dense_4/Tensordot/ShapeShapernn/transpose_1:y:0*
T0*
_output_shapes
:a
dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ฿
dense_4/Tensordot/GatherV2_1GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/axes:output:0*dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ผ
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_4/Tensordot/transpose	Transposernn/transpose_1:y:0!dense_4/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????ข
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????ข
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????c
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ว
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????k
IdentityIdentitydense_4/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????ล
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp%^my_cnn_block/StatefulPartitionedCall.^rnn/my_lstm_cell/dense/BiasAdd/ReadVariableOp-^rnn/my_lstm_cell/dense/MatMul/ReadVariableOp0^rnn/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp/^rnn/my_lstm_cell/dense_1/MatMul/ReadVariableOp0^rnn/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp/^rnn/my_lstm_cell/dense_2/MatMul/ReadVariableOp0^rnn/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp/^rnn/my_lstm_cell/dense_3/MatMul/ReadVariableOp
^rnn/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp2L
$my_cnn_block/StatefulPartitionedCall$my_cnn_block/StatefulPartitionedCall2^
-rnn/my_lstm_cell/dense/BiasAdd/ReadVariableOp-rnn/my_lstm_cell/dense/BiasAdd/ReadVariableOp2\
,rnn/my_lstm_cell/dense/MatMul/ReadVariableOp,rnn/my_lstm_cell/dense/MatMul/ReadVariableOp2b
/rnn/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp/rnn/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp2`
.rnn/my_lstm_cell/dense_1/MatMul/ReadVariableOp.rnn/my_lstm_cell/dense_1/MatMul/ReadVariableOp2b
/rnn/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp/rnn/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp2`
.rnn/my_lstm_cell/dense_2/MatMul/ReadVariableOp.rnn/my_lstm_cell/dense_2/MatMul/ReadVariableOp2b
/rnn/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp/rnn/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp2`
.rnn/my_lstm_cell/dense_3/MatMul/ReadVariableOp.rnn/my_lstm_cell/dense_3/MatMul/ReadVariableOp2
	rnn/while	rnn/while:V R
3
_output_shapes!
:?????????

_user_specified_namex
เ[
๛
while_body_18723
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_my_lstm_cell_dense_matmul_readvariableop_resource_0:'H
:while_my_lstm_cell_dense_biasadd_readvariableop_resource_0:M
;while_my_lstm_cell_dense_1_matmul_readvariableop_resource_0:'J
<while_my_lstm_cell_dense_1_biasadd_readvariableop_resource_0:M
;while_my_lstm_cell_dense_2_matmul_readvariableop_resource_0:'J
<while_my_lstm_cell_dense_2_biasadd_readvariableop_resource_0:M
;while_my_lstm_cell_dense_3_matmul_readvariableop_resource_0:'J
<while_my_lstm_cell_dense_3_biasadd_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_my_lstm_cell_dense_matmul_readvariableop_resource:'F
8while_my_lstm_cell_dense_biasadd_readvariableop_resource:K
9while_my_lstm_cell_dense_1_matmul_readvariableop_resource:'H
:while_my_lstm_cell_dense_1_biasadd_readvariableop_resource:K
9while_my_lstm_cell_dense_2_matmul_readvariableop_resource:'H
:while_my_lstm_cell_dense_2_biasadd_readvariableop_resource:K
9while_my_lstm_cell_dense_3_matmul_readvariableop_resource:'H
:while_my_lstm_cell_dense_3_biasadd_readvariableop_resource:ข/while/my_lstm_cell/dense/BiasAdd/ReadVariableOpข.while/my_lstm_cell/dense/MatMul/ReadVariableOpข1while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOpข0while/my_lstm_cell/dense_1/MatMul/ReadVariableOpข1while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOpข0while/my_lstm_cell/dense_2/MatMul/ReadVariableOpข1while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOpข0while/my_lstm_cell/dense_3/MatMul/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ฆ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0n
,while/my_lstm_cell/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :์
'while/my_lstm_cell/concatenate_1/concatConcatV20while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_25while/my_lstm_cell/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????'จ
.while/my_lstm_cell/dense/MatMul/ReadVariableOpReadVariableOp9while_my_lstm_cell_dense_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ล
while/my_lstm_cell/dense/MatMulMatMul0while/my_lstm_cell/concatenate_1/concat:output:06while/my_lstm_cell/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ฆ
/while/my_lstm_cell/dense/BiasAdd/ReadVariableOpReadVariableOp:while_my_lstm_cell_dense_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0ม
 while/my_lstm_cell/dense/BiasAddBiasAdd)while/my_lstm_cell/dense/MatMul:product:07while/my_lstm_cell/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
 while/my_lstm_cell/dense/SigmoidSigmoid)while/my_lstm_cell/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
while/my_lstm_cell/multiply/mulMulwhile_placeholder_3$while/my_lstm_cell/dense/Sigmoid:y:0*
T0*'
_output_shapes
:?????????ฌ
0while/my_lstm_cell/dense_1/MatMul/ReadVariableOpReadVariableOp;while_my_lstm_cell_dense_1_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ษ
!while/my_lstm_cell/dense_1/MatMulMatMul0while/my_lstm_cell/concatenate_1/concat:output:08while/my_lstm_cell/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ช
1while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOpReadVariableOp<while_my_lstm_cell_dense_1_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0ว
"while/my_lstm_cell/dense_1/BiasAddBiasAdd+while/my_lstm_cell/dense_1/MatMul:product:09while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
"while/my_lstm_cell/dense_1/SigmoidSigmoid+while/my_lstm_cell/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????ฌ
0while/my_lstm_cell/dense_2/MatMul/ReadVariableOpReadVariableOp;while_my_lstm_cell_dense_2_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ษ
!while/my_lstm_cell/dense_2/MatMulMatMul0while/my_lstm_cell/concatenate_1/concat:output:08while/my_lstm_cell/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ช
1while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOpReadVariableOp<while_my_lstm_cell_dense_2_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0ว
"while/my_lstm_cell/dense_2/BiasAddBiasAdd+while/my_lstm_cell/dense_2/MatMul:product:09while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
while/my_lstm_cell/dense_2/TanhTanh+while/my_lstm_cell/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????ง
!while/my_lstm_cell/multiply/mul_1Mul&while/my_lstm_cell/dense_1/Sigmoid:y:0#while/my_lstm_cell/dense_2/Tanh:y:0*
T0*'
_output_shapes
:?????????ฃ
while/my_lstm_cell/add_1/addAddV2#while/my_lstm_cell/multiply/mul:z:0%while/my_lstm_cell/multiply/mul_1:z:0*
T0*'
_output_shapes
:?????????ฌ
0while/my_lstm_cell/dense_3/MatMul/ReadVariableOpReadVariableOp;while_my_lstm_cell_dense_3_matmul_readvariableop_resource_0*
_output_shapes

:'*
dtype0ษ
!while/my_lstm_cell/dense_3/MatMulMatMul0while/my_lstm_cell/concatenate_1/concat:output:08while/my_lstm_cell/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ช
1while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOpReadVariableOp<while_my_lstm_cell_dense_3_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0ว
"while/my_lstm_cell/dense_3/BiasAddBiasAdd+while/my_lstm_cell/dense_3/MatMul:product:09while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
"while/my_lstm_cell/dense_3/SigmoidSigmoid+while/my_lstm_cell/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????~
"while/my_lstm_cell/activation/TanhTanh while/my_lstm_cell/add_1/add:z:0*
T0*'
_output_shapes
:?????????ช
!while/my_lstm_cell/multiply/mul_2Mul&while/my_lstm_cell/activation/Tanh:y:0&while/my_lstm_cell/dense_3/Sigmoid:y:0*
T0*'
_output_shapes
:?????????ฮ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder%while/my_lstm_cell/multiply/mul_2:z:0*
_output_shapes
: *
element_dtype0:้่าM
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
: 
while/Identity_4Identity%while/my_lstm_cell/multiply/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:?????????}
while/Identity_5Identity while/my_lstm_cell/add_1/add:z:0^while/NoOp*
T0*'
_output_shapes
:?????????ไ

while/NoOpNoOp0^while/my_lstm_cell/dense/BiasAdd/ReadVariableOp/^while/my_lstm_cell/dense/MatMul/ReadVariableOp2^while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp1^while/my_lstm_cell/dense_1/MatMul/ReadVariableOp2^while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp1^while/my_lstm_cell/dense_2/MatMul/ReadVariableOp2^while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp1^while/my_lstm_cell/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"z
:while_my_lstm_cell_dense_1_biasadd_readvariableop_resource<while_my_lstm_cell_dense_1_biasadd_readvariableop_resource_0"x
9while_my_lstm_cell_dense_1_matmul_readvariableop_resource;while_my_lstm_cell_dense_1_matmul_readvariableop_resource_0"z
:while_my_lstm_cell_dense_2_biasadd_readvariableop_resource<while_my_lstm_cell_dense_2_biasadd_readvariableop_resource_0"x
9while_my_lstm_cell_dense_2_matmul_readvariableop_resource;while_my_lstm_cell_dense_2_matmul_readvariableop_resource_0"z
:while_my_lstm_cell_dense_3_biasadd_readvariableop_resource<while_my_lstm_cell_dense_3_biasadd_readvariableop_resource_0"x
9while_my_lstm_cell_dense_3_matmul_readvariableop_resource;while_my_lstm_cell_dense_3_matmul_readvariableop_resource_0"v
8while_my_lstm_cell_dense_biasadd_readvariableop_resource:while_my_lstm_cell_dense_biasadd_readvariableop_resource_0"t
7while_my_lstm_cell_dense_matmul_readvariableop_resource9while_my_lstm_cell_dense_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"จ
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :?????????:?????????: : : : : : : : : : 2b
/while/my_lstm_cell/dense/BiasAdd/ReadVariableOp/while/my_lstm_cell/dense/BiasAdd/ReadVariableOp2`
.while/my_lstm_cell/dense/MatMul/ReadVariableOp.while/my_lstm_cell/dense/MatMul/ReadVariableOp2f
1while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp1while/my_lstm_cell/dense_1/BiasAdd/ReadVariableOp2d
0while/my_lstm_cell/dense_1/MatMul/ReadVariableOp0while/my_lstm_cell/dense_1/MatMul/ReadVariableOp2f
1while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp1while/my_lstm_cell/dense_2/BiasAdd/ReadVariableOp2d
0while/my_lstm_cell/dense_2/MatMul/ReadVariableOp0while/my_lstm_cell/dense_2/MatMul/ReadVariableOp2f
1while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp1while/my_lstm_cell/dense_3/BiasAdd/ReadVariableOp2d
0while/my_lstm_cell/dense_3/MatMul/ReadVariableOp0while/my_lstm_cell/dense_3/MatMul/ReadVariableOp: 
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
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: "ต	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ป
serving_defaultง
G
input_1<
serving_default_input_1:0?????????@
output_14
StatefulPartitionedCall:0?????????tensorflow/serving/predict:ดะ
่
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
ฆ
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
ส
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
ุ
(trace_0
)trace_1
*trace_2
+trace_32ํ
-__inference_my_lstm_model_layer_call_fn_16475
-__inference_my_lstm_model_layer_call_fn_17537
-__inference_my_lstm_model_layer_call_fn_17570
-__inference_my_lstm_model_layer_call_fn_16893ฎ
ฅฒก
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaultsข
p 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 z(trace_0z)trace_1z*trace_2z+trace_3
ฤ
,trace_0
-trace_1
.trace_2
/trace_32ู
H__inference_my_lstm_model_layer_call_and_return_conditional_losses_17787
H__inference_my_lstm_model_layer_call_and_return_conditional_losses_18004
H__inference_my_lstm_model_layer_call_and_return_conditional_losses_16931
H__inference_my_lstm_model_layer_call_and_return_conditional_losses_16969ฎ
ฅฒก
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaultsข
p 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 z,trace_0z-trace_1z.trace_2z/trace_3
หBศ
 __inference__wrapped_model_15711input_1"
ฒ
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
ภ
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6conv_layers
7call"
_tf_keras_layer
ฅ
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses"
_tf_keras_layer
ฐ
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses
		layer"
_tf_keras_layer
น
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
ร
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
ป
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
๋
biter

cbeta_1

dbeta_2
	edecay
flearning_ratemmmmmmmm?mกmขmฃmคmฅmฆvงvจvฉvชvซvฌvญvฎvฏvฐvฑvฒvณvด"
	optimizer

gtrace_0
htrace_12แ
__inference_call_17252
__inference_call_17469ฎ
ฅฒก
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaultsข
p 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 zgtrace_0zhtrace_1
,
iserving_default"
signature_map
':%2conv2d/kernel
:2conv2d/bias
):'2conv2d_1/kernel
:2conv2d_1/bias
/:-'2rnn/my_lstm_cell/dense/kernel
):'2rnn/my_lstm_cell/dense/bias
1:/'2rnn/my_lstm_cell/dense_1/kernel
+:)2rnn/my_lstm_cell/dense_1/bias
1:/'2rnn/my_lstm_cell/dense_2/kernel
+:)2rnn/my_lstm_cell/dense_2/bias
1:/'2rnn/my_lstm_cell/dense_3/kernel
+:)2rnn/my_lstm_cell/dense_3/bias
 :2dense_4/kernel
:2dense_4/bias
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
๎B๋
-__inference_my_lstm_model_layer_call_fn_16475input_1"ฎ
ฅฒก
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaultsข
p 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
่Bๅ
-__inference_my_lstm_model_layer_call_fn_17537x"ฎ
ฅฒก
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaultsข
p 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
่Bๅ
-__inference_my_lstm_model_layer_call_fn_17570x"ฎ
ฅฒก
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaultsข
p 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๎B๋
-__inference_my_lstm_model_layer_call_fn_16893input_1"ฎ
ฅฒก
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaultsข
p 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
B
H__inference_my_lstm_model_layer_call_and_return_conditional_losses_17787x"ฎ
ฅฒก
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaultsข
p 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
B
H__inference_my_lstm_model_layer_call_and_return_conditional_losses_18004x"ฎ
ฅฒก
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaultsข
p 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
B
H__inference_my_lstm_model_layer_call_and_return_conditional_losses_16931input_1"ฎ
ฅฒก
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaultsข
p 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
B
H__inference_my_lstm_model_layer_call_and_return_conditional_losses_16969input_1"ฎ
ฅฒก
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaultsข
p 

kwonlyargs 
kwonlydefaults
 
annotationsช *
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
ญ
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
ศ
otrace_0
ptrace_12
,__inference_my_cnn_block_layer_call_fn_18137
,__inference_my_cnn_block_layer_call_fn_18150ฒ
ฉฒฅ
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
annotationsช *
 zotrace_0zptrace_1
?
qtrace_0
rtrace_12ว
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_18210
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_18270ฒ
ฉฒฅ
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
annotationsช *
 zqtrace_0zrtrace_1
.
s0
t1"
trackable_list_wrapper

utrace_0
vtrace_12ๅ
__inference_call_18064
__inference_call_18124ฒ
ฉฒฅ
FullArgSpec(
args 
jself
jinput

jtraining
varargs
 
varkw
 
defaultsข

 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 zutrace_0zvtrace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ญ
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
?
|trace_02฿
8__inference_global_average_pooling2d_layer_call_fn_18275ข
ฒ
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
annotationsช *
 z|trace_0

}trace_02๚
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_18281ข
ฒ
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
annotationsช *
 z}trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ฐ
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
แ
trace_0
trace_12ฆ
0__inference_time_distributed_layer_call_fn_18286
0__inference_time_distributed_layer_call_fn_18291ฟ
ถฒฒ
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
annotationsช *
 ztrace_0ztrace_1

trace_0
trace_12?
K__inference_time_distributed_layer_call_and_return_conditional_losses_18308
K__inference_time_distributed_layer_call_and_return_conditional_losses_18325ฟ
ถฒฒ
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
annotationsช *
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
ฒ
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
?
trace_02?
,__inference_my_lstm_cell_layer_call_fn_18352ฌ
ฃฒ
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
annotationsช *
 ztrace_0

trace_02๘
G__inference_my_lstm_cell_layer_call_and_return_conditional_losses_18395ฌ
ฃฒ
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
annotationsช *
 ztrace_0
ซ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ม
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
ซ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ม
?	variables
กtrainable_variables
ขregularization_losses
ฃ	keras_api
ค__call__
+ฅ&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
ม
ฆ	variables
งtrainable_variables
จregularization_losses
ฉ	keras_api
ช__call__
+ซ&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
ซ
ฌ	variables
ญtrainable_variables
ฎregularization_losses
ฏ	keras_api
ฐ__call__
+ฑ&call_and_return_all_conditional_losses"
_tf_keras_layer
ม
ฒ	variables
ณtrainable_variables
ดregularization_losses
ต	keras_api
ถ__call__
+ท&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
ซ
ธ	variables
นtrainable_variables
บregularization_losses
ป	keras_api
ผ__call__
+ฝ&call_and_return_all_conditional_losses"
_tf_keras_layer
)
พ	keras_api"
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
ฟ
ฟstates
ภnon_trainable_variables
มlayers
ยmetrics
 รlayer_regularization_losses
ฤlayer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
๏
ลtrace_0
ฦtrace_1
วtrace_2
ศtrace_32?
#__inference_rnn_layer_call_fn_18416
#__inference_rnn_layer_call_fn_18437
#__inference_rnn_layer_call_fn_18458
#__inference_rnn_layer_call_fn_18479ๅ
?ฒุ
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
annotationsช *
 zลtrace_0zฦtrace_1zวtrace_2zศtrace_3
?
ษtrace_0
สtrace_1
หtrace_2
ฬtrace_32่
>__inference_rnn_layer_call_and_return_conditional_losses_18653
>__inference_rnn_layer_call_and_return_conditional_losses_18827
>__inference_rnn_layer_call_and_return_conditional_losses_19001
>__inference_rnn_layer_call_and_return_conditional_losses_19175ๅ
?ฒุ
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
annotationsช *
 zษtrace_0zสtrace_1zหtrace_2zฬtrace_3
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
ฒ
อnon_trainable_variables
ฮlayers
ฯmetrics
 ะlayer_regularization_losses
ัlayer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
ํ
าtrace_02ฮ
'__inference_dense_4_layer_call_fn_19184ข
ฒ
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
annotationsช *
 zาtrace_0

ำtrace_02้
B__inference_dense_4_layer_call_and_return_conditional_losses_19214ข
ฒ
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
annotationsช *
 zำtrace_0
P
ิ	variables
ี	keras_api
	 total
	!count"
_tf_keras_metric
a
ึ	variables
ื	keras_api
	"total
	#count
ุ
_fn_kwargs"
_tf_keras_metric
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ัBฮ
__inference_call_17252x"ฎ
ฅฒก
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaultsข
p 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
ัBฮ
__inference_call_17469x"ฎ
ฅฒก
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaultsข
p 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
สBว
#__inference_signature_wrapper_17504input_1"
ฒ
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
annotationsช *
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
๏B์
,__inference_my_cnn_block_layer_call_fn_18137input"ฒ
ฉฒฅ
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
annotationsช *
 
๏B์
,__inference_my_cnn_block_layer_call_fn_18150input"ฒ
ฉฒฅ
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
annotationsช *
 
B
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_18210input"ฒ
ฉฒฅ
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
annotationsช *
 
B
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_18270input"ฒ
ฉฒฅ
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
annotationsช *
 
ไ
ู	variables
ฺtrainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias
!฿_jit_compiled_convolution_op"
_tf_keras_layer
ไ
เ	variables
แtrainable_variables
โregularization_losses
ใ	keras_api
ไ__call__
+ๅ&call_and_return_all_conditional_losses

kernel
bias
!ๆ_jit_compiled_convolution_op"
_tf_keras_layer
ูBึ
__inference_call_18064input"ฒ
ฉฒฅ
FullArgSpec(
args 
jself
jinput

jtraining
varargs
 
varkw
 
defaultsข

 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
ูBึ
__inference_call_18124input"ฒ
ฉฒฅ
FullArgSpec(
args 
jself
jinput

jtraining
varargs
 
varkw
 
defaultsข

 

kwonlyargs 
kwonlydefaults
 
annotationsช *
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
์B้
8__inference_global_average_pooling2d_layer_call_fn_18275inputs"ข
ฒ
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
annotationsช *
 
B
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_18281inputs"ข
ฒ
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
annotationsช *
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
B?
0__inference_time_distributed_layer_call_fn_18286inputs"ฟ
ถฒฒ
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
annotationsช *
 
B?
0__inference_time_distributed_layer_call_fn_18291inputs"ฟ
ถฒฒ
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
annotationsช *
 
B
K__inference_time_distributed_layer_call_and_return_conditional_losses_18308inputs"ฟ
ถฒฒ
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
annotationsช *
 
B
K__inference_time_distributed_layer_call_and_return_conditional_losses_18325inputs"ฟ
ถฒฒ
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
annotationsช *
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
?B๛
,__inference_my_lstm_cell_layer_call_fn_18352inputsstates/0states/1"ฌ
ฃฒ
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
annotationsช *
 
B
G__inference_my_lstm_cell_layer_call_and_return_conditional_losses_18395inputsstates/0states/1"ฌ
ฃฒ
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
annotationsช *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
็non_trainable_variables
่layers
้metrics
 ๊layer_regularization_losses
๋layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
จ2ฅข
ฒ
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
annotationsช *
 
จ2ฅข
ฒ
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
annotationsช *
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
ธ
์non_trainable_variables
ํlayers
๎metrics
 ๏layer_regularization_losses
๐layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
จ2ฅข
ฒ
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
annotationsช *
 
จ2ฅข
ฒ
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
annotationsช *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
๑non_trainable_variables
๒layers
๓metrics
 ๔layer_regularization_losses
๕layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
จ2ฅข
ฒ
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
annotationsช *
 
จ2ฅข
ฒ
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
annotationsช *
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
ธ
๖non_trainable_variables
๗layers
๘metrics
 ๙layer_regularization_losses
๚layer_metrics
?	variables
กtrainable_variables
ขregularization_losses
ค__call__
+ฅ&call_and_return_all_conditional_losses
'ฅ"call_and_return_conditional_losses"
_generic_user_object
จ2ฅข
ฒ
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
annotationsช *
 
จ2ฅข
ฒ
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
annotationsช *
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
ธ
๛non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
ฆ	variables
งtrainable_variables
จregularization_losses
ช__call__
+ซ&call_and_return_all_conditional_losses
'ซ"call_and_return_conditional_losses"
_generic_user_object
จ2ฅข
ฒ
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
annotationsช *
 
จ2ฅข
ฒ
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
annotationsช *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ฌ	variables
ญtrainable_variables
ฎregularization_losses
ฐ__call__
+ฑ&call_and_return_all_conditional_losses
'ฑ"call_and_return_conditional_losses"
_generic_user_object
จ2ฅข
ฒ
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
annotationsช *
 
จ2ฅข
ฒ
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
annotationsช *
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
ธ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ฒ	variables
ณtrainable_variables
ดregularization_losses
ถ__call__
+ท&call_and_return_all_conditional_losses
'ท"call_and_return_conditional_losses"
_generic_user_object
จ2ฅข
ฒ
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
annotationsช *
 
จ2ฅข
ฒ
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
annotationsช *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ธ	variables
นtrainable_variables
บregularization_losses
ผ__call__
+ฝ&call_and_return_all_conditional_losses
'ฝ"call_and_return_conditional_losses"
_generic_user_object
จ2ฅข
ฒ
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
annotationsช *
 
จ2ฅข
ฒ
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
annotationsช *
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
B
#__inference_rnn_layer_call_fn_18416inputs/0"ๅ
?ฒุ
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
annotationsช *
 
B
#__inference_rnn_layer_call_fn_18437inputs/0"ๅ
?ฒุ
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
annotationsช *
 
B
#__inference_rnn_layer_call_fn_18458inputs"ๅ
?ฒุ
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
annotationsช *
 
B
#__inference_rnn_layer_call_fn_18479inputs"ๅ
?ฒุ
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
annotationsช *
 
ทBด
>__inference_rnn_layer_call_and_return_conditional_losses_18653inputs/0"ๅ
?ฒุ
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
annotationsช *
 
ทBด
>__inference_rnn_layer_call_and_return_conditional_losses_18827inputs/0"ๅ
?ฒุ
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
annotationsช *
 
ตBฒ
>__inference_rnn_layer_call_and_return_conditional_losses_19001inputs"ๅ
?ฒุ
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
annotationsช *
 
ตBฒ
>__inference_rnn_layer_call_and_return_conditional_losses_19175inputs"ๅ
?ฒุ
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
annotationsช *
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
?Bุ
'__inference_dense_4_layer_call_fn_19184inputs"ข
ฒ
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
annotationsช *
 
๖B๓
B__inference_dense_4_layer_call_and_return_conditional_losses_19214inputs"ข
ฒ
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
annotationsช *
 
.
 0
!1"
trackable_list_wrapper
.
ิ	variables"
_generic_user_object
.
"0
#1"
trackable_list_wrapper
.
ึ	variables"
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
ธ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ู	variables
ฺtrainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
จ2ฅข
ฒ
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
annotationsช *
 
จ2ฅข
ฒ
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
annotationsช *
 
ด2ฑฎ
ฃฒ
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
annotationsช *
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
ธ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
เ	variables
แtrainable_variables
โregularization_losses
ไ__call__
+ๅ&call_and_return_all_conditional_losses
'ๅ"call_and_return_conditional_losses"
_generic_user_object
จ2ฅข
ฒ
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
annotationsช *
 
จ2ฅข
ฒ
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
annotationsช *
 
ด2ฑฎ
ฃฒ
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
annotationsช *
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
,:*2Adam/conv2d/kernel/m
:2Adam/conv2d/bias/m
.:,2Adam/conv2d_1/kernel/m
 :2Adam/conv2d_1/bias/m
4:2'2$Adam/rnn/my_lstm_cell/dense/kernel/m
.:,2"Adam/rnn/my_lstm_cell/dense/bias/m
6:4'2&Adam/rnn/my_lstm_cell/dense_1/kernel/m
0:.2$Adam/rnn/my_lstm_cell/dense_1/bias/m
6:4'2&Adam/rnn/my_lstm_cell/dense_2/kernel/m
0:.2$Adam/rnn/my_lstm_cell/dense_2/bias/m
6:4'2&Adam/rnn/my_lstm_cell/dense_3/kernel/m
0:.2$Adam/rnn/my_lstm_cell/dense_3/bias/m
%:#2Adam/dense_4/kernel/m
:2Adam/dense_4/bias/m
,:*2Adam/conv2d/kernel/v
:2Adam/conv2d/bias/v
.:,2Adam/conv2d_1/kernel/v
 :2Adam/conv2d_1/bias/v
4:2'2$Adam/rnn/my_lstm_cell/dense/kernel/v
.:,2"Adam/rnn/my_lstm_cell/dense/bias/v
6:4'2&Adam/rnn/my_lstm_cell/dense_1/kernel/v
0:.2$Adam/rnn/my_lstm_cell/dense_1/bias/v
6:4'2&Adam/rnn/my_lstm_cell/dense_2/kernel/v
0:.2$Adam/rnn/my_lstm_cell/dense_2/bias/v
6:4'2&Adam/rnn/my_lstm_cell/dense_3/kernel/v
0:.2$Adam/rnn/my_lstm_cell/dense_3/bias/v
%:#2Adam/dense_4/kernel/v
:2Adam/dense_4/bias/vฌ
 __inference__wrapped_model_15711<ข9
2ข/
-*
input_1?????????
ช "7ช4
2
output_1&#
output_1?????????
__inference_call_17252j:ข7
0ข-
'$
x?????????
p
ช "?????????
__inference_call_17469j:ข7
0ข-
'$
x?????????
p 
ช "?????????
__inference_call_18064l>ข;
4ข1
+(
input?????????
p
ช "$!?????????
__inference_call_18124l>ข;
4ข1
+(
input?????????
p 
ช "$!?????????ช
B__inference_dense_4_layer_call_and_return_conditional_losses_19214d3ข0
)ข&
$!
inputs?????????
ช ")ข&

0?????????
 
'__inference_dense_4_layer_call_fn_19184W3ข0
)ข&
$!
inputs?????????
ช "??????????
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_18281RขO
HขE
C@
inputs4????????????????????????????????????
ช ".ข+
$!
0??????????????????
 ณ
8__inference_global_average_pooling2d_layer_call_fn_18275wRขO
HขE
C@
inputs4????????????????????????????????????
ช "!??????????????????ฤ
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_18210y>ข;
4ข1
+(
input?????????
p 
ช "1ข.
'$
0?????????
 ฤ
G__inference_my_cnn_block_layer_call_and_return_conditional_losses_18270y>ข;
4ข1
+(
input?????????
p
ช "1ข.
'$
0?????????
 
,__inference_my_cnn_block_layer_call_fn_18137l>ข;
4ข1
+(
input?????????
p 
ช "$!?????????
,__inference_my_cnn_block_layer_call_fn_18150l>ข;
4ข1
+(
input?????????
p
ช "$!?????????ษ
G__inference_my_lstm_cell_layer_call_and_return_conditional_losses_18395?|ขy
rขo
 
inputs?????????
KขH
"
states/0?????????
"
states/1?????????
ช "sขp
iขf

0/0?????????
EB

0/1/0?????????

0/1/1?????????
 
,__inference_my_lstm_cell_layer_call_fn_18352ํ|ขy
rขo
 
inputs?????????
KขH
"
states/0?????????
"
states/1?????????
ช "cข`

0?????????
A>

1/0?????????

1/1?????????ษ
H__inference_my_lstm_model_layer_call_and_return_conditional_losses_16931}@ข=
6ข3
-*
input_1?????????
p 
ช ")ข&

0?????????
 ษ
H__inference_my_lstm_model_layer_call_and_return_conditional_losses_16969}@ข=
6ข3
-*
input_1?????????
p
ช ")ข&

0?????????
 ร
H__inference_my_lstm_model_layer_call_and_return_conditional_losses_17787w:ข7
0ข-
'$
x?????????
p 
ช ")ข&

0?????????
 ร
H__inference_my_lstm_model_layer_call_and_return_conditional_losses_18004w:ข7
0ข-
'$
x?????????
p
ช ")ข&

0?????????
 ก
-__inference_my_lstm_model_layer_call_fn_16475p@ข=
6ข3
-*
input_1?????????
p 
ช "?????????ก
-__inference_my_lstm_model_layer_call_fn_16893p@ข=
6ข3
-*
input_1?????????
p
ช "?????????
-__inference_my_lstm_model_layer_call_fn_17537j:ข7
0ข-
'$
x?????????
p 
ช "?????????
-__inference_my_lstm_model_layer_call_fn_17570j:ข7
0ข-
'$
x?????????
p
ช "?????????ึ
>__inference_rnn_layer_call_and_return_conditional_losses_18653SขP
IขF
41
/,
inputs/0??????????????????

 
p 

 

 
ช "2ข/
(%
0??????????????????
 ึ
>__inference_rnn_layer_call_and_return_conditional_losses_18827SขP
IขF
41
/,
inputs/0??????????????????

 
p

 

 
ช "2ข/
(%
0??????????????????
 ผ
>__inference_rnn_layer_call_and_return_conditional_losses_19001zCข@
9ข6
$!
inputs?????????

 
p 

 

 
ช ")ข&

0?????????
 ผ
>__inference_rnn_layer_call_and_return_conditional_losses_19175zCข@
9ข6
$!
inputs?????????

 
p

 

 
ช ")ข&

0?????????
 ฎ
#__inference_rnn_layer_call_fn_18416SขP
IขF
41
/,
inputs/0??????????????????

 
p 

 

 
ช "%"??????????????????ฎ
#__inference_rnn_layer_call_fn_18437SขP
IขF
41
/,
inputs/0??????????????????

 
p

 

 
ช "%"??????????????????
#__inference_rnn_layer_call_fn_18458mCข@
9ข6
$!
inputs?????????

 
p 

 

 
ช "?????????
#__inference_rnn_layer_call_fn_18479mCข@
9ข6
$!
inputs?????????

 
p

 

 
ช "?????????บ
#__inference_signature_wrapper_17504GขD
ข 
=ช:
8
input_1-*
input_1?????????"7ช4
2
output_1&#
output_1?????????า
K__inference_time_distributed_layer_call_and_return_conditional_losses_18308LขI
Bข?
52
inputs&??????????????????
p 

 
ช "2ข/
(%
0??????????????????
 า
K__inference_time_distributed_layer_call_and_return_conditional_losses_18325LขI
Bข?
52
inputs&??????????????????
p

 
ช "2ข/
(%
0??????????????????
 ฉ
0__inference_time_distributed_layer_call_fn_18286uLขI
Bข?
52
inputs&??????????????????
p 

 
ช "%"??????????????????ฉ
0__inference_time_distributed_layer_call_fn_18291uLขI
Bข?
52
inputs&??????????????????
p

 
ช "%"??????????????????