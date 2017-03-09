import matplotlib.pyplot as plt
import sys
import math

# filename = 'train_details12.log'

filename = sys.argv[1]
num_training = int(sys.argv[2])
num_val = int(sys.argv[3])
batch_size = int(sys.argv[4])
checkpoint_every = int(sys.argv[5])
prefix = sys.argv[6]

num_batches = int(math.floor(num_training/batch_size))
num_batches_val = int(math.floor(num_val / batch_size))

f = open(filename)
all_train_loss = []
all_val_loss = []
for line in f:
    if 'Training' not in line and 'Validation' not in line:
        print(line)
    elif 'Training' in line:
        # Read training loss
        train_loss = float(line.split(':')[4].strip())
        all_train_loss.append(train_loss)
    else:
        # Read validation loss
        val_loss = float(line.split(':')[4].strip())
        all_val_loss.append(val_loss)
f.close()

# Plot Training and Validation loss in same graph
train_loss_iter =[]
for i in range(0,int(len(all_train_loss)/num_batches)):
    train_loss_iter.append( (sum(all_train_loss[i*num_batches : (i+1)*num_batches])/num_batches) )

val_loss_iter =[]
for i in range(0,int(len(all_val_loss)/num_batches_val)):
    val_loss_iter.append( (sum(all_val_loss[i*num_batches_val : (i+1)*num_batches_val])/num_batches_val) )

plt.plot(train_loss_iter,label='Training Error')
iter_val = range(checkpoint_every,checkpoint_every*(len(val_loss_iter)+1),checkpoint_every)
plt.plot(iter_val,val_loss_iter,label='Validation Error')
plt.legend()
plt.savefig(prefix+'/err_analysis.png')
print(val_loss_iter)
# python plot_train_val.py experiment1/train.log 1100 60 20 5 experiment1
