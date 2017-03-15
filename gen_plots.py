
# coding: utf-8

# In[50]:

import matplotlib.pyplot as plt
import sys
import math

# filename = 'train_details12.log'

filename = sys.argv[1]
num_training = int(sys.argv[2])
validation_size = int(sys.argv[3])
batch_size = int(sys.argv[4])
checkpoint_every = int(sys.argv[5])
prefix = sys.argv[6]

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


# In[51]:

plt.plot(all_train_loss)
plt.ylabel('Loss')
plt.xlabel('Iteration')
plt.savefig(prefix+'/train_loss_analysis.png')
#plt.show()


# In[52]:

plt.plot(all_val_loss)
plt.ylabel('Loss')
plt.xlabel('Iteration')
plt.savefig(prefix+'/validatoin_loss_analysis.png')
#plt.show()


# In[53]:

train_loss_iter = list()
validation_loss_iter = list()

print 'Total train loss points: ' + str(len(all_train_loss))
print 'Total validation loss points: ' + str(len(all_val_loss))
print 'Checkpoint every: ' + str(checkpoint_every)
print 'validation size: ' + str(validation_size)

print 'Number of training iterations: ' + str(int(math.floor(len(all_train_loss)/checkpoint_every)))
print 'Number of validation iterations: ' + str(int(math.floor(len(all_val_loss)/validation_size)))

for idx in range(int(math.floor(len(all_train_loss)/checkpoint_every))):
    train_loss_iter.append( (sum(all_train_loss[idx*checkpoint_every : (idx+1)*checkpoint_every])/checkpoint_every) )
for idx in range(int(math.floor(len(all_val_loss)/validation_size))):
    validation_loss_iter.append( (sum(all_val_loss[idx*validation_size : (idx+1)*validation_size])/validation_size) )


# In[55]:

plt.plot(train_loss_iter,label='Training Error')
plt.plot(validation_loss_iter,label='Validation Error')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.savefig(prefix+'/error_analysis.png')
#plt.show()



# In[ ]:



