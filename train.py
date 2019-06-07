import torch
import torch.optim as optim
import model
import preprocessing
from tqdm import tqdm
import argparse
import numpy as np
import os

# parse command line argument
def parse_args():
    '''
    Parse the TopoLSTM parameters
    '''
    parser = argparse.ArgumentParser(description="Run TopoLSTM")

    parser.add_argument('--data_dir',default='',type=str,help='data path')
    parser.add_argument('--maxlen',default=30,type=int,help='max sequence lengths')
    parser.add_argument('--batch_size',default=256,type=int,help='batch size')
    parser.add_argument('--hidden_size',default=128,type=int,help='embedding size')
    parser.add_argument('--epochs',default=100,type=int,help='iteration numbers')
    parser.add_argument('--learning_rate',default=0.001,type=float,help='learning rate')

    return parser.parse_args()

# read argument
args = parse_args()

# test data only saw 4 nodes instead of all nodes
train_nodes = preprocessing.process_dataset(args.data_dir,'train',maxlen=args.maxlen)
test_nodes = preprocessing.process_test(args.data_dir)
seen_nodes = train_nodes | test_nodes
print('%d seen nodes.' % len(seen_nodes))

# write seen nodes into file
filename = os.path.join(args.data_dir, 'seen_nodes.txt')
with open(filename, 'w') as f:
    for v in seen_nodes:
        f.write('%s\n' % v)

# read graph and node index
G, node_to_index = preprocessing.load_graph(args.data_dir)

# transform data into input we need
examples = preprocessing.load_examples(args.data_dir,dataset='train',G=G,node_index=node_to_index,maxlen=args.maxlen)

# sort example to accelerate training and handle different sequence lengths
lengths = []
for example in examples:
    lengths.append(len(example['sequence']))
sorted_length_index = sorted(range(len(lengths)), key=lambda k: lengths[k])
sorted_examples = [examples[index] for index in sorted_length_index]

# claim TopoLSTM
topolstm = model.TopoLSTM(data_dir=args.data_dir,node_index = node_to_index,hidden_size = args.hidden_size).cuda()
loss_function = model.Penalty_cross_entropy()
loader = preprocessing.Loader(sorted_examples,batch_size=args.batch_size,shuffle_data=False)
optimizer = optim.Adam(topolstm.parameters(),lr=args.learning_rate)
batch_number = len(examples) // args.batch_size + 1

# Start training
for epoch in tqdm(range(args.epochs)):
    mean_loss = []
    for step in tqdm(range(batch_number)):
        topolstm.zero_grad()
        sequence_matrix, sequence_mask_matrix, topo_mask_matrix, label = loader.__call__()
        sequence_matrix = torch.tensor(sequence_matrix,dtype=torch.long).cuda()
        sequence_mask_matrix = torch.tensor(sequence_mask_matrix,dtype=torch.float).cuda()
        topo_mask_matrix = torch.tensor(topo_mask_matrix,dtype=torch.float).cuda()
        label = torch.tensor(label,dtype=torch.long).cuda()
        predict_result = topolstm(sequence_matrix,sequence_mask_matrix,topo_mask_matrix)
        penalty_term = list(topolstm.parameters())[1:]
        loss = loss_function(predict_result,label,penalty_term)
        loss.backward()
        mean_loss.append(loss.cpu().detach().numpy())
        optimizer.step()
    mean_loss = np.mean(mean_loss)
    print('Epochs:{},Loss:{:5f}'.format(epoch,mean_loss))

torch.save(topolstm,'topolstm.pkl')