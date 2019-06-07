import torch
import torch.nn as nn
from model import TopoLSTM
from model import TopoLSTMCell
import numpy as np
import preprocessing
import argparse

# parse command line argument
def parse_args():
    '''
    Parse the TopoLSTM parameters
    '''
    parser = argparse.ArgumentParser(description="Predict Test")

    parser.add_argument('--data_dir',default='',type=str,help='data path')

    return parser.parse_args()

# prepare test data
def prepare_test_data(tuples):
    '''
    produces a mini-batch of data in format required by model.
    '''
    seqs = tuples['sequence']
    lengths = len(seqs)
    n_timesteps = lengths
    n_samples = 1

    # prepare sequences data
    seqs_matrix = np.zeros((n_timesteps, n_samples))
    seqs_matrix[: lengths, 0] = seqs

    # prepare topo-masks data
    topo_masks = tuples['topo_mask']
    topo_masks_tensor = np.zeros(
        (n_timesteps, n_samples, n_timesteps))
    topo_masks_tensor[: lengths, 0, : lengths] = topo_masks
    # prepare sequence masks
    seq_masks_matrix = np.zeros((n_timesteps, n_samples))
    seq_masks_matrix[: lengths, 0] = 1.
    
    return (seqs_matrix,
            seq_masks_matrix,
            topo_masks_tensor
           )

# exclude activated node has been predicted
def select_certain_node_softmax(data,drop_node_index):
    total_index = [i for i in range(data.shape[1])]
    inactive_node_index = []
    for i in total_index:
        if i not in drop_node_index:
            inactive_node_index.append(i)
    prob = np.exp(data) / (np.sum(np.exp(data[:,inactive_node_index]),axis=1).reshape(-1,1))
    prob[:,drop_node_index] = 0
    return prob

# read argument
args = parse_args()
# read graph and node index
G, node_to_index = preprocessing.load_graph(args.data_dir)
# load pretrained model
model = torch.load('topolstm.pkl')
index_to_node = {index:node for node, index in node_to_index.items()}

print('Start peredicting')
with open('test.csv') as f:
    for line in f:
        line = line.strip().split(',')
        node_to_index = [node_to_index[node] for node in line]
        with torch.no_grad():
            for i in range(97):
                examples = preprocessing.convert_cascade_to_examples(node_to_index,G=G,inference=True)
                seq_matrix_test, seq_masks_matrix_test, topo_masks_tensor_test = prepare_test_data(examples)

                seq_matrix_test = torch.tensor(seq_matrix_test,dtype=torch.long).cuda()
                seq_masks_matrix_test = torch.tensor(seq_masks_matrix_test,dtype=torch.float).cuda()
                topo_masks_tensor_test = torch.tensor(topo_masks_tensor_test,dtype=torch.float).cuda()
                final_layer = model(seq_matrix_test,seq_masks_matrix_test,topo_masks_tensor_test)
                final_layer = final_layer.cpu().detach().numpy()
                prob = select_certain_node_softmax(final_layer,node_to_index)
                predict_node = np.argmax(prob,axis=1)[-1]
                node_to_index.append(predict_node)
            final_answer = [index_to_node[index] for index in node_to_index]
            with open('answer.csv','a+') as f:
                final_answer = ','.join(final_answer)
                f.write(final_answer)
                f.write('\n')