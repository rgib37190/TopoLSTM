# TopoLSTM
Offer Pytorch(1.1.0) implementation

Our data preprocessing and topolstm structure refer to 
[this repositary](https://github.com/vwz/topolstm)

## How to Use ?

##### execute the following command to start training.
    
    python train.py --data_dir '' --max_len 30 --batch_size 256

##### You can set parameter you want to tune your model(training)
    --data_dir : your data directory
    --max_len  : max diffusion path lengths
    --batch_size : batch_size
    --hidden_size : your embedding size
    --epochs : training epochs
    --learning_rate : optimizer learning rate

##### execute the following command to start testing.

    python test.py --data_dir ''

##### You can set parameter you want to tune your model(testing)
    --data_dir : your data directory
    
## File Desription
    preprocessing.py : preprocessing model need inputs
    model.py : TopoLSTM implementation
    train.py : Training TopoLSTM
    test.py : Predict our test data

## Embedding Data

Using node2vec to get embedding:
[data link](https://drive.google.com/drive/folders/1HeutDaYU9XiZov-wEPCagX_xZCHHxLz5?usp=sharing)

## Citing
    @inproceedings{WangZLC17,
      author    = {Jia Wang and
               Vincent W. Zheng and
               Zemin Liu and
               Kevin Chen-Chuan Chang},
      title     = {Topological Recurrent Neural Network for Diffusion Prediction},
      booktitle = {ICDM},
      pages     = {475--484},
      year      = {2017}
    }
 
