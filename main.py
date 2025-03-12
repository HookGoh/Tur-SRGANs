''' @author: Wu
'''
from turSRGANs import *

# WIND - LR-MR
#-------------------------------------------------------------

data_type = 'tur'
data_path = 'data_in_example/train_HIT.tfrecord'
data_path1 = 'data_in_example/test_tbl.tfrecord'
model_path = None
r = [5]
mu_sig = None


if __name__ == '__main__':

    turSRGANs = turSRGANs(data_type=data_type, mu_sig=mu_sig)
    
    
    model_dir = turSRGANs.pretrain(r=r,
                                   data_path=data_path,
                                   model_path=model_path,
                                   batch_size=8)
    
    model_dir = turSRGANs.train(r=r,
                                data_path=data_path,
                                model_path=model_dir,
                                batch_size=8)
    
    turSRGANs.test(r=r,
                   data_path=data_path1,
                   model_path=model_dir,
                   batch_size=16,
                   plot_data=False)
    

