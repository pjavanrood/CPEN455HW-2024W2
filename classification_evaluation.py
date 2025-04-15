'''
This code is used to evaluate the classification accuracy of the trained model.
You should at least guarantee this code can run without any error on validation set.
And whether this code can run is the most important factor for grading.
We provide the remaining code, all you should do are, and you can't modify other code:
1. Replace the random classifier with your trained model.(line 69-72)
2. modify the get_label function to get the predicted label.(line 23-29)(just like Leetcode solutions, the args of the function can't be changed)

REQUIREMENTS:
- You should save your model to the path 'models/conditional_pixelcnn.pth'
- You should Print the accuracy of the model on validation set, when we evaluate your code, we will use test set to evaluate the accuracy
'''
from torchvision import datasets, transforms
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
import csv
import os
import pandas as pd
import logging
import torch
NUM_CLASSES = len(my_bidict)

#TODO: Begin of your code
def get_label(model, model_input, device):
    model_output = model(model_input)
    return torch.argmax(model_output, dim=1)
    # Write your code here, replace the random classifier with your trained model
    # and return the predicted label, which is a tensor of shape (batch_size,)
    
    batch_size = model_input.shape[0]
    log_likelihoods = torch.zeros((batch_size, len(my_bidict)), device=model_input.device)
    for class_label, i in my_bidict.items():
      # breakpoint()
      model_output = model(model_input, [class_label])
      loss = discretized_mix_logistic_loss_batch(model_input, model_output)
      log_likelihoods[:, i] = loss
    
    return torch.argmin(log_likelihoods, dim=1)
# End of your code

def classifier(model, data_loader, device):
    model.eval()
    acc_tracker = ratio_tracker()
    for batch_idx, item in enumerate(tqdm(data_loader)):
        model_input, categories = item
        model_input = model_input.to(device)
        original_label = [my_bidict[item] if item in my_bidict else 4 for item in categories]
        original_label = torch.tensor(original_label, dtype=torch.int64).to(device)
        answer = get_label(model, model_input, device)
        correct_num = torch.sum(answer == original_label)
        acc_tracker.update(correct_num.item(), model_input.shape[0])
    
    return acc_tracker.get_ratio()


def predict_classification(model, data_loader, device):
    model.eval()
    all_answers = []
    for batch_idx, item in enumerate(tqdm(data_loader)):
        model_input, categories = item
        model_input = model_input.to(device)
        answer = get_label(model, model_input, device)
        all_answers.extend(answer.tolist())
    
    return all_answers
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=32, help='Batch size for inference')
    parser.add_argument('-m', '--mode', type=str,
                        default='validation', help='Mode for the dataset')
    
    args = parser.parse_args()
    pprint(args.__dict__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers':0, 'pin_memory':True, 'drop_last':False}

    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    dataloader = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                            mode = args.mode, 
                                                            transform=ds_transforms), 
                                             batch_size=args.batch_size, 
                                             shuffle=False, 
                                             **kwargs)

    #TODO:Begin of your code
    # logging.basicConfig(level=logging.DEBUG)
    # logger = logging.getLogger()
    # logger.setLevel(logging.DEBUG)
    #You should replace the random classifier with your trained model
    # model_path = os.path.join(os.path.dirname(__file__), 'models/pcnn_cpen455_from_scratch_29.pth')
    # model = ClassifierWrapper(model_path, NUM_CLASSES)
    # model = PixelCNN(nr_resnet=5, nr_filters=80,
    #         input_channels=3, nr_logistic_mix=5)
    # model.load_state_dict(torch.load('models_backup_f40_l5_r5/pcnn_cpen455_f80_l5_r5.pth'))
    # model = model.to(device)
    # model = model.eval()


    model = PixelCNNClassifier(nr_resnet=5, nr_filters=80,
                input_channels=3, nr_logistic_mix=5)
    model.load_state_dict(torch.load('models/model_Classifier_f80_l5_r5_scratch_119.pth'))
    #End of your code
    
    model = model.to(device)
    #Attention: the path of the model is fixed to './models/conditional_pixelcnn.pth'
    #You should save your model to this path
    # model_path = os.path.join(os.path.dirname(__file__), 'models/conditional_pixelcnn.pth')
    # if os.path.exists(model_path):
    #     model.load_state_dict(torch.load(model_path))
    #     print('model parameters loaded')
    # else:
    #     raise FileNotFoundError(f"Model file not found at {model_path}")
    model.eval()
    
    # acc = classifier(model = model, data_loader = dataloader, device = device)
    # print(f"Accuracy: {acc}")
    prediction = predict_classification(model = model, data_loader = dataloader, device = device)

    # # print(args.mode)

    test_df = pd.read_csv("test.csv", header=None)
    test_df[1] = prediction

    test_df.to_csv('test_pred_model_Classifier_f80_l5_r5_scratch_119.csv', header=None, index=False)

    print(prediction)

        
        