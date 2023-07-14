import numpy as np
import torch
from torch.autograd import Variable

from models.roll_to_midi import Generator
from models.make_wav import MIDISynth

def video_to_roll_inference(model, video_info, frames_with5):
    min_key, max_key = 15, 65
    threshold = 0.4

    batch_size = 32
    preds_roll, preds_logit = [], []
    for idx in range(0, len(frames_with5), batch_size):
        batch_frames = torch.stack([frames_with5[i] for i in range(idx, min(len(frames_with5), idx+batch_size))])
        pred_logits = model(batch_frames)
        pred_roll = torch.sigmoid(pred_logits) >= threshold   
        numpy_pred_roll = pred_roll.cpu().detach().numpy().astype(np.int_)
        numpy_pred_logit = pred_logits.cpu().detach().numpy()
        
        for roll, logit in zip(numpy_pred_roll, numpy_pred_logit):
            preds_roll.append(roll)
            preds_logit.append(logit)

    preds_roll = np.asarray(preds_roll).squeeze()
    if preds_roll.shape[0] != video_info['video_select_frame']:
        temp = np.zeros((video_info['video_select_frame'], max_key-min_key+1))
        temp[:preds_roll.shape[0], :] = preds_roll[:video_info['video_select_frame'], :]
        preds_roll = temp
    
    roll = np.zeros((video_info['video_select_frame'], 88))
    roll[:, min_key:max_key+1] = preds_roll
    
    preds_logit = np.asarray(preds_logit).squeeze()
    if preds_logit.shape[0] != video_info['video_select_frame']:
        temp = np.zeros((video_info['video_select_frame'], max_key-min_key+1))
        temp[:preds_logit.shape[0], :] = preds_logit[:video_info['video_select_frame'], :]
        preds_logit = temp

    logit = np.zeros((video_info['video_select_frame'], 88))
    logit[:, min_key:max_key+1] = preds_logit
    
    wav, pm = MIDISynth(roll=roll, midi=None, frame=video_info['video_select_frame'], is_midi=False).process_roll()

    return roll, logit, wav, pm

def roll_to_midi_inference(_model, video_info, logit):
    min_key, max_key = 15, 65
    frame = 50
    input_shape = (1, max_key - min_key + 1, 2 * frame)
    
    model = Generator(input_shape).cuda()
    model.load_state_dict(_model['state_dict_G'])
    
    data = [torch.from_numpy(logit[i:i+50]) for i in range(0, len(logit), 50)]

    final_data = []    
    for i in range(0, len(data), 2):
        if i + 1 < len(data):
            one_roll = data[i]
            two_roll = data[i+1]
            final_roll = torch.cat([one_roll, two_roll], dim=0)
            final_data.append(final_roll)

    results = []
    for i, data in enumerate(final_data):
        roll = torch.unsqueeze(torch.unsqueeze(torch.sigmoid(data.T.float().cuda()), dim=0), dim=0)
        with torch.no_grad():
            model.eval()
            
            roll = roll.type(torch.cuda.FloatTensor)
            roll_ = Variable(roll)
            
            gen_img = model(roll_)
            gen_img = gen_img >= 0.5

            numpy_pre_label = gen_img.cpu().detach().numpy().astype(np.int_)
            numpy_pre_label = np.transpose(numpy_pre_label.squeeze(), (1, 0))

            results.append(numpy_pre_label[:frame, :])
            results.append(numpy_pre_label[frame:, :])
    
    midi = np.concatenate(results, axis=0)
    
    wav, pm = MIDISynth(roll=None, midi=midi, frame=midi.shape[0], is_midi=True).process_midi()
    
    return wav