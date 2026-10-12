import torch
import cv2
import json
import os
import numpy as np
from PIL import Image
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def run_batch_test():
    model = create_model('./models/cldm_v15.yaml').cuda()
    sd = load_state_dict('/media/Storage3/rhw/gen/Siamese-Diffusionmain/outputs/merged_pytorch_model.pth', location='cuda:0')
    sd = {k: v for k, v in sd.items() if not k.startswith('model_ema')}
    model.load_state_dict(sd, strict=False)
    sampler = DDIMSampler(model)

    with open('/media/Storage3/rhw/data/data_origan/consep/test/dataset_prompts_5class.json', 'r') as f:
        prompts_dict = json.load(f)

    for file_name, prompt in prompts_dict.items():
        mask_path = os.path.join('/media/Storage3/rhw/data/data_origan/consep/test/masks', file_name)
        
        source = cv2.imread(mask_path)
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        source = cv2.resize(source, (256, 256)) 
        source = source.astype(np.float32) / 255.0
        
        control = torch.from_numpy(source).cuda().permute(2, 0, 1).unsqueeze(0)
        
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt])]}
        un_cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([""])]}
        

        shape = (4, 256 // 8, 256 // 8)
        samples, _ = sampler.sample(30, 1, shape, cond, unconditional_guidance_scale=6.5, unconditional_conditioning=un_cond)
        

        x_samples = model.decode_first_stage(samples)
        x_samples = (torch.clamp((x_samples + 1.0) / 2.0, 0.0, 1.0) * 255).cpu().numpy().transpose(0, 2, 3, 1)
        Image.fromarray(x_samples[0].astype(np.uint8)).save(f"./results_consep1/gen_{file_name}")

if __name__ == "__main__":
    run_batch_test()