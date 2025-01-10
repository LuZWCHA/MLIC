import torch

def extract_model_state_dict(checkpoint_path, output_path):
    checkpoint = torch.load(checkpoint_path)
    model_state_dict = checkpoint['state_dict']
    for key in list(model_state_dict.keys()):
        if key.startswith('module.'):
            model_state_dict[key[7:]] = model_state_dict.pop(key)
            
        # Remove encoder: h_a, g_a
        if "h_a" in key:
            model_state_dict.pop(key)
            print(key)
        if "g_a" in key:
            model_state_dict.pop(key)
            print(key)
            
    torch.save({"state_dict": model_state_dict}, output_path)
    print(f"Model state dict saved to {output_path}")
    
    
if __name__ == "__main__":
    extract_model_state_dict("/nasdata2/private/zwlu/compress/naic2024/MLIC/MLIC++/playground/experiments/mlicpp_small_dec_mse_q1_finetune_ddp/checkpoints/checkpoint_best_loss.pth.tar", "model_submit.pth")