import torch
import sys
import os

# Add current dir to path so it can find bertVAE
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import transformers.models.bert.modeling_bert as modeling_bert
    if not hasattr(modeling_bert, 'BertSdpaSelfAttention'):
        print("Patching BertSdpaSelfAttention into transformers...")
        modeling_bert.BertSdpaSelfAttention = modeling_bert.BertSelfAttention
except ImportError:
    pass

def fix_model(model_path, output_path):
    print(f"Loading {model_path}...")
    # Load the whole object
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Save just the state dict and architecture params if possible, 
    # but the user's score_email_csv.py expects the full object.
    # To keep it compatible with their script, we save it back as is, 
    # but now it's 'fixed' in memory. However, the pickle still contains the reference.
    
    # Instead, let's just save the state_dict to a separate file 
    # and we will modify score_email_csv.py to use state_dict loading.
    state_dict_path = model_path.replace('.pkl', '_state_dict.pt')
    torch.save(model.state_dict(), state_dict_path)
    print(f"Saved state dict to {state_dict_path}")

if __name__ == "__main__":
    smoke_model = "/scratch3/che489/FC-W2-SoK/ssc_LLM_phishing_survey/Visualization/persuasion_strategy_wvae/output/cialdini_wvae_smoke/model.pkl"
    if os.path.exists(smoke_model):
        fix_model(smoke_model, smoke_model)
