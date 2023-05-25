from torch.utils.mobile_optimizer import optimize_for_mobile
import torch
from model import Model

def optimizeSave():
    # Load in the model
    model = Model()
    model.load_state_dict(torch.load("model.pkl", \
        map_location=torch.device("cpu")))
    model.eval() # Put the model in inference mode
    
    # Generate some random noise
    X = torch.distributions.uniform.Uniform(-10000, \
        10000).sample((4, 2))
    
    # Generate the optimized model
    traced_script_module = torch.jit.trace(model, X)
    traced_script_module_optimized = optimize_for_mobile(\
        traced_script_module)
    
    # Save the optimzied model
    traced_script_module_optimized._save_for_lite_interpreter(\
        "model.pt")
    
if __name__=="__main__":
    optimizeSave()