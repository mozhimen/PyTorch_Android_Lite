from model import Model
import torch

def main():
    # Create the model
    model = Model()
    
    # Create 4 random noise vectors
    # meaning we want 4 random numbers
    X = torch.distributions.uniform.Uniform(-10000,\
        10000).sample((4, 2))
    
    # Send the noise vectors through the model
    # to get the argmax outputs
    outputs = model(X)
    
    # Print the outputs
    for o in outputs:
        print(f"{o.item()} ")
    
    # Save the model to a file named model.pkl
    torch.save(model.state_dict(), "model.pkl")

if __name__=="__main__":
    main()