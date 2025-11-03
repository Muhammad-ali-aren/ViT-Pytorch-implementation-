def print_parameters(model):
    print(sum([p.numel() for p in model.parameters() if p.requires_grad]))