# example usage
from captum.attr import IntegratedGradients


test_dataset = YourDataset(test_data, test_labels)  # Replace with your actual test dataset preparation
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # Batch size can be 1 for detailed analysis

dataiter = iter(test_loader)
inputs, labels = next(dataiter)


def attribute_image_features(model, inputs, target=None):
    model.eval()
    inputs.requires_grad = True
    
    ig = IntegratedGradients(model)
    # If target is None, IG will compute attributions for the most likely class
    # Otherwise, specify the target class as an integer
    target = labels if target is None else target
    attributions, delta = ig.attribute(inputs=inputs, target=target, return_convergence_delta=True)
    
    return attributions

# Example usage with a single test example
attributions = attribute_image_features(best_clf_model, inputs, target=labels.item())

# Process and analyze your attributions here
