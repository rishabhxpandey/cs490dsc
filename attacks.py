import torch 
import torch.nn.functional as F
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
import copy


# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Clip values to 0-255 to maintain pixel value range
    torch.clamp(perturbed_image, 0, 255)
    # Return the perturbed image
    return perturbed_image

def deepfool_attack(image, model, overshoot=0.2, max_iterations=50):  
    # Copy the image data as an object to preserve gradient. This will be perturbed 
    # rather than the original image data
    x = copy.deepcopy(image)

    # Get the actual predictions from the model
    output_actual, _ = model(x)
    # Probability of the label's classification
    label_probability = output_actual[0][7].item()

    # Actual label value (number)
    _, label_actual = torch.max(output_actual.data, 1)

    # Store reference of original image
    original_img = copy.deepcopy(image)
    # Initialize weights w
    w = torch.zeros_like(image)
    # Initialize perturbations
    r_total = torch.zeros_like(image)

    # Variables to limit perturbation. Perturbation ends after a fixed number of iterations
    # or when the poisoned images' predicted label no longer matches the clean images' predicted label
    iter = 0
    k_i = label_actual

    while k_i == label_actual and iter < max_iterations:
        iter += 1
        
        # Argmin l
        l = float('inf')
        output_actual[:, label_actual].backward(retain_graph=True)
        # Extract Gradient of Image w.r.t. true prediction
        grad_original = x.grad.clone() 

        for k, class_prob in enumerate(output_actual.squeeze()):
            # Check all other class predictions
            if k != label_actual.item():
                # Extract Gradient of Image w.r.t. every other class except predicted
                model.zero_grad()
                output_actual[:, k].backward(retain_graph=True)
                x.retain_grad()
                curr_grad = x.grad.clone()
                # calculate l for the current class
                w_k = curr_grad - grad_original
                f_k = class_prob.item() - label_probability

                l_k = abs(f_k)/ torch.norm(w_k.view(-1))
                # pick minimum l
                if l_k < l:
                    l = l_k
                    w = w_k

        # Calculate the perturbation
        r_i = (1 + 1e-4) * w / torch.norm(w.view(-1))
        r_total += r_i

        # Perturb image
        x = original_img + (1+overshoot)*r_total
        x.retain_grad()
        output_actual, _ = model(x)
        _, k_i = torch.max(output_actual.data, 1)

    return x, k_i, r_total, iter

