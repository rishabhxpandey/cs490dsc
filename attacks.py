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

def deepfool_attack(image, model, overshoot=0.2, max_iterations=10):  
    x = copy.deepcopy(image)

    output_actual, _ = model(x)
    label_probability = output_actual[0][7].item()
    print(f"Original Label: {label_probability}")

    _, label_actual = torch.max(output_actual.data, 1)


    original_img = copy.deepcopy(image)
    w = torch.zeros_like(image)
    r_total = torch.zeros_like(image)

    iter = 0
    k_i = label_actual

    while k_i == label_actual and iter < max_iterations:
        iter += 1
        print(f"Iteration: {iter}")
        l = float('inf')
        output_actual[:, label_actual].backward(retain_graph=True)

        grad_original = x.grad.clone()        
        move_towards = label_actual

        for k, class_prob in enumerate(output_actual.squeeze()):
            print("Index:", k, "Class Probability:", class_prob.item())
            if k != label_actual.item():
                model.zero_grad()
                output_actual[:, k].backward(retain_graph=True)
                x.retain_grad()
                curr_grad = x.grad.clone()
                w_k = grad_original - curr_grad
                f_k = class_prob.item() - label_probability

                l_k = abs(f_k)/ torch.norm(w_k.view(-1))

                # print(l_k)
                if l_k < l:
                    move_towards = k
                    l = l_k
                    w = w_k

        print(f"Chose Index: {move_towards}")    
        print(torch.norm(w.view(-1)))
        r_i = (1 + 1e-4) * w / torch.norm(w.view(-1))
        r_total += r_i

        x = original_img + (1+overshoot)*r_total
        x.retain_grad()
        output_actual, _ = model(x)
        _, k_i = torch.max(output_actual.data, 1)
    print(r_total)
    print(x - original_img)
    print(f"New output: {k_i}")
    return x

# print(label_probabilities[label_actual].backward(retain_graph=True))
        # print("GRADEITN")
        # print(label_probabilities[label_actual].grad)
        # print(grad_original)