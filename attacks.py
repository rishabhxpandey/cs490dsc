import torch 
import torch.nn.functional as F
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
import copy



def fgsm_attack(image, epsilon, data_grad):
    """
    Author: Sai Coumar
    Description: Perturbs an image using the Fast Gradient Sign Method attack
    Attack Type: White Box

    Parameters:
    - image: The pytorch tensor of an image to perturb
    - epsilon: A constant used to control the magnitude of perturbation

    Returns:
    - Perturbed image

    Literature:
    - https://arxiv.org/abs/1412.6572
    """
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Clip values to 0-255 to maintain pixel value range
    torch.clamp(perturbed_image, 0, 255)
    # Return the perturbed image
    return perturbed_image

def deepfool_attack(image, model, overshoot=0.02, max_iterations=50):  
    """
    Author: Sai Coumar
    Description: Perturbs an image using the DeepFool attack
    Attack Type: White Box

    Parameters:
    - image: The pytorch tensor of an image to perturb
    - model: The classifier model to attack
    - overshoot: Hyperparameter to edge the perturbation past the minimal amount needed 
    to perturb the image just to be safe the perturbation crosses the decision boundary
    - max_iterations: Hyperparameter to limit resources to finite value 

    Returns:
    - Perturbed image

    Literature:
    - https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Moosavi-Dezfooli_DeepFool_A_Simple_CVPR_2016_paper.pdf
    """
    # Copy the image data as an object to preserve gradient. This will be perturbed 
    # rather than the original image data
    x = copy.deepcopy(image)

    # Get the actual predictions from the model
    output_actual, _ = model(x)
    # Probability of the label's classification
    # print(output_actual)
    

    # Actual label value (number)
    _, label_actual = torch.max(output_actual.data, 1)
    label_probability = output_actual[0][label_actual].item()
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

def pgd_attack(image, model, init_pred, epsilon, alpha=2,  max_iterations=50):
    """
    Author: Sai Coumar
    Description: Perturbs an image using the Projected Gradient Descent attack
    Attack Type: White Box

    Parameters:
    - image: The pytorch tensor of an image to perturb
    - model: The classifier model to attack
    - init_pred: True classified label given by the model before any attacks
    - epsilon: A hyperparameter that defines the epsilon-ball threshold that the perturbed image
    must stay confined to in order to retain percievability
    - alpha: Step size hyperparameter to control the magnitude of perturbation
    - max_iterations: Hyperparameter to limit resources to finite value  

    Returns:
    - Perturbed image

    Literature:
    - https://arxiv.org/abs/1412.6572
    """

    # Note: other examples may use alpha and epsilon divided by 255 
    # because they normalize pixels to 1.
    # Since normalization wasn't used we use integers 
    perturbed_image = image
    output_final = None
    for _ in range(max_iterations):
        # Predict on perturbed image
        output, _ = model(perturbed_image)
        output_final = output
        # _, pred = torch.max(output.data, 1)

        # Compare loss of true prediction vs outputted prediction
        loss = F.cross_entropy(output, init_pred)
        model.zero_grad()
        loss.backward() 

        # Extract gradient
        sign_data_grad = image.grad.data.sign()
        # print(sign_data_grad.size())

        # Perturb image
        perturbed_image = image  + alpha * sign_data_grad
        
        # Clipping to epislon ball
        eta = torch.clamp(perturbed_image - image, min=-epsilon, max=epsilon)
        perturbed_image = torch.clamp(perturbed_image + eta, min=0, max=255)

    return output_final, perturbed_image

def nes_attack(image, model, init_pred, init_labels, epsilon, alpha=2,  max_iterations=1):
    """
    Author: Sai Coumar
    Description: Perturbs an image using the Natural Evolution Strategies (Finite Differences Variant) attack
    Attack Type: Score-Based Black Box

    Parameters:
    - image: The pytorch tensor of an image to perturb
    - model: The classifier model to attack
    - init_pred: True classified label given by the model before any attacks
    - epsilon: A hyperparameter that defines the epsilon-ball threshold that the perturbed image
    must stay confined to in order to retain percievability
    - alpha: Step size hyperparameter to control the magnitude of perturbation
    - max_iterations: Hyperparameter to limit resources to finite value  

    Returns:
    - Perturbed image

    Literature:
    - https://arxiv.org/pdf/1804.08598.pdf
    """
    def create_multivariate_gaussian(mean, cov_matrix):
        # Create a MultivariateNormal distribution with the specified mean and covariance matrix
        mv_normal = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov_matrix)
        # Sample from the distribution
        sample = mv_normal.sample()
        return sample
        
    def nes_estimation(sign_data_grad_actual, model, label_actual, sigma, n, image):
        N = image.size()[2]

        g = torch.zeros(1, 1, N, N)
        g = g.to(device)
        # print(g.size())
        for _ in range(n):
            
            ui = create_multivariate_gaussian(torch.zeros(N), torch.eye(N)).unsqueeze(0).unsqueeze(0)
            ui = ui.to(device)

            p_plus = model(image + sigma * ui)[0][0][label_actual].item()
            p_minus = model(image - sigma * ui)[0][0][label_actual].item()

            g += p_plus * ui
            g -= p_minus * ui
            
            print("diff: ", torch.norm((sign_data_grad_actual - g).view(-1)).item(), " ,prob +: ", p_plus, " ,prob -: ", p_minus)
          
        return (1 / (2 * n * sigma)) * g

    
    # # Note: other examples may use alpha and epsilon divided by 255 
    # # because they normalize pixels to 1.
    # # Since normalization wasn't used we use integers 
    perturbed_image = image
    output_final = None
    

    # max_iterations = 1
    for _ in range(max_iterations):
        # Predict on perturbed image
        output, _ = model(perturbed_image)
        output_final = output

        _, label_actual = torch.max(output.data, 1)
        label_probability = output[0][init_pred].item()
        # print(label_probability)
        loss = F.cross_entropy(output, init_pred)
        model.zero_grad()
        loss.backward() 

        # Extract gradient
        sign_data_grad_actual = image.grad.data.sign()
        # Extract gradient
        sign_data_grad = nes_estimation(sign_data_grad_actual, model, init_pred, sigma=0.001, n=100, image = perturbed_image)
        sign_data_grad = sign_data_grad.to(device)
        # print(torch.norm((sign_data_grad_actual - sign_data_grad).view(-1)))
        print(label_actual[0].item(), init_pred[0].item(), label_probability, " ,diff: ", torch.norm((sign_data_grad_actual - sign_data_grad).view(-1)).item())

        # Perturb image
        perturbed_image = image + alpha * sign_data_grad
        
        # Clipping to epislon ball
        eta = torch.clamp(perturbed_image - image, min=-epsilon, max=epsilon)
        perturbed_image = torch.clamp(perturbed_image + eta, min=0, max=255)
        # print(perturbed_image)

    return output_final, perturbed_image
    
    # def nes_estimation(sign_data_grad_actual, model, label_actual, labels, sigma, n, image):
    #     N = image.size()[2]
    #     grads = []
    #     final_losses = []


    #     g = torch.zeros(1, 1, N, N)
    #     g = g.to(device)
    #     # print(g.size())
    #     for _ in range(n):
            
    #         ui = create_multivariate_gaussian(torch.zeros(N), torch.eye(N)).unsqueeze(0).unsqueeze(0)
    #         ui = ui.to(device)
    #         # print(model(image + sigma * ui)[0][0])
    #         print(ui.size())
    #         noise = torch.cat([ui, -ui], dim=0)
    #         print(noise.size())
    #         eval_points = image + sigma * noise
    #         output, _ = model(eval_points)
    #         print(output, labels)
    #         loss = F.cross_entropy(output, labels)
    #         print(loss)
    #         losses_tiled = loss.view(1, 1, 1, 1).expand(1,1,N,N)
    #         print(losses_tiled)
    #         grads.append(losses_tiled * noise)
    #         # g = (grad_plus - grad_minus) * ui
    #         # grads.append(g)
    #         # print("diff: ", torch.norm((sign_data_grad_actual - g).view(-1)).item(), " ,prob +: ", grad_plus, " ,prob -: ", grad_minus)
            
    #     g = torch.sum(torch.stack(grads), dim=0)
    #     return (1 / (2 * n * sigma)) * g

    # perturbed_image = image
    # output_final = None
    # for _ in range(max_iterations):
    #     # Predict on perturbed image
    #     output, _ = model(perturbed_image)
    #     output_final = output
    #     # _, pred = torch.max(output.data, 1)

    #     # Compare loss of true prediction vs outputted prediction
        
    #     # Extract gradient
    #     sign_data_grad = nes_estimation(model, init_pred, sigma=0.001, n=50, image = perturbed_image)
    #     # print(sign_data_grad.size())

    #     # Perturb image
    #     perturbed_image = image  + alpha * sign_data_grad
        
    #     # Clipping to epislon ball
    #     eta = torch.clamp(perturbed_image - image, min=-epsilon, max=epsilon)
    #     perturbed_image = torch.clamp(perturbed_image + eta, min=0, max=255)

    # return output_final, perturbed_image