from model_architectures import *
from attacks import * 
import csv
from multiprocessing import Pool

class Curator():
    def store_data(self, filename, data, color = False):
        if color:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Write header row
                header = ['IntegerValue'] + [f'R_Pixel_{i}' for i in range(32*32)]+ [f'G_Pixel_{i}' for i in range(32*32)]+ [f'B_Pixel_{i}' for i in range(32*32)]
                writer.writerow(header)

                # Write data rows
                for tuple_data in data:
                    integer_value, _, array = tuple_data
                    flattened_images = array.reshape(array.shape[0], -1)

                    # Concatenate the color channels
                    concatenated_channels = np.concatenate(flattened_images, axis=0)
                    row = [integer_value] + list(concatenated_channels)
                    writer.writerow(row)
        else:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # Write header row
                header = ['IntegerValue'] + [f'Pixel_{i}' for i in range(28*28)]
                writer.writerow(header)

                # Write data rows
                for tuple_data in data:
                    integer_value, _, array = tuple_data
                    flattened_array = array.flatten()
                    row = [integer_value] + list(flattened_array)
                    
                    writer.writerow(row)

    def curate_fgsm(self, model, test_loader, epsilon):
        correct = 0
        total = 0

        adv_examples = []
        batch = 0
        
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            batch += 1
            for image, label in zip(images, labels):
                image = image.unsqueeze(0)
                label = label.unsqueeze(0)
                image.requires_grad = True
                output, _ = model(image)

                _, init_pred = torch.max(output.data, 1)

                if not torch.equal(init_pred, label):
                    total +=1 
                    continue
                
                loss = F.nll_loss(output, label)
                model.zero_grad()
                loss.backward()
                data_grad = image.grad.data
                perturbed_data = fgsm_attack(image, epsilon, data_grad)

                output_final, _ = model(perturbed_data)
                _, final_pred = torch.max(output_final.data, 1)
                if torch.equal(final_pred, label):
                    correct += 1
                else: 
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
                total +=1 
            print(f'Batch {batch} Completed. # of Adversarial Examples: {len(adv_examples)}')
            
        accuracy = correct / total
        return accuracy, adv_examples
    def curate_pgd(self, model, test_loader, epsilon, alpha):
        correct = 0
        total = 0

        adv_examples = []
        batch = 0
        
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            batch += 1
            # print(f"Batch: {batch}, Epsilon: {epsilon}, Correct: {correct}")
            for image, label in zip(images, labels):
                image = image.unsqueeze(0)
                label = label.unsqueeze(0)
                image.requires_grad = True
                output, _ = model(image)

                _, init_pred = torch.max(output.data, 1)

                if not torch.equal(init_pred, label):
                    total +=1 
                    continue
                
                

                output_final, perturbed_data = pgd_attack(image, model, init_pred, epsilon, alpha)
                _, final_pred = torch.max(output_final.data, 1)
                if torch.equal(final_pred, label):
                    correct += 1
                else:
                    # Save some adv examples for visualization later
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
                # print(f"{correct}/{total}")
                total +=1
            
            print(f'Batch {batch} Completed. # of Adversarial Examples: {len(adv_examples)}') 
                # break
            # break

        accuracy = correct / total
        # print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {total} = {accuracy}")
        return accuracy, adv_examples
    
    def curate_deepfool(self, model, test_loader, overshoot=0.02):
        correct = 0
        total = 0

        adv_examples = []
        batch = 0
        
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            batch += 1
            # print(f"Batch: {batch}, Correct: {correct}")
            for image, label in zip(images, labels):
                # print("image number: ",total)
                image = image.unsqueeze(0)
                label = label.unsqueeze(0)
                image.requires_grad = True
                output, _ = model(image)

                # print(outputs)

                _, init_pred = torch.max(output.data, 1)

                if not torch.equal(init_pred, label):
                    total +=1 
                    continue
                
                perturbed_image, final_pred, r_total, iter = deepfool_attack(image, model, overshoot=0.02, max_iterations=100)
                # print(f"Perturbed Iteration: {iter}")
                if torch.equal(final_pred, label):
                    correct += 1
                else:
                    adv_ex = perturbed_image.squeeze().detach().cpu().numpy()
                    adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
                total +=1 
            print(f'Batch {batch} Completed. # of Adversarial Examples: {len(adv_examples)}')    

        accuracy = correct / total
        # print(f"Test Accuracy = {correct} / {total} = {accuracy}")
        return accuracy, adv_examples

    def curate_jsma(self, model, test_loader, theta = 0.1):
        model.eval()

        pred_list = []
        correct = 0
        total = 0
        adv_examples = []
        batch = 0

        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            batch += 1
            for image, label in zip(images, labels):
                image = image.unsqueeze(0)
                label = label.unsqueeze(0)

                output, _ = model(image)

                # print(outputs)

                _, init_pred = torch.max(output.data, 1)
                
                if not torch.equal(init_pred, label):
                    total +=1 
                    continue 
                
                target = torch.topk(output,2).indices[0][1].item()
                advimages = jsma_attack(model, image, target, 10, theta = theta)
                output_adv, _ = model(advimages)
                
                _, prediction_adv = torch.max(output_adv.data, 1)

                if torch.equal(prediction_adv, label):
                    correct += 1
                else:
                    adv_ex = advimages.squeeze().detach().cpu().numpy()

                    adv_examples.append( (init_pred.item(), prediction_adv.item(), adv_ex) )
                    pred_list.append(prediction_adv)
                        
                            
                total +=1 
                # print(correct, "/", total)  
                # if (total>0):
                #     break
            print(f'Batch {batch} Completed. # of Adversarial Examples: {len(adv_examples)}')    
        # print('Accuracy of test text: %f %%' % ((float(correct) / total) * 100))
        accuracy = correct / total
        return accuracy, adv_examples
    def curate_cw(self, model, test_loader, targeted=False, target_label=0, c=0.75, alpha=0.01, kappa= 0, max_iterations=50, mnist = 1):
        model.eval()

        pred_list = []
        correct = 0
        total = 0
        adv_examples = []
        batch = 0
        
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            batch += 1
            for image, label in zip(images, labels):
                image = image.unsqueeze(0)
                label = label.unsqueeze(0)

                output, _ = model(image)

                # print(outputs)

                _, init_pred = torch.max(output.data, 1)
                

                
                advimages = cw_attack(image, model, init_pred, targeted, target_label, c, alpha, kappa, max_iterations)
                output_adv, _ = model(advimages)
                # advimages = advimages[0,:,:,:] / 255

                _, prediction_adv = torch.max(output_adv.data, 1)

                if torch.equal(prediction_adv, label):
                    correct += 1
                else:
                    # Save some adv examples for visualization later
                
                    if not targeted: 
                        #if prediction_adv not in pred_list:
                        adv_ex = advimages.squeeze().detach().cpu().numpy()
                        
                        #if not mnist:
                            #adv_ex = adv_ex/255
                        
                        adv_examples.append( (init_pred.item(), prediction_adv.item(), adv_ex) )
                        pred_list.append(prediction_adv)
                    else:
                        adv_ex = advimages.squeeze().detach().cpu().numpy()
                        # print(init_pred.item())
                        adv_examples.append( (init_pred.item(), prediction_adv.item(), adv_ex) )
                    
                            
                total +=1 
                # print(correct, "/", total)  

                # if total-correct > 100:
                #     break
        
        accuracy = correct / total
            
                
        print(f'Batch {batch} Completed. # of Adversarial Examples: {len(adv_examples)}')    
        # print('Accuracy of test text: %f %%' % ((float(correct) / total) * 100))
        return accuracy, adv_examples