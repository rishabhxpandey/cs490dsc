from model_architectures import *
from attacks import * 
import csv

class Curator():
    def store_data(self, filename,data):
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